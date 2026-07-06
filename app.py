import streamlit as st
import sqlite3
import pandas as pd
import asyncio
import uuid
import os

from datetime import date
from typing import Optional

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.skills import models
from google.adk.tools import FunctionTool
from google.adk.tools.skill_toolset import SkillToolset
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService


# ============================================================
# PAGE CONFIG — must be the first Streamlit command in the script
# ============================================================
st.set_page_config(
    page_title="AgroScan AI Farm Manager",
    page_icon="🐔",
    layout="centered"
)

# ============================================================
# CONSTANTS
# ============================================================
DATABASE_NAME = "agroscan.db"
EXCEL_FILE = "agroscan_farm_records_july2025_june2026.xlsx"


# ============================================================
# DATABASE INITIALIZATION FUNCTIONS
# Defined at module level. create_farm_records_table() is
# idempotent and safe to call on every cold start. The Excel
# import only runs if the table is empty (see below), since the
# database lives at the CONTAINER level, shared across every
# visitor until the container restarts (Path A: data does not
# survive a redeploy/sleep cycle — known, accepted trade-off).
# ============================================================

def create_farm_records_table():
    connection = sqlite3.connect(DATABASE_NAME)
    cursor = connection.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS farm_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        record_date DATE UNIQUE NOT NULL,
        bird_count INTEGER NOT NULL,
        crates_collected INTEGER NOT NULL,
        feed_consumed_kg REAL NOT NULL,
        revenue REAL NOT NULL,
        expenses REAL NOT NULL,
        notes TEXT
    );
    """)
    connection.commit()
    connection.close()


def initialize_farm_record_book(excel_path):
    df = pd.read_excel(excel_path)
    connection = sqlite3.connect(DATABASE_NAME)
    cursor = connection.cursor()

    imported = 0
    skipped = 0

    for _, row in df.iterrows():
        try:
            cursor.execute("""
            INSERT INTO farm_records (
                record_date, bird_count, crates_collected,
                feed_consumed_kg, revenue, expenses, notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                str(row["record_date"]),
                int(row["bird_count"]),
                int(row["crates_collected"]),
                float(row["feed_consumed_kg"]),
                float(row["revenue"]),
                float(row["expenses"]),
                row["notes"] if pd.notna(row["notes"]) else None
            ))
            imported += 1
        except sqlite3.IntegrityError:
            skipped += 1

    connection.commit()
    connection.close()
    return imported, skipped


def get_record_count():
    connection = sqlite3.connect(DATABASE_NAME)
    cursor = connection.cursor()
    cursor.execute("SELECT COUNT(*) FROM farm_records")
    count = cursor.fetchone()[0]
    connection.close()
    return count


# Run database setup on every cold start (not per-session).
create_farm_records_table()

if get_record_count() == 0:
    _imported, _skipped = initialize_farm_record_book(EXCEL_FILE)
    st.session_state.import_summary = f"Imported {_imported} historical records."
else:
    st.session_state.setdefault("import_summary", None)


# ============================================================
# ONE-TIME AGENT/RUNNER SETUP
# Everything inside this block builds the Skills, Tools, Agent,
# and Runner — but only runs ONCE per browser session. On every
# subsequent script rerun (i.e. every message sent), this block
# is skipped, and the already-built objects are pulled from
# st.session_state instead.
# ============================================================

if "initialized" not in st.session_state:
    st.session_state.initialized = False

if not st.session_state.initialized:

    # --------------------------------------------------------
    # SKILL 1 — Farm Manager Core
    # --------------------------------------------------------
    farm_manager_core_skill = models.Skill(

        frontmatter=models.Frontmatter(
            name="farm-manager-core",
            description=(
                "Defines AgroScan AI Farm Manager's identity, "
                "communication style and overall user experience."
            ),
        ),

        instructions="""
You are AgroScan AI Farm Manager.

You are the intelligent virtual manager of a poultry farm.

Your responsibility is to coordinate AgroScan's capabilities
to help farmers manage their farms through natural conversation.

Communication Style

• Friendly
• Professional
• Practical
• Clear
• Confident

Never mention:

• Skills
• Tools
• Tool calls
• Internal reasoning
• System architecture

Remain in character as AgroScan AI Farm Manager.

For greetings:

Introduce yourself warmly and briefly explain how you can help.

For casual conversation:

Respond naturally without referring to yourself as
an AI model, language model or ChatGPT.

If the farmer requests a capability that AgroScan does not
yet support, politely explain that it will be available in
a future version.

Never invent farm records or agricultural information.
""",

        resources=models.Resources(
            references={
                "identity.md": """
# AgroScan AI Farm Manager

AgroScan is an intelligent poultry farm management system.

Its goal is to help poultry farmers through natural conversation,
while internally coordinating multiple specialised capabilities.
"""
            }
        )
    )

    # --------------------------------------------------------
    # SKILL 2 — Farm Record Management
    # --------------------------------------------------------
    farm_record_skill = models.Skill(

        frontmatter=models.Frontmatter(
            name="farm-record-management",
            description=(
                "Records and manages daily poultry farm production "
                "records and historical farm data."
            ),
        ),

        instructions="""
You are AgroScan's Farm Record Specialist.

Your responsibility is maintaining the farm record book.

RECORDING DATA

When the farmer provides daily production information, call the
record_daily_farm_data tool. Only include the fields the farmer
actually mentioned — omit anything they didn't state.

The tool's result includes an "action" field ("recorded" or
"updated") and a "previous_values" field. When reporting back:

• If action is "recorded", clearly state a new record was created.
• If action is "updated", explicitly name which field(s) changed,
  comparing "previous_values" to the new values. Do not just repeat
  the full record — call out what is actually different.

Missing values inherit from today's own existing record if one
exists, otherwise from the most recent prior record.

LOOKING UP A SINGLE RECORD

You have two distinct tools for retrieving past data:

• get_farm_record(record_date) — use this for a SPECIFIC date,
  including relative terms like "yesterday", "last Tuesday", or
  "the 3rd of July" once you have converted them into an exact
  YYYY-MM-DD date. This is an EXACT match only. If it reports
  found=False, tell the farmer honestly that no record exists for
  that exact date. Never substitute a different date's data.

• get_most_recent_farm_record() — use this when the farmer asks for
  their "last" or "most recent" record without naming a specific
  date. This always returns the latest entry that exists, whatever
  date that is.

SUMMARIZING A PERIOD

• get_farm_summary(start_date, end_date) — use this when the farmer
  asks about totals or profit/loss over a period. Convert relative
  periods (this month, last week, etc.) into exact start and end
  dates before calling this tool.

The result includes total_crates, total_feed_kg, total_revenue,
total_expenses, net_profit, and days_recorded. ALWAYS check
days_recorded first: if it is 0, no data exists for that period at
all — say so honestly rather than reporting a profit/loss of zero as
if it were real performance.

GENERAL RULES

All monetary values must be reported using the ₦ (Naira) symbol,
never $ or any other currency symbol.

Revenue is calculated automatically.

Only one record should exist for each day.

Never invent production figures.

Never invent revenue.

Never simulate tool execution.

Always wait for the tool result before responding.

If the tool reports an error, communicate that error honestly.
"""
    )

    # --------------------------------------------------------
    # FarmRecordService
    # --------------------------------------------------------

    class FarmRecordService:

        CRATE_PRICE = 3500

        def __init__(self, database_name):
            self.database_name = database_name

        def get_connection(self):
            return sqlite3.connect(self.database_name)

        def get_total_records(self):
            connection = self.get_connection()
            cursor = connection.cursor()
            cursor.execute("SELECT COUNT(*) FROM farm_records")
            total = cursor.fetchone()[0]
            connection.close()
            return total

        def record_exists(self, record_date):
            connection = self.get_connection()
            cursor = connection.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM farm_records WHERE record_date=?",
                (record_date,)
            )
            exists = cursor.fetchone()[0] > 0
            connection.close()
            return exists

        def get_record_by_date(self, record_date):
            connection = self.get_connection()
            connection.row_factory = sqlite3.Row
            cursor = connection.cursor()
            cursor.execute(
                "SELECT * FROM farm_records WHERE record_date=?",
                (record_date,)
            )
            row = cursor.fetchone()
            connection.close()
            return dict(row) if row else None

        def get_previous_record(self, record_date):
            connection = self.get_connection()
            connection.row_factory = sqlite3.Row
            cursor = connection.cursor()
            cursor.execute(
                """
                SELECT * FROM farm_records
                WHERE record_date < ?
                ORDER BY record_date DESC
                LIMIT 1
                """,
                (record_date,)
            )
            row = cursor.fetchone()
            connection.close()
            return dict(row) if row else None

        def get_most_recent_record(self):
            connection = self.get_connection()
            connection.row_factory = sqlite3.Row
            cursor = connection.cursor()
            cursor.execute(
                """
                SELECT * FROM farm_records
                ORDER BY record_date DESC
                LIMIT 1
                """
            )
            row = cursor.fetchone()
            connection.close()
            return dict(row) if row else None

        def get_summary(self, start_date, end_date):
            connection = self.get_connection()
            connection.row_factory = sqlite3.Row
            cursor = connection.cursor()
            cursor.execute(
                """
                SELECT
                    COUNT(*) AS days_recorded,
                    COALESCE(SUM(crates_collected), 0) AS total_crates,
                    COALESCE(SUM(feed_consumed_kg), 0) AS total_feed_kg,
                    COALESCE(SUM(revenue), 0) AS total_revenue,
                    COALESCE(SUM(expenses), 0) AS total_expenses
                FROM farm_records
                WHERE record_date BETWEEN ? AND ?
                """,
                (start_date, end_date)
            )
            row = cursor.fetchone()
            connection.close()

            result = dict(row)
            result["net_profit"] = result["total_revenue"] - result["total_expenses"]
            result["start_date"] = start_date
            result["end_date"] = end_date
            return result

        def get_all_records(self):
            connection = self.get_connection()
            connection.row_factory = sqlite3.Row
            cursor = connection.cursor()
            cursor.execute("SELECT * FROM farm_records ORDER BY record_date")
            rows = cursor.fetchall()
            connection.close()
            return [dict(row) for row in rows]

        def validate_daily_record(self, bird_count, crates_collected):
            eggs = crates_collected * 30
            maximum_eggs = bird_count * 0.95
            if eggs > maximum_eggs:
                return (
                    False,
                    f"{crates_collected} crates ({eggs} eggs) appears "
                    f"unrealistic for {bird_count} birds."
                )
            return True, "Validation passed."

        def record_daily_farm_data(
            self, crates_collected=None, bird_count=None,
            feed_consumed_kg=None, expenses=None,
            notes=None, record_date=None
        ):
            if record_date is None:
                record_date = date.today().isoformat()

            existing_record = self.get_record_by_date(record_date)
            previous_day_record = self.get_previous_record(record_date)

            reference_record = existing_record or previous_day_record

            if reference_record:
                if crates_collected is None:
                    crates_collected = reference_record["crates_collected"]
                if bird_count is None:
                    bird_count = reference_record["bird_count"]
                if feed_consumed_kg is None:
                    feed_consumed_kg = reference_record["feed_consumed_kg"]
                if expenses is None:
                    expenses = reference_record["expenses"]
                if notes is None:
                    notes = reference_record["notes"]
            else:
                if crates_collected is None:
                    raise ValueError("crates_collected is required for the first record.")
                if bird_count is None:
                    raise ValueError("bird_count is required for the first record.")
                if feed_consumed_kg is None:
                    raise ValueError("feed_consumed_kg is required for the first record.")
                if expenses is None:
                    raise ValueError("expenses is required for the first record.")

            revenue = crates_collected * self.CRATE_PRICE

            valid, message = self.validate_daily_record(bird_count, crates_collected)
            if not valid:
                return {"success": False, "message": message}

            connection = self.get_connection()
            cursor = connection.cursor()

            if existing_record:
                cursor.execute(
                    """
                    UPDATE farm_records
                    SET bird_count=?, crates_collected=?, feed_consumed_kg=?,
                        revenue=?, expenses=?, notes=?
                    WHERE record_date=?
                    """,
                    (bird_count, crates_collected, feed_consumed_kg,
                     revenue, expenses, notes, record_date)
                )
                action = "updated"
            else:
                cursor.execute(
                    """
                    INSERT INTO farm_records(
                        record_date, bird_count, crates_collected,
                        feed_consumed_kg, revenue, expenses, notes
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (record_date, bird_count, crates_collected,
                     feed_consumed_kg, revenue, expenses, notes)
                )
                action = "recorded"

            connection.commit()
            connection.close()

            return {
                "success": True,
                "action": action,
                "record_date": record_date,
                "previous_values": existing_record,
                "bird_count": bird_count,
                "crates_collected": crates_collected,
                "feed_consumed_kg": feed_consumed_kg,
                "revenue": revenue,
                "expenses": expenses,
                "notes": notes,
                "message": f"Farm record successfully {action}."
            }

    farm_service = FarmRecordService(DATABASE_NAME)

    # --------------------------------------------------------
    # STRING-COERCION HELPERS
    # Groq sometimes sends numeric tool arguments as strings.
    # These tolerate that, and strip common currency formatting.
    # --------------------------------------------------------

    def _clean_numeric_string(value):
        cleaned = value.strip().lower()
        if cleaned in ("", "null", "none"):
            return None
        for symbol in ["₦", "$", ",", "naira", "usd"]:
            cleaned = cleaned.replace(symbol, "")
        return cleaned.strip()

    def _to_int(value, field_name):
        if value is None:
            return None
        if isinstance(value, str):
            cleaned = _clean_numeric_string(value)
            if cleaned is None or cleaned == "":
                return None
            try:
                return int(float(cleaned))
            except ValueError:
                raise ValueError(f"Could not interpret '{value}' as a whole number for {field_name}.")
        return int(value)

    def _to_float(value, field_name):
        if value is None:
            return None
        if isinstance(value, str):
            cleaned = _clean_numeric_string(value)
            if cleaned is None or cleaned == "":
                return None
            try:
                return float(cleaned)
            except ValueError:
                raise ValueError(f"Could not interpret '{value}' as a number for {field_name}.")
        return float(value)

    # --------------------------------------------------------
    # AGENT ACTION 1 — Record or update daily farm data
    # --------------------------------------------------------

    def record_daily_farm_data(
        crates_collected: Optional[str] = None,
        bird_count: Optional[str] = None,
        feed_consumed_kg: Optional[str] = None,
        expenses: Optional[str] = None,
        notes: Optional[str] = None,
        record_date: Optional[str] = None,
    ):
        """
        Record or update daily poultry farm production.

        Use this action whenever the farmer provides
        daily production information.

        NOTE: All numeric fields are accepted as text and converted
        internally to numbers, to tolerate models that pass numeric
        values as strings.

        If record_date is not provided, it defaults to today's date
        automatically — do not ask the farmer for the date unless
        they are referring to a specific past day.

        Business Rules

        - If a record already exists for the date, it is updated —
          only the fields provided are changed; all other fields,
          including crates_collected, keep their current values.
        - If no record exists yet for the date, missing fields are
          inherited from the most recent prior record. The very
          first record ever created requires all fields.
        - Revenue is calculated automatically.
        - The result includes 'previous_values' (the record's state
          before this call, or None if this created a new record).
        """
        parsed_crates = _to_int(crates_collected, "crates_collected")
        parsed_bird_count = _to_int(bird_count, "bird_count")
        parsed_feed = _to_float(feed_consumed_kg, "feed_consumed_kg")
        parsed_expenses = _to_float(expenses, "expenses")

        return farm_service.record_daily_farm_data(
            crates_collected=parsed_crates,
            bird_count=parsed_bird_count,
            feed_consumed_kg=parsed_feed,
            expenses=parsed_expenses,
            notes=notes,
            record_date=record_date,
        )

    # --------------------------------------------------------
    # AGENT ACTION 2 — Exact-date lookup
    # --------------------------------------------------------

    def get_farm_record(record_date: str):
        """
        Retrieve the farm record for one exact calendar date.

        Use this when the farmer asks about a SPECIFIC date — including
        relative terms like "yesterday" or "last Tuesday" that you have
        already converted into an exact date (YYYY-MM-DD) before calling
        this tool.

        This performs an EXACT match only. If no record exists for that
        exact date, it returns a clear "no record found" result — it does
        NOT fall back to the nearest available date.

        Args:
            record_date: The exact date to look up, in YYYY-MM-DD format.
        """
        record = farm_service.get_record_by_date(record_date)

        if record is None:
            return {
                "found": False,
                "record_date": record_date,
                "message": f"No farm record was found for {record_date}."
            }

        return {
            "found": True,
            "record_date": record_date,
            "record": record
        }

    # --------------------------------------------------------
    # AGENT ACTION 3 — Most recent record
    # --------------------------------------------------------

    def get_most_recent_farm_record():
        """
        Retrieve the single most recent farm record in the entire record
        book, regardless of how many days ago it was.

        Use this when the farmer asks something like "what's my last
        record?" or "show me my most recent entry" — situations where
        they want the latest available data, not a specific date.
        """
        record = farm_service.get_most_recent_record()

        if record is None:
            return {"found": False, "message": "No farm records exist yet."}

        return {"found": True, "record": record}

    # --------------------------------------------------------
    # AGENT ACTION 4 — Period summary
    # --------------------------------------------------------

    def get_farm_summary(start_date: str, end_date: str):
        """
        Get a summary of farm performance over a date range (inclusive
        of both start_date and end_date).

        Use this when the farmer asks about totals or profit/loss over
        a period. Convert relative period references into exact
        YYYY-MM-DD start and end dates BEFORE calling this tool.

        Returns total crates collected, total feed consumed, total
        revenue, total expenses, net profit, and days_recorded. If
        days_recorded is 0, no data exists for that range at all.

        Args:
            start_date: Start of the range, in YYYY-MM-DD format.
            end_date: End of the range, in YYYY-MM-DD format.
        """
        return farm_service.get_summary(start_date, end_date)

    # --------------------------------------------------------
    # WRAP AS ADK TOOLS
    # --------------------------------------------------------

    farm_record_tool = FunctionTool(record_daily_farm_data)
    farm_record_lookup_tool = FunctionTool(get_farm_record)
    most_recent_record_tool = FunctionTool(get_most_recent_farm_record)
    farm_summary_tool = FunctionTool(get_farm_summary)

    # --------------------------------------------------------
    # COMBINED SKILLTOOLSET
    # One SkillToolset per Agent, both skills together.
    # additional_tools left empty — all four tools are
    # registered directly on the Agent instead.
    # --------------------------------------------------------

    agroscan_toolset = SkillToolset(
        skills=[
            farm_manager_core_skill,
            farm_record_skill,
        ],
        additional_tools=[]
    )

    # --------------------------------------------------------
    # LOAD GROQ API KEY FROM STREAMLIT SECRETS
    # Must run before the Agent/LiteLlm object is created.
    # --------------------------------------------------------

    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

    # --------------------------------------------------------
    # AGENT
    # --------------------------------------------------------

    today_str = date.today().isoformat()

    farm_manager_agent = Agent(

        model=LiteLlm(
            model="groq/meta-llama/llama-4-scout-17b-16e-instruct"
        ),

        name="farm_manager",

        description=(
            "An intelligent poultry farm management system "
            "that assists farmers using specialized capabilities."
        ),

        instruction=f"""
You are AgroScan AI Farm Manager.

You are the single point of interaction for the farmer.

Today's date is {today_str}. Use this to resolve any relative date
or period the farmer mentions (e.g. "yesterday", "this month", "last
week", "three days ago") into exact YYYY-MM-DD date(s) BEFORE calling
any tool. Tools only accept exact dates — never pass a relative term
directly to a tool.

Your responsibility is to help manage poultry farms by
using the available Skills and Tools behind the scenes.

GENERAL RULES

• Never expose internal implementation details.

• Never mention Skills.

• Never mention Tool calls.

• Never mention FunctionTools.

• Never invent farm records.

• Never invent production figures.

• Never invent revenue.

• Treat the Farm Record Book as the single source of truth.

• To record or update daily farm data, call the
record_daily_farm_data tool directly. This tool is
always available.

• To look up a specific date's record, call get_farm_record
directly with an exact date. This tool is always available.

• To find the most recent record on file (when the farmer doesn't
name a specific date), call get_most_recent_farm_record directly.
This tool is always available.

• To summarize performance over a period (totals, profit/loss),
call get_farm_summary directly with an exact start and end date.
This tool is always available.

• All monetary values must be reported using the ₦ (Naira) symbol,
never $ or any other currency symbol.

• Load the farm-record-management skill to guide how you
interpret and communicate about farm records, lookups, and
summaries.

• Load the farm-manager-core skill to guide your identity,
tone, and communication style.

• Never simulate tool execution.

• Wait for tool results before responding.

• If required information is missing,
ask only for the missing information.

Maintain a friendly, professional and practical tone.
""",

        tools=[
            farm_record_tool,
            farm_record_lookup_tool,
            most_recent_record_tool,
            farm_summary_tool,
            agroscan_toolset,
        ]
    )

    # --------------------------------------------------------
    # SESSION SERVICE, SESSION, AND RUNNER
    # user_id and session_id are generated uniquely per browser
    # session — important if multiple people use the app at once.
    # --------------------------------------------------------

    session_service = InMemorySessionService()

    unique_user_id = str(uuid.uuid4())

    agroscan_session = session_service.create_session_sync(
        app_name="agroscan_app",
        user_id=unique_user_id
    )

    runner = Runner(
        app_name="agroscan_app",
        agent=farm_manager_agent,
        session_service=session_service
    )

    # --------------------------------------------------------
    # STORE EVERYTHING IN SESSION STATE
    # --------------------------------------------------------

    st.session_state.runner = runner
    st.session_state.session_service = session_service
    st.session_state.agroscan_session = agroscan_session
    st.session_state.user_id = unique_user_id
    st.session_state.chat_history = []

    st.session_state.initialized = True


# ============================================================
# ASYNC BRIDGE
# runner.run_debug(...) is async; Streamlit's script executes
# synchronously, so this wraps the call.
# ============================================================

def run_agent_turn(message: str):
    return asyncio.run(
        st.session_state.runner.run_debug(
            message,
            user_id=st.session_state.user_id,
            session_id=st.session_state.agroscan_session.id,
            quiet=True
        )
    )


# ============================================================
# PAGE HEADER
# ============================================================

st.title("🐔 AgroScan AI Farm Manager")
st.caption("Your intelligent poultry farm management assistant")

if st.session_state.get("import_summary"):
    st.info(st.session_state.import_summary)


# ============================================================
# RENDER EXISTING CHAT HISTORY
# ============================================================

for role, text in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(text)


# ============================================================
# HANDLE NEW MESSAGE
# ============================================================

user_message = st.chat_input("Talk to AgroScan about your farm...")

if user_message:

    st.session_state.chat_history.append(("user", user_message))
    with st.chat_message("user"):
        st.markdown(user_message)

    with st.chat_message("assistant"):
        with st.spinner("AgroScan is thinking..."):
            try:
                events = run_agent_turn(user_message)
                final_event = events[-1]

                if final_event.content and final_event.content.parts:
                    response = " ".join(
                        part.text
                        for part in final_event.content.parts
                        if part.text
                    )
                else:
                    response = "No response was generated."

            except Exception as e:
                response = (
                    "I ran into an issue processing that. "
                    "Could you try rephrasing, or ask again?"
                )
                st.session_state.setdefault("last_error", str(e))

            st.markdown(response)

    st.session_state.chat_history.append(("assistant", response))
