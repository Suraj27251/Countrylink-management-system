import os
import json
import logging
import re
import uuid
import pymysql
import pymysql.cursors
import fcntl
import glob
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Optional, List, Tuple
from flask import Flask, request, jsonify
import requests
from groq import Groq

# Environment variables are set directly in cPanel's Python App manager.
# No .env file or python-dotenv needed on the server.

app = Flask(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
VERIFY_TOKEN     = os.environ.get("VERIFY_TOKEN")
WHATSAPP_TOKEN   = os.environ.get("WHATSAPP_TOKEN")
PHONE_NUMBER_ID  = os.environ.get("PHONE_NUMBER_ID")
GROQ_API_KEY     = os.environ.get("GROQ_API_KEY")
DB_HOST     = os.environ.get("MYSQL_DB_HOST", "localhost")
DB_NAME     = os.environ.get("MYSQL_DB_NAME", "countrylinks_user_database")
DB_USER     = os.environ.get("MYSQL_DB_USER")
DB_PASSWORD = os.environ.get("MYSQL_DB_PASSWORD")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    filename="whatsapp_agent.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

# ── Groq Setup ───────────────────────────────────────────────────────────────
# Client initialized lazily inside _call_groq so missing env var at startup is safe

SYSTEM_PROMPT = """You are a WhatsApp customer support assistant EXCLUSIVELY for Countrylink Broadband Services.

You ONLY help with topics directly related to Countrylink Broadband:
- Internet plans, pricing, upgrades, and availability
- OTT add-on packages
- Technical troubleshooting (slow speed, disconnection, router issues)
- Bill payment, due dates, invoice lookup, and account information
- New connection requests and installation scheduling
- Reporting outages and service complaints
- Raising complaints about service issues

Current Countrylink plan information:
Internet-only packages:
1. Limited 50 Mbps: ₹499, 50 Mbps speed, 500 GB data
2. Unlimited 50 Mbps: ₹800, unlimited data, 50 Mbps speed
3. Unlimited 100 Mbps: ₹900, unlimited data, 100 Mbps speed
4. Unlimited 200 Mbps: ₹999, unlimited data, 200 Mbps speed
5. Unlimited 300 Mbps: ₹1099, unlimited data, 300 Mbps speed

Internet package notes:
- GST is excluded from the listed prices. Tell customers final payable amount may include GST.
- Free 5G Wi-Fi router and free installation are subject to a technical and feasibility survey.
- All listed internet packages use fiber connectivity.
- Quarterly plans have a 20% discount. If a customer asks for final quarterly billing, explain that support will confirm the exact payable total including GST.
- More plans may be available; ask customers with special requirements to contact support.

OTT add-on packages:
1. Premium: ₹200 monthly. Includes Jio Hotstar, SonyLIV, Discovery Plus, and 10+ more apps.
2. VIP: ₹350 monthly. Includes Jio Hotstar, SonyLIV, ZEE5, Discovery Plus, and 10+ more apps.
3. Prime: ₹1999 quarterly. Includes Jio Hotstar, Amazon Prime Video, SonyLIV, ZEE5, Discovery Plus, and 10+ more apps.

Sales/support contact:
- Phone/WhatsApp: 9765009850 or 9765005851

INTENT HANDLING INSTRUCTIONS:

1. COMPLAINT RAISING:
   When a customer reports an issue (slow speed, not working, outage, disconnected, etc.), the system will automatically detect this and start collecting their details (name, mobile, complaint description). You do NOT need to handle complaint collection yourself — the system does it via multi-turn prompts. However, if the customer describes an issue conversationally without triggering the complaint flow, empathize and ask if they'd like to raise a formal complaint.

2. INVOICE / BILLING LOOKUP:
   When a customer asks about their bill, invoice, payment status, or due date, the system will ask for their registered mobile number and look up their invoices automatically. You do NOT need to fabricate invoice details. If the system hasn't triggered the invoice flow, ask the customer for their registered 10-digit mobile number.

3. NEW CONNECTION REQUEST:
   When a customer wants a new broadband connection, the system will automatically detect this and collect their details (name, mobile, area/address). You do NOT need to handle this yourself. If the customer asks about availability in their area, let them know a feasibility survey will be done after they submit their request.

Guidelines:
- Be friendly, concise, and professional.
- Keep messages short and WhatsApp-friendly (no heavy markdown).
- Always respond in the same language the customer writes in.
- Every successful complaint confirmation MUST include this exact standalone line: "Track your complaint here: https://countrylinks.in/track". This tracking URL is mandatory in every complaint confirmation, not optional.
- When asked about plans, recommend based on the customer's usage:
  * Basic browsing/limited use: Limited 50 Mbps ₹499 plan.
  * Regular home use: Unlimited 50 Mbps ₹800 or Unlimited 100 Mbps ₹900 plan.
  * Work from home, gaming, streaming, or many users: Unlimited 200 Mbps ₹999 or Unlimited 300 Mbps ₹1099 plan.
  * OTT entertainment: suggest the suitable OTT add-on package.
- Clearly mention GST exclusion and feasibility-survey conditions when sharing package prices.
- For invoice, bill, due date, or payment-status questions, ask the customer for their registered 10-digit mobile number for authentication before sharing invoice details.
- If you don't know something specific (like area feasibility, appointment slots, exact GST-inclusive bill, outage ETA, or account details not returned from the invoice lookup), ask for their registered mobile number/address and tell them support will confirm, or ask them to call 9765009850 / 9765005851.
- Do not invent unavailable offers, app names beyond "10+ more", due dates, service availability, invoice details, payment status, or account details.

CRITICAL RULE — OFF-TOPIC MESSAGES:
If the customer's message is NOT related to Countrylink Broadband services, internet plans, billing, technical support, or OTT packages, you MUST respond with ONLY this message (translated to their language if needed):
"I'm Countrylink's broadband support assistant and can only help with internet plans, billing, and technical queries. For other help, please call 9765009850 or 9765005851. 😊"
Do NOT attempt to answer general knowledge questions, news, entertainment, jokes, coding, or any topic outside Countrylink Broadband services — even if you know the answer.
"""

# ── File Paths ────────────────────────────────────────────────────────────────
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE      = os.path.join(BASE_DIR, "conversation_history.json")
INVOICE_AUTH_FILE = os.path.join(BASE_DIR, "invoice_auth_requests.json")
PAYMENT_VERIFY_FILE = os.path.join(BASE_DIR, "payment_verify_requests.json")
ONBOARDING_FILE = os.path.join(BASE_DIR, "onboarding_requests.json")
PENDING_ACTIONS_FILE = os.path.join(BASE_DIR, "pending_actions.json")

# ── Repo 1 API Configuration ─────────────────────────────────────────────────
REPO1_BASE_URL       = os.environ.get("REPO1_BASE_URL", "https://countrylinks.in")
AGENT_INTERNAL_TOKEN = os.environ.get("AGENT_INTERNAL_TOKEN", "")

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_HISTORY_MESSAGES = 20    # Per user: last 10 exchanges
MAX_USERS_IN_HISTORY = 500   # FIX #9 — cap total users to prevent unbounded file growth
MAX_PROCESSED_IDS    = 1000  # FIX #4 — deduplication set cap
GEMINI_TIMEOUT_SECS  = 15    # FIX #10 — abort if Gemini takes too long

# FIX #4 — In-memory set to deduplicate incoming WhatsApp messages
# (Meta sometimes sends the same webhook event twice)
PROCESSED_MESSAGE_IDS: set = set()

INVOICE_KEYWORDS = (
    "invoice", "bill", "billing", "unpaid",
    "due", "due date", "amount", "outstanding"
)

PAYMENT_VERIFY_KEYWORDS = (
    "paid", "payment done", "recharge done", "transferred", "upi",
    "gpay", "phonepe", "paytm", "screenshot", "receipt", "transaction",
)

PAYMENT_PAGE_KEYWORDS = (
    "pay", "payment", "recharge", "renew", "renewal", "how to pay",
    "pay online", "online payment", "pay bill",
)

COMPLAINT_KEYWORDS = (
    "complaint", "slow speed", "not working", "outage", "disconnected",
    "no internet", "speed issue", "buffering", "lag", "down",
    "network issue", "connection problem", "wifi not working",
    "router issue", "no signal", "internet down", "speed slow",
    "very slow", "net nahi chal raha", "internet band", "speed kam",
    "net nahi aa raha", "problem", "issue",
)

NEW_CONNECTION_KEYWORDS = (
    "new connection", "new customer", "new broadband", "want connection",
    "apply connection", "broadband connection", "new internet",
    "want internet", "get broadband", "start connection", "need internet",
    "need wifi", "want wifi", "naya connection", "internet chahiye",
    "broadband chahiye", "connection lena hai", "wifi lena hai",
    "install internet", "setup internet", "get fiber", "want fiber",
)


# ── Thread-safe JSON helpers (FIX #2) ────────────────────────────────────────
def _load_json_file(path: str) -> dict:
    """Load a JSON file with a shared read lock."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            try:
                return json.load(f)
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
    except (json.JSONDecodeError, IOError) as e:
        logging.error(f"Could not load {path}: {e}")
        return {}


def _save_json_file(path: str, data: dict):
    """Save a JSON file with an exclusive write lock."""
    try:
        with open(path, "w") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                json.dump(data, f, indent=2)
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
    except IOError as e:
        logging.error(f"Could not save {path}: {e}")


# ── Conversation History ──────────────────────────────────────────────────────
def load_history() -> dict:
    return _load_json_file(HISTORY_FILE)


def save_history(history: dict):
    # FIX #9 — prune oldest users when the history grows too large
    if len(history) > MAX_USERS_IN_HISTORY:
        keys = list(history.keys())
        for k in keys[: MAX_USERS_IN_HISTORY // 2]:
            del history[k]
        logging.info("Pruned conversation history (exceeded MAX_USERS_IN_HISTORY)")
    _save_json_file(HISTORY_FILE, history)


# ── Invoice Auth State ────────────────────────────────────────────────────────
def load_invoice_auth_requests() -> dict:
    return _load_json_file(INVOICE_AUTH_FILE)


def save_invoice_auth_requests(requests_map: dict):
    _save_json_file(INVOICE_AUTH_FILE, requests_map)


def set_invoice_auth_pending(user_id: str):
    m = load_invoice_auth_requests()
    m[user_id] = {"step": "awaiting_mobile", "data": {}}
    save_invoice_auth_requests(m)


def clear_invoice_auth_pending(user_id: str):
    m = load_invoice_auth_requests()
    if user_id in m:
        del m[user_id]
        save_invoice_auth_requests(m)


def is_invoice_auth_pending(user_id: str) -> bool:
    return bool(load_invoice_auth_requests().get(user_id))




# ── Payment Verification State ───────────────────────────────────────────────
def load_payment_verify_requests() -> dict:
    return _load_json_file(PAYMENT_VERIFY_FILE)


def save_payment_verify_requests(requests_map: dict):
    _save_json_file(PAYMENT_VERIFY_FILE, requests_map)


def set_payment_verify_pending(user_id: str, data: dict = None):
    requests_map = load_payment_verify_requests()
    requests_map[user_id] = {
        "step": "awaiting_mobile",
        "data": data or {},
    }
    save_payment_verify_requests(requests_map)


def get_payment_verify_pending(user_id: str) -> Optional[dict]:
    return load_payment_verify_requests().get(user_id)


def clear_payment_verify_pending(user_id: str):
    requests_map = load_payment_verify_requests()
    if user_id in requests_map:
        del requests_map[user_id]
        save_payment_verify_requests(requests_map)


# ── Onboarding State ─────────────────────────────────────────────────────────
def load_onboarding_requests() -> dict:
    return _load_json_file(ONBOARDING_FILE)


def save_onboarding_requests(requests_map: dict):
    _save_json_file(ONBOARDING_FILE, requests_map)


def set_onboarding_state(user_id: str, step: str, data: dict = None):
    requests_map = load_onboarding_requests()
    requests_map[user_id] = {
        "step": step,
        "data": data or {},
    }
    save_onboarding_requests(requests_map)


def get_onboarding_state(user_id: str) -> Optional[dict]:
    return load_onboarding_requests().get(user_id)


def clear_onboarding_state(user_id: str):
    requests_map = load_onboarding_requests()
    if user_id in requests_map:
        del requests_map[user_id]
        save_onboarding_requests(requests_map)


# ── Pending Actions State (Multi-Turn) ────────────────────────────────────────
def load_pending_actions() -> dict:
    return _load_json_file(PENDING_ACTIONS_FILE)


def save_pending_actions(actions: dict):
    _save_json_file(PENDING_ACTIONS_FILE, actions)


def get_pending_action(user_id: str) -> Optional[dict]:
    actions = load_pending_actions()
    return actions.get(user_id)


def set_pending_action(user_id: str, action_type: str, collected: dict = None):
    """
    Set a pending multi-turn action for a user.
    action_type: 'complaint' or 'new_connection'
    collected: dict of fields already collected (e.g. {'name': 'John'})
    """
    collected = collected or {}
    if action_type == "complaint":
        if "name" not in collected:
            step = "awaiting_name"
        elif "mobile" not in collected:
            step = "awaiting_mobile"
        elif "complaint" not in collected:
            step = "awaiting_issue"
        else:
            step = "ready_to_submit"
    elif action_type == "new_connection":
        if "name" not in collected:
            step = "awaiting_name"
        elif "mobile" not in collected:
            step = "awaiting_mobile"
        elif "area" not in collected:
            step = "awaiting_area"
        else:
            step = "ready_to_submit"
    else:
        step = "pending"

    actions = load_pending_actions()
    actions[user_id] = {
        "type": action_type,
        "step": step,
        "data": collected,
        "collected": collected
    }
    save_pending_actions(actions)


def clear_pending_action(user_id: str):
    actions = load_pending_actions()
    if user_id in actions:
        del actions[user_id]
        save_pending_actions(actions)


# ── Intent Detection ──────────────────────────────────────────────────────────
def is_complaint_request(message: str) -> bool:
    """Detect if the message is a complaint/issue report."""
    msg_lower = message.lower()
    return any(kw in msg_lower for kw in COMPLAINT_KEYWORDS)


def is_new_connection_request(message: str) -> bool:
    """Detect if the message is a new connection request."""
    msg_lower = message.lower()
    return any(kw in msg_lower for kw in NEW_CONNECTION_KEYWORDS)


def is_payment_verification_request(message: str, message_type: str = "text") -> bool:
    """Detect payment screenshots/receipts that need account verification."""
    if message_type == "image":
        return True
    msg_lower = message.lower()
    return any(kw in msg_lower for kw in PAYMENT_VERIFY_KEYWORDS)


def is_payment_page_request(message: str) -> bool:
    """Detect general payment/recharge intent for the online payment page."""
    msg_lower = message.lower()
    return any(kw in msg_lower for kw in PAYMENT_PAGE_KEYWORDS)


def looks_like_phone_number_or_numeric(value: str) -> bool:
    stripped = value.strip()
    digits = re.sub(r"\D", "", stripped)
    return stripped.isdigit() or len(digits) >= 10


def is_valid_person_name(value: str) -> bool:
    """Require alphabetic characters so phone numbers cannot be accepted as names."""
    stripped = value.strip()
    if not stripped or looks_like_phone_number_or_numeric(stripped):
        return False
    return bool(re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ]", stripped))


# ── API Calls to Repo 1 ──────────────────────────────────────────────────────
def raise_complaint_via_api(name: str, mobile: str, complaint: str) -> Optional[str]:
    """
    Insert complaint directly into MySQL complaints table.
    Returns None on success, error message on failure.
    """
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            # Generate a unique complaint ID
            complaint_id = f"CMP-{uuid.uuid4().hex[:8].upper()}"

            cursor.execute(
                """
                INSERT INTO complaints (
                    complaint_id,
                    customer_name,
                    customer_phone,
                    complaint_subject,
                    complaint_description,
                    category,
                    status,
                    escalation_level,
                    created_at,
                    updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
                """,
                (
                    complaint_id,
                    name,
                    mobile,
                    complaint[:100],  # Use first 100 chars as subject
                    complaint,
                    "General",  # Default category
                    "open",
                    "low",
                )
            )
        conn.close()
        logging.info(f"Complaint {complaint_id} raised for {mobile}")
        return complaint_id  # Return the complaint ID on success
    except Exception as e:
        logging.error(f"Failed to raise complaint for {mobile}: {e}")
        return None


def raise_new_connection_via_api(name: str, mobile: str, area: str) -> Optional[str]:
    """
    POST new connection request to Repo 1's /api/new-connection-request.
    Returns None on success, error message on failure.
    """
    url = f"{REPO1_BASE_URL}/api/new-connection-request"
    headers = {
        "Content-Type": "application/json",
        "X-Internal-Token": AGENT_INTERNAL_TOKEN
    }
    payload = {
        "name": name,
        "mobile": mobile,
        "area": area
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        logging.info(f"New connection request raised for {mobile}: {data}")
        return None  # Success
    except requests.RequestException as e:
        logging.error(f"Failed to raise new connection for {mobile}: {e}")
        return str(e)


# ── Multi-Turn Conversation Handler ──────────────────────────────────────────
def handle_pending_action(user_id: str, user_message: str, customer_name: str = "Customer") -> Optional[str]:
    """
    Handle multi-turn collection for complaints and new connections.
    Returns a reply if the pending action is being handled, None otherwise.
    """
    pending = get_pending_action(user_id)
    if not pending:
        return None

    action_type = pending["type"]
    collected = pending.get("collected") or pending.get("data") or {}

    # Allow user to cancel
    cancel_words = ("cancel", "nevermind", "never mind", "stop", "nahi", "ruko", "band karo")
    if user_message.lower().strip() in cancel_words:
        clear_pending_action(user_id)
        return "No problem, I've cancelled that. How else can I help you? 😊"

    if action_type == "complaint":
        return _handle_complaint_collection(user_id, user_message, collected, customer_name)
    elif action_type == "new_connection":
        return _handle_new_connection_collection(user_id, user_message, collected, customer_name)

    return None


def _handle_complaint_collection(
    user_id: str, user_message: str, collected: dict, customer_name: str
) -> str:
    """Collect complaint details step by step."""

    # Step 1: Collect name
    if "name" not in collected:
        if not is_valid_person_name(user_message):
            return "That looks like a number, not a name. Please share your full name first."
        collected["name"] = user_message.strip()
        set_pending_action(user_id, "complaint", collected)
        return "Got it. Now please share your registered 10-digit mobile number."

    # Step 2: Collect mobile
    if "mobile" not in collected:
        mobile = extract_mobile_number(user_message)
        if not mobile:
            return "Please provide a valid 10-digit mobile number (e.g., 9876543210)."
        collected["mobile"] = mobile
        set_pending_action(user_id, "complaint", collected)
        return "Thanks. Now please describe your complaint or issue in detail."

    # Step 3: Collect complaint description
    if "complaint" not in collected:
        collected["complaint"] = user_message.strip()
        set_pending_action(user_id, "complaint", collected)

        # All fields collected — submit
        complaint_id = raise_complaint_via_api(
            collected["name"],
            collected["mobile"],
            collected["complaint"]
        )

        clear_pending_action(user_id)

        if not complaint_id:
            return (
                "Sorry, I couldn't register your complaint right now. "
                "Please try again or call 9765009850 / 9765005851 for help."
            )

        return (
            f"✅ Your complaint has been registered successfully!\n\n"
            f"Complaint ID: {complaint_id}\n"
            f"Name: {collected['name']}\n"
            f"Mobile: {collected['mobile']}\n"
            f"Issue: {collected['complaint']}\n"
            f"Track your complaint here: https://countrylinks.in/track\n\n"
            f"Our team will look into this and get back to you soon. "
            f"For urgent issues, call 9765009850 / 9765005851."
        )

    return None


def _handle_new_connection_collection(
    user_id: str, user_message: str, collected: dict, customer_name: str
) -> str:
    """Collect new connection details step by step."""

    # Step 1: Collect name
    if "name" not in collected:
        collected["name"] = user_message.strip()
        set_pending_action(user_id, "new_connection", collected)
        return "Great! Please share your 10-digit mobile number."

    # Step 2: Collect mobile
    if "mobile" not in collected:
        mobile = extract_mobile_number(user_message)
        if not mobile:
            return "Please provide a valid 10-digit mobile number (e.g., 9876543210)."
        collected["mobile"] = mobile
        set_pending_action(user_id, "new_connection", collected)
        return "Thanks! Now please share your area or full address where you need the connection."

    # Step 3: Collect area/address
    if "area" not in collected:
        collected["area"] = user_message.strip()
        set_pending_action(user_id, "new_connection", collected)

        # All fields collected — submit
        error = raise_new_connection_via_api(
            collected["name"],
            collected["mobile"],
            collected["area"]
        )

        clear_pending_action(user_id)

        if error:
            return (
                "Sorry, I couldn't submit your connection request right now. "
                "Please try again or call 9765009850 / 9765005851."
            )

        return (
            f"✅ Your new connection request has been submitted!\n\n"
            f"Name: {collected['name']}\n"
            f"Mobile: {collected['mobile']}\n"
            f"Area: {collected['area']}\n\n"
            f"Our team will contact you shortly to schedule a feasibility survey and installation. "
            f"For queries, call 9765009850 / 9765005851."
        )

    return None


# ── Invoice Helpers ───────────────────────────────────────────────────────────
def is_invoice_request(message: str) -> bool:
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in INVOICE_KEYWORDS)


def extract_mobile_number(message: str) -> Optional[str]:
    """Extract a 10-digit Indian mobile number from a message."""
    candidates = re.findall(r"(?:\+?91[-\s]?)?[6-9]\d(?:[-\s]?\d){8}", message)
    for candidate in candidates:
        digits = re.sub(r"\D", "", candidate)
        if len(digits) >= 10:
            return digits[-10:]
    return None


def normalize_sql_phone(column_name: str) -> str:
    """MySQL version — strip formatting chars and compare last 10 digits."""
    return (
        f"RIGHT(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE("
        f"COALESCE({column_name}, ''), ' ', ''), '-', ''), '+', ''), '(', ''), ')', ''), 10)"
    )


def get_db_connection() -> pymysql.connections.Connection:
    """Open a MySQL connection using cPanel env vars."""
    return pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
    )



def get_customer_by_mobile(mobile_number: str) -> Optional[dict]:
    """Find a Zoho customer by registered mobile or phone number."""
    if not DB_USER or not DB_PASSWORD:
        raise pymysql.Error(
            "DB_USER / DB_PASSWORD not configured in cPanel env vars"
        )

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            mobile_expr = normalize_sql_phone("mobile")
            phone_expr = normalize_sql_phone("phone")
            cur.execute(
                f"""
                SELECT contact_name, company_name, mobile, phone, status, outstanding_amount
                FROM zoho_customers
                WHERE {mobile_expr} = %s
                   OR {phone_expr} = %s
                ORDER BY id DESC
                LIMIT 1
                """,
                (mobile_number, mobile_number),
            )
            return cur.fetchone()
    finally:
        conn.close()


def get_conversation_mode(phone: str) -> str:
    """Check human_takeover flag. Returns 'human' or 'ai'."""
    try:
        # Normalize incoming phone
        digits = re.sub(r'\D', '', str(phone))

        # Remove India country code if present
        if len(digits) >= 12 and digits.startswith('91'):
            digits = digits[2:]

        normalized = digits[-10:] if len(digits) >= 10 else digits

        conn = get_db_connection()

        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT human_takeover
                FROM whatsapp_conversations
                WHERE phone = %s
                   OR phone = %s
                   OR RIGHT(phone, 10) = %s
                LIMIT 1
                """,
                (phone, normalized, normalized)
            )

            row = cursor.fetchone()

        conn.close()

        # Debug log
        logging.info(
            f"Toggle check: phone={phone}, "
            f"normalized={normalized}, row={row}"
        )

        if row and int(row.get('human_takeover', 0)) == 1:
            return 'human'

        return 'ai'

    except Exception as e:
        logging.error(f"get_conversation_mode error: {e}")
        return 'ai'


def get_customer_and_invoices(
    mobile_number: str, limit: int = 5
) -> Tuple[Optional[dict], List[dict]]:
    """Fetch customer AND invoices in one MySQL connection (always closed)."""

    if not DB_USER or not DB_PASSWORD:
        raise pymysql.Error(
            "DB_USER / DB_PASSWORD not configured in cPanel env vars"
        )

    conn = get_db_connection()

    try:
        with conn.cursor() as cur:
            mobile_expr = normalize_sql_phone("mobile")
            phone_expr = normalize_sql_phone("phone")

            customer_query = f"""
                SELECT
                    zoho_contact_id,
                    contact_name,
                    company_name,
                    mobile,
                    phone,
                    outstanding_amount,
                    status
                FROM zoho_customers
                WHERE {mobile_expr} = %s
                   OR {phone_expr} = %s
                ORDER BY id DESC
                LIMIT 1
            """

            cur.execute(customer_query, (mobile_number, mobile_number))

            customer = cur.fetchone()

            if not customer:
                logging.info(
                    f"No customer found for mobile={mobile_number}"
                )
                return None, []

            logging.info(
                f"Customer found: mobile={mobile_number}, "
                f"zoho_contact_id={customer.get('zoho_contact_id')}"
            )

            invoices_query = """
                SELECT
                    invoice_number,
                    customer_name,
                    status,
                    total,
                    invoice_date,
                    due_date,
                    created_at
                FROM invoices
                WHERE zoho_contact_id = %s
                ORDER BY COALESCE(invoice_date, created_at) DESC, id DESC
                LIMIT %s
            """

            cur.execute(
                invoices_query,
                (customer["zoho_contact_id"], limit)
            )

            invoices = cur.fetchall()

            logging.info(
                f"Fetched {len(invoices)} invoices for "
                f"mobile={mobile_number}"
            )

        return customer, list(invoices)

    except Exception as e:
        logging.error(
            f"get_customer_and_invoices error for "
            f"{mobile_number}: {e}"
        )
        raise

    finally:
        conn.close()
        
def format_currency(value) -> str:
    try:
        return f"₹{float(value):.2f}"
    except (TypeError, ValueError):
        return "amount not available"


def format_invoice_lookup_reply(mobile_number: str) -> str:
    try:
        customer, invoices = get_customer_and_invoices(mobile_number)
    except pymysql.Error as e:
        logging.error(f"Invoice DB error for {mobile_number}: {e}")
        return (
            "Sorry, I cannot access invoice details right now. "
            "Please call 9765009850 / 9765005851 for billing support."
        )

    if not customer:
        return (
            "Sorry, I could not find an account with that mobile number. "
            "Please double-check or call 9765009850 / 9765005851 for help."
        )

    customer_name     = customer["contact_name"] or customer["company_name"] or "customer"
    outstanding_amount = format_currency(customer["outstanding_amount"])

    if not invoices:
        return (
            f"Thanks, {customer_name}. I found your account, but no invoices are "
            "available right now. Please call 9765009850 / 9765005851 for confirmation."
        )

    lines = [
        f"Thanks, {customer_name}. I found your account.",
        f"Outstanding amount: {outstanding_amount}",
        "Latest invoices:",
    ]
    for index, invoice in enumerate(invoices, start=1):
        invoice_number = invoice["invoice_number"] or "N/A"
        status         = invoice["status"]         or "status not available"
        total          = format_currency(invoice["total"])
        invoice_date   = invoice["invoice_date"]   or "date not available"
        due_date       = invoice["due_date"]        or "due date not available"
        lines.append(
            f"{index}) Invoice {invoice_number}: {total}, {status}, "
            f"invoice date {invoice_date}, due date {due_date}"
        )
    lines.append(
        "For payment confirmation or detailed statement, call 9765009850 / 9765005851."
    )
    return "\n".join(lines)


def get_invoice_reply_if_applicable(
    user_id: str, user_message: str
) -> Optional[str]:
    mobile_number     = extract_mobile_number(user_message)
    invoice_requested = is_invoice_request(user_message)

    # Customer sent mobile number — look up invoice (covers both first-time and auth-pending)
    if mobile_number and (invoice_requested or is_invoice_auth_pending(user_id)):
        reply = format_invoice_lookup_reply(mobile_number)
        clear_invoice_auth_pending(user_id)
        return reply

    # Customer mentioned invoice/bill but no mobile yet — ask for it
    if invoice_requested:
        set_invoice_auth_pending(user_id)
        return (
            "Sure, I can check your invoice details. "
            "Please send your registered 10-digit mobile number for authentication."
        )

    return None



def get_payment_verify_reply_if_applicable(
    user_id: str, user_message: str, message_type: str = "text"
) -> Optional[str]:
    """Handle payment receipt/screenshot verification before invoice lookup."""
    pending = get_payment_verify_pending(user_id)
    mobile_number = extract_mobile_number(user_message)

    if pending:
        if mobile_number:
            try:
                customer = get_customer_by_mobile(mobile_number)
            except pymysql.Error as e:
                logging.error(f"Payment verification DB error for {mobile_number}: {e}")
                clear_payment_verify_pending(user_id)
                return (
                    "Sorry, I cannot verify payment details right now. "
                    "Please call 9765009850 / 9765005851 for assistance."
                )

            clear_payment_verify_pending(user_id)
            if customer:
                contact_name = (
                    customer.get("contact_name")
                    or customer.get("company_name")
                    or "customer"
                )
                return (
                    f"✅ We found your account for {contact_name}. Your payment is being verified. "
                    "Our team will update your account within 2-4 hours. "
                    "For urgent help call 9765009850."
                )

            return (
                "❌ We couldn't find an account with that number. "
                "Please call 9765009850 / 9765005851 for assistance."
            )

        return "Thank you! To verify your payment, please share your registered 10-digit mobile number."

    if is_payment_verification_request(user_message, message_type):
        set_payment_verify_pending(user_id)
        return "Thank you! To verify your payment, please share your registered 10-digit mobile number."

    return None


def get_onboarding_reply_if_applicable(
    user_id: str, user_message: str, customer_name: str = "Customer"
) -> Optional[str]:
    """Handle new customer onboarding with name and area state."""
    state = get_onboarding_state(user_id)

    if state:
        step = state.get("step")
        data = state.get("data") or {}

        if step == "awaiting_name":
            if not is_valid_person_name(user_message):
                return "That looks like a number, not a name. Please share your full name first."
            data["name"] = user_message.strip()
            set_onboarding_state(user_id, "awaiting_area", data)
            return f"Thanks {data['name']}! Please share your area/locality so we can check availability."

        if step == "awaiting_area":
            area = user_message.strip()
            if not area:
                return "Please share your area/locality so we can check availability."
            data["area"] = area
            mobile = re.sub(r"\D", "", user_id)[-10:]
            error = raise_new_connection_via_api(data["name"], mobile, data["area"])
            clear_onboarding_state(user_id)
            if error:
                logging.error(f"Onboarding request submit failed for {user_id}: {error}")
            return (
                "Great! Our team will contact you within 24 hours to confirm availability "
                "and schedule installation. For faster assistance call 9765009850. "
                "You can also check our plans at https://countrylinks.in"
            )

    if is_new_connection_request(user_message):
        set_onboarding_state(user_id, "awaiting_name", {})
        return "Welcome to Countrylink Broadband! 🎉 Please share your full name."

    return None


def get_payment_page_reply_if_applicable(
    user_message: str, user_id: str = None
) -> Optional[str]:
    # Do not send the payment page for receipt/screenshot reports; those
    # belong to payment verification even when substring keywords overlap.
    if is_payment_verification_request(user_message, "text"):
        return None
    if user_id and get_payment_verify_pending(user_id):
        return None
    if is_payment_page_request(user_message):
        return "You can pay online here: https://countrylinks.in/payment"
    return None


def get_structured_reply_if_applicable(
    user_id: str,
    user_message: str,
    customer_name: str = "Customer",
    message_type: str = "text",
) -> Optional[str]:
    """
    Unified intent handler. Checks in required priority order:
    payment_verify → invoice → onboarding → payment_page_keywords →
    complaint → Groq/off-topic fallback.
    """

    # Legacy pending actions remain supported so in-progress users are not dropped.
    pending = get_pending_action(user_id)
    if pending and pending.get("type") in ("complaint", "new_connection"):
        pending_reply = handle_pending_action(user_id, user_message, customer_name)
        if pending_reply:
            return pending_reply

    payment_verify_reply = get_payment_verify_reply_if_applicable(
        user_id, user_message, message_type
    )
    if payment_verify_reply:
        return payment_verify_reply

    invoice_reply = get_invoice_reply_if_applicable(user_id, user_message)
    if invoice_reply:
        return invoice_reply

    onboarding_reply = get_onboarding_reply_if_applicable(
        user_id, user_message, customer_name
    )
    if onboarding_reply:
        return onboarding_reply

    payment_page_reply = get_payment_page_reply_if_applicable(user_message, user_id)
    if payment_page_reply:
        return payment_page_reply

    if is_complaint_request(user_message):
        set_pending_action(user_id, "complaint", {})
        return (
            "I'm sorry to hear you're facing an issue. Let me help you raise a complaint.\n\n"
            "Please share your full name to get started."
        )

    return None


# ── Topic Relevance Pre-Filter ────────────────────────────────────────────────
RELEVANT_KEYWORDS = (
    # Plans & pricing
    "plan", "price", "pricing", "mbps", "speed", "data", "fiber", "fibre",
    "broadband", "internet", "connection", "recharge", "upgrade", "package",
    "ott", "hotstar", "sonyliv", "zee5", "prime", "netflix", "wifi", "wi-fi",
    "router", "installation", "install", "new connection", "new customer",
    "new broadband", "want connection", "apply connection",
    "broadband connection", "new internet",
    # Billing
    "bill", "billing", "invoice", "payment", "paid", "unpaid", "due",
    "amount", "receipt", "outstanding", "renewal", "renew", "pay",
    "how to pay", "pay online", "online payment", "pay bill", "payment done",
    "recharge done", "transferred", "upi", "gpay", "phonepe", "paytm",
    "screenshot", "transaction",
    # Support & Complaints
    "slow", "disconnect", "not working", "down", "outage", "issue", "problem",
    "complaint", "help", "support", "contact", "number", "repair", "technician",
    "buffering", "lag", "no internet", "speed issue", "network issue",
    "connection problem", "wifi not working", "router issue", "no signal",
    "internet down", "speed slow", "very slow",
    # Hindi/Marathi complaint keywords
    "net nahi chal raha", "internet band", "speed kam", "net nahi aa raha",
    # New connection
    "new connection", "new customer", "want connection", "apply connection",
    "broadband connection", "new internet", "want internet", "get broadband",
    "new broadband", "naya connection", "internet chahiye", "broadband chahiye",
    "connection lena hai", "wifi lena hai", "get fiber", "want fiber",
    # Greetings / generic (always let these through to Groq)
    "hi", "hello", "hii", "hey", "helo", "namaste", "namaskar",
    "good morning", "good evening", "good afternoon",
    "thank", "thanks", "ok", "okay", "sure", "yes", "no",
    # Account / auth
    "account", "mobile", "number", "customer", "id",
)

OFF_TOPIC_REPLY = (
    "I'm Countrylink's broadband support assistant and can only help with "
    "internet plans, billing, and technical queries. 😊\n"
    "For other help, please call 9765009850 or 9765005851."
)

def is_relevant_to_countrylink(message: str) -> bool:
    """
    Returns True if the message is likely related to Countrylink services.
    Short messages (greetings, confirmations) are always allowed through.
    """
    msg_lower = message.lower().strip()

    # Always allow very short messages (greetings, yes/no replies)
    if len(msg_lower) <= 20:
        return True

    # Allow if any relevant keyword found
    return any(kw in msg_lower for kw in RELEVANT_KEYWORDS)


# ── Groq Response ────────────────────────────────────────────────────────────
def _call_groq(user_id: str, user_message: str) -> str:
    """Inner function that calls the Groq API (run inside a thread)."""
    groq_client  = Groq(api_key=GROQ_API_KEY)  # created per-call; env var safe
    all_history  = load_history()
    user_history = all_history.get(user_id, [])

    # Build messages: system prompt + conversation history + new message
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add conversation history (Groq uses role: user/assistant)
    for entry in user_history:
        role    = "assistant" if entry["role"] == "model" else entry["role"]
        content = entry["parts"][0]["text"] if isinstance(entry["parts"][0], dict) else entry["parts"][0]
        messages.append({"role": role, "content": content})

    # Add current user message
    messages.append({"role": "user", "content": user_message})

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        max_completion_tokens=1024,
        temperature=0.7,
    )
    reply = response.choices[0].message.content

    # Save to history
    user_history.append({"role": "user",  "parts": [{"text": user_message}]})
    user_history.append({"role": "model", "parts": [{"text": reply}]})

    if len(user_history) > MAX_HISTORY_MESSAGES:
        user_history = user_history[-MAX_HISTORY_MESSAGES:]

    all_history[user_id] = user_history
    save_history(all_history)
    return reply


def get_groq_response(user_id: str, user_message: str) -> str:
    """Calls Groq API with timeout protection."""
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_call_groq, user_id, user_message)
            try:
                return future.result(timeout=GEMINI_TIMEOUT_SECS)
            except FuturesTimeout:
                logging.warning(f"Groq timeout for user {user_id}")
                return (
                    "Taking a bit longer than usual — please try again in a moment "
                    "or call 9765009850 / 9765005851 for immediate help."
                )
    except Exception as e:
        logging.error(f"Groq error for {user_id}: {e}")
        return (
            "Sorry, I'm having trouble right now. "
            "Please try again or call our support line."
        )
        
# ── Send WhatsApp Message ─────────────────────────────────────────────────────
def send_whatsapp_message(to: str, message: str):
    url     = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type":  "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to":   to,
        "type": "text",
        "text": {"body": message},
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=10)
        resp.raise_for_status()
        logging.info(f"Message sent to {to}")
    except requests.RequestException as e:
        logging.error(f"Failed to send message to {to}: {e}")


# ── Webhook Verification (GET) ────────────────────────────────────────────────
@app.route("/webhook", methods=["GET"])
def verify_webhook():
    mode      = request.args.get("hub.mode")
    token     = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        logging.info("Webhook verified successfully")
        return challenge, 200

    logging.warning("Webhook verification failed")
    return "Forbidden", 403


# ── Incoming Message Handler (POST) ──────────────────────────────────────────
@app.route("/webhook", methods=["POST"])
def handle_webhook():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"status": "ok"}), 200

    try:
        for entry in data.get("entry", []):
            for change in entry.get("changes", []):
                value = change.get("value", {})

                # Ignore delivery/read receipts
                if "statuses" in value:
                    continue

                for message in value.get("messages", []):
                    # ── FIX #4: Deduplicate — Meta sometimes sends webhooks twice ──
                    msg_id = message.get("id")
                    if msg_id:
                        if msg_id in PROCESSED_MESSAGE_IDS:
                            logging.info(f"Duplicate webhook ignored: {msg_id}")
                            continue
                        PROCESSED_MESSAGE_IDS.add(msg_id)
                        # Cap memory usage of the deduplication set
                        if len(PROCESSED_MESSAGE_IDS) > MAX_PROCESSED_IDS:
                            ids_list = list(PROCESSED_MESSAGE_IDS)
                            PROCESSED_MESSAGE_IDS.clear()
                            PROCESSED_MESSAGE_IDS.update(
                                ids_list[MAX_PROCESSED_IDS // 2 :]
                            )

                    from_number = message.get("from")
                    msg_type    = message.get("type")

                    # ── HUMAN/AI TOGGLE CHECK ──────────────────────
                    if get_conversation_mode(from_number) == 'human':
                        logging.info(f"Human mode active for {from_number} — skipping AI reply")
                        continue

                    if msg_type == "text":
                        user_text = message["text"]["body"]
                        logging.info(f"Received from {from_number}: {user_text[:80]}")
                        ai_reply = get_structured_reply_if_applicable(from_number, user_text, message_type="text")
                        if not ai_reply:
                            if not is_relevant_to_countrylink(user_text):
                                logging.info(f"Off-topic blocked from {from_number}: {user_text[:60]}")
                                ai_reply = OFF_TOPIC_REPLY
                            else:
                                ai_reply = get_groq_response(from_number, user_text)
                        send_whatsapp_message(from_number, ai_reply)
                    elif msg_type in ("image", "audio", "video", "document"):
                        if msg_type == "image":
                            ai_reply = get_structured_reply_if_applicable(
                                from_number, "", message_type="image"
                            )
                        else:
                            ai_reply = None
                        send_whatsapp_message(
                            from_number,
                            ai_reply or (
                                "Thanks for your message! I can only read text right now. "
                                "Please type your query and I'll help you."
                            ),
                        )
    except Exception as e:
        logging.error(f"Webhook processing error: {e}")
    # Always return 200 so Meta does not retry
    return jsonify({"status": "ok"}), 200

# ── Internal AI Endpoint (For Management Repo) ───────────────────────────────
@app.route("/webhook-ai", methods=["POST"])
def webhook_ai():
    try:
        data = request.get_json(silent=True) or {}

        user_text     = (data.get("message") or "").strip()
        from_number   = (data.get("phone") or "").strip()
        customer_name = (data.get("name") or "Customer").strip()
        message_type  = (data.get("message_type") or data.get("type") or "text").strip().lower()

        logging.info(
            f"Internal AI request from {from_number} ({customer_name}): "
            f"type={message_type} text={user_text[:80]}"
        )

        if not from_number or (not user_text and message_type != "image"):
            return jsonify({
                "status": "error",
                "reply": "Missing message or phone"
            }), 400

        # Structured intent handling follows the required flow order.
        ai_reply = get_structured_reply_if_applicable(
            from_number, user_text, customer_name, message_type=message_type
        )

        # AI fallback with pre-filter
        if not ai_reply:
            if not is_relevant_to_countrylink(user_text):
                logging.info(f"Off-topic blocked from {from_number}: {user_text[:60]}")
                ai_reply = OFF_TOPIC_REPLY
            else:
                ai_reply = get_groq_response(from_number, user_text)

        return jsonify({
            "status": "success",
            "reply": ai_reply
        }), 200

    except Exception as e:
        logging.error(f"/webhook-ai error: {e}")
        return jsonify({
            "status": "error",
            "reply": "Sorry, AI service unavailable right now."
        }), 500
# ── Health Check ──────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "running", "service": "Countrylink WhatsApp Agent"}), 200


if __name__ == "__main__":
    app.run(debug=False, port=5000)
