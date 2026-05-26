"""
Countrylink WhatsApp AI Agent — COMPLETE app.py
================================================
Drop this into your agent repo. It integrates with the management system
via the shared MySQL database (countrylinks_user_database).

Integration points:
- Reads whatsapp_conversations.human_takeover to check AI/Human mode
- /webhook-ai endpoint called by management system for AI replies
- /set-mode and /get-mode endpoints for dashboard toggle
- Same DB credentials as management system
"""

import os
import json
import logging
import re
import pymysql
import pymysql.cursors
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Optional, List, Tuple

from flask import Flask, request, jsonify
import requests
from groq import Groq


app = Flask(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

VERIFY_TOKEN     = os.environ.get("VERIFY_TOKEN")
WHATSAPP_TOKEN   = os.environ.get("WHATSAPP_TOKEN")
PHONE_NUMBER_ID  = os.environ.get("PHONE_NUMBER_ID")
GROQ_API_KEY     = os.environ.get("GROQ_API_KEY")

DB_HOST     = os.environ.get("DB_HOST", "localhost")
DB_NAME     = os.environ.get("DB_NAME", "countrylinks_user_database")
DB_USER     = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")

INTERNAL_API_TOKEN = os.environ.get("AGENT_INTERNAL_TOKEN", "")

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    filename="whatsapp_agent.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

# ── Constants ─────────────────────────────────────────────────────────────────

MAX_HISTORY_MESSAGES = 20
MAX_USERS_IN_HISTORY = 500
MAX_PROCESSED_IDS    = 1000
GROQ_TIMEOUT_SECS    = 15

PROCESSED_MESSAGE_IDS: set = set()

INVOICE_KEYWORDS = (
    "invoice", "bill", "billing", "payment", "paid", "unpaid",
    "due", "due date", "amount", "receipt", "outstanding"
)

# ── System Prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a WhatsApp customer support assistant EXCLUSIVELY for Countrylink Broadband Services.

You ONLY help with topics directly related to Countrylink Broadband:
- Internet plans, pricing, upgrades, and availability
- OTT add-on packages
- Technical troubleshooting (slow speed, disconnection, router issues)
- Bill payment, due dates, invoice lookup, and account information
- New connection requests and installation scheduling
- Reporting outages and service complaints

Current Countrylink plan information:

Internet-only packages:
1. Limited 50 Mbps: ₹499, 50 Mbps speed, 500 GB data
2. Unlimited 50 Mbps: ₹800, unlimited data, 50 Mbps speed
3. Unlimited 100 Mbps: ₹900, unlimited data, 100 Mbps speed
4. Unlimited 200 Mbps: ₹999, unlimited data, 200 Mbps speed
5. Unlimited 300 Mbps: ₹1099, unlimited data, 300 Mbps speed

Internet package notes:
- GST is excluded from the listed prices.
- Free 5G Wi-Fi router and free installation are subject to a technical and feasibility survey.
- All listed internet packages use fiber connectivity.
- Quarterly plans have a 20% discount.

OTT add-on packages:
1. Premium: ₹200 monthly. Includes Jio Hotstar, SonyLIV, Discovery Plus, and 10+ more apps.
2. VIP: ₹350 monthly. Includes Jio Hotstar, SonyLIV, ZEE5, Discovery Plus, and 10+ more apps.
3. Prime: ₹1999 quarterly. Includes Jio Hotstar, Amazon Prime Video, SonyLIV, ZEE5, Discovery Plus, and 10+ more apps.

Sales/support contact: 9765009850 or 9765005851

Guidelines:
- Be friendly, concise, and professional.
- Keep messages short and WhatsApp-friendly.
- Always respond in the same language the customer writes in.
- For invoice/bill questions, ask for registered 10-digit mobile number first.
- Do not invent details you don't have.

CRITICAL RULE — OFF-TOPIC MESSAGES:
If the message is NOT related to Countrylink Broadband, respond ONLY with:
"I'm Countrylink's broadband support assistant and can only help with internet plans, billing, and technical queries. For other help, please call 9765009850 or 9765005851. 😊"
"""

# ── File Paths ────────────────────────────────────────────────────────────────

BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE      = os.path.join(BASE_DIR, "conversation_history.json")
INVOICE_AUTH_FILE = os.path.join(BASE_DIR, "invoice_auth_requests.json")

# ── JSON Helpers ──────────────────────────────────────────────────────────────

def _load_json_file(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logging.error(f"Could not load {path}: {e}")
        return {}


def _save_json_file(path: str, data: dict):
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    except IOError as e:
        logging.error(f"Could not save {path}: {e}")


# ── Conversation History ──────────────────────────────────────────────────────

def load_history() -> dict:
    return _load_json_file(HISTORY_FILE)


def save_history(history: dict):
    if len(history) > MAX_USERS_IN_HISTORY:
        keys = list(history.keys())
        for k in keys[: MAX_USERS_IN_HISTORY // 2]:
            del history[k]
    _save_json_file(HISTORY_FILE, history)


# ── Invoice Auth State ────────────────────────────────────────────────────────

def load_invoice_auth_requests() -> dict:
    return _load_json_file(INVOICE_AUTH_FILE)

def save_invoice_auth_requests(m: dict):
    _save_json_file(INVOICE_AUTH_FILE, m)

def set_invoice_auth_pending(user_id: str):
    m = load_invoice_auth_requests()
    m[user_id] = True
    save_invoice_auth_requests(m)

def clear_invoice_auth_pending(user_id: str):
    m = load_invoice_auth_requests()
    if user_id in m:
        del m[user_id]
        save_invoice_auth_requests(m)

def is_invoice_auth_pending(user_id: str) -> bool:
    return bool(load_invoice_auth_requests().get(user_id))


# ── Database ──────────────────────────────────────────────────────────────────

def get_db_connection() -> pymysql.connections.Connection:
    return pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# HUMAN/AI TOGGLE — reads from whatsapp_conversations.human_takeover
# This is the same table the management system writes to via the toggle button
# ══════════════════════════════════════════════════════════════════════════════

def get_conversation_mode(phone: str) -> str:
    """
    Returns 'human' if human_takeover=1 in whatsapp_conversations, else 'ai'.
    Defaults to 'ai' on any error or missing row.
    """
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT human_takeover FROM whatsapp_conversations WHERE phone = %s LIMIT 1",
                (phone,)
            )
            row = cursor.fetchone()
        conn.close()

        if row and int(row.get('human_takeover', 0)) == 1:
            return 'human'
        return 'ai'
    except Exception as e:
        logging.error(f"get_conversation_mode error for {phone}: {e}")
        return 'ai'


# ── Invoice Helpers ───────────────────────────────────────────────────────────

def is_invoice_request(message: str) -> bool:
    return any(kw in message.lower() for kw in INVOICE_KEYWORDS)

def extract_mobile_number(message: str) -> Optional[str]:
    candidates = re.findall(r"(?:\+?91[-\s]?)?[6-9]\d(?:[-\s]?\d){8}", message)
    for candidate in candidates:
        digits = re.sub(r"\D", "", candidate)
        if len(digits) >= 10:
            return digits[-10:]
    return None

def normalize_sql_phone(column_name: str) -> str:
    return (
        f"RIGHT(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE("
        f"COALESCE({column_name}, ''), ' ', ''), '-', ''), '+', ''), '(', ''), ')', ''), 10)"
    )

def get_customer_and_invoices(mobile_number: str, limit: int = 5) -> Tuple[Optional[dict], List[dict]]:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            mobile_expr = normalize_sql_phone("mobile")
            phone_expr  = normalize_sql_phone("phone")
            cur.execute(f"""
                SELECT zoho_contact_id, contact_name, company_name, mobile, phone, outstanding_amount, status
                FROM zoho_customers
                WHERE {mobile_expr} = %s OR {phone_expr} = %s
                ORDER BY id DESC LIMIT 1
            """, (mobile_number, mobile_number))
            customer = cur.fetchone()
            if not customer:
                return None, []

            cur.execute("""
                SELECT invoice_number, customer_name, status, total, invoice_date, due_date, created_at
                FROM invoices
                WHERE zoho_contact_id = %s
                ORDER BY COALESCE(invoice_date, created_at) DESC, id DESC
                LIMIT %s
            """, (customer["zoho_contact_id"], limit))
            return customer, list(cur.fetchall())
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
        return "Sorry, I cannot access invoice details right now. Please call 9765009850 / 9765005851."

    if not customer:
        return "Sorry, I could not find an account with that mobile number. Please double-check or call 9765009850 / 9765005851."

    name = customer["contact_name"] or customer["company_name"] or "customer"
    outstanding = format_currency(customer["outstanding_amount"])

    if not invoices:
        return f"Thanks, {name}. I found your account, but no invoices are available right now. Please call 9765009850 / 9765005851."

    lines = [f"Thanks, {name}. I found your account.", f"Outstanding amount: {outstanding}", "Latest invoices:"]
    for i, inv in enumerate(invoices, 1):
        lines.append(f"{i}) Invoice {inv['invoice_number'] or 'N/A'}: {format_currency(inv['total'])}, {inv['status'] or 'N/A'}, due {inv['due_date'] or 'N/A'}")
    lines.append("For payment confirmation, call 9765009850 / 9765005851.")
    return "\n".join(lines)

def get_invoice_reply_if_applicable(user_id: str, user_message: str) -> Optional[str]:
    mobile = extract_mobile_number(user_message)
    invoice_req = is_invoice_request(user_message)

    if mobile and (invoice_req or is_invoice_auth_pending(user_id)):
        reply = format_invoice_lookup_reply(mobile)
        clear_invoice_auth_pending(user_id)
        return reply

    if invoice_req:
        set_invoice_auth_pending(user_id)
        return "Sure, I can check your invoice details. Please send your registered 10-digit mobile number for authentication."

    return None


# ── Topic Filter ──────────────────────────────────────────────────────────────

RELEVANT_KEYWORDS = (
    "plan", "price", "pricing", "mbps", "speed", "data", "fiber", "fibre",
    "broadband", "internet", "connection", "recharge", "upgrade", "package",
    "ott", "hotstar", "sonyliv", "zee5", "prime", "netflix", "wifi", "wi-fi",
    "router", "installation", "install", "new connection",
    "bill", "billing", "invoice", "payment", "paid", "unpaid", "due",
    "amount", "receipt", "outstanding", "renewal", "renew",
    "slow", "disconnect", "not working", "down", "outage", "issue", "problem",
    "complaint", "help", "support", "contact", "number", "repair", "technician",
    "hi", "hello", "hii", "hey", "helo", "namaste", "namaskar",
    "good morning", "good evening", "good afternoon",
    "thank", "thanks", "ok", "okay", "sure", "yes", "no",
    "account", "mobile", "number", "customer", "id",
)

OFF_TOPIC_REPLY = (
    "I'm Countrylink's broadband support assistant and can only help with "
    "internet plans, billing, and technical queries. 😊\n"
    "For other help, please call 9765009850 or 9765005851."
)

def is_relevant_to_countrylink(message: str) -> bool:
    msg_lower = message.lower().strip()
    if len(msg_lower) <= 20:
        return True
    return any(kw in msg_lower for kw in RELEVANT_KEYWORDS)


# ── Groq AI ───────────────────────────────────────────────────────────────────

def _call_groq(user_id: str, user_message: str) -> str:
    groq_client = Groq(api_key=GROQ_API_KEY)
    all_history = load_history()
    user_history = all_history.get(user_id, [])

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for entry in user_history:
        role = "assistant" if entry["role"] == "model" else entry["role"]
        content = entry["parts"][0]["text"] if isinstance(entry["parts"][0], dict) else entry["parts"][0]
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_message})

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        max_completion_tokens=1024,
        temperature=0.7,
    )
    reply = response.choices[0].message.content

    user_history.append({"role": "user",  "parts": [{"text": user_message}]})
    user_history.append({"role": "model", "parts": [{"text": reply}]})
    if len(user_history) > MAX_HISTORY_MESSAGES:
        user_history = user_history[-MAX_HISTORY_MESSAGES:]
    all_history[user_id] = user_history
    save_history(all_history)
    return reply


def get_groq_response(user_id: str, user_message: str) -> str:
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_call_groq, user_id, user_message)
            try:
                return future.result(timeout=GROQ_TIMEOUT_SECS)
            except FuturesTimeout:
                logging.warning(f"Groq timeout for {user_id}")
                return "Taking a bit longer than usual — please try again or call 9765009850 / 9765005851."
    except Exception as e:
        logging.error(f"Groq error for {user_id}: {e}")
        return "Sorry, I'm having trouble right now. Please try again or call our support line."


# ── Send WhatsApp Message ─────────────────────────────────────────────────────

def send_whatsapp_message(to: str, message: str):
    url = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"}
    payload = {"messaging_product": "whatsapp", "to": to, "type": "text", "text": {"body": message}}
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=10)
        resp.raise_for_status()
        logging.info(f"Message sent to {to}")
    except requests.RequestException as e:
        logging.error(f"Failed to send message to {to}: {e}")


# ── Webhook GET (Meta verification) ──────────────────────────────────────────

@app.route("/webhook", methods=["GET"])
def verify_webhook():
    mode      = request.args.get("hub.mode")
    token     = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")
    if mode == "subscribe" and token == VERIFY_TOKEN:
        logging.info("Webhook verified")
        return challenge, 200
    logging.warning("Webhook verification failed")
    return "Forbidden", 403


# ── Webhook POST (Incoming messages) ─────────────────────────────────────────

@app.route("/webhook", methods=["POST"])
def handle_webhook():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"status": "ok"}), 200

    try:
        for entry in data.get("entry", []):
            for change in entry.get("changes", []):
                value = change.get("value", {})
                if "statuses" in value:
                    continue

                for message in value.get("messages", []):
                    msg_id = message.get("id")
                    if msg_id:
                        if msg_id in PROCESSED_MESSAGE_IDS:
                            continue
                        PROCESSED_MESSAGE_IDS.add(msg_id)
                        if len(PROCESSED_MESSAGE_IDS) > MAX_PROCESSED_IDS:
                            ids_list = list(PROCESSED_MESSAGE_IDS)
                            PROCESSED_MESSAGE_IDS.clear()
                            PROCESSED_MESSAGE_IDS.update(ids_list[MAX_PROCESSED_IDS // 2:])

                    from_number = message.get("from")
                    msg_type    = message.get("type")

                    # ═══ HUMAN/AI TOGGLE CHECK ═══
                    mode = get_conversation_mode(from_number)
                    if mode == 'human':
                        logging.info(f"Human mode active for {from_number}, skipping AI")
                        continue

                    # ═══ AI MODE ═══
                    if msg_type == "text":
                        user_text = message["text"]["body"]
                        logging.info(f"From {from_number}: {user_text[:80]}")

                        ai_reply = get_invoice_reply_if_applicable(from_number, user_text)
                        if not ai_reply:
                            if not is_relevant_to_countrylink(user_text):
                                ai_reply = OFF_TOPIC_REPLY
                            else:
                                ai_reply = get_groq_response(from_number, user_text)
                        send_whatsapp_message(from_number, ai_reply)

                    elif msg_type in ("image", "audio", "video", "document"):
                        send_whatsapp_message(from_number,
                            "Thanks for your message! I can only read text right now. Please type your query.")

    except Exception as e:
        logging.error(f"Webhook error: {e}")

    return jsonify({"status": "ok"}), 200


# ── Internal AI Endpoint (called by management system) ────────────────────────

@app.route("/webhook-ai", methods=["POST"])
def webhook_ai():
    try:
        token = request.headers.get("X-Internal-Token", "")
        if INTERNAL_API_TOKEN and token != INTERNAL_API_TOKEN:
            return jsonify({"status": "error", "reply": "Unauthorized"}), 403

        data = request.get_json(silent=True) or {}
        user_text   = (data.get("message") or "").strip()
        from_number = (data.get("phone") or "").strip()

        if not user_text or not from_number:
            return jsonify({"status": "error", "reply": "Missing message or phone"}), 400

        ai_reply = get_invoice_reply_if_applicable(from_number, user_text)
        if not ai_reply:
            if not is_relevant_to_countrylink(user_text):
                ai_reply = OFF_TOPIC_REPLY
            else:
                ai_reply = get_groq_response(from_number, user_text)

        return jsonify({"status": "success", "reply": ai_reply}), 200
    except Exception as e:
        logging.error(f"/webhook-ai error: {e}")
        return jsonify({"status": "error", "reply": "AI service unavailable."}), 500


# ── Toggle Endpoints (management dashboard can also call these directly) ──────

@app.route("/set-mode", methods=["POST"])
def set_mode():
    token = request.headers.get("X-Internal-Token", "")
    if INTERNAL_API_TOKEN and token != INTERNAL_API_TOKEN:
        return jsonify({"error": "Unauthorized"}), 403

    data = request.get_json(silent=True) or {}
    phone = (data.get("phone") or "").strip()
    mode  = (data.get("mode") or "").strip().lower()

    if not phone:
        return jsonify({"error": "phone required"}), 400
    if mode not in ("ai", "human"):
        return jsonify({"error": "mode must be ai or human"}), 400

    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                "UPDATE whatsapp_conversations SET human_takeover = %s, updated_at = NOW() WHERE phone = %s",
                (1 if mode == "human" else 0, phone)
            )
        conn.commit()
        conn.close()
        return jsonify({"status": "success", "phone": phone, "mode": mode}), 200
    except Exception as e:
        logging.error(f"set-mode error: {e}")
        return jsonify({"error": "DB error"}), 500


@app.route("/get-mode", methods=["GET"])
def get_mode():
    token = request.headers.get("X-Internal-Token", "")
    if INTERNAL_API_TOKEN and token != INTERNAL_API_TOKEN:
        return jsonify({"error": "Unauthorized"}), 403

    phone = (request.args.get("phone") or "").strip()
    if not phone:
        return jsonify({"error": "phone required"}), 400
    return jsonify({"phone": phone, "mode": get_conversation_mode(phone)}), 200


# ── Health ────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "running", "service": "Countrylink WhatsApp Agent"}), 200


if __name__ == "__main__":
    app.run(debug=False, port=5000)
