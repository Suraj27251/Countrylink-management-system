import os
import logging
import hmac
import hashlib
import json
import re
from pathlib import Path
import csv
import pickle
import socket
import threading
import time
import mimetypes
import subprocess
import uuid
import requests
import mysql.connector
from mysql.connector import Error as MySQLError
from flask import Flask, render_template, request, redirect, url_for, jsonify, session, flash
import sqlite3
from datetime import datetime
from collections import defaultdict
from functools import wraps
from werkzeug.utils import secure_filename
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-change-me')

DB_PATH = os.environ.get('DATABASE_PATH')
if not DB_PATH:
    default_path = Path(app.root_path) / 'database' / 'complaints.db'
    DB_PATH = str(default_path if default_path.exists() else Path(app.root_path) / 'complaints.db')

def get_db_connection():
    timeout = int(os.environ.get('SQLITE_BUSY_TIMEOUT_SECONDS', '30'))
    conn = sqlite3.connect(DB_PATH, timeout=timeout)
    try:
        conn.execute("PRAGMA busy_timeout = 30000")
        conn.execute("PRAGMA journal_mode = WAL")
    except sqlite3.DatabaseError:
        app.logger.warning("Unable to apply SQLite PRAGMA settings.", exc_info=True)
    return conn

WHATSAPP_API_VERSION = os.environ.get('WHATSAPP_API_VERSION', 'v20.0')
WEBHOOK_VERIFY_TOKEN = os.environ.get('WEBHOOK_VERIFY_TOKEN', '')
WHATSAPP_MEDIA_DIR = Path(app.root_path) / 'static' / 'uploads' / 'whatsapp'
WEBHOOK_ASYNC_PROCESSING = os.environ.get('WEBHOOK_ASYNC_PROCESSING', 'true').strip().lower() in {'1', 'true', 'yes', 'on'}
WHATSAPP_WEBHOOK_LOG_PATH = os.environ.get(
    'WHATSAPP_WEBHOOK_LOG_PATH',
    str(Path.home() / 'whatsapp_webhook.log')
)
AUTO_NOTIFICATION_TEMPLATE_NAME = 'notification_team'
AUTO_NOTIFICATION_TEMPLATE_LANGUAGE = 'en'
AUTO_NOTIFICATION_RECIPIENTS = ['8149912379', '8055782345']
REQUIRED_WHATSAPP_ENV_VARS = (
    'META_ACCESS_TOKEN',
    'PHONE_NUMBER_ID',
    'WEBHOOK_VERIFY_TOKEN',
    'WHATSAPP_API_VERSION',
)
DEFAULT_ENQUIRY_FLOW_NAME = os.environ.get('WHATSAPP_ENQUIRY_FLOW_NAME', 'enquiry').strip() or 'enquiry'
WHATSAPP_DB_WRITE_LOCK = threading.Lock()
WHATSAPP_WEBHOOK_LOGGER = None
WHATSAPP_WEBHOOK_LOGGER_PATH = None


def get_whatsapp_webhook_logger():
    global WHATSAPP_WEBHOOK_LOGGER, WHATSAPP_WEBHOOK_LOGGER_PATH
    if WHATSAPP_WEBHOOK_LOGGER is not None:
        return WHATSAPP_WEBHOOK_LOGGER

    logger = logging.getLogger('whatsapp_webhook')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        candidate_paths = [
            Path(WHATSAPP_WEBHOOK_LOG_PATH),
            Path.home() / 'whatsapp_webhook.log',
            Path('/tmp/whatsapp_webhook.log'),
        ]
        file_handler_attached = False
        for log_path in candidate_paths:
            try:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open('a', encoding='utf-8'):
                    pass
                file_handler = logging.FileHandler(log_path, encoding='utf-8')
                file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
                logger.addHandler(file_handler)
                WHATSAPP_WEBHOOK_LOGGER_PATH = str(log_path)
                file_handler_attached = True
                break
            except OSError:
                app.logger.warning("Could not open webhook log file at %s", log_path, exc_info=True)

        if not file_handler_attached:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
            logger.addHandler(stream_handler)
            WHATSAPP_WEBHOOK_LOGGER_PATH = 'stream-only'
            app.logger.warning("WhatsApp webhook logger is using stream-only fallback (no writable file path found).")
    WHATSAPP_WEBHOOK_LOGGER = logger
    return logger


def log_whatsapp_env_warnings():
    missing_vars = [key for key in REQUIRED_WHATSAPP_ENV_VARS if not os.environ.get(key)]
    if missing_vars:
        app.logger.warning("Missing WhatsApp environment variables: %s", ', '.join(missing_vars))


log_whatsapp_env_warnings()

RAZORPAY_API_BASE = 'https://api.razorpay.com/v1'
ZOHO_TOKEN_URL = "https://accounts.zoho.in/oauth/v2/token"
ZOHO_TIMEOUT_SECONDS = int(os.environ.get("ZOHO_TIMEOUT_SECONDS", "30"))
ZOHO_MAX_RETRIES = int(os.environ.get("ZOHO_MAX_RETRIES", "3"))
ZOHO_RETRY_BACKOFF = float(os.environ.get("ZOHO_RETRY_BACKOFF", "1.0"))
MYSQL_DB_HOST = os.environ.get("MYSQL_DB_HOST", "localhost")
MYSQL_DB_NAME = os.environ.get("MYSQL_DB_NAME", "countrylinks_user_database")
MYSQL_DB_USER = os.environ.get("MYSQL_DB_USER", "countrylinks_Suraj27251")
MYSQL_DB_PASSWORD = os.environ.get("MYSQL_DB_PASSWORD", "")


def create_retryable_session():
    retry = Retry(
        total=ZOHO_MAX_RETRIES,
        connect=ZOHO_MAX_RETRIES,
        read=ZOHO_MAX_RETRIES,
        backoff_factor=ZOHO_RETRY_BACKOFF,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset({"GET", "POST"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def get_zoho_access_token():
    org_client_id = os.environ.get("ZOHO_CLIENT_ID")
    org_client_secret = os.environ.get("ZOHO_CLIENT_SECRET")
    refresh_token = os.environ.get("ZOHO_REFRESH_TOKEN")
    if not org_client_id or not org_client_secret or not refresh_token:
        raise RuntimeError("Missing Zoho OAuth credentials in environment variables.")

    payload = {
        "refresh_token": refresh_token,
        "client_id": org_client_id,
        "client_secret": org_client_secret,
        "grant_type": "refresh_token",
    }
    session = create_retryable_session()
    try:
        response = session.post(ZOHO_TOKEN_URL, data=payload, timeout=ZOHO_TIMEOUT_SECONDS)
        response.raise_for_status()
        token_data = response.json()
    except requests.RequestException as exc:
        app.logger.error("Failed to refresh Zoho access token: %s", exc, exc_info=True)
        raise

    access_token = token_data.get("access_token")
    if not access_token:
        app.logger.error("Zoho token response missing access_token: %s", token_data)
        raise RuntimeError("Could not retrieve Zoho access token.")
    return access_token


def get_all_zoho_customers():
    org_id = os.environ.get("ZOHO_ORG_ID")
    api_domain = os.environ.get("ZOHO_API_DOMAIN")
    if not org_id or not api_domain:
        raise RuntimeError("Missing ZOHO_ORG_ID or ZOHO_API_DOMAIN in environment variables.")

    access_token = get_zoho_access_token()
    session = create_retryable_session()
    headers = {"Authorization": f"Zoho-oauthtoken {access_token}"}
    page = 1
    per_page = 200
    customers = []

    while True:
        url = f"{api_domain.rstrip('/')}/books/v3/contacts"
        params = {
            "organization_id": org_id,
            "page": page,
            "per_page": per_page,
        }
        try:
            response = session.get(url, headers=headers, params=params, timeout=ZOHO_TIMEOUT_SECONDS)
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            app.logger.error("Failed to fetch Zoho customers on page %s: %s", page, exc, exc_info=True)
            raise

        current_batch = payload.get("contacts", [])
        customers.extend(current_batch)
        app.logger.info("Fetched Zoho contacts page %s: %s records", page, len(current_batch))

        if not current_batch or len(current_batch) < per_page:
            break
        page += 1

    app.logger.info("Total Zoho customers fetched: %s", len(customers))
    return customers


def save_customers_to_db(customers):
    if not customers:
        return 0, 0

    inserted = 0
    updated = 0
    conn = None
    cursor = None

    query = """
        INSERT INTO zoho_customers (
            zoho_contact_id,
            contact_name,
            company_name,
            email,
            phone,
            mobile,
            status,
            currency_code,
            outstanding_amount,
            created_time,
            last_modified_time
        ) VALUES (
            %(zoho_contact_id)s,
            %(contact_name)s,
            %(company_name)s,
            %(email)s,
            %(phone)s,
            %(mobile)s,
            %(status)s,
            %(currency_code)s,
            %(outstanding_amount)s,
            %(created_time)s,
            %(last_modified_time)s
        )
        ON DUPLICATE KEY UPDATE
            contact_name = VALUES(contact_name),
            company_name = VALUES(company_name),
            email = VALUES(email),
            phone = VALUES(phone),
            mobile = VALUES(mobile),
            status = VALUES(status),
            currency_code = VALUES(currency_code),
            outstanding_amount = VALUES(outstanding_amount),
            created_time = VALUES(created_time),
            last_modified_time = VALUES(last_modified_time)
    """

    try:
        conn = mysql.connector.connect(
            host=MYSQL_DB_HOST,
            database=MYSQL_DB_NAME,
            user=MYSQL_DB_USER,
            password=MYSQL_DB_PASSWORD,
        )
        cursor = conn.cursor()

        for customer in customers:
            data = {
                "zoho_contact_id": customer.get("contact_id"),
                "contact_name": customer.get("contact_name"),
                "company_name": customer.get("company_name"),
                "email": customer.get("email"),
                "phone": customer.get("phone"),
                "mobile": customer.get("mobile"),
                "status": customer.get("status"),
                "currency_code": customer.get("currency_code"),
                "outstanding_amount": customer.get("outstanding_receivable_amount"),
                "created_time": customer.get("created_time"),
                "last_modified_time": customer.get("last_modified_time"),
            }
            if not data["zoho_contact_id"]:
                app.logger.warning("Skipping customer without contact_id: %s", customer)
                continue

            cursor.execute(query, data)
            if cursor.rowcount == 1:
                inserted += 1
            elif cursor.rowcount == 2:
                updated += 1

        conn.commit()
        app.logger.info("Zoho customer sync committed. Inserted=%s Updated=%s", inserted, updated)
        return inserted, updated
    except MySQLError as exc:
        if conn and conn.is_connected():
            conn.rollback()
        app.logger.error("Database error while saving Zoho customers: %s", exc, exc_info=True)
        raise
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()


def _get_first_env(*keys):
    for key in keys:
        value = os.environ.get(key)
        if value:
            return value.strip()
    return ''


def get_razorpay_key_id():
    return _get_first_env('RAZORPAY_KEY_ID', 'RAZORPAY_KEY', 'RAZORPAY_API_KEY', 'KEY_ID')


def get_razorpay_key_secret():
    return _get_first_env('RAZORPAY_KEY_SECRET', 'RAZORPAY_SECRET', 'RAZORPAY_API_SECRET', 'KEY_SECRET')


def get_missing_razorpay_env_keys():
    missing = []
    if not get_razorpay_key_id():
        missing.append('RAZORPAY_KEY_ID')
    if not get_razorpay_key_secret():
        missing.append('RAZORPAY_KEY_SECRET')
    return missing

def razorpay_config_ready():
    return bool(get_razorpay_key_id() and get_razorpay_key_secret())


def create_razorpay_order(amount_paise, receipt, notes=None):
    key_id = get_razorpay_key_id()
    key_secret = get_razorpay_key_secret()
    if not key_id or not key_secret:
        raise RuntimeError('Razorpay credentials are not configured')

    payload = {
        'amount': int(amount_paise),
        'currency': 'INR',
        'receipt': receipt,
        'payment_capture': 1,
    }
    if notes:
        payload['notes'] = notes

    response = requests.post(
        f"{RAZORPAY_API_BASE}/orders",
        auth=(key_id, key_secret),
        json=payload,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def verify_razorpay_signature(order_id, payment_id, signature):
    key_secret = get_razorpay_key_secret()
    if not key_secret:
        return False
    message = f"{order_id}|{payment_id}".encode('utf-8')
    expected = hmac.new(key_secret.encode('utf-8'), message, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


# ---- WhatsApp helpers ----
def normalize_mobile(raw_mobile):
    if not raw_mobile:
        return ''
    digits = ''.join(ch for ch in str(raw_mobile).strip() if ch.isdigit())
    # Normalize India numbers like 91XXXXXXXXXX into a consistent 10-digit key
    # so webhook numbers and locally stored contacts match reliably.
    if len(digits) == 12 and digits.startswith('91'):
        digits = digits[2:]
    return digits


def conversation_mobile_key(raw_mobile):
    digits = normalize_mobile(raw_mobile)
    if len(digits) >= 10:
        return digits[-10:]
    return digits


def mobiles_equivalent(left_mobile, right_mobile):
    left = normalize_mobile(left_mobile)
    right = normalize_mobile(right_mobile)
    if not left or not right:
        return False
    if left == right:
        return True
    return conversation_mobile_key(left) == conversation_mobile_key(right)


def whatsapp_config_ready():
    return bool(
        os.environ.get('META_ACCESS_TOKEN')
        and os.environ.get('PHONE_NUMBER_ID')
        and os.environ.get('WABA_ID')
    )


def get_whatsapp_headers():
    token = os.environ.get('META_ACCESS_TOKEN')
    if not token:
        raise RuntimeError("Missing META_ACCESS_TOKEN")
    return {"Authorization": f"Bearer {token}"}


def get_whatsapp_phone_number_id():
    phone_number_id = os.environ.get('PHONE_NUMBER_ID')
    if not phone_number_id:
        raise RuntimeError("Missing PHONE_NUMBER_ID")
    return phone_number_id


def get_whatsapp_waba_id():
    waba_id = os.environ.get('WABA_ID')
    if not waba_id:
        raise RuntimeError("Missing WABA_ID")
    return waba_id


def detect_message_type(mime_type):
    if mime_type and mime_type.startswith('image/'):
        return 'image'
    if mime_type and mime_type.startswith('video/'):
        return 'video'
    if mime_type and mime_type.startswith('audio/'):
        return 'audio'
    return 'document'


def ensure_media_dir():
    WHATSAPP_MEDIA_DIR.mkdir(parents=True, exist_ok=True)


def save_media_file(file_storage, fallback_name=None):
    ensure_media_dir()
    original_name = secure_filename(file_storage.filename or '') or fallback_name or 'attachment'
    unique_prefix = uuid.uuid4().hex
    saved_name = f"{unique_prefix}_{original_name}"
    file_path = WHATSAPP_MEDIA_DIR / saved_name
    file_storage.save(file_path)
    return file_path, saved_name


def download_whatsapp_media(media_id, fallback_name=None):
    headers = get_whatsapp_headers()
    url = f"https://graph.facebook.com/{WHATSAPP_API_VERSION}/{media_id}"
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    payload = response.json()
    download_url = payload.get('url')
    mime_type = payload.get('mime_type')

    if not download_url:
        raise RuntimeError("Missing download URL from WhatsApp")

    file_response = requests.get(download_url, headers=headers, timeout=60)
    file_response.raise_for_status()

    ensure_media_dir()
    ext = mimetypes.guess_extension(mime_type or '') or ''
    safe_name = secure_filename(fallback_name or '')
    if safe_name:
        filename = safe_name
    else:
        filename = f"{media_id}{ext or '.bin'}"
    file_path = WHATSAPP_MEDIA_DIR / filename
    if file_path.exists():
        filename = f"{media_id}_{uuid.uuid4().hex}{ext or '.bin'}"
        file_path = WHATSAPP_MEDIA_DIR / filename
    with open(file_path, 'wb') as file_handle:
        file_handle.write(file_response.content)
    return file_path, filename, mime_type


def extract_text_body(message):
    return (
        message.get('text', {}).get('body')
        or message.get('button', {}).get('text')
        or message.get('interactive', {}).get('button_reply', {}).get('title')
        or message.get('interactive', {}).get('list_reply', {}).get('title')
        or ''
    )


def process_incoming_message(message, metadata):
    message_type = message.get('type', 'text')
    message_id = message.get('id')
    app.logger.info("Processing inbound WhatsApp message id=%s type=%s", message_id or 'no-id', message_type)

    timestamp_unix = message.get('timestamp')
    created_at = datetime.fromtimestamp(int(timestamp_unix)).strftime('%Y-%m-%d %H:%M:%S') if timestamp_unix else datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    from_id = message.get('from', '')
    contact = metadata.get('contacts_by_wa', {}).get(from_id, {})
    profile = contact.get('profile', {}) if isinstance(contact, dict) else {}
    name = (profile.get('name') or metadata.get('display_phone_number') or 'Unknown').strip()
    mobile = normalize_mobile((contact.get('wa_id') if isinstance(contact, dict) else '') or from_id)

    text_body = extract_text_body(message)
    media_id = None
    media_url = None
    media_mime_type = None
    file_name = None
    latitude = None
    longitude = None

    if message_type == 'location':
        location_info = message.get('location', {})
        latitude = location_info.get('latitude')
        longitude = location_info.get('longitude')
        location_name = location_info.get('name') or ''
        location_address = location_info.get('address') or ''
        if location_name and location_address:
            text_body = f"{location_name}\n{location_address}"
        else:
            text_body = location_name or location_address or text_body

    if message_type in {'image', 'video', 'audio', 'document'}:
        media_info = message.get(message_type, {})
        media_id = media_info.get('id')
        text_body = media_info.get('caption', '') or text_body
        file_name = media_info.get('filename') if message_type == 'document' else None
        media_mime_type = media_info.get('mime_type')

        if media_id and whatsapp_config_ready():
            try:
                downloaded_path, stored_name, mime_type = download_whatsapp_media(media_id, file_name)
                media_url = f"uploads/whatsapp/{stored_name}"
                media_mime_type = mime_type or media_mime_type
                app.logger.info("Downloaded WhatsApp media id=%s to %s", media_id, downloaded_path)
            except Exception as exc:
                app.logger.warning("Failed to download WhatsApp media id=%s: %s", media_id, exc)

    if not text_body:
        text_body = safe_message_preview(message_type, text_body)

    if not mobile:
        app.logger.error(
            "Skipping inbound message id=%s due to missing sender mobile. Raw from=%s payload=%s",
            message_id or 'no-id',
            from_id,
            json.dumps(message, ensure_ascii=False)[:1000],
        )
        return

    inserted_new_message = False
    is_new_chat = False
    try:
        # Serialize inbound DB writes with status updates/poll reads to reduce
        # race windows where the poll request can miss just-arriving rows.
        with WHATSAPP_DB_WRITE_LOCK:
            conn = get_db_connection()
            c = conn.cursor()
            c.execute(
                """
                SELECT 1
                FROM whatsapp_messages
                WHERE mobile = ? AND direction = 'inbound'
                LIMIT 1
                """,
                (mobile,),
            )
            is_new_chat = c.fetchone() is None

            if message_id:
                c.execute(
                    """
                    INSERT OR IGNORE INTO whatsapp_messages
                    (message_id, name, mobile, direction, message_type, text, media_id, media_url, media_mime_type, file_name, latitude, longitude, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        message_id,
                        name or None,
                        mobile,
                        'inbound',
                        message_type,
                        text_body,
                        media_id,
                        media_url,
                        media_mime_type,
                        file_name,
                        latitude,
                        longitude,
                        created_at,
                    ),
                )
                inserted_new_message = c.rowcount > 0
            else:
                c.execute(
                    """
                    INSERT INTO whatsapp_messages
                    (message_id, name, mobile, direction, message_type, text, media_id, media_url, media_mime_type, file_name, latitude, longitude, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        None,
                        name or None,
                        mobile,
                        'inbound',
                        message_type,
                        text_body,
                        media_id,
                        media_url,
                        media_mime_type,
                        file_name,
                        latitude,
                        longitude,
                        created_at,
                    ),
                )
                inserted_new_message = True
            conn.commit()
            conn.close()

        app.logger.info(
            "Stored inbound WhatsApp message id=%s mobile=%s",
            message_id or 'no-id',
            mobile
        )

        # =========================
        # MySQL AI Inbox Sync
        # =========================

        try:
            mysql_conn = mysql.connector.connect(
                host=MYSQL_DB_HOST,
                database=MYSQL_DB_NAME,
                user=MYSQL_DB_USER,
                password=MYSQL_DB_PASSWORD,
            )

            mysql_cursor = mysql_conn.cursor(dictionary=True)

            # Find existing conversation
            mysql_cursor.execute("""
                SELECT id
                FROM whatsapp_conversations
                WHERE phone = %s
                LIMIT 1
            """, (mobile,))

            conversation = mysql_cursor.fetchone()

            if conversation:
                conversation_id = conversation["id"]

            else:
                mysql_cursor.execute("""
                    INSERT INTO whatsapp_conversations (
                        phone,
                        customer_name,
                        last_message,
                        last_message_at,
                        unread_count,
                        ai_enabled,
                        created_at,
                        updated_at
                    )
                    VALUES (%s, %s, %s, NOW(), 1, 1, NOW(), NOW())
                """, (
                    mobile,
                    name,
                    text_body
                ))

                mysql_conn.commit()

                conversation_id = mysql_cursor.lastrowid

            # Save customer message
            mysql_cursor.execute("""
                INSERT INTO whatsapp_messages (
                    conversation_id,
                    whatsapp_message_id,
                    sender_type,
                    phone,
                    message_text,
                    message_type,
                    media_url,
                    status,
                    created_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
            """, (
                conversation_id,
                message_id,
                'customer',
                mobile,
                text_body,
                message_type,
                media_url,
                'received'
            ))

            # Update conversation
            mysql_cursor.execute("""
                UPDATE whatsapp_conversations
                SET
                    last_message = %s,
                    last_message_at = NOW(),
                    unread_count = unread_count + 1,
                    updated_at = NOW()
                WHERE id = %s
            """, (
                text_body,
                conversation_id
            ))

            mysql_conn.commit()

            mysql_cursor.close()
            mysql_conn.close()

            app.logger.info(
                "MySQL inbox sync success mobile=%s conversation_id=%s",
                mobile,
                conversation_id
            )

        except Exception:
            app.logger.exception(
                "MySQL inbox sync failed for mobile=%s",
                mobile
            )

    except Exception:
        app.logger.exception(
            "Failed storing inbound WhatsApp message id=%s mobile=%s",
            message_id or 'no-id',
            mobile,
        )
        raise
def process_whatsapp_webhook_payload(data):
    webhook_logger = get_whatsapp_webhook_logger()
    try:
        app.logger.info("Webhook POST processing started: object=%s entries=%s", data.get('object'), len(data.get('entry', [])))
        webhook_logger.info("processing_started object=%s entries=%s", data.get('object'), len(data.get('entry', [])))
        for entry in data.get('entry', []):
            for change in entry.get('changes', []):
                value = change.get('value', {})
                contacts = value.get('contacts', [])
                messages = value.get('messages', [])
                statuses = value.get('statuses', [])
                webhook_logger.info(
                    "change_received field=%s messages=%s statuses=%s",
                    change.get('field'),
                    len(messages or []),
                    len(statuses or []),
                )
                metadata = {
                    'display_phone_number': value.get('metadata', {}).get('display_phone_number', ''),
                    'contacts_by_wa': {
                        contact.get('wa_id'): contact
                        for contact in contacts
                        if contact.get('wa_id')
                    },
                }

                for message in messages:
                    try:
                        process_incoming_message(message, metadata)
                    except Exception:
                        app.logger.exception("Failed processing incoming message event.")
                        webhook_logger.exception("incoming_message_processing_failed")

                for status_event in statuses:
                    try:
                        process_message_status_event(status_event)
                    except Exception:
                        app.logger.exception("Failed processing message status event.")
                        webhook_logger.exception("status_event_processing_failed")
    except Exception:
        app.logger.exception("Webhook payload parsing failed.")
        webhook_logger.exception("payload_parsing_failed")


def parse_webhook_request_json():
    webhook_logger = get_whatsapp_webhook_logger()
    data = request.get_json(force=True, silent=True)
    if data:
        webhook_logger.info("request_json_parsed_via_flask")
        return data

    raw_body = (request.get_data(cache=False, as_text=True) or '').strip()
    if not raw_body:
        webhook_logger.warning("request_body_empty_or_unreadable")
        return None

    try:
        webhook_logger.info("request_json_parsed_via_raw_body")
        return json.loads(raw_body)
    except json.JSONDecodeError:
        app.logger.warning("Webhook body was not valid JSON. body_preview=%s", raw_body[:500])
        webhook_logger.warning("request_json_invalid body_preview=%s", raw_body[:500])
        return None


def store_webhook_audit_event(data, note):
    webhook_logger = get_whatsapp_webhook_logger()
    try:
        payload = json.dumps(data, ensure_ascii=False)[:2000] if data is not None else None
        entries = data.get('entry', []) if isinstance(data, dict) else []
        message_events = 0
        status_events = 0
        for entry in entries:
            for change in entry.get('changes', []):
                value = change.get('value', {})
                message_events += len(value.get('messages', []) or [])
                status_events += len(value.get('statuses', []) or [])

        conn = get_db_connection()
        c = conn.cursor()
        c.execute(
            '''
            INSERT INTO whatsapp_webhook_audit
            (note, message_events, status_events, payload_preview, created_at)
            VALUES (?, ?, ?, ?, ?)
            ''',
            (note, message_events, status_events, payload, datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
        )
        conn.commit()
        conn.close()
        webhook_logger.info("audit_saved note=%s message_events=%s status_events=%s", note, message_events, status_events)
    except Exception:
        app.logger.exception("Failed to store webhook audit event.")
        webhook_logger.exception("audit_save_failed note=%s", note)


def process_message_status_event(status_event):
    message_id = status_event.get('id')
    status = (status_event.get('status') or '').strip().lower() or 'unknown'
    timestamp_unix = status_event.get('timestamp')
    status_time = datetime.fromtimestamp(int(timestamp_unix)).strftime('%Y-%m-%d %H:%M:%S') if timestamp_unix else datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    errors = status_event.get('errors') or []
    reason = ''
    if errors:
        first_error = errors[0] or {}
        reason = first_error.get('details') or first_error.get('title') or first_error.get('message') or ''

    app.logger.info("Message status update received: message_id=%s status=%s", message_id or 'no-id', status)
    if not message_id:
        return

    with WHATSAPP_DB_WRITE_LOCK:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute(
            """
            UPDATE whatsapp_messages
            SET delivery_status = ?, error_reason = ?, created_at = CASE WHEN created_at IS NULL THEN ? ELSE created_at END
            WHERE message_id = ?
            """,
            (status, reason or None, status_time, message_id),
        )
        conn.commit()
        conn.close()


def send_auto_notification_templates(inbound_message_id, sender_mobile):
    if not whatsapp_config_ready():
        app.logger.warning(
            "Skipping auto template send for inbound message id=%s because WhatsApp configuration is incomplete.",
            inbound_message_id or 'no-id',
        )
        return

    chat_key = sender_mobile

    conn = get_db_connection()
    c = conn.cursor()
    for raw_mobile in AUTO_NOTIFICATION_RECIPIENTS:
        target_mobile = normalize_mobile(raw_mobile)
        if not target_mobile:
            app.logger.warning("Skipping invalid auto template recipient mobile=%s", raw_mobile)
            continue

        c.execute(
            """
            SELECT 1
            FROM whatsapp_template_notifications
            WHERE chat_mobile = ? AND target_mobile = ?
            LIMIT 1
            """,
            (chat_key, target_mobile),
        )
        if c.fetchone():
            app.logger.info(
                "Auto template already handled for chat mobile=%s to mobile=%s",
                chat_key,
                target_mobile,
            )
            continue

        c.execute(
            """
            INSERT INTO whatsapp_template_notifications
            (inbound_message_id, chat_mobile, target_mobile, template_name, language_code, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                inbound_message_id,
                chat_key,
                target_mobile,
                AUTO_NOTIFICATION_TEMPLATE_NAME,
                AUTO_NOTIFICATION_TEMPLATE_LANGUAGE,
                'pending',
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            ),
        )

        try:
            result = send_whatsapp_template_message(
                target_mobile,
                AUTO_NOTIFICATION_TEMPLATE_NAME,
                AUTO_NOTIFICATION_TEMPLATE_LANGUAGE,
            )
            outbound_message_id = (result.get('messages') or [{}])[0].get('id')
            sent_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            c.execute(
                """
                UPDATE whatsapp_template_notifications
                SET status = ?, outbound_message_id = ?, error_reason = ?, sent_at = ?
                WHERE chat_mobile = ? AND target_mobile = ?
                """,
                ('sent', outbound_message_id, None, sent_at, chat_key, target_mobile),
            )
            c.execute(
                """
                INSERT INTO whatsapp_messages
                (message_id, name, mobile, direction, message_type, text, delivery_status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    outbound_message_id,
                    'System',
                    target_mobile,
                    'outbound',
                    'template',
                    f"Template: {AUTO_NOTIFICATION_TEMPLATE_NAME}",
                    'accepted',
                    sent_at,
                ),
            )
        except Exception as exc:
            error_reason = str(exc)
            c.execute(
                """
                UPDATE whatsapp_template_notifications
                SET status = ?, error_reason = ?, sent_at = ?
                WHERE chat_mobile = ? AND target_mobile = ?
                """,
                ('failed', error_reason, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), chat_key, target_mobile),
            )
            app.logger.error(
                "Failed auto template send for chat mobile=%s to %s: %s",
                chat_key,
                target_mobile,
                exc,
            )

    conn.commit()
    conn.close()


def safe_message_preview(message_type, text):
    if text:
        return text
    return {
        "image": "📷 Photo",
        "video": "🎥 Video",
        "audio": "🎵 Audio",
        "document": "📄 Document",
        "sticker": "🧩 Sticker",
        "reaction": "😊 Reaction",
        "location": "📍 Location",
        "contacts": "👤 Contact",
        "unknown": "New message",
    }.get(message_type, "New message")


def load_whatsapp_rows(conn):
    # Ensure dictionary-like access (row["id"], row["mobile"], etc.) works
    # regardless of which caller created the connection.
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("""
        SELECT id, message_id, name, mobile, direction, message_type, text, media_url, file_name, media_mime_type, latitude, longitude, delivery_status, error_reason, created_at
        FROM whatsapp_messages
        ORDER BY datetime(created_at) DESC, id DESC
    """)
    rows = c.fetchall()
    legacy_mode = False
    if not rows:
        legacy_mode = True
        c.execute("""
            SELECT id, name, mobile, complaint AS text, created_at
            FROM complaints
            WHERE source = 'WhatsApp'
            ORDER BY datetime(created_at) DESC, id DESC
        """)
        legacy_rows = c.fetchall()
        rows = [
            {
                "id": row["id"],
                "message_id": None,
                "name": row["name"],
                "mobile": normalize_mobile(row["mobile"]),
                "direction": "inbound",
                "message_type": "text",
                "text": row["text"],
                "media_url": None,
                "file_name": None,
                "media_mime_type": None,
                "latitude": None,
                "longitude": None,
                "delivery_status": None,
                "error_reason": None,
                "created_at": row["created_at"],
            }
            for row in legacy_rows
        ]
    return rows, legacy_mode


def _is_usable_contact_name(name):
    normalized = (name or '').strip().lower()
    return bool(normalized and normalized not in {'unknown', '.', 'agent'})


def build_whatsapp_contacts(rows):
    latest_by_mobile = {}
    preferred_name_by_mobile = {}

    def value(row, key, default=None):
        if isinstance(row, dict):
            return row.get(key, default)
        try:
            return row[key]
        except Exception:
            return default

    for row in rows:
        mobile_raw = normalize_mobile(value(row, "mobile", ''))
        mobile_key = conversation_mobile_key(mobile_raw)
        if not mobile_key:
            continue

        if mobile_key not in latest_by_mobile:
            latest_by_mobile[mobile_key] = {
                "mobile": mobile_raw,
                "name": value(row, "name") or mobile_raw,
                "preview": safe_message_preview(value(row, "message_type", 'unknown'), value(row, "text")),
                "created_at": value(row, "created_at"),
            }

        row_name = value(row, "name")
        if value(row, "direction") == 'inbound' and _is_usable_contact_name(row_name):
            preferred_name_by_mobile[mobile_key] = row_name

    for mobile_key, contact in latest_by_mobile.items():
        if preferred_name_by_mobile.get(mobile_key):
            contact["name"] = preferred_name_by_mobile[mobile_key]

    return sorted(latest_by_mobile.values(), key=lambda item: item["created_at"], reverse=True)


def upload_media_to_whatsapp(file_path, mime_type):
    headers = get_whatsapp_headers()
    phone_number_id = get_whatsapp_phone_number_id()
    url = f"https://graph.facebook.com/{WHATSAPP_API_VERSION}/{phone_number_id}/media"
    with open(file_path, 'rb') as file_handle:
        response = requests.post(
            url,
            headers=headers,
            files={'file': (file_path.name, file_handle, mime_type)},
            data={'messaging_product': 'whatsapp', 'type': mime_type},
            timeout=60,
        )
    response.raise_for_status()
    return response.json().get('id')


def send_whatsapp_message(to_number, message_type, text=None, media_id=None, file_name=None):
    headers = get_whatsapp_headers()
    phone_number_id = get_whatsapp_phone_number_id()
    url = f"https://graph.facebook.com/{WHATSAPP_API_VERSION}/{phone_number_id}/messages"
    payload = {
        'messaging_product': 'whatsapp',
        'to': to_number,
        'type': message_type,
    }
    if message_type == 'text':
        payload['text'] = {'body': text or ''}
    elif message_type == 'document':
        payload['document'] = {'id': media_id}
        if file_name:
            payload['document']['filename'] = file_name
        if text:
            payload['document']['caption'] = text
    else:
        payload[message_type] = {'id': media_id}
        if text and message_type in {'image', 'video'}:
            payload[message_type]['caption'] = text
    response = requests.post(url, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def send_whatsapp_template_message(to_number, template_name, language_code, components=None):
    headers = get_whatsapp_headers()
    phone_number_id = get_whatsapp_phone_number_id()
    url = f"https://graph.facebook.com/{WHATSAPP_API_VERSION}/{phone_number_id}/messages"
    payload = {
        'messaging_product': 'whatsapp',
        'to': to_number,
        'type': 'template',
        'template': {
            'name': template_name,
            'language': {'code': language_code},
        }
    }
    if components:
        payload['template']['components'] = components
    response = requests.post(url, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def send_whatsapp_interactive_message(to_number, interactive_payload):
    headers = get_whatsapp_headers()
    phone_number_id = get_whatsapp_phone_number_id()
    url = f"https://graph.facebook.com/{WHATSAPP_API_VERSION}/{phone_number_id}/messages"
    payload = {
        'messaging_product': 'whatsapp',
        'to': to_number,
        'type': 'interactive',
        'interactive': interactive_payload,
    }
    response = requests.post(url, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_whatsapp_templates(limit=100):
    headers = get_whatsapp_headers()
    waba_id = get_whatsapp_waba_id()
    url = f"https://graph.facebook.com/{WHATSAPP_API_VERSION}/{waba_id}/message_templates"
    response = requests.get(
        url,
        headers=headers,
        params={
            'limit': limit,
            'fields': 'name,language,status,category,components',
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json().get('data', [])


def fetch_whatsapp_flows(limit=100):
    headers = get_whatsapp_headers()
    waba_id = get_whatsapp_waba_id()
    url = f"https://graph.facebook.com/{WHATSAPP_API_VERSION}/{waba_id}/flows"
    response = requests.get(
        url,
        headers=headers,
        params={
            'limit': limit,
            'fields': 'id,name,status,categories,updated_at,validation_errors',
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json().get('data', [])


def load_cached_whatsapp_flows(conn):
    c = conn.cursor()
    c.execute(
        """
        SELECT flow_id, flow_name, status, categories, updated_at
        FROM whatsapp_flows
        ORDER BY COALESCE(flow_name, flow_id) COLLATE NOCASE
        """
    )
    flows = []
    for row in c.fetchall():
        categories = []
        if row[3]:
            try:
                categories = json.loads(row[3])
            except json.JSONDecodeError:
                categories = []
        flows.append(
            {
                'id': row[0],
                'name': row[1] or row[0],
                'status': row[2] or '',
                'categories': categories,
                'updated_at': row[4],
            }
        )
    return flows


def sync_whatsapp_flows_to_db(limit=100):
    fetched_flows = fetch_whatsapp_flows(limit=limit)
    synced_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    conn = get_db_connection()
    c = conn.cursor()
    for flow in fetched_flows:
        flow_id = str(flow.get('id') or '').strip()
        if not flow_id:
            continue
        flow_name = (flow.get('name') or '').strip()
        status = (flow.get('status') or '').strip()
        categories = flow.get('categories')
        if not isinstance(categories, list):
            categories = []
        c.execute(
            """
            INSERT INTO whatsapp_flows (flow_id, flow_name, status, categories, raw_json, synced_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(flow_id) DO UPDATE SET
                flow_name=excluded.flow_name,
                status=excluded.status,
                categories=excluded.categories,
                raw_json=excluded.raw_json,
                synced_at=excluded.synced_at,
                updated_at=excluded.updated_at
            """,
            (
                flow_id,
                flow_name,
                status,
                json.dumps(categories),
                json.dumps(flow),
                synced_at,
                (flow.get('updated_at') or '').strip() or synced_at,
            ),
        )
    conn.commit()
    flows = load_cached_whatsapp_flows(conn)
    conn.close()
    return flows


def find_whatsapp_flow(flow_name):
    normalized = (flow_name or '').strip().lower()
    if not normalized:
        return None

    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        """
        SELECT flow_id, flow_name, status
        FROM whatsapp_flows
        WHERE LOWER(COALESCE(flow_name, '')) = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (normalized,),
    )
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    return {'id': row[0], 'name': row[1] or row[0], 'status': row[2] or ''}

# ---- AI categorization setup ----
MODEL_PATH = 'complaint_model.pkl'
CATEGORIES = ["Speed Issue", "Connection Down", "Billing", "Installation", "Other"]
_model = None
_vectorizer = None

# --- Pinging setup ---
PING_IPS = ["103.149.126.10", "36.50.163.244", "154.84.251.178"]
ping_results = {ip: "Checking..." for ip in PING_IPS}


def ping_host(ip: str) -> str:
    """Check reachability via TCP (port 80) instead of ICMP ping."""
    try:
        sock = socket.create_connection((ip, 80), timeout=2)
        sock.close()
        return "Online"
    except Exception:
        return "Offline"


def ping_loop():
    """Background thread: continuously check IPs and update results."""
    while True:
        for ip in PING_IPS:
            ping_results[ip] = ping_host(ip)
        time.sleep(10)  # refresh every 10s

# Ensure we only launch the ping thread once
_ping_thread_started = False


def _start_ping_thread():
    global _ping_thread_started
    if not _ping_thread_started:
        threading.Thread(target=ping_loop, daemon=True).start()
        _ping_thread_started = True


def _launch_ping_thread():
    _start_ping_thread()


def _tokenize_complaint(text: str):
    return [token for token in re.findall(r"[a-z0-9]+", (text or '').lower()) if len(token) > 2]


def _load_or_train_model():
    """Load a lightweight local keyword model or train one from sample data."""
    global _model, _vectorizer
    if _model and _vectorizer:
        return

    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, 'rb') as f:
                model, vectorizer = pickle.load(f)
            if isinstance(model, dict) and vectorizer == 'keyword_v1':
                _model, _vectorizer = model, vectorizer
                return
        except Exception:
            pass

    training_file = os.path.join('setup', 'complaint_training_data.csv')
    if not os.path.exists(training_file):
        _model = {'category_totals': {}, 'token_scores': {}}
        _vectorizer = 'keyword_v1'
        return

    token_scores = {category: {} for category in CATEGORIES}
    category_totals = {category: 0 for category in CATEGORIES}

    with open(training_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            category = (row.get('category') or 'Other').strip()
            if category not in token_scores:
                category = 'Other'
            tokens = _tokenize_complaint(row.get('complaint', ''))
            for token in tokens:
                token_scores[category][token] = token_scores[category].get(token, 0) + 1
                category_totals[category] += 1

    _model = {
        'category_totals': category_totals,
        'token_scores': token_scores,
    }
    _vectorizer = 'keyword_v1'

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump((_model, _vectorizer), f)


def predict_category(text: str) -> str:
    """Predict complaint category using keyword score matching."""
    _load_or_train_model()
    if not _model:
        return 'Other'

    tokens = _tokenize_complaint(text)
    if not tokens:
        return 'Other'

    token_scores = _model.get('token_scores', {})
    category_totals = _model.get('category_totals', {})
    best_category = 'Other'
    best_score = 0

    for category in CATEGORIES:
        category_score = 0
        cat_token_map = token_scores.get(category, {})
        for token in tokens:
            category_score += cat_token_map.get(token, 0)
        if category_totals.get(category):
            category_score = category_score / category_totals[category]
        if category_score > best_score:
            best_score = category_score
            best_category = category

    return best_category if best_score > 0 else 'Other'


# Make {{ current_year }} available in all templates
@app.context_processor
def inject_year():
    return {'current_year': datetime.now().year}


# --- Auth helpers ---
def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get('user_id'):
            flash('Please log in to continue.', 'warning')
            return redirect(url_for('auth.login'))
        return view(*args, **kwargs)
    return wrapped


# --- Register auth blueprint (requires auth.py in project root) ---
from auth import auth_bp
app.register_blueprint(auth_bp)


# Initialize DB
def init_db():
    conn = get_db_connection()
    c = conn.cursor()

    # Users table
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT DEFAULT 'user',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS complaints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            mobile TEXT NOT NULL,
            complaint TEXT NOT NULL,
            category TEXT DEFAULT 'Other',
            status TEXT DEFAULT 'Pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            source TEXT DEFAULT 'Web'
        )
    ''')

    try:
        c.execute("ALTER TABLE complaints ADD COLUMN source TEXT DEFAULT 'Web'")
    except sqlite3.OperationalError:
        pass
    for column_def in [
        "name TEXT",
        "mobile TEXT",
        "complaint TEXT",
        "category TEXT DEFAULT 'Other'",
        "status TEXT DEFAULT 'Pending'",
        "created_at TEXT",
    ]:
        try:
            c.execute(f"ALTER TABLE complaints ADD COLUMN {column_def}")
        except sqlite3.OperationalError:
            pass
    c.execute("UPDATE complaints SET created_at = COALESCE(created_at, CURRENT_TIMESTAMP)")

    c.execute('''
        CREATE TABLE IF NOT EXISTS whatsapp_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id TEXT,
            name TEXT,
            mobile TEXT NOT NULL,
            direction TEXT NOT NULL,
            message_type TEXT NOT NULL,
            text TEXT,
            media_id TEXT,
            media_url TEXT,
            media_mime_type TEXT,
            file_name TEXT,
            latitude REAL,
            longitude REAL,
            delivery_status TEXT,
            error_reason TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    try:
        c.execute("ALTER TABLE whatsapp_messages ADD COLUMN message_id TEXT")
    except sqlite3.OperationalError:
        pass
    c.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_whatsapp_messages_message_id ON whatsapp_messages(message_id)")
    for column_def in [
        "name TEXT",
        "direction TEXT NOT NULL DEFAULT 'inbound'",
        "message_type TEXT NOT NULL DEFAULT 'text'",
        "text TEXT",
        "media_url TEXT",
        "file_name TEXT",
        "media_mime_type TEXT",
        "latitude REAL",
        "longitude REAL",
        "delivery_status TEXT",
        "error_reason TEXT",
        "created_at TEXT DEFAULT CURRENT_TIMESTAMP",
    ]:
        try:
            c.execute(f"ALTER TABLE whatsapp_messages ADD COLUMN {column_def}")
        except sqlite3.OperationalError:
            pass

    c.execute('''
        CREATE TABLE IF NOT EXISTS whatsapp_template_notifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            inbound_message_id TEXT,
            chat_mobile TEXT NOT NULL,
            target_mobile TEXT NOT NULL,
            template_name TEXT NOT NULL,
            language_code TEXT NOT NULL,
            outbound_message_id TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            error_reason TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            sent_at TEXT,
            UNIQUE (chat_mobile, target_mobile)
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS whatsapp_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            invoice_id TEXT,
            invoice_number TEXT,
            customer_name TEXT,
            phone TEXT,
            template_name TEXT,
            status TEXT DEFAULT 'sent',
            error_message TEXT,
            message_id TEXT,
            attempts INTEGER DEFAULT 1,
            total_amount REAL,
            due_date TEXT,
            sent_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT
        )
    ''')
    c.execute("CREATE INDEX IF NOT EXISTS idx_whatsapp_logs_sent_at ON whatsapp_logs(sent_at)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_whatsapp_logs_status ON whatsapp_logs(status)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_whatsapp_logs_invoice_status_date ON whatsapp_logs(invoice_id, status, sent_at)")

    c.execute('''
        CREATE TABLE IF NOT EXISTS whatsapp_webhook_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            note TEXT,
            message_events INTEGER DEFAULT 0,
            status_events INTEGER DEFAULT 0,
            payload_preview TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')


    c.execute('''
        CREATE TABLE IF NOT EXISTS whatsapp_flows (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            flow_id TEXT NOT NULL UNIQUE,
            flow_name TEXT,
            status TEXT,
            categories TEXT,
            raw_json TEXT,
            synced_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    c.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_whatsapp_flows_flow_id ON whatsapp_flows(flow_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_whatsapp_flows_name ON whatsapp_flows(flow_name)")

    try:
        c.execute("ALTER TABLE whatsapp_template_notifications ADD COLUMN chat_mobile TEXT")
    except sqlite3.OperationalError:
        pass

    c.execute('''
        CREATE TABLE IF NOT EXISTS connection_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            mobile TEXT NOT NULL,
            area TEXT,
            status TEXT DEFAULT 'Pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    for column_def in [
        "name TEXT",
        "mobile TEXT",
        "area TEXT",
        "status TEXT DEFAULT 'Pending'",
        "created_at TEXT",
    ]:
        try:
            c.execute(f"ALTER TABLE connection_requests ADD COLUMN {column_def}")
        except sqlite3.OperationalError:
            pass
    c.execute("UPDATE connection_requests SET created_at = COALESCE(created_at, CURRENT_TIMESTAMP)")

    c.execute('''CREATE TABLE IF NOT EXISTS stock (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        item_type TEXT,
        description TEXT,
        quantity INTEGER,
        date TEXT
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS issued_stock (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        device TEXT,
        recipient TEXT,
        date TEXT,
        note TEXT,
        payment_mode TEXT,
        status TEXT
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS staff_attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        first_name TEXT,
        last_name TEXT,
        date TEXT,
        time TEXT,
        action TEXT,
        note TEXT
    )''')

    conn.commit()
    conn.close()


@app.before_request
def before_request():
    init_db()
    _launch_ping_thread()


# --- updated ping status route ---
@app.route('/ping-status')
@login_required
def ping_status():
    """Return boolean status for each IP (True = online)."""
    statuses = {ip: (result == "Online") for ip, result in ping_results.items()}
    return jsonify(statuses)

# ==============================
#   EXISTING ROUTES
# ==============================

@app.route('/')
def landing():
    return render_template('index.html')


@app.route('/dashboard')
@login_required
def dashboard():
    conn = get_db_connection()
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    complaint_columns = {row[1] for row in c.execute("PRAGMA table_info(complaints)").fetchall()}
    connection_columns = {row[1] for row in c.execute("PRAGMA table_info(connection_requests)").fetchall()}

    c.execute('SELECT COUNT(*) FROM complaints')
    total = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM complaints WHERE status = 'Pending'")
    pending = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM complaints WHERE status = 'Resolved'")
    resolved = c.fetchone()[0]

    has_mobile = "mobile" in complaint_columns
    has_created_at = "created_at" in complaint_columns

    if has_mobile and has_created_at:
        c.execute('''
        SELECT * FROM (
            SELECT * FROM complaints
            ORDER BY created_at DESC
        )
        GROUP BY mobile
        ORDER BY created_at DESC
        LIMIT 50
    ''')
        recent_complaints_raw = c.fetchall()
    elif has_mobile:
        c.execute('''
        SELECT * FROM (
            SELECT * FROM complaints
            ORDER BY id DESC
        )
        GROUP BY mobile
        ORDER BY id DESC
        LIMIT 50
    ''')
        recent_complaints_raw = c.fetchall()
    else:
        order_column = "created_at" if has_created_at else "id"
        c.execute(f"SELECT * FROM complaints ORDER BY {order_column} DESC LIMIT 50")
        recent_complaints_raw = c.fetchall()

    priority_complaints = []
    for comp in recent_complaints_raw:
        mobile = comp['mobile'] if has_mobile else None
        if mobile and has_created_at:
            c.execute("""
                SELECT COUNT(*) FROM complaints
                WHERE mobile = ? AND date(created_at) >= date('now', '-30 day')
            """, (mobile,))
        elif mobile:
            c.execute("SELECT COUNT(*) FROM complaints WHERE mobile = ?", (mobile,))
        else:
            c.execute("SELECT 1")
        count = c.fetchone()[0]
        priority = "High" if count >= 3 else "Medium" if count == 2 else "Low"
        priority_complaints.append(dict(comp) | {'priority': priority})

    pending_connection_order = "created_at" if "created_at" in connection_columns else "id"
    pending_connection_fields = [
        "id",
        "name" if "name" in connection_columns else "'' AS name",
        "mobile" if "mobile" in connection_columns else "'' AS mobile",
        "area" if "area" in connection_columns else "'' AS area",
        "status" if "status" in connection_columns else "'Pending' AS status",
        "created_at" if "created_at" in connection_columns else "NULL AS created_at",
    ]
    c.execute(
        f"SELECT {', '.join(pending_connection_fields)} "
        f"FROM connection_requests "
        f"WHERE {'status = ?' if 'status' in connection_columns else '1=1'} "
        f"ORDER BY {pending_connection_order} DESC LIMIT 5",
        ("Pending",) if "status" in connection_columns else ()
    )
    pending_connections = c.fetchall()

    if "status" in connection_columns:
        c.execute("SELECT COUNT(*) FROM connection_requests WHERE status = 'Pending'")
    else:
        c.execute("SELECT COUNT(*) FROM connection_requests")
    pending_connection_count = c.fetchone()[0]

    device_types = ['Switch', 'WAN Router', 'ONT Router', 'ONU']
    stock_summary = {}
    for device in device_types:
        c.execute("SELECT SUM(quantity) FROM stock WHERE item_type = ?", (device,))
        stock = c.fetchone()[0] or 0
        c.execute("SELECT COUNT(*) FROM issued_stock WHERE device = ?", (device,))
        issued = c.fetchone()[0] or 0
        stock_summary[device] = {'stock': stock, 'issued': issued, 'available': stock - issued}

    conn.close()
    return render_template(
        'dashboard.html',
        total=total,
        pending=pending,
        resolved=resolved,
        recent_complaints=priority_complaints,
        pending_connections=pending_connections,
        pending_connection_count=pending_connection_count,
        stock_summary=stock_summary,
        categories=CATEGORIES,
    )


@app.route('/submit', methods=['POST'])
def submit():
    name = request.form['name']
    mobile = request.form['mobile']
    complaint = request.form['complaint']
    category = predict_category(complaint)
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        "INSERT INTO complaints (name, mobile, complaint, category, source) VALUES (?, ?, ?, ?, ?)",
        (name, mobile, complaint, category, 'Web')
    )
    conn.commit()
    conn.close()
    return redirect(url_for('dashboard'))


@app.route('/track', methods=['GET', 'POST'])
def track():
    complaints = []
    status = None

    if request.method == 'POST':
        mobile = request.form.get('mobile', '').strip()
        name = request.form.get('name', '').strip()

        conn = get_db_connection()
        c = conn.cursor()

        if mobile and name:
            c.execute("""
                SELECT id, name, mobile, complaint, status, created_at 
                FROM complaints 
                WHERE mobile = ? AND name LIKE ? 
                ORDER BY created_at DESC
            """, (mobile, f"%{name}%"))
        elif mobile:
            c.execute("""
                SELECT id, name, mobile, complaint, status, created_at 
                FROM complaints 
                WHERE mobile = ? 
                ORDER BY created_at DESC
            """, (mobile,))
        elif name:
            c.execute("""
                SELECT id, name, mobile, complaint, status, created_at 
                FROM complaints 
                WHERE name LIKE ? 
                ORDER BY created_at DESC
            """, (f"%{name}%",))

        complaints = c.fetchall()

        status_priority = {"Registered": 0, "Pending": 1, "Assigned": 2, "Complete": 3, "Resolved": 3}
        if complaints:
            worst_status_value = max(status_priority.get(comp[4], 0) for comp in complaints)
            status = [key for key, value in status_priority.items() if value == worst_status_value][0]

        conn.close()

    return render_template('track.html', complaints=complaints, status=status)


@app.route('/payment')
def payment():
    return render_template('payment.html', razorpay_key_id=get_razorpay_key_id())


@app.route('/api/payments/razorpay/order', methods=['POST'])
def create_payment_order():
    data = request.get_json(silent=True) or {}
    try:
        amount_rupees = float(data.get('amount', 0))
    except (TypeError, ValueError):
        return jsonify({'error': 'Invalid amount'}), 400

    if amount_rupees < 10:
        return jsonify({'error': 'Amount must be at least ₹10'}), 400

    amount_paise = int(round(amount_rupees * 100))

    if not razorpay_config_ready():
        missing_keys = get_missing_razorpay_env_keys()
        return jsonify({'error': 'Razorpay is not configured on server', 'missing': missing_keys}), 503

    plan_name = (data.get('plan_name') or '').strip()
    billing_cycle = (data.get('billing_cycle') or '').strip()
    receipt = f"cl-{uuid.uuid4().hex[:12]}"
    notes = {'source': 'payment_page'}
    if plan_name:
        notes['plan_name'] = plan_name
    if billing_cycle:
        notes['billing_cycle'] = billing_cycle

    try:
        order = create_razorpay_order(amount_paise, receipt, notes=notes)
        return jsonify({
            'id': order.get('id'),
            'amount': order.get('amount', amount_paise),
            'currency': order.get('currency', 'INR'),
            'key': get_razorpay_key_id(),
        })
    except requests.RequestException as exc:
        app.logger.exception('Failed to create Razorpay order: %s', exc)
        return jsonify({'error': 'Unable to create Razorpay order'}), 502


@app.route('/api/payments/razorpay/verify', methods=['POST'])
def verify_payment_signature():
    data = request.get_json(silent=True) or {}
    order_id = (data.get('razorpay_order_id') or '').strip()
    payment_id = (data.get('razorpay_payment_id') or '').strip()
    signature = (data.get('razorpay_signature') or '').strip()

    if not order_id or not payment_id or not signature:
        return jsonify({'error': 'Missing payment verification fields'}), 400

    verified = verify_razorpay_signature(order_id, payment_id, signature)
    return jsonify({'verified': verified}), (200 if verified else 400)


@app.route('/update_status/<int:complaint_id>/<status>')
@login_required
def update_status(complaint_id, status):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("UPDATE complaints SET status = ? WHERE id = ?", (status, complaint_id))
    conn.commit()
    conn.close()
    return redirect(url_for('dashboard'))




@app.route('/update_complaints_bulk', methods=['POST'])
@login_required
def update_complaints_bulk():
    action = (request.form.get('action') or '').strip().lower()
    selected_ids = request.form.getlist('selected_ids[]')

    valid_ids = []
    for complaint_id in selected_ids:
        try:
            valid_ids.append(int(complaint_id))
        except (TypeError, ValueError):
            continue

    if not valid_ids:
        flash('No complaints selected.', 'warning')
        return redirect(url_for('dashboard'))

    placeholders = ','.join('?' for _ in valid_ids)
    conn = get_db_connection()
    c = conn.cursor()

    if action == 'resolve':
        c.execute(
            f"UPDATE complaints SET status = 'Resolved' WHERE id IN ({placeholders})",
            valid_ids
        )
        flash(f"Marked {c.rowcount} complaint(s) as resolved.", 'success')
    elif action == 'delete':
        c.execute(
            f"DELETE FROM complaints WHERE id IN ({placeholders})",
            valid_ids
        )
        flash(f"Deleted {c.rowcount} complaint(s).", 'success')
    else:
        conn.close()
        flash('Invalid bulk action.', 'error')
        return redirect(url_for('dashboard'))

    conn.commit()
    conn.close()
    return redirect(url_for('dashboard'))


@app.route('/webhook', methods=['GET', 'POST'])
@app.route('/webhook/whatsapp', methods=['GET', 'POST'])
def webhook():
    webhook_logger = get_whatsapp_webhook_logger()
    app.logger.info("Webhook hit: method=%s path=%s", request.method, request.path)
    webhook_logger.info("webhook_hit method=%s path=%s remote=%s", request.method, request.path, request.remote_addr)

    if request.method == 'GET':
        mode = request.args.get('hub.mode')
        challenge = request.args.get('hub.challenge')
        verify_token = request.args.get('hub.verify_token')
        app.logger.info("Webhook verification attempt: mode=%s token_present=%s", mode, bool(verify_token))
        webhook_logger.info("verification_attempt mode=%s token_present=%s", mode, bool(verify_token))
        if mode != 'subscribe':
            app.logger.warning("Webhook verification failed: invalid hub.mode=%s", mode)
            webhook_logger.warning("verification_failed_invalid_mode mode=%s", mode)
            return 'Invalid mode', 403
        if not WEBHOOK_VERIFY_TOKEN:
            app.logger.warning("Webhook verification failed: WEBHOOK_VERIFY_TOKEN is not configured.")
            webhook_logger.warning("verification_failed_missing_verify_token")
            return 'Missing verify token', 500
        if verify_token != WEBHOOK_VERIFY_TOKEN:
            app.logger.warning("Webhook verification failed: invalid token.")
            webhook_logger.warning("verification_failed_invalid_token")
            return 'Verification failed', 403
        app.logger.info("Webhook verification handshake accepted.")
        webhook_logger.info("verification_success")
        return challenge or '', 200

    if request.method == 'POST':
        try:
            data = parse_webhook_request_json()
            if not data:
                app.logger.warning("Webhook received no JSON payload.")
                webhook_logger.warning("post_no_json_payload")
                store_webhook_audit_event(None, 'empty-or-invalid-json')
                return 'ok', 200

            store_webhook_audit_event(data, 'received')
            if WEBHOOK_ASYNC_PROCESSING:
                app.logger.info("Webhook POST accepted. Processing asynchronously.")
                webhook_logger.info("post_processing_async")
                threading.Thread(target=process_whatsapp_webhook_payload, args=(data,), daemon=True).start()
            else:
                app.logger.info("Webhook POST accepted. Processing synchronously because WEBHOOK_ASYNC_PROCESSING is disabled.")
                webhook_logger.info("post_processing_sync")
                process_whatsapp_webhook_payload(data)
            return 'ok', 200

        except Exception:
            app.logger.exception("Webhook processing failed.")
            webhook_logger.exception("post_handler_exception")
            store_webhook_audit_event(None, 'handler-exception')
            return 'ok', 200


@app.route('/webhook/debug', methods=['GET'])
@login_required
def webhook_debug():
    get_whatsapp_webhook_logger()
    log_path = WHATSAPP_WEBHOOK_LOGGER_PATH or WHATSAPP_WEBHOOK_LOG_PATH
    log_file_exists = Path(log_path).exists() if log_path and log_path != 'stream-only' else False
    conn = get_db_connection()
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT COUNT(*) AS total FROM whatsapp_messages")
    total_messages = c.fetchone()['total']
    c.execute("SELECT COUNT(*) AS inbound_total FROM whatsapp_messages WHERE direction = 'inbound'")
    inbound_total = c.fetchone()['inbound_total']
    c.execute("SELECT message_id, mobile, message_type, created_at FROM whatsapp_messages ORDER BY id DESC LIMIT 5")
    latest = [dict(row) for row in c.fetchall()]
    c.execute("""
        SELECT id, note, message_events, status_events, created_at
        FROM whatsapp_webhook_audit
        ORDER BY id DESC
        LIMIT 10
    """)
    webhook_audit = [dict(row) for row in c.fetchall()]
    conn.close()
    app.logger.info("Webhook debug route checked. total=%s inbound=%s", total_messages, inbound_total)
    return jsonify({
        'status': 'ok',
        'total_messages': total_messages,
        'inbound_messages': inbound_total,
        'latest_messages': latest,
        'latest_webhook_events': webhook_audit,
        'webhook_log_path': log_path,
        'webhook_log_file_exists': log_file_exists,
    })


@app.after_request
def set_default_json_header(response):
    if request.path.startswith('/flow-endpoint'):
        response.headers['Content-Type'] = 'application/json'
    return response


def parse_flow_payload(data):
    if not isinstance(data, dict):
        return None, "Invalid JSON payload"

    name = str(data.get("name", "")).strip()
    mobile = str(data.get("mobile", "")).strip()
    complaint = str(data.get("complaint", "")).strip()

    if not all([name, mobile, complaint]):
        return None, "Missing required fields"

    return {"name": name, "mobile": mobile, "complaint": complaint}, None


@app.route('/flow-endpoint', methods=['POST'])
def flow_endpoint():
    data = request.get_json(silent=True)
    payload, error = parse_flow_payload(data)
    if error:
        return jsonify({"error": error}), 400

    category = predict_category(payload["complaint"])
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        "INSERT INTO complaints (name, mobile, complaint, category) VALUES (?, ?, ?, ?)",
        (payload["name"], payload["mobile"], payload["complaint"], category)
    )
    conn.commit()
    conn.close()
    return jsonify({"status": "received"}), 200


@app.route('/complaints', endpoint='complaints_page')
@login_required
def view_complaints():
    conn = get_db_connection()
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("""
        SELECT id, name, mobile, complaint, status, created_at, source 
        FROM complaints 
        ORDER BY created_at DESC LIMIT 100
    """)
    complaints = c.fetchall()

    c.execute("""
        SELECT
            mobile,
            SUM(CASE WHEN date(created_at) >= date('now', '-30 day') THEN 1 ELSE 0 END) AS count_30d,
            SUM(CASE WHEN date(created_at) >= date('now', '-30 day') AND status != 'Resolved' THEN 1 ELSE 0 END) AS unresolved_30d,
            SUM(CASE WHEN date(created_at) >= date('now', '-7 day') THEN 1 ELSE 0 END) AS count_7d,
            SUM(
                CASE
                    WHEN date(created_at) >= date('now', '-14 day')
                         AND date(created_at) < date('now', '-7 day')
                    THEN 1
                    ELSE 0
                END
            ) AS count_prev_7d,
            MAX(created_at) AS last_seen
        FROM complaints
        GROUP BY mobile
    """)
    stats_rows = c.fetchall()

    customer_stats = {}
    for row in stats_rows:
        score = (row["count_30d"] or 0) * 2 + (row["unresolved_30d"] or 0) * 3 + (row["count_7d"] or 0) * 2
        if score >= 12:
            risk_level = "Highly Disturbed"
        elif score >= 7:
            risk_level = "At Risk"
        else:
            risk_level = "Normal"

        if (row["count_7d"] or 0) > (row["count_prev_7d"] or 0):
            trend = "Rising"
        elif (row["count_7d"] or 0) < (row["count_prev_7d"] or 0):
            trend = "Falling"
        else:
            trend = "Stable"

        customer_stats[row["mobile"]] = {
            "count_30d": row["count_30d"] or 0,
            "unresolved_30d": row["unresolved_30d"] or 0,
            "count_7d": row["count_7d"] or 0,
            "count_prev_7d": row["count_prev_7d"] or 0,
            "last_seen": row["last_seen"],
            "score": score,
            "risk_level": risk_level,
            "trend": trend,
        }

    complaints_enriched = []
    for complaint in complaints:
        mobile = complaint["mobile"]
        stats = customer_stats.get(mobile, {})
        complaints_enriched.append({
            "id": complaint["id"],
            "name": complaint["name"],
            "mobile": mobile,
            "complaint": complaint["complaint"],
            "status": complaint["status"],
            "created_at": complaint["created_at"],
            "source": complaint["source"],
            "risk_level": stats.get("risk_level", "Normal"),
            "trend": stats.get("trend", "Stable"),
            "score": stats.get("score", 0),
        })

    high_risk_customers = [
        {
            "mobile": mobile,
            "risk_level": stats["risk_level"],
            "trend": stats["trend"],
            "score": stats["score"],
            "count_30d": stats["count_30d"],
            "unresolved_30d": stats["unresolved_30d"],
            "last_seen": stats["last_seen"],
        }
        for mobile, stats in customer_stats.items()
        if stats["risk_level"] in {"Highly Disturbed", "At Risk"}
    ]
    high_risk_customers.sort(key=lambda item: item["score"], reverse=True)
    conn.close()
    return render_template(
        'complaints.html',
        complaints=complaints_enriched,
        high_risk_customers=high_risk_customers[:5],
        high_risk_total=sum(1 for stats in customer_stats.values() if stats["risk_level"] == "Highly Disturbed"),
        at_risk_total=sum(1 for stats in customer_stats.values() if stats["risk_level"] == "At Risk"),
    )


@app.route('/whatsapp')
@login_required
def whatsapp_complaints():
    conn = get_db_connection()
    conn.row_factory = sqlite3.Row
    rows, legacy_mode = load_whatsapp_rows(conn)
    contacts = build_whatsapp_contacts(rows)
    active_mobile = normalize_mobile(request.args.get("mobile")) or (contacts[0]["mobile"] if contacts else None)
    app.logger.info("WhatsApp inbox loaded with %s contacts. Active mobile: %s", len(contacts), active_mobile or "none")

    messages = []
    active_name = None
    last_inbox_message_id = max((row["id"] for row in rows), default=0)
    if active_mobile:
        messages = [row for row in rows if mobiles_equivalent(row["mobile"], active_mobile)]
        messages = sorted(messages, key=lambda item: (item["created_at"], item["id"]))
        active_name = contacts[0]["name"] if contacts and mobiles_equivalent(contacts[0]["mobile"], active_mobile) else None
        if not active_name and messages:
            active_name = messages[0]["name"] or active_mobile

    conn.close()
    return render_template(
        "whatsapp.html",
        contacts=contacts,
        messages=messages,
        active_mobile=active_mobile,
        active_name=active_name or "",
        config_ready=whatsapp_config_ready(),
        phone_number_id=os.environ.get('PHONE_NUMBER_ID'),
        waba_id=os.environ.get('WABA_ID'),
        last_inbox_message_id=last_inbox_message_id,
    )


@app.route('/api/whatsapp/messages')
@login_required
def whatsapp_messages_api():
    mobile = normalize_mobile(request.args.get('mobile', ''))
    since_id_raw = (request.args.get('since_id') or '').strip()
    since_inbox_id_raw = (request.args.get('since_inbox_id') or '').strip()
    include_contacts = request.args.get('include_contacts', '0') == '1'

    def parse_non_negative_int(raw_value, field_name):
        if raw_value == '':
            return 0
        try:
            value = int(raw_value)
            if value < 0:
                raise ValueError("negative value")
            return value
        except Exception:
            app.logger.warning(
                "Invalid %s='%s' received in WhatsApp API; defaulting to 0",
                field_name,
                raw_value,
            )
            return 0

    since_id = parse_non_negative_int(since_id_raw, 'since_id')
    since_inbox_id = parse_non_negative_int(since_inbox_id_raw, 'since_inbox_id')

    contacts = []
    messages = []
    inbox_messages = []
    latest_inbox_id = 0
    active_name = ""

    # =========================
    # MYSQL AI CRM MODE
    # =========================

    mysql_conn = mysql.connector.connect(
        host=MYSQL_DB_HOST,
        database=MYSQL_DB_NAME,
        user=MYSQL_DB_USER,
        password=MYSQL_DB_PASSWORD,
    )

    mysql_cursor = mysql_conn.cursor(dictionary=True)

    if include_contacts:

        mysql_cursor.execute("""
            SELECT
                phone AS mobile,
                customer_name AS name,
                last_message AS text,
                last_message_at AS created_at
            FROM whatsapp_conversations
            ORDER BY updated_at DESC
        """)

        raw_contacts = mysql_cursor.fetchall()

        contacts = []

        for idx, row in enumerate(raw_contacts, start=1):
            contacts.append({
                "id": idx,
                "name": row.get("name") or row.get("mobile"),
                "mobile": normalize_mobile(row.get("mobile")),
                "text": row.get("text") or "",
                "created_at": row.get("created_at"),
                "message_type": "text",
                "direction": "inbound",
            })

    mysql_cursor.execute("""
        SELECT COALESCE(MAX(id), 0) AS latest_id
        FROM whatsapp_messages
    """)

    latest_row = mysql_cursor.fetchone()
    latest_inbox_id = latest_row["latest_id"] if latest_row else 0

    if mobile:

        mysql_cursor.execute("""
            SELECT
                id,
                whatsapp_message_id AS message_id,
                phone AS mobile,
                sender_type,
                message_text AS text,
                message_type,
                media_url,
                status,
                created_at
            FROM whatsapp_messages
            WHERE phone = %s
            ORDER BY created_at ASC, id ASC
        """, (mobile,))

        db_messages = mysql_cursor.fetchall()

        for msg in db_messages:

            sender_type = msg.get("sender_type", "customer")

            direction = (
                "inbound"
                if sender_type == "customer"
                else "outbound"
            )

            messages.append({
                "id": msg["id"],
                "message_id": msg.get("message_id"),
                "name": active_name or mobile,
                "mobile": normalize_mobile(msg["mobile"]),
                "direction": direction,
                "from_me": direction == "outbound",
                "message_type": msg.get("message_type") or "text",
                "text": msg.get("text") or "",
                "media_url": msg.get("media_url"),
                "media_public_url": url_for(
                    'static',
                    filename=msg["media_url"],
                    _external=True
                ) if msg.get("media_url") else None,
                "file_name": None,
                "media_mime_type": None,
                "latitude": None,
                "longitude": None,
                "delivery_status": msg.get("status"),
                "error_reason": None,
                "created_at": msg.get("created_at"),
            })

        mysql_cursor.execute("""
            SELECT customer_name
            FROM whatsapp_conversations
            WHERE phone = %s
            LIMIT 1
        """, (mobile,))

        active_contact = mysql_cursor.fetchone()

        if active_contact:
            active_name = active_contact.get("customer_name") or mobile

    mysql_cursor.close()
    mysql_conn.close()

    last_message_id = messages[-1]["id"] if messages else None

    response_payload = {
        "contacts": contacts if include_contacts else [],
        "messages": messages,
        "inbox_messages": [],
        "active_mobile": mobile,
        "active_name": active_name,
        "last_message_id": last_message_id,
        "last_inbox_message_id": latest_inbox_id,
        "legacy_mode": False,
    }

    return jsonify(response_payload)
@app.route('/api/whatsapp/logs')
@login_required
def api_whatsapp_logs():
    date_filter = (request.args.get('date') or 'today').strip().lower()
    status_filter = (request.args.get('status') or 'all').strip().lower()

    allowed_date_filters = {'today', 'last7days'}
    allowed_statuses = {'all', 'sent', 'delivered', 'read', 'failed'}

    if date_filter not in allowed_date_filters:
        return jsonify({'error': 'Invalid date filter'}), 422
    if status_filter not in allowed_statuses:
        return jsonify({'error': 'Invalid status filter'}), 422

    # Use MySQL date functions instead of SQLite
    if date_filter == 'today':
        date_clause = "DATE(sent_at) = CURDATE()"
    else:
        date_clause = "sent_at >= NOW() - INTERVAL 7 DAY"

    status_clause = ""
    params = []
    if status_filter != 'all':
        status_clause = " AND status = %s"
        params.append(status_filter)

    try:
        conn = mysql.connector.connect(
            host=MYSQL_DB_HOST,
            database=MYSQL_DB_NAME,
            user=MYSQL_DB_USER,
            password=MYSQL_DB_PASSWORD,
        )
        cursor = conn.cursor(dictionary=True)

        summary_query = f"""
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN status = 'delivered' THEN 1 ELSE 0 END) AS delivered,
                SUM(CASE WHEN status = 'read' THEN 1 ELSE 0 END) AS read_count,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed,
                SUM(CASE WHEN status = 'sent' THEN 1 ELSE 0 END) AS sent_only
            FROM whatsapp_logs
            WHERE {date_clause}{status_clause}
        """
        cursor.execute(summary_query, tuple(params))
        summary = cursor.fetchone()

        rows_query = f"""
            SELECT customer_name, invoice_id, 
                   COALESCE(invoice_id, 'N/A') AS invoice_number, 
                   total AS total, status, sent_at
            FROM whatsapp_logs
            WHERE {date_clause}{status_clause}
            ORDER BY sent_at DESC
            LIMIT 200
        """
        cursor.execute(rows_query, tuple(params))
        rows = cursor.fetchall()

        # Convert datetime objects to strings for JSON serialization
        for row in rows:
            if isinstance(row.get('sent_at'), datetime):
                row['sent_at'] = row['sent_at'].strftime('%Y-%m-%d %H:%M:%S')

        return jsonify({
            'summary': {
                'total_sent_today': int((summary['total'] if summary else 0) or 0),
                'delivered': int((summary['delivered'] if summary else 0) or 0),
                'read': int((summary['read_count'] if summary else 0) or 0),
                'failed': int((summary['failed'] if summary else 0) or 0),
                'sent': int((summary['sent_only'] if summary else 0) or 0),
            },
            'rows': rows
        })
    except MySQLError as e:
        app.logger.exception("Failed to load WhatsApp logs from MySQL.")
        return jsonify({
            'summary': {
                'total_sent_today': 0,
                'delivered': 0,
                'read': 0,
                'failed': 0,
                'sent': 0,
            },
            'rows': []
        }), 200
    finally:
        if conn:
            cursor.close()
            conn.close()


@app.route('/api/whatsapp/send', methods=['POST'])
@login_required
def send_whatsapp():
    if not whatsapp_config_ready():
        return jsonify({"error": "WhatsApp configuration missing."}), 400

    mobile_raw = request.form.get('mobile', '').strip()
    mobile = normalize_mobile(mobile_raw)
    message_text = request.form.get('message', '').strip()
    attachment = request.files.get('attachment')

    if not mobile:
        return jsonify({"error": "Mobile number is required."}), 400
    if not attachment and not message_text:
        return jsonify({"error": "Message or attachment is required."}), 400

    message_type = 'text'
    media_id = None
    media_url = None
    media_mime_type = None
    file_name = None
    latitude = None
    longitude = None
    message_id = None

    try:
        if attachment:
            file_path, stored_name = save_media_file(attachment)
            file_name = attachment.filename or stored_name
            media_mime_type = attachment.mimetype or mimetypes.guess_type(file_path.name)[0] or 'application/octet-stream'
            message_type = detect_message_type(media_mime_type)
            media_id = upload_media_to_whatsapp(file_path, media_mime_type)
            response_payload = send_whatsapp_message(mobile, message_type, text=message_text or None, media_id=media_id, file_name=file_name)
            message_id = (response_payload.get('messages') or [{}])[0].get('id')
            media_url = f"uploads/whatsapp/{stored_name}"
        else:
            response_payload = send_whatsapp_message(mobile, 'text', text=message_text)
            message_id = (response_payload.get('messages') or [{}])[0].get('id')
    except Exception as exc:
        app.logger.error("Failed to send WhatsApp message to %s: %s", mobile, exc)
        return jsonify({"error": f"Failed to send message: {exc}"}), 500

    created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO whatsapp_messages
        (message_id, name, mobile, direction, message_type, text, media_id, media_url, media_mime_type, file_name, latitude, longitude, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            message_id,
            session.get('user_name', 'Agent'),
            mobile,
            'outbound',
            message_type,
            message_text,
            media_id,
            media_url,
            media_mime_type,
            file_name,
            latitude,
            longitude,
            created_at,
        ),
    )
    conn.commit()
    conn.close()

    return jsonify({"status": "sent"}), 200


@app.route('/api/whatsapp/templates')
@login_required
def whatsapp_templates_api():
    if not whatsapp_config_ready():
        return jsonify({"error": "WhatsApp configuration missing."}), 400

    try:
        templates = fetch_whatsapp_templates(limit=100)
        return jsonify({"data": templates}), 200
    except Exception as exc:
        app.logger.error("Failed to load WhatsApp templates: %s", exc)
        return jsonify({"error": f"Failed to fetch templates: {exc}"}), 500


@app.route('/api/whatsapp/flows')
@login_required
def whatsapp_flows_api():
    if not whatsapp_config_ready():
        return jsonify({"error": "WhatsApp configuration missing."}), 400

    force_sync = request.args.get('sync', '0') == '1'
    try:
        if force_sync:
            flows = sync_whatsapp_flows_to_db(limit=100)
        else:
            conn = get_db_connection()
            flows = load_cached_whatsapp_flows(conn)
            conn.close()
            if not flows:
                flows = sync_whatsapp_flows_to_db(limit=100)
        return jsonify({"data": flows}), 200
    except Exception as exc:
        app.logger.error("Failed to load WhatsApp flows: %s", exc)
        return jsonify({"error": f"Failed to fetch flows: {exc}"}), 500


@app.route('/api/whatsapp/flows/sync', methods=['POST'])
@login_required
def sync_whatsapp_flows_api():
    if not whatsapp_config_ready():
        return jsonify({"error": "WhatsApp configuration missing."}), 400

    try:
        flows = sync_whatsapp_flows_to_db(limit=100)
        return jsonify({"status": "synced", "count": len(flows), "data": flows}), 200
    except Exception as exc:
        app.logger.error("Failed to sync WhatsApp flows: %s", exc)
        return jsonify({"error": f"Failed to sync flows: {exc}"}), 500


@app.route('/api/check-availability', methods=['POST'])
def check_availability():
    payload = request.get_json(silent=True) or {}
    mobile = normalize_mobile((payload.get('mobile') or '').strip())
    flow_name = (payload.get('flow_name') or DEFAULT_ENQUIRY_FLOW_NAME).strip()

    if not mobile:
        return jsonify({"error": "mobile is required."}), 400
    if not whatsapp_config_ready():
        return jsonify({"error": "WhatsApp configuration missing."}), 400

    flow = find_whatsapp_flow(flow_name)
    if not flow:
        try:
            sync_whatsapp_flows_to_db(limit=100)
            flow = find_whatsapp_flow(flow_name)
        except Exception as exc:
            app.logger.error("Unable to sync WhatsApp flows while checking availability: %s", exc)
            return jsonify({"error": f"Unable to sync flows: {exc}"}), 500

    if not flow:
        return jsonify({"error": f"Flow '{flow_name}' not found after sync. Please publish it in Meta first."}), 404

    body_text = (payload.get('body') or 'Please complete this enquiry form to check service availability.').strip()[:1024]
    cta_text = (payload.get('button_text') or 'Check Availability').strip()[:20] or 'Check Availability'
    interactive_payload = {
        'type': 'flow',
        'body': {'text': body_text},
        'action': {
            'name': 'flow',
            'parameters': {
                'flow_message_version': '3',
                'flow_id': flow['id'],
                'flow_cta': cta_text,
                'mode': 'published',
            }
        }
    }

    try:
        result = send_whatsapp_interactive_message(mobile, interactive_payload)
        message_id = (result.get('messages') or [{}])[0].get('id')
    except Exception as exc:
        app.logger.error("Failed to send availability flow to %s: %s", mobile, exc)
        return jsonify({"error": f"Failed to send availability flow: {exc}"}), 500

    created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO whatsapp_messages
        (message_id, name, mobile, direction, message_type, text, delivery_status, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            message_id,
            'System',
            mobile,
            'outbound',
            'interactive',
            f"Availability flow sent ({flow.get('name')})",
            'accepted',
            created_at,
        ),
    )
    conn.commit()
    conn.close()

    return jsonify({"status": "sent", "mobile": mobile, "flow": flow}), 200


@app.route('/api/whatsapp/send-template', methods=['POST'])
@login_required
def send_whatsapp_template_api():
    if not whatsapp_config_ready():
        return jsonify({"error": "WhatsApp configuration missing."}), 400

    payload = request.get_json(silent=True) or {}
    mobile = normalize_mobile((payload.get('mobile') or '').strip())
    template_name = (payload.get('template_name') or '').strip()
    language_code = (payload.get('language_code') or 'en_US').strip()
    components = payload.get('components') or []
    template_preview = (payload.get('template_preview') or '').strip()

    if not mobile or not template_name:
        return jsonify({"error": "mobile and template_name are required."}), 400

    try:
        result = send_whatsapp_template_message(
            mobile,
            template_name,
            language_code,
            components=components if isinstance(components, list) else [],
        )
        message_id = (result.get('messages') or [{}])[0].get('id')
        send_status = 'accepted'
        send_error_reason = None
    except Exception as exc:
        app.logger.error("Failed to send WhatsApp template to %s: %s", mobile, exc)
        message_id = None
        send_status = 'failed'
        send_error_reason = str(exc)

    created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO whatsapp_messages
        (message_id, name, mobile, direction, message_type, text, media_id, media_url, media_mime_type, file_name, delivery_status, error_reason, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            message_id,
            session.get('user_name', 'Agent'),
            mobile,
            'outbound',
            'template',
            template_preview or f"Template: {template_name}",
            None,
            None,
            None,
            None,
            send_status,
            send_error_reason,
            created_at,
        ),
    )
    conn.commit()
    conn.close()
    if send_status == 'failed':
        return jsonify({
            "error": f"Failed to send template: {send_error_reason}",
            "stored": True,
            "mobile": mobile,
        }), 500
    return jsonify({"status": "sent"}), 200


@app.route('/api/whatsapp/send-interactive', methods=['POST'])
@login_required
def send_whatsapp_interactive_api():
    if not whatsapp_config_ready():
        return jsonify({"error": "WhatsApp configuration missing."}), 400

    payload = request.get_json(silent=True) or {}
    mobile = normalize_mobile((payload.get('mobile') or '').strip())
    interactive_type = (payload.get('interactive_type') or 'button').strip().lower()
    header = (payload.get('header') or '').strip()
    body_text = (payload.get('body') or '').strip()
    footer = (payload.get('footer') or '').strip()
    interactive_payload = {}

    if not mobile or not body_text:
        return jsonify({"error": "mobile and body are required."}), 400

    if interactive_type == 'button':
        buttons = payload.get('buttons') or []
        if not isinstance(buttons, list) or not 1 <= len(buttons) <= 3:
            return jsonify({"error": "buttons must be an array with 1 to 3 items."}), 400

        normalized_buttons = []
        for idx, button in enumerate(buttons):
            title = (button.get('title') or '').strip()[:20]
            if not title:
                return jsonify({"error": f"Button {idx + 1} title is required."}), 400
            normalized_buttons.append({
                'type': 'reply',
                'reply': {
                    'id': (button.get('id') or f'btn_{idx + 1}').strip()[:256],
                    'title': title,
                }
            })

        interactive_payload = {
            'type': 'button',
            'body': {'text': body_text[:1024]},
            'action': {'buttons': normalized_buttons}
        }
    elif interactive_type == 'list':
        list_button_text = (payload.get('list_button_text') or '').strip()[:20]
        sections = payload.get('sections') or []
        if not list_button_text:
            return jsonify({"error": "list_button_text is required for list interactive messages."}), 400
        if not isinstance(sections, list) or not sections:
            return jsonify({"error": "sections must be a non-empty array for list interactive messages."}), 400

        normalized_sections = []
        total_rows = 0
        for section_idx, section in enumerate(sections):
            rows = section.get('rows') or []
            if not isinstance(rows, list) or not rows:
                return jsonify({"error": f"Section {section_idx + 1} must have at least one row."}), 400

            normalized_rows = []
            for row_idx, row in enumerate(rows):
                row_title = (row.get('title') or '').strip()[:24]
                if not row_title:
                    return jsonify({"error": f"Section {section_idx + 1}, row {row_idx + 1} title is required."}), 400
                normalized_row = {
                    'id': (row.get('id') or f'row_{section_idx + 1}_{row_idx + 1}').strip()[:200],
                    'title': row_title,
                }
                row_description = (row.get('description') or '').strip()[:72]
                if row_description:
                    normalized_row['description'] = row_description
                normalized_rows.append(normalized_row)
                total_rows += 1

            section_title = (section.get('title') or '').strip()[:24]
            normalized_section = {'rows': normalized_rows}
            if section_title:
                normalized_section['title'] = section_title
            normalized_sections.append(normalized_section)

        if total_rows > 10:
            return jsonify({"error": "Total rows across sections cannot exceed 10."}), 400

        interactive_payload = {
            'type': 'list',
            'body': {'text': body_text[:1024]},
            'action': {
                'button': list_button_text,
                'sections': normalized_sections,
            }
        }
    elif interactive_type == 'cta_url':
        button_text = (payload.get('button_text') or '').strip()[:20]
        button_url = (payload.get('button_url') or '').strip()
        if not button_text or not button_url:
            return jsonify({"error": "button_text and button_url are required for CTA URL messages."}), 400
        interactive_payload = {
            'type': 'cta_url',
            'body': {'text': body_text[:1024]},
            'action': {
                'name': 'cta_url',
                'parameters': {
                    'display_text': button_text,
                    'url': button_url,
                }
            }
        }
    elif interactive_type == 'flow':
        flow_id = (payload.get('flow_id') or '').strip()
        button_text = (payload.get('button_text') or '').strip()[:20]
        if not flow_id:
            return jsonify({"error": "flow_id is required for flow interactive messages."}), 400
        if not button_text:
            return jsonify({"error": "button_text is required for flow interactive messages."}), 400
        interactive_payload = {
            'type': 'flow',
            'body': {'text': body_text[:1024]},
            'action': {
                'name': 'flow',
                'parameters': {
                    'flow_message_version': '3',
                    'flow_id': flow_id,
                    'flow_cta': button_text,
                    'mode': 'published',
                }
            }
        }
    else:
        return jsonify({"error": "Unsupported interactive_type. Use button, list, cta_url, or flow."}), 400

    if header:
        interactive_payload['header'] = {'type': 'text', 'text': header[:60]}
    if footer:
        interactive_payload['footer'] = {'text': footer[:60]}

    try:
        result = send_whatsapp_interactive_message(mobile, interactive_payload)
        message_id = (result.get('messages') or [{}])[0].get('id')
    except Exception as exc:
        app.logger.error("Failed to send WhatsApp interactive message to %s: %s", mobile, exc)
        return jsonify({"error": f"Failed to send interactive message: {exc}"}), 500

    created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO whatsapp_messages
        (message_id, name, mobile, direction, message_type, text, media_id, media_url, media_mime_type, file_name, latitude, longitude, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            message_id,
            session.get('user_name', 'Agent'),
            mobile,
            'outbound',
            'interactive',
            body_text,
            None,
            None,
            None,
            None,
            None,
            None,
            created_at,
        ),
    )
    conn.commit()
    conn.close()
    return jsonify({"status": "sent"}), 200


@app.route('/new-connections')
@login_required
def new_connections():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT id, name, mobile, area, status, created_at FROM connection_requests ORDER BY created_at DESC")
    connections = c.fetchall()
    conn.close()
    return render_template('connection.html', connections=connections)


@app.route('/api/new-connection-request', methods=['POST'])
def api_new_connection_request():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid or missing JSON"}), 400

    name = data.get("name")
    mobile = data.get("mobile")
    area = data.get("area")

    if not all([name, mobile]):
        return jsonify({"error": "Missing name or mobile"}), 400

    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        "INSERT INTO connection_requests (name, mobile, area) VALUES (?, ?, ?)",
        (name, mobile, area)
    )
    conn.commit()
    conn.close()

    return jsonify({"status": "received"}), 200


@app.route('/update-connection-status/<int:connection_id>', methods=['POST'])
@login_required
def update_connection_status(connection_id):
    new_status = request.form['status']
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("UPDATE connection_requests SET status = ? WHERE id = ?", (new_status, connection_id))
    conn.commit()
    conn.close()
    return redirect(url_for('new_connections'))

@app.route('/stock', methods=['GET', 'POST'])
@login_required
def stock():
    conn = get_db_connection()
    c = conn.cursor()

    if request.method == 'POST':
        if 'item_type' in request.form and 'quantity' in request.form:
            item_type = request.form.get('item_type', '').strip()
            description = request.form.get('description', '').strip()
            quantity = int(request.form.get('quantity') or 0)
            stock_date = request.form.get('stock_date') or datetime.now().strftime('%Y-%m-%d')

            if item_type and quantity > 0:
                c.execute("SELECT id FROM stock WHERE item_type = ? AND description = ?", (item_type, description))
                existing = c.fetchone()

                if existing:
                    c.execute(
                        "UPDATE stock SET quantity = quantity + ?, date = ? WHERE id = ?",
                        (quantity, stock_date, existing[0])
                    )
                else:
                    c.execute(
                        "INSERT INTO stock (item_type, description, quantity, date) VALUES (?, ?, ?, ?)",
                        (item_type, description, quantity, stock_date)
                    )
                conn.commit()

        elif 'device' in request.form:
            device = request.form.get('device', '').strip()
            recipient = request.form.get('recipient', '').strip()
            date = request.form.get('date') or datetime.now().strftime('%Y-%m-%d')
            note = request.form.get('note', '').strip()
            payment_mode = request.form.get('payment_mode', '').strip()
            status = request.form.get('status', '').strip()

            if device and recipient and payment_mode and status:
                c.execute('''
                    INSERT INTO issued_stock (device, recipient, date, note, payment_mode, status)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (device, recipient, date, note, payment_mode, status))
                conn.commit()

    c.execute("SELECT item_type, description, quantity, date FROM stock ORDER BY id DESC")
    stock_items = c.fetchall()

    c.execute("SELECT device, recipient, date, note, payment_mode, status FROM issued_stock ORDER BY id DESC LIMIT 20")
    issued_items = c.fetchall()

    conn.close()
    return render_template('stock.html', stock_items=stock_items, issued_items=issued_items)


@app.route('/hr', endpoint='hr_dashboard')
@login_required
def hr_page():
    conn = get_db_connection()
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute("SELECT first_name, last_name, date, time, action, note FROM staff_attendance ORDER BY date DESC, time DESC")
    records = c.fetchall()

    c.execute('''
        SELECT first_name || ' ' || last_name AS name,
               COUNT(DISTINCT date) AS total_days,
               SUM(CASE WHEN action = 'Log in' THEN 1 ELSE 0 END) AS present,
               SUM(CASE WHEN action = 'Absent' THEN 1 ELSE 0 END) AS absent,
               SUM(CASE WHEN action = 'Log in' THEN 1 ELSE 0 END) AS login,
               SUM(CASE WHEN action = 'Log out' THEN 1 ELSE 0 END) AS logout
        FROM staff_attendance
        GROUP BY first_name, last_name
    ''')
    summary = c.fetchall()

    conn.close()
    return render_template('hr.html', records=records, summary=summary)


@app.route('/update_salary', methods=['POST'])
@login_required
def update_salary():
    return redirect(url_for('hr_dashboard'))


@app.route('/staff-attendance-webhook', methods=['POST'])
def staff_attendance_webhook():
    if request.is_json:
        data = request.get_json()
        first_name = data.get('first_name')
        last_name = data.get('last_name')
        date = data.get('date')
        time = data.get('time')
        action = data.get('action')
        note = data.get('note')
    else:
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')
        date = request.form.get('date')
        time = request.form.get('time')
        action = request.form.get('action')
        note = request.form.get('note')

    if not all([first_name, last_name, date, time, action]):
        return jsonify({"error": "Missing required fields"}), 400

    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        INSERT INTO staff_attendance (first_name, last_name, date, time, action, note)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (first_name, last_name, date, time, action, note))
    conn.commit()
    conn.close()

    return jsonify({"status": "attendance saved"}), 200


@app.route('/delete_complaint/<int:complaint_id>', methods=['DELETE'])
@login_required
def delete_complaint(complaint_id):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("DELETE FROM complaints WHERE id=?", (complaint_id,))
    conn.commit()
    conn.close()
    return jsonify({"status": "success"})




@app.route('/ping')
def ping():
    return 'pong', 200


@app.route('/api/zoho/sync-customers', methods=['POST'])
def sync_zoho_customers():
    try:
        customers = get_all_zoho_customers()
        inserted, updated = save_customers_to_db(customers)
        return jsonify({
            "status": "ok",
            "fetched": len(customers),
            "inserted": inserted,
            "updated": updated
        })
    except Exception as e:
        app.logger.error("Zoho sync failed: %s", e, exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


def normalize_invoice_phone(raw_phone):
    digits = ''.join(ch for ch in str(raw_phone or '') if ch.isdigit())
    if len(digits) == 10:
        digits = f"91{digits}"
    elif len(digits) == 11 and digits.startswith('0'):
        digits = f"91{digits[1:]}"
    if len(digits) == 12 and digits.startswith('91'):
        return digits
    return ''


def format_due_date_for_whatsapp(due_date_value):
    if not due_date_value:
        return ''
    try:
        if isinstance(due_date_value, datetime):
            due_dt = due_date_value.date()
        else:
            due_dt = datetime.strptime(str(due_date_value), '%Y-%m-%d').date()
        return f"{due_dt.day} {due_dt.strftime('%B %Y')}"
    except ValueError:
        return str(due_date_value)


def send_payment_overdue_whatsapp(phone, customer_name, plan_name, amount, due_date_str):
    api_version = os.environ.get('WHATSAPP_API_VERSION', 'v19.0')
    phone_number_id = os.environ.get('PHONE_NUMBER_ID')
    token = os.environ.get('META_ACCESS_TOKEN')
    if not phone_number_id:
        raise RuntimeError('Missing PHONE_NUMBER_ID')
    if not token:
        raise RuntimeError('Missing META_ACCESS_TOKEN')

    url = f"https://graph.facebook.com/{api_version}/{phone_number_id}/messages"
    payload = {
        "messaging_product": "whatsapp",
        "to": phone,
        "type": "template",
        "template": {
            "name": "payment_overdue_2",
            "language": {"code": "en"},
            "components": [{
                "type": "body",
                "parameters": [
                    {"type": "text", "text": customer_name},
                    {"type": "text", "text": plan_name},
                    {"type": "text", "text": amount},
                    {"type": "text", "text": due_date_str},
                ]
            }]
        }
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    response = requests.post(url, json=payload, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()


def log_invoice_whatsapp_attempt(
    cursor,
    invoice_id,
    invoice_number,
    customer_name,
    phone,
    status,
    total_amount=None,
    due_date=None,
    message_id=None,
    error_message=None,
):
    cursor.execute(
        """
        INSERT INTO whatsapp_logs
            (invoice_id, invoice_number, customer_name, phone, template_name, status, error_message, message_id, attempts, total_amount, due_date)
        VALUES
            (%s, %s, %s, %s, 'payment_overdue_2', %s, %s, %s, 1, %s, %s)
        ON DUPLICATE KEY UPDATE
            status = VALUES(status),
            error_message = VALUES(error_message),
            message_id = VALUES(message_id),
            attempts = attempts + 1,
            total_amount = VALUES(total_amount),
            due_date = VALUES(due_date),
            updated_at = CURRENT_TIMESTAMP
        """,
        (
            invoice_id,
            invoice_number,
            customer_name,
            phone,
            status,
            (error_message or '')[:500] if error_message else None,
            message_id,
            total_amount,
            due_date,
        ),
    )


@app.route('/invoices', endpoint='invoices_page')
@login_required
def invoices():
    invoices_list = []
    summary = {"total_overdue": 0, "total_amount_due": 0, "very_overdue": 0}
    no_phone_count = 0
    conn = None
    cursor = None

    try:
        conn = mysql.connector.connect(
            host=MYSQL_DB_HOST,
            database=MYSQL_DB_NAME,
            user=MYSQL_DB_USER,
            password=MYSQL_DB_PASSWORD,
        )
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT
                i.invoice_id,
                i.invoice_number,
                i.customer_name,
                i.plan_name,
                i.total,
                i.due_date,
                i.status,
                i.zoho_contact_id,
                COALESCE(zc.mobile, zc.phone, '') AS customer_phone,
                zc.email AS customer_email
            FROM invoices i
            LEFT JOIN zoho_customers zc ON zc.zoho_contact_id = i.zoho_contact_id
            WHERE i.status IN ('overdue', 'sent', 'unpaid')
            ORDER BY i.due_date ASC
            """
        )
        invoices_list = cursor.fetchall() or []

        cursor.execute(
            """
            SELECT
                COUNT(*) AS total_overdue,
                SUM(total) AS total_amount_due,
                SUM(CASE WHEN DATEDIFF(CURDATE(), due_date) > 30 THEN 1 ELSE 0 END) AS very_overdue
            FROM invoices
            WHERE status IN ('overdue', 'sent', 'unpaid')
            """
        )
        summary_row = cursor.fetchone() or {}
        summary = {
            "total_overdue": int(summary_row.get("total_overdue") or 0),
            "total_amount_due": float(summary_row.get("total_amount_due") or 0),
            "very_overdue": int(summary_row.get("very_overdue") or 0),
        }

        cursor.execute(
            """
            SELECT t.invoice_id, t.status, t.sent_at
            FROM whatsapp_logs t
            INNER JOIN (
                SELECT invoice_id, MAX(sent_at) AS latest_sent_at
                FROM whatsapp_logs
                GROUP BY invoice_id
            ) s ON s.invoice_id = t.invoice_id AND s.latest_sent_at = t.sent_at
            """
        )
        latest_logs = {row["invoice_id"]: row for row in (cursor.fetchall() or [])}

        now_utc = datetime.utcnow()
        for row in invoices_list:
            normalized_phone = normalize_invoice_phone(row.get("customer_phone"))
            row["normalized_phone"] = normalized_phone
            row["has_valid_phone"] = bool(normalized_phone)
            if not normalized_phone:
                no_phone_count += 1
            log_row = latest_logs.get(row.get("invoice_id"))
            if not log_row:
                row["last_sent_label"] = "Not sent"
                continue
            last_status = (log_row.get("status") or '').lower()
            sent_at = log_row.get("sent_at")
            if last_status == "failed":
                row["last_sent_label"] = "Failed"
            elif last_status == "delivered":
                row["last_sent_label"] = "Delivered"
            elif sent_at:
                diff_hours = max(0, int((now_utc - sent_at).total_seconds() // 3600))
                row["last_sent_label"] = "Sent just now" if diff_hours == 0 else f"Sent {diff_hours}h ago"
            else:
                row["last_sent_label"] = "Sent"
    except MySQLError as exc:
        app.logger.error("MySQL error: %s", exc)
        flash("Could not load invoice data. Please try again.", "error")
        invoices_list = []
        summary = {"total_overdue": 0, "total_amount_due": 0, "very_overdue": 0}
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()

    return render_template(
        'invoices.html',
        invoices=invoices_list,
        summary=summary,
        no_phone_count=no_phone_count,
    )


@app.route('/api/invoices/manual-fetch', methods=['POST'])
@login_required
def manual_fetch_invoices():
    """
    Manually trigger the existing PHP Zoho invoice sync script, then allow staff to refresh /invoices.
    """
    script_path = Path(app.root_path) / 'sync_zoho_invoices.php'
    if not script_path.exists():
        return jsonify({"status": "error", "message": "Sync script not found."}), 404

    try:
        result = subprocess.run(
            ['php', str(script_path)],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
            cwd=app.root_path,
        )
        stdout_text = (result.stdout or '').strip()
        stderr_text = (result.stderr or '').strip()
        if result.returncode != 0:
            app.logger.error(
                "Manual invoice fetch failed. exit=%s stderr=%s",
                result.returncode,
                stderr_text[:500],
            )
            return jsonify({
                "status": "error",
                "message": "Manual fetch failed. Please try again.",
                "details": stderr_text[:500] or stdout_text[:500],
            }), 500

        app.logger.info("Manual invoice fetch completed successfully.")
        return jsonify({
            "status": "success",
            "message": "Invoice data fetched successfully. Refreshing list.",
            "output": stdout_text[:500],
        }), 200
    except subprocess.TimeoutExpired:
        app.logger.error("Manual invoice fetch timed out after 180 seconds.")
        return jsonify({"status": "error", "message": "Manual fetch timed out. Please try again."}), 504
    except Exception as exc:
        app.logger.error("Manual invoice fetch error: %s", exc, exc_info=True)
        return jsonify({"status": "error", "message": "Could not trigger manual fetch."}), 500


@app.route('/api/invoices/send-whatsapp', methods=['POST'])
@login_required
def send_invoice_whatsapp():
    payload = request.get_json(silent=True) or {}
    required_fields = ['invoice_id', 'phone', 'customer_name', 'plan_name', 'amount', 'due_date']
    missing = [field for field in required_fields if not str(payload.get(field, '')).strip()]
    if missing:
        return jsonify({"status": "error", "message": f"Missing fields: {', '.join(missing)}"}), 400

    invoice_id = str(payload.get('invoice_id')).strip()
    phone_raw = str(payload.get('phone')).strip()
    customer_name = str(payload.get('customer_name')).strip()
    plan_name = str(payload.get('plan_name')).strip()
    amount_value = str(payload.get('amount')).strip()
    due_date_value = str(payload.get('due_date')).strip()
    invoice_number = str(payload.get('invoice_number') or invoice_id).strip()

    normalized_phone = normalize_invoice_phone(phone_raw)
    if not normalized_phone:
        return jsonify({"status": "error", "message": "Invalid phone number"}), 400

    amount_display = amount_value if amount_value.startswith('₹') else f"₹{amount_value}"
    due_date_str = format_due_date_for_whatsapp(due_date_value)

    conn = None
    cursor = None
    try:
        response_json = send_payment_overdue_whatsapp(
            normalized_phone,
            customer_name,
            plan_name,
            amount_display,
            due_date_str,
        )
        message_id = (response_json.get('messages') or [{}])[0].get('id')
        total_amount = None
        try:
            total_amount = float(str(amount_value).replace('₹', '').replace(',', '').strip())
        except (TypeError, ValueError):
            total_amount = None

        conn = mysql.connector.connect(
            host=MYSQL_DB_HOST,
            database=MYSQL_DB_NAME,
            user=MYSQL_DB_USER,
            password=MYSQL_DB_PASSWORD,
        )
        cursor = conn.cursor(dictionary=True)
        log_invoice_whatsapp_attempt(
            cursor=cursor,
            invoice_id=invoice_id,
            invoice_number=invoice_number,
            customer_name=customer_name,
            phone=normalized_phone,
            status='sent',
            total_amount=total_amount,
            due_date=due_date_value,
            message_id=message_id,
        )
        conn.commit()
        app.logger.info("Invoice WhatsApp sent for invoice_id=%s to %s", invoice_id, normalized_phone)
        return jsonify({"status": "success", "message_id": message_id, "phone": normalized_phone}), 200
    except requests.RequestException as exc:
        app.logger.error("WhatsApp API error for invoice_id=%s: %s", invoice_id, exc)
        error_text = str(exc)
        try:
            conn = mysql.connector.connect(
                host=MYSQL_DB_HOST,
                database=MYSQL_DB_NAME,
                user=MYSQL_DB_USER,
                password=MYSQL_DB_PASSWORD,
            )
            cursor = conn.cursor(dictionary=True)
            log_invoice_whatsapp_attempt(
                cursor=cursor,
                invoice_id=invoice_id,
                invoice_number=invoice_number,
                customer_name=customer_name,
                phone=normalized_phone,
                status='failed',
                due_date=due_date_value,
                error_message=error_text,
            )
            conn.commit()
        except MySQLError as db_exc:
            app.logger.error("MySQL error: %s", db_exc)
        return jsonify({"status": "error", "message": error_text}), 500
    except MySQLError as exc:
        app.logger.error("MySQL error: %s", exc)
        return jsonify({"status": "error", "message": "Could not log send attempt."}), 500
    except Exception as exc:
        app.logger.error("Unexpected invoice send error: %s", exc, exc_info=True)
        return jsonify({"status": "error", "message": str(exc)}), 500
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()


@app.route('/api/invoices/send-bulk-whatsapp', methods=['POST'])
@login_required
def send_bulk_invoice_whatsapp():
    payload = request.get_json(silent=True) or {}
    invoice_ids = payload.get('invoice_ids') or []
    if not isinstance(invoice_ids, list) or not invoice_ids:
        return jsonify({"status": "error", "message": "invoice_ids must be a non-empty array"}), 400

    cleaned_ids = [str(item).strip() for item in invoice_ids if str(item).strip()]
    if not cleaned_ids:
        return jsonify({"status": "error", "message": "invoice_ids must be a non-empty array"}), 400

    sent_count = 0
    failed_count = 0
    results = []
    conn = None
    cursor = None

    try:
        conn = mysql.connector.connect(
            host=MYSQL_DB_HOST,
            database=MYSQL_DB_NAME,
            user=MYSQL_DB_USER,
            password=MYSQL_DB_PASSWORD,
        )
        cursor = conn.cursor(dictionary=True)
        placeholders = ','.join(['%s'] * len(cleaned_ids))
        cursor.execute(
            f"""
            SELECT
                i.invoice_id,
                i.invoice_number,
                i.customer_name,
                i.plan_name,
                i.total,
                i.due_date,
                COALESCE(zc.mobile, zc.phone, '') AS customer_phone
            FROM invoices i
            LEFT JOIN zoho_customers zc ON zc.zoho_contact_id = i.zoho_contact_id
            WHERE i.invoice_id IN ({placeholders})
            """,
            tuple(cleaned_ids),
        )
        invoice_rows = cursor.fetchall() or []
        invoice_map = {row["invoice_id"]: row for row in invoice_rows}

        for invoice_id in cleaned_ids:
            row = invoice_map.get(invoice_id)
            if not row:
                failed_count += 1
                results.append({"invoice_id": invoice_id, "status": "failed", "phone": ""})
                continue

            normalized_phone = normalize_invoice_phone(row.get("customer_phone"))
            if not normalized_phone:
                failed_count += 1
                results.append({"invoice_id": invoice_id, "status": "failed", "phone": ""})
                log_invoice_whatsapp_attempt(
                    cursor=cursor,
                    invoice_id=row.get("invoice_id"),
                    invoice_number=row.get("invoice_number"),
                    customer_name=row.get("customer_name"),
                    phone='',
                    status='failed',
                    total_amount=row.get("total"),
                    due_date=row.get("due_date"),
                    error_message='Invalid or missing phone number',
                )
                continue

            amount_value = float(row.get("total") or 0)
            amount_display = f"₹{amount_value:,.2f}"
            due_date_str = format_due_date_for_whatsapp(row.get("due_date"))

            try:
                response_json = send_payment_overdue_whatsapp(
                    normalized_phone,
                    (row.get("customer_name") or '').strip(),
                    (row.get("plan_name") or '').strip(),
                    amount_display,
                    due_date_str,
                )
                message_id = (response_json.get('messages') or [{}])[0].get('id')
                sent_count += 1
                results.append({"invoice_id": invoice_id, "status": "sent", "phone": normalized_phone})
                log_invoice_whatsapp_attempt(
                    cursor=cursor,
                    invoice_id=row.get("invoice_id"),
                    invoice_number=row.get("invoice_number"),
                    customer_name=row.get("customer_name"),
                    phone=normalized_phone,
                    status='sent',
                    total_amount=amount_value,
                    due_date=row.get("due_date"),
                    message_id=message_id,
                )
            except Exception as exc:
                failed_count += 1
                results.append({"invoice_id": invoice_id, "status": "failed", "phone": normalized_phone})
                log_invoice_whatsapp_attempt(
                    cursor=cursor,
                    invoice_id=row.get("invoice_id"),
                    invoice_number=row.get("invoice_number"),
                    customer_name=row.get("customer_name"),
                    phone=normalized_phone,
                    status='failed',
                    total_amount=amount_value,
                    due_date=row.get("due_date"),
                    error_message=str(exc),
                )
                app.logger.error("Bulk WhatsApp send failed for invoice_id=%s: %s", invoice_id, exc)

        conn.commit()
        return jsonify({"sent": sent_count, "failed": failed_count, "results": results}), 200
    except MySQLError as exc:
        app.logger.error("MySQL error: %s", exc)
        return jsonify({"status": "error", "message": "Could not process bulk send."}), 500
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()


@app.route('/api/invoices/check-sent/<invoice_id>')
@login_required
def check_invoice_sent_status(invoice_id):
    conn = None
    cursor = None
    try:
        conn = mysql.connector.connect(
            host=MYSQL_DB_HOST,
            database=MYSQL_DB_NAME,
            user=MYSQL_DB_USER,
            password=MYSQL_DB_PASSWORD,
        )
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT
                MAX(CASE WHEN DATE(sent_at) = CURDATE() THEN 1 ELSE 0 END) AS sent_today,
                SUBSTRING_INDEX(GROUP_CONCAT(status ORDER BY sent_at DESC), ',', 1) AS last_status
            FROM whatsapp_logs
            WHERE invoice_id = %s
            """,
            (invoice_id,),
        )
        row = cursor.fetchone() or {}
        return jsonify({
            "sent_today": bool(row.get("sent_today")),
            "last_status": row.get("last_status"),
        }), 200
    except MySQLError as exc:
        app.logger.error("MySQL error: %s", exc)
        return jsonify({"sent_today": False, "last_status": None}), 200
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()
@app.route('/api/whatsapp/toggle-ai', methods=['POST'])
def toggle_whatsapp_ai():
    try:
        data = request.get_json(force=True)

        phone = str(data.get('phone', '')).strip()
        human_takeover = int(data.get('human_takeover', 0))

        if not phone:
            return jsonify({
                'success': False,
                'error': 'Phone is required'
            }), 400

        conn = mysql.connector.connect(
            host=MYSQL_DB_HOST,
            database=MYSQL_DB_NAME,
            user=MYSQL_DB_USER,
            password=MYSQL_DB_PASSWORD,
        )

        cursor = conn.cursor()

        cursor.execute("""
            UPDATE whatsapp_conversations
            SET
                human_takeover = %s,
                updated_at = NOW()
            WHERE phone = %s
        """, (
            human_takeover,
            phone
        ))

        conn.commit()

        cursor.close()
        conn.close()

        return jsonify({
            'success': True,
            'phone': phone,
            'human_takeover': human_takeover,
            'mode': 'human' if human_takeover else 'ai'
        })

    except Exception as e:
        app.logger.exception("Failed to toggle WhatsApp AI mode")

        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    try:
        zoho_customers = get_all_zoho_customers()
        inserted_count, updated_count = save_customers_to_db(zoho_customers)
        print(
            f"Zoho customer sync complete. Total fetched: {len(zoho_customers)}, "
            f"inserted: {inserted_count}, updated: {updated_count}"
        )
    except Exception as exc:
        app.logger.error("Zoho customer sync failed during startup: %s", exc, exc_info=True)
        print(f"Zoho customer sync failed: {exc}")
    _start_ping_thread()
    app.run(debug=True)
