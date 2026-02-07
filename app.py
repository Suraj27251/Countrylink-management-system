import os
import zipfile
from pathlib import Path
import csv
import pickle
import socket
import threading
import time
import mimetypes
import uuid
from flask import Flask, render_template, request, redirect, url_for, jsonify, session, flash
from werkzeug.utils import secure_filename
import sqlite3
from datetime import datetime
from collections import defaultdict
from functools import wraps
import requests

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-change-me')

DB_PATH = os.environ.get('DATABASE_PATH')
if not DB_PATH:
    default_path = Path(app.root_path) / 'database' / 'complaints.db'
    DB_PATH = str(default_path if default_path.exists() else Path(app.root_path) / 'complaints.db')

def get_db_connection():
    return sqlite3.connect(DB_PATH)

WEBHOOK_VERIFY_TOKEN = os.environ.get('WEBHOOK_VERIFY_TOKEN')

WHATSAPP_API_VERSION = os.environ.get('WHATSAPP_API_VERSION', 'v20.0')
WHATSAPP_MEDIA_DIR = Path(app.root_path) / 'static' / 'uploads' / 'whatsapp'


def ensure_provider_assets():
    base_dir = Path(app.root_path)
    zip_path = base_dir / 'providerHTML-main.zip'
    target_dir = base_dir / 'static' / 'provider' / 'providerHTML-main'
    if target_dir.exists():
        return
    if not zip_path.exists():
        return
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(target_dir.parent)


ensure_provider_assets()

# ---- WhatsApp helpers ----
def normalize_mobile(raw_mobile):
    if not raw_mobile:
        return ''
    digits = ''.join(ch for ch in str(raw_mobile).strip() if ch.isdigit())
    return digits


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
    c = conn.cursor()
    c.execute("""
        SELECT id, message_id, name, mobile, direction, message_type, text, media_url, file_name, media_mime_type, created_at
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
                "created_at": row["created_at"],
            }
            for row in legacy_rows
        ]
    return rows, legacy_mode


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


def _load_or_train_model():
    """Load a trained model or train a new one from sample data."""
    global _model, _vectorizer
    if _model and _vectorizer:
        return
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            _model, _vectorizer = pickle.load(f)
        return

    training_file = os.path.join('setup', 'complaint_training_data.csv')
    if not os.path.exists(training_file):
        return

    texts, labels = [], []
    with open(training_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            texts.append(row.get('complaint', ''))
            labels.append(row.get('category', 'Other'))

    if not texts:
        return

    _vectorizer = TfidfVectorizer()
    X = _vectorizer.fit_transform(texts)
    _model = MultinomialNB()
    _model.fit(X, labels)

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump((_model, _vectorizer), f)


def predict_category(text: str) -> str:
    """Predict complaint category with confidence threshold."""
    _load_or_train_model()
    if not _model or not _vectorizer:
        return 'Other'

    probs = _model.predict_proba(_vectorizer.transform([text]))[0]
    max_prob = probs.max()
    if max_prob < 0.5:
        return 'Other'
    return _model.classes_[probs.argmax()]


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
    try:
        c.execute("ALTER TABLE complaints ADD COLUMN category TEXT DEFAULT 'Other'")
    except sqlite3.OperationalError:
        pass

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
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    try:
        c.execute("ALTER TABLE whatsapp_messages ADD COLUMN message_id TEXT")
    except sqlite3.OperationalError:
        pass
    c.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_whatsapp_messages_message_id ON whatsapp_messages(message_id)")

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

    c.execute('SELECT COUNT(*) FROM complaints')
    total = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM complaints WHERE status = 'Pending'")
    pending = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM complaints WHERE status = 'Resolved'")
    resolved = c.fetchone()[0]

    c.execute('''
    SELECT * FROM (
        SELECT * FROM complaints
        WHERE source != 'WhatsApp' OR source IS NULL
        ORDER BY created_at DESC
    )
    GROUP BY mobile
    ORDER BY created_at DESC
    LIMIT 50
''')
    recent_complaints_raw = c.fetchall()

    priority_complaints = []
    for comp in recent_complaints_raw:
        mobile = comp['mobile']
        c.execute("""
            SELECT COUNT(*) FROM complaints
            WHERE mobile = ? AND date(created_at) >= date('now', '-30 day')
        """, (mobile,))
        count = c.fetchone()[0]
        priority = "High" if count >= 3 else "Medium" if count == 2 else "Low"
        priority_complaints.append(dict(comp) | {'priority': priority})

    c.execute("SELECT id, name, mobile, area, status, created_at FROM connection_requests WHERE status = 'Pending' ORDER BY created_at DESC LIMIT 5")
    pending_connections = c.fetchall()

    c.execute("SELECT COUNT(*) FROM connection_requests WHERE status = 'Pending'")
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


@app.route('/update_status/<int:complaint_id>/<status>')
@login_required
def update_status(complaint_id, status):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("UPDATE complaints SET status = ? WHERE id = ?", (status, complaint_id))
    conn.commit()
    conn.close()
    return redirect(url_for('dashboard'))


@app.route('/webhook', methods=['GET', 'POST'])
@app.route('/webhook/whatsapp', methods=['GET', 'POST'])
def webhook():
    if request.method == 'GET':
        challenge = request.args.get('hub.challenge')
        verify_token = request.args.get('hub.verify_token')
        if WEBHOOK_VERIFY_TOKEN and verify_token != WEBHOOK_VERIFY_TOKEN:
            app.logger.warning("Webhook verification failed: invalid token.")
            return 'Verification failed', 403
        return challenge or '', 200

    if request.method == 'POST':
        try:
            data = request.get_json(force=True, silent=True)
            if not data:
                app.logger.warning("Webhook received no JSON payload.")
                return jsonify({"error": "No JSON data received"}), 400

            for entry in data.get('entry', []):
                for change in entry.get('changes', []):
                    value = change.get('value', {})
                    contacts = value.get('contacts', [])
                    messages = value.get('messages', [])

                    if messages:
                        contacts_by_wa = {
                            contact.get('wa_id'): contact
                            for contact in contacts
                            if contact.get('wa_id')
                        }
                        for msg in messages:
                            message_type = msg.get('type', 'text')
                            message_id = msg.get('id')
                            timestamp_unix = msg.get('timestamp')
                            created_at = datetime.fromtimestamp(int(timestamp_unix)).strftime('%Y-%m-%d %H:%M:%S') if timestamp_unix else datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                            from_id = msg.get('from', '')
                            contact = contacts_by_wa.get(from_id)
                            name = contact.get('profile', {}).get('name', 'Unknown') if contact else 'Unknown'
                            mobile = normalize_mobile(contact.get('wa_id', '') if contact else from_id)

                            text_body = extract_text_body(msg)
                            media_id = None
                            media_url = None
                            media_mime_type = None
                            file_name = None

                            if message_type in {'image', 'video', 'audio', 'document'}:
                                media_info = msg.get(message_type, {})
                                media_id = media_info.get('id')
                                text_body = media_info.get('caption', '') or text_body
                                file_name = media_info.get('filename') if message_type == 'document' else None
                                media_mime_type = media_info.get('mime_type')

                                if media_id and whatsapp_config_ready():
                                    try:
                                        downloaded_path, stored_name, mime_type = download_whatsapp_media(media_id, file_name)
                                        media_url = f"uploads/whatsapp/{stored_name}"
                                        media_mime_type = mime_type or media_mime_type
                                    except Exception as exc:
                                        app.logger.warning("Failed to download WhatsApp media %s: %s", media_id, exc)
                                        media_url = None
                            if not text_body:
                                text_body = safe_message_preview(message_type, text_body)

                            if mobile.strip():
                                conn = get_db_connection()
                                c = conn.cursor()
                                if message_id:
                                    c.execute(
                                        """
                                        INSERT OR IGNORE INTO whatsapp_messages
                                        (message_id, name, mobile, direction, message_type, text, media_id, media_url, media_mime_type, file_name, created_at)
                                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                        """,
                                        (
                                            message_id,
                                            name.strip() if name else None,
                                            mobile,
                                            'inbound',
                                            message_type,
                                            text_body,
                                            media_id,
                                            media_url,
                                            media_mime_type,
                                            file_name,
                                            created_at,
                                        ),
                                    )
                                else:
                                    c.execute(
                                        """
                                        INSERT INTO whatsapp_messages
                                        (message_id, name, mobile, direction, message_type, text, media_id, media_url, media_mime_type, file_name, created_at)
                                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                        """,
                                        (
                                            None,
                                            name.strip() if name else None,
                                            mobile,
                                            'inbound',
                                            message_type,
                                            text_body,
                                            media_id,
                                            media_url,
                                            media_mime_type,
                                            file_name,
                                            created_at,
                                        ),
                                    )
                                conn.commit()
                                conn.close()

                            if message_type == 'text' and name.strip() and name.strip() != '.' and mobile.strip() and text_body.strip():
                                category = predict_category(text_body)
                                conn = get_db_connection()
                                c = conn.cursor()
                                c.execute(
                                    """
                                    INSERT INTO complaints (name, mobile, complaint, category, status, created_at, source)
                                    VALUES (?, ?, ?, ?, ?, ?, ?)
                                    """,
                                    (name, mobile, text_body, category, 'Pending', created_at, 'WhatsApp')
                                )
                                conn.commit()
                                conn.close()
            return jsonify({"status": "Message received"}), 200

        except Exception:
            app.logger.exception("Webhook processing failed.")
            return jsonify({"error": "Webhook processing failed"}), 500


@app.after_request
def set_default_json_header(response):
    if request.path.startswith('/webhook') or request.path.startswith('/webhook/whatsapp') or request.path.startswith('/flow-endpoint'):
        response.headers['Content-Type'] = 'application/json'
    return response


@app.route('/flow-endpoint', methods=['POST'])
def flow_endpoint():
    data = request.get_json()
    name = data.get("name")
    mobile = data.get("mobile")
    complaint = data.get("complaint")

    if not all([name, mobile, complaint]):
        return jsonify({"error": "Missing required fields"}), 400

    category = predict_category(complaint)
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        "INSERT INTO complaints (name, mobile, complaint, category) VALUES (?, ?, ?, ?)",
        (name, mobile, complaint, category)
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
        WHERE source != 'WhatsApp'
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
        WHERE source != 'WhatsApp'
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

    latest_by_mobile = {}
    for row in rows:
        mobile = normalize_mobile(row["mobile"])
        if mobile in latest_by_mobile:
            continue
        preview = safe_message_preview(row["message_type"], row["text"])
        latest_by_mobile[mobile] = {
            "mobile": mobile,
            "name": row["name"] or mobile,
            "preview": preview,
            "created_at": row["created_at"],
        }

    contacts = sorted(latest_by_mobile.values(), key=lambda item: item["created_at"], reverse=True)
    active_mobile = normalize_mobile(request.args.get("mobile")) or (contacts[0]["mobile"] if contacts else None)

    messages = []
    active_name = None
    if active_mobile:
        messages = [row for row in rows if normalize_mobile(row["mobile"]) == active_mobile]
        messages = sorted(messages, key=lambda item: (item["created_at"], item["id"]))
        active_name = contacts[0]["name"] if contacts and contacts[0]["mobile"] == active_mobile else None
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
    )


@app.route('/api/whatsapp/messages')
@login_required
def whatsapp_messages_api():
    mobile = normalize_mobile(request.args.get('mobile', ''))
    since_id = request.args.get('since_id', type=int)
    include_contacts = request.args.get('include_contacts', '0') == '1'

    conn = get_db_connection()
    conn.row_factory = sqlite3.Row
    rows, legacy_mode = load_whatsapp_rows(conn)

    latest_by_mobile = {}
    for row in rows:
        row_mobile = normalize_mobile(row["mobile"])
        if row_mobile in latest_by_mobile:
            continue
        latest_by_mobile[row_mobile] = {
            "mobile": row_mobile,
            "name": row["name"] or row_mobile,
            "preview": safe_message_preview(row["message_type"], row["text"]),
            "created_at": row["created_at"],
        }
    contacts = sorted(latest_by_mobile.values(), key=lambda item: item["created_at"], reverse=True)

    messages = []
    active_name = ""
    if mobile:
        messages = [row for row in rows if normalize_mobile(row["mobile"]) == mobile]
        if since_id:
            messages = [row for row in messages if row["id"] > since_id]
        messages = sorted(messages, key=lambda item: (item["created_at"], item["id"]))
        active_name = next((contact["name"] for contact in contacts if contact["mobile"] == mobile), "")
        if not active_name and messages:
            active_name = messages[0]["name"] or mobile

    conn.close()

    def serialize_message(msg):
        return {
            "id": msg["id"],
            "message_id": msg.get("message_id") if isinstance(msg, dict) else msg["message_id"],
            "name": msg["name"],
            "mobile": normalize_mobile(msg["mobile"]),
            "direction": msg["direction"],
            "message_type": msg["message_type"],
            "text": msg["text"],
            "media_url": msg["media_url"],
            "file_name": msg["file_name"],
            "media_mime_type": msg.get("media_mime_type") if isinstance(msg, dict) else msg["media_mime_type"],
            "created_at": msg["created_at"],
        }

    serialized_messages = [serialize_message(msg) for msg in messages]
    last_message_id = serialized_messages[-1]["id"] if serialized_messages else None

    return jsonify({
        "contacts": contacts if include_contacts else [],
        "messages": serialized_messages,
        "active_mobile": mobile,
        "active_name": active_name,
        "last_message_id": last_message_id,
        "legacy_mode": legacy_mode,
    })


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
        (message_id, name, mobile, direction, message_type, text, media_id, media_url, media_mime_type, file_name, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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


@app.route('/update_whatsapp_bulk', methods=['POST'])
@login_required
def update_whatsapp_bulk():
    action = request.form.get('action')
    ids = request.form.getlist('selected_ids[]')

    if not ids:
        return redirect(url_for('dashboard'))

    conn = get_db_connection()
    c = conn.cursor()

    if action == 'resolve':
        c.executemany("UPDATE complaints SET status = 'Resolved' WHERE id = ?", [(i,) for i in ids])
    elif action == 'delete':
        c.executemany("DELETE FROM complaints WHERE id = ?", [(i,) for i in ids])

    conn.commit()
    conn.close()
    return redirect(url_for('dashboard'))


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


if __name__ == '__main__':
    _start_ping_thread()
    app.run(debug=True)
