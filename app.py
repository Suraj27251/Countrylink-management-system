import os
import zipfile
from pathlib import Path
import csv
import pickle
import socket
import threading
import time
from flask import Flask, render_template, request, redirect, url_for, jsonify, session, flash
import sqlite3
from datetime import datetime
from collections import defaultdict
from functools import wraps

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


@app.after_request
def set_default_json_header(response):
    if request.path.startswith('/flow-endpoint'):
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


@app.route('/update_complaints_bulk', methods=['POST'])
@login_required
def update_complaints_bulk():
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


@app.route('/ping')
def ping():
    return 'pong', 200


if __name__ == '__main__':
    _start_ping_thread()
    app.run(debug=True)
