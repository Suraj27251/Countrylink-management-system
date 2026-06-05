from flask import Blueprint, render_template, request, redirect, url_for, flash, session
import os
from pathlib import Path
import sqlite3
from hmac import compare_digest

import mysql.connector
from werkzeug.security import generate_password_hash, check_password_hash

auth_bp = Blueprint('auth', __name__)

MYSQL_DB_HOST = os.environ.get('MYSQL_DB_HOST', 'localhost')
MYSQL_DB_NAME = os.environ.get('MYSQL_DB_NAME', 'countrylinks_user_database')
MYSQL_DB_USER = os.environ.get('MYSQL_DB_USER', 'countrylinks_Suraj27251')
MYSQL_DB_PASSWORD = os.environ.get('MYSQL_DB_PASSWORD', '')

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = os.environ.get('DATABASE_PATH')
if not DB_PATH:
    default_path = BASE_DIR / 'database' / 'complaints.db'
    DB_PATH = str(default_path if default_path.exists() else BASE_DIR / 'complaints.db')


def get_db_connection():
    return sqlite3.connect(DB_PATH)


def get_db():
    return get_db_connection()


def get_mysql_connection():
    return mysql.connector.connect(
        host=MYSQL_DB_HOST,
        database=MYSQL_DB_NAME,
        user=MYSQL_DB_USER,
        password=MYSQL_DB_PASSWORD,
        charset='utf8mb4',
        collation='utf8mb4_unicode_ci',
        use_unicode=True,
        time_zone='+05:30',
    )


@auth_bp.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        if not name or not email or not password:
            flash('All fields are required.', 'error')
            return render_template('auth/signup.html')

        with get_db() as conn:
            c = conn.cursor()
            c.execute('SELECT id FROM users WHERE email=?', (email,))
            if c.fetchone():
                flash('Email already registered. Try logging in.', 'error')
                return redirect(url_for('auth.login'))
            c.execute(
                'INSERT INTO users(name, email, password_hash, role) VALUES (?,?,?,?)',
                (name, email, generate_password_hash(password), 'user')
            )
            conn.commit()

        flash('Account created. Please log in.', 'success')
        return redirect(url_for('auth.login'))

    return render_template('auth/signup.html')

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    return _handle_login(require_admin=False)


@auth_bp.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    return _handle_login(require_admin=True)


def _handle_login(require_admin=False):
    login_title = 'Admin Login' if require_admin else 'Welcome Back'
    login_button_label = 'Login as Admin' if require_admin else 'Login'
    login_context = {
        'login_title': login_title,
        'login_button_label': login_button_label,
        'login_endpoint': 'auth.admin_login' if require_admin else 'auth.login',
        'show_signup_link': not require_admin,
        'login_input_type': 'text' if require_admin else 'email',
        'login_input_placeholder': 'User ID or Email' if require_admin else 'Email',
        'login_input_autocomplete': 'username' if require_admin else 'email',
    }

    if request.method == 'POST':
        identifier = request.form.get('email', '').strip()
        password = request.form.get('password', '')

        if require_admin:
            try:
                user = _get_mysql_user(identifier)
            except mysql.connector.Error:
                flash('Login service unavailable. Please try again later.', 'error')
                return render_template('auth/login.html', **login_context)

            if not user or not _password_matches(user.get('password'), password):
                flash('Invalid user ID or password.', 'error')
                return render_template('auth/login.html', **login_context)

            if _is_inactive_user(user):
                flash('This user account is inactive.', 'error')
                return render_template('auth/login.html', **login_context)

            user_name = user.get('name') or user.get('email') or f"User {user.get('id')}"
            user_role = (user.get('role') or 'admin').strip().lower()
            session['user_id'] = user.get('id')
            session['user_name'] = user_name
            session['user_email'] = user.get('email') or ''
            session['user_role'] = user_role
            session['auth_source'] = 'mysql_users'
            if user.get('permissions'):
                session['permissions'] = _parse_permissions(user['permissions'])
            flash(f'Welcome back, {user_name}!', 'success')
            return redirect(url_for('dashboard'))

        email = identifier.lower()
        with get_db() as conn:
            c = conn.cursor()
            c.execute('PRAGMA table_info(users)')
            columns = [column[1] for column in c.fetchall()]
            selected_columns = ['id', 'name', 'email', 'password_hash', 'role']
            if 'permissions' in columns:
                selected_columns.append('permissions')
            c.execute(
                f"SELECT {', '.join(selected_columns)} FROM users WHERE email=?",
                (email,)
            )
            row = c.fetchone()

        if not row or not check_password_hash(row[3], password):
            flash('Invalid email or password.', 'error')
            return render_template('auth/login.html', **login_context)

        user_role = (row[4] or 'user').strip().lower()
        session['user_id'] = row[0]
        session['user_name'] = row[1]
        session['user_email'] = row[2]
        session['user_role'] = user_role
        if len(row) > 5 and row[5]:
            session['permissions'] = _parse_permissions(row[5])
        flash(f'Welcome back, {row[1]}!', 'success')
        return redirect(url_for('dashboard'))

    return render_template('auth/login.html', **login_context)


def _get_mysql_user(identifier):
    if not identifier:
        return None

    conn = get_mysql_connection()
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute('SHOW COLUMNS FROM users')
        columns = {row['Field'] for row in cursor.fetchall()}
        selectable_columns = [
            column for column in ('id', 'name', 'email', 'phone', 'password', 'status', 'role', 'permissions')
            if column in columns
        ]
        if not {'id', 'password'}.issubset(columns) or not selectable_columns:
            return None

        where_clauses = []
        params = []
        if 'email' in columns:
            where_clauses.append('email = %s')
            params.append(identifier.lower())
        if 'phone' in columns:
            where_clauses.append('phone = %s')
            params.append(identifier)
        if 'id' in columns and identifier.isdigit():
            where_clauses.append('id = %s')
            params.append(int(identifier))
        if not where_clauses:
            return None

        select_clause = ', '.join(f'`{column}`' for column in selectable_columns)
        cursor.execute(
            f"SELECT {select_clause} FROM users WHERE {' OR '.join(where_clauses)} LIMIT 1",
            tuple(params),
        )
        return cursor.fetchone()
    finally:
        conn.close()


def _password_matches(stored_password, submitted_password):
    stored_password = str(stored_password or '')
    submitted_password = str(submitted_password or '')
    if not stored_password or not submitted_password:
        return False

    hash_prefixes = ('pbkdf2:', 'scrypt:')
    if stored_password.startswith(hash_prefixes):
        return check_password_hash(stored_password, submitted_password)
    return compare_digest(stored_password, submitted_password)


def _is_inactive_user(user):
    status = str(user.get('status') or '').strip().lower()
    return status in {'inactive', 'disabled', 'blocked', 'suspended', '0', 'no', 'false'}


def _parse_permissions(raw_permissions):
    """Normalize database permissions stored as JSON or comma-separated text."""
    if not raw_permissions:
        return []
    if isinstance(raw_permissions, str):
        try:
            import json
            decoded = json.loads(raw_permissions)
        except (TypeError, ValueError):
            decoded = raw_permissions
        if isinstance(decoded, str):
            return [perm.strip() for perm in decoded.split(',') if perm.strip()]
        raw_permissions = decoded
    if isinstance(raw_permissions, (list, tuple, set)):
        return [str(perm).strip() for perm in raw_permissions if str(perm).strip()]
    return []

@auth_bp.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('auth.login'))
