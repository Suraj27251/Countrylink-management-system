from flask import Blueprint, render_template, request, redirect, url_for, flash, session
import os
from pathlib import Path
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

auth_bp = Blueprint('auth', __name__)

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = os.environ.get('DATABASE_PATH')
if not DB_PATH:
    default_path = BASE_DIR / 'database' / 'complaints.db'
    DB_PATH = str(default_path if default_path.exists() else BASE_DIR / 'complaints.db')


def get_db_connection():
    return sqlite3.connect(DB_PATH)


def get_db():
    return get_db_connection()

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

    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')

        with get_db() as conn:
            c = conn.cursor()
            c.execute('SELECT id, name, email, password_hash, role FROM users WHERE email=?', (email,))
            row = c.fetchone()

        if not row or not check_password_hash(row[3], password):
            flash('Invalid email or password.', 'error')
            return render_template('auth/login.html', login_title=login_title, login_button_label=login_button_label)

        if require_admin and row[4] != 'admin':
            flash('Admin access required for this login.', 'error')
            return render_template('auth/login.html', login_title=login_title, login_button_label=login_button_label)

        session['user_id'] = row[0]
        session['user_name'] = row[1]
        session['user_email'] = row[2]
        session['user_role'] = row[4]
        flash(f'Welcome back, {row[1]}!', 'success')
        if require_admin:
            return redirect(url_for('dashboard'))
        return redirect(url_for('landing'))

    return render_template('auth/login.html', login_title=login_title, login_button_label=login_button_label)

@auth_bp.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('auth.login'))
