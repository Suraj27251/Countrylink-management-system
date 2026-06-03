"""
Internal AI Integration APIs

Provides secure machine-readable endpoints for the WhatsApp AI Agent to
perform customer lookups, invoice queries, complaint creation/tracking, and
payment-verification workflow. All endpoints require internal authentication
via the `X-Internal-Token` header except admin-only approve/reject which
require `X-Admin-Token`.

This module intentionally reuses existing MySQL connection factory from
`app.get_mysql_connection` and existing database tables; it avoids
duplicating CRM business rules and keeps responses compact and machine
friendly.
"""
import os
import re
import logging
from datetime import datetime
from flask import Blueprint, request, jsonify

logger = logging.getLogger(__name__)

ai_bp = Blueprint("ai", __name__, url_prefix="/api")


def _internal_auth_required():
    token = request.headers.get("X-Internal-Token", "")
    expected = os.environ.get("AGENT_INTERNAL_TOKEN", "")
    return bool(token and expected and token == expected)


def _admin_auth_required():
    token = request.headers.get("X-Admin-Token", "")
    expected = os.environ.get("ADMIN_API_TOKEN", "")
    return bool(token and expected and token == expected)


def _normalize_mobile_to_last10(raw: str) -> str:
    if not raw:
        return ""
    digits = re.sub(r"\D", "", raw)
    if len(digits) >= 10:
        return digits[-10:]
    return digits


def _format_e164_from_last10(last10: str) -> str:
    if not last10:
        return ""
    return f"+91{last10}"


@ai_bp.before_request
def _require_internal_token():
    # Allow open endpoints if explicitly necessary in future; by default require internal token
    if request.path.startswith("/api/payment-verification/approve") or request.path.startswith("/api/payment-verification/reject"):
        # admin endpoints handled separately
        return None
    if not _internal_auth_required():
        return jsonify({"success": False, "error": "Authentication required (X-Internal-Token)"}), 401


@ai_bp.route("/customer/by-mobile", methods=["GET"]) 
def customer_by_mobile():
    mobile = (request.args.get("mobile") or request.args.get("phone") or "").strip()
    if not mobile:
        return jsonify({"success": False, "error": "mobile parameter is required"}), 400

    last10 = _normalize_mobile_to_last10(mobile)
    try:
        from app import get_mysql_connection
        conn = get_mysql_connection()
        cur = conn.cursor(dictionary=True)

        # Find customer by last 10 digits in mobile or phone
        cur.execute(
            """
            SELECT zoho_contact_id, contact_name, company_name, mobile, phone, outstanding_amount, status
            FROM zoho_customers
            WHERE (mobile IS NOT NULL AND mobile LIKE CONCAT('%', %s))
               OR (phone IS NOT NULL AND phone LIKE CONCAT('%', %s))
            ORDER BY id DESC
            LIMIT 1
            """,
            (last10, last10),
        )
        cust = cur.fetchone()
        cur.close()
        conn.close()
    except Exception as exc:
        logger.exception("customer lookup failed")
        return jsonify({"success": False, "error": "internal_error"}), 500

    if not cust:
        return jsonify({"success": True, "customer": None}), 200

    normalized = _format_e164_from_last10(_normalize_mobile_to_last10(cust.get("mobile") or cust.get("phone") or last10))
    response = {
        "success": True,
        "customer": {
            "name": (cust.get("contact_name") or cust.get("company_name") or "").strip(),
            "mobile": normalized,
            "status": cust.get("status") or "",
            "outstanding_amount": str(cust.get("outstanding_amount")) if cust.get("outstanding_amount") is not None else None,
        }
    }
    return jsonify(response), 200


@ai_bp.route("/customer/invoices", methods=["GET"]) 
def customer_invoices():
    mobile = (request.args.get("mobile") or "").strip()
    limit = int(request.args.get("limit", 5))
    if not mobile:
        return jsonify({"success": False, "error": "mobile parameter is required"}), 400

    last10 = _normalize_mobile_to_last10(mobile)
    try:
        from app import get_mysql_connection
        conn = get_mysql_connection()
        cur = conn.cursor(dictionary=True)

        # Find contact id first
        cur.execute(
            "SELECT zoho_contact_id FROM zoho_customers WHERE (mobile LIKE CONCAT('%', %s) OR phone LIKE CONCAT('%', %s)) ORDER BY id DESC LIMIT 1",
            (last10, last10),
        )
        row = cur.fetchone()
        if not row:
            cur.close()
            conn.close()
            return jsonify({"success": True, "invoices": []}), 200

        contact_id = row.get("zoho_contact_id")
        cur.execute(
            "SELECT invoice_number, status, total, invoice_date, due_date FROM invoices WHERE zoho_contact_id = %s ORDER BY COALESCE(invoice_date, created_at) DESC LIMIT %s",
            (contact_id, limit),
        )
        invs = cur.fetchall()
        cur.close()
        conn.close()
    except Exception:
        logger.exception("failed fetching invoices")
        return jsonify({"success": False, "error": "internal_error"}), 500

    return jsonify({"success": True, "invoices": invs}), 200


@ai_bp.route("/customer/status", methods=["GET"]) 
def customer_status():
    mobile = (request.args.get("mobile") or "").strip()
    if not mobile:
        return jsonify({"success": False, "error": "mobile parameter is required"}), 400

    last10 = _normalize_mobile_to_last10(mobile)
    try:
        from app import get_mysql_connection
        conn = get_mysql_connection()
        cur = conn.cursor(dictionary=True)
        cur.execute(
            "SELECT id, customer_name, mobile, plan_name, status, outstanding_amount FROM customers WHERE (mobile LIKE CONCAT('%', %s) OR phone LIKE CONCAT('%', %s)) ORDER BY id DESC LIMIT 1",
            (last10, last10),
        )
        cust = cur.fetchone()
        cur.close()
        conn.close()
    except Exception:
        logger.exception("customer status lookup failed")
        return jsonify({"success": False, "error": "internal_error"}), 500

    if not cust:
        return jsonify({"success": True, "customer": None}), 200

    mobile_e164 = _format_e164_from_last10(_normalize_mobile_to_last10(cust.get("mobile") or cust.get("phone") or last10))
    return jsonify({
        "success": True,
        "customer": {
            "name": cust.get("customer_name") or "",
            "mobile": mobile_e164,
            "status": cust.get("status") or "",
            "plan_name": cust.get("plan_name") or "",
            "outstanding_amount": str(cust.get("outstanding_amount")) if cust.get("outstanding_amount") is not None else None,
        }
    }), 200


# -------------------- Complaints --------------------
@ai_bp.route("/complaints/create", methods=["POST"]) 
def create_complaint():
    if not _internal_auth_required():
        return jsonify({"success": False, "error": "Authentication required"}), 401

    data = request.get_json(silent=True) or {}
    name = (data.get("name") or data.get("customer_name") or "").strip()
    mobile = (data.get("mobile") or data.get("phone") or "").strip()
    complaint = (data.get("complaint") or data.get("description") or "").strip()
    source = (data.get("source") or "whatsapp_ai").strip()

    if not all([name, mobile, complaint]):
        return jsonify({"success": False, "error": "name, mobile and complaint are required"}), 400

    last10 = _normalize_mobile_to_last10(mobile)
    complaint_id = f"CMP-{uuid4_hex()[:8].upper()}"

    try:
        from app import get_mysql_connection
        conn = get_mysql_connection()
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO complaints (complaint_id, customer_name, customer_phone, complaint_subject, complaint_description, category, status, escalation_level, source, created_at, updated_at)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())""",
            (
                complaint_id,
                name,
                last10,
                complaint[:100],
                complaint,
                "General",
                "open",
                "low",
                source,
            ),
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception:
        logger.exception("Failed to create complaint")
        return jsonify({"success": False, "error": "internal_error"}), 500

    return jsonify({"success": True, "complaint_id": complaint_id, "tracking_url": "https://countrylinks.in/track"}), 201


def uuid4_hex():
    import uuid
    return uuid.uuid4().hex


@ai_bp.route("/complaints/<string:complaint_id>", methods=["GET"]) 
def get_complaint(complaint_id):
    if not _internal_auth_required():
        return jsonify({"success": False, "error": "Authentication required"}), 401

    try:
        from app import get_mysql_connection
        conn = get_mysql_connection()
        cur = conn.cursor(dictionary=True)
        cur.execute(
            "SELECT complaint_id, customer_name, customer_phone, complaint_subject, complaint_description, status, escalation_level, created_at, updated_at FROM complaints WHERE complaint_id = %s OR id = %s LIMIT 1",
            (complaint_id, complaint_id),
        )
        comp = cur.fetchone()
        cur.close()
        conn.close()
    except Exception:
        logger.exception("Failed to fetch complaint")
        return jsonify({"success": False, "error": "internal_error"}), 500

    if not comp:
        return jsonify({"success": False, "error": "not_found"}), 404

    return jsonify({"success": True, "complaint": comp}), 200


@ai_bp.route("/complaints/customer/<string:mobile>", methods=["GET"]) 
def get_complaints_by_customer(mobile):
    if not _internal_auth_required():
        return jsonify({"success": False, "error": "Authentication required"}), 401

    last10 = _normalize_mobile_to_last10(mobile)
    try:
        from app import get_mysql_connection
        conn = get_mysql_connection()
        cur = conn.cursor(dictionary=True)
        cur.execute(
            "SELECT complaint_id, customer_name, customer_phone, complaint_subject, status, created_at, updated_at FROM complaints WHERE customer_phone = %s ORDER BY created_at DESC LIMIT 20",
            (last10,),
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
    except Exception:
        logger.exception("Failed to fetch complaints for customer")
        return jsonify({"success": False, "error": "internal_error"}), 500

    return jsonify({"success": True, "complaints": rows}), 200


# -------------------- Payment verification --------------------
@ai_bp.route('/payment-verification/create', methods=['POST'])
def payment_verification_create():
    if not _internal_auth_required():
        return jsonify({"success": False, "error": "Authentication required"}), 401

    data = request.get_json(silent=True) or {}
    mobile = (data.get('mobile') or data.get('phone') or '').strip()
    screenshot = (data.get('screenshot_ref') or data.get('screenshot') or '').strip()
    amount = data.get('amount')
    utr = (data.get('utr') or data.get('reference') or '').strip()

    if not mobile or not screenshot:
        return jsonify({"success": False, "error": "mobile and screenshot_ref are required"}), 400

    last10 = _normalize_mobile_to_last10(mobile)
    try:
        from app import get_mysql_connection
        conn = get_mysql_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO payment_verification_requests (customer_mobile, screenshot_ref, amount, utr_reference, status, created_at) VALUES (%s, %s, %s, %s, 'pending', NOW())",
            (last10, screenshot, amount, utr),
        )
        conn.commit()
        inserted_id = cur.lastrowid
        cur.close()
        conn.close()
    except Exception:
        logger.exception('Failed to create payment verification request')
        return jsonify({"success": False, "error": "internal_error"}), 500

    return jsonify({"success": True, "id": inserted_id}), 201


@ai_bp.route('/payment-verification/pending', methods=['GET'])
def payment_verification_pending():
    if not _internal_auth_required():
        return jsonify({"success": False, "error": "Authentication required"}), 401
    try:
        from app import get_mysql_connection
        conn = get_mysql_connection()
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT id, customer_mobile, screenshot_ref, amount, utr_reference, status, created_at FROM payment_verification_requests WHERE status = 'pending' ORDER BY created_at ASC LIMIT 100")
        rows = cur.fetchall()
        cur.close()
        conn.close()
    except Exception:
        logger.exception('Failed to fetch pending payment verifications')
        return jsonify({"success": False, "error": "internal_error"}), 500

    return jsonify({"success": True, "pending": rows}), 200


@ai_bp.route('/payment-verification/approve', methods=['POST'])
def payment_verification_approve():
    # Admin-only: AI must NOT be able to approve payments
    if not _admin_auth_required():
        return jsonify({"success": False, "error": "Admin authentication required"}), 401
    data = request.get_json(silent=True) or {}
    req_id = data.get('id')
    admin = data.get('admin') or 'admin'
    remarks = data.get('remarks') or ''
    if not req_id:
        return jsonify({"success": False, "error": "id is required"}), 400
    try:
        from app import get_mysql_connection
        conn = get_mysql_connection()
        cur = conn.cursor()
        cur.execute("UPDATE payment_verification_requests SET status = 'approved', verified_by = %s, admin_remarks = %s, verified_at = NOW() WHERE id = %s", (admin, remarks, req_id))
        conn.commit()
        cur.close()
        conn.close()
    except Exception:
        logger.exception('Failed to approve payment verification')
        return jsonify({"success": False, "error": "internal_error"}), 500
    return jsonify({"success": True}), 200


@ai_bp.route('/payment-verification/reject', methods=['POST'])
def payment_verification_reject():
    if not _admin_auth_required():
        return jsonify({"success": False, "error": "Admin authentication required"}), 401
    data = request.get_json(silent=True) or {}
    req_id = data.get('id')
    admin = data.get('admin') or 'admin'
    remarks = data.get('remarks') or ''
    if not req_id:
        return jsonify({"success": False, "error": "id is required"}), 400
    try:
        from app import get_mysql_connection
        conn = get_mysql_connection()
        cur = conn.cursor()
        cur.execute("UPDATE payment_verification_requests SET status = 'rejected', verified_by = %s, admin_remarks = %s, verified_at = NOW() WHERE id = %s", (admin, remarks, req_id))
        conn.commit()
        cur.close()
        conn.close()
    except Exception:
        logger.exception('Failed to reject payment verification')
        return jsonify({"success": False, "error": "internal_error"}), 500
    return jsonify({"success": True}), 200


# -------------------- Serviceability --------------------
@ai_bp.route('/serviceability', methods=['GET'])
def serviceability():
    area = (request.args.get('area') or '').strip()
    if not area:
        return jsonify({"success": False, "error": "area parameter is required"}), 400
    try:
        from app import get_mysql_connection
        conn = get_mysql_connection()
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT COUNT(*) AS cnt FROM customers WHERE area LIKE %s", (f"%{area}%",))
        cnt = cur.fetchone().get('cnt', 0)
        plans = []
        if cnt:
            cur.execute("SELECT DISTINCT plan_name FROM customers WHERE area LIKE %s LIMIT 10", (f"%{area}%",))
            plans = [r.get('plan_name') for r in cur.fetchall() if r.get('plan_name')]
        cur.close()
        conn.close()
    except Exception:
        logger.exception('Serviceability lookup failed')
        return jsonify({"success": False, "error": "internal_error"}), 500

    return jsonify({
        "available": bool(cnt),
        "area": area,
        "estimated_installation_time": "2-4 hours" if cnt else None,
        "plans_available": plans,
    }), 200
