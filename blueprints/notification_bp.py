"""
Notification Blueprint — alert delivery and acknowledgment endpoints.

Flask Blueprint registered at /api/notifications/
Routes use existing Flask session authentication from auth.py.

Requirements: 25.1, 25.2, 25.3, 25.4, 25.5, 25.6, 25.7
"""

import logging
from functools import wraps

from flask import Blueprint, jsonify, request, session

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Flask Blueprint
# ---------------------------------------------------------------------------
notification_bp = Blueprint("notifications", __name__, url_prefix="/api/notifications")


def _require_auth(f):
    """Decorator: require Flask session authentication."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("user_id"):
            return jsonify({"error": "Authentication required."}), 401
        return f(*args, **kwargs)
    return decorated


def _get_engine():
    """Get NotificationEngine instance using app's MySQL connection factory."""
    from app import get_mysql_connection
    from services.notification_engine import NotificationEngine
    return NotificationEngine(get_mysql_connection)


# ------------------------------------------------------------------
# Get unacknowledged notifications (Requirement 25.6)
# ------------------------------------------------------------------

@notification_bp.route("/unacknowledged", methods=["GET"])
@_require_auth
def get_unacknowledged():
    """
    Get all unacknowledged notifications for the current operator.

    Query Parameters:
        limit (int): Max notifications to return (default 50).

    Returns JSON list of notification objects sorted by severity then date.
    """
    operator_name = session.get("username", session.get("user_id"))
    limit = request.args.get("limit", 50, type=int)

    engine = _get_engine()
    notifications = engine.get_unacknowledged(
        operator_name=str(operator_name) if operator_name else None,
        limit=limit,
    )

    # Serialize datetime objects for JSON response
    for notif in notifications:
        if notif.get("created_at"):
            notif["created_at"] = notif["created_at"].isoformat() if hasattr(notif["created_at"], "isoformat") else str(notif["created_at"])

    return jsonify({"notifications": notifications, "count": len(notifications)}), 200


# ------------------------------------------------------------------
# Get unacknowledged count (for badge display)
# ------------------------------------------------------------------

@notification_bp.route("/count", methods=["GET"])
@_require_auth
def get_unacknowledged_count():
    """
    Get count of unacknowledged notifications for badge indicator.

    Returns JSON with count field.
    """
    engine = _get_engine()
    count = engine.get_unacknowledged_count()
    return jsonify({"count": count}), 200


# ------------------------------------------------------------------
# Acknowledge notification (Requirement 25.6)
# ------------------------------------------------------------------

@notification_bp.route("/<int:notification_id>/acknowledge", methods=["POST"])
@_require_auth
def acknowledge_notification(notification_id):
    """
    Mark a notification as acknowledged by the current operator.

    Returns success status.
    """
    operator_name = session.get("username", session.get("user_id"))

    engine = _get_engine()
    result = engine.acknowledge(notification_id, str(operator_name))

    if result:
        return jsonify({"success": True, "message": "Notification acknowledged."}), 200
    else:
        return jsonify({"success": False, "message": "Notification not found or already acknowledged."}), 404
