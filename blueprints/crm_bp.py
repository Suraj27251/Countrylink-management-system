"""
CRM Panel Blueprint — customer profile, notes, tags, interaction timeline.

Flask Blueprint registered at /api/crm/
Routes use existing Flask session authentication from auth.py.

Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 19.6, 23.7
"""

import logging
from functools import wraps

from flask import Blueprint, jsonify, request, session

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Flask Blueprint
# ---------------------------------------------------------------------------
crm_bp = Blueprint("crm", __name__, url_prefix="/api/crm")


def _require_auth(f):
    """Decorator: require Flask session authentication."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("user_id"):
            return jsonify({"error": "Authentication required."}), 401
        return f(*args, **kwargs)
    return decorated


def _get_service():
    """Get CRMService instance using app's MySQL connection factory."""
    from app import get_mysql_connection
    from services.crm import CRMService
    return CRMService(get_mysql_connection)


# ------------------------------------------------------------------
# Customer Profile (Requirement 7.1, 19.6, 23.7)
# ------------------------------------------------------------------

@crm_bp.route("/customers/<int:customer_id>/profile", methods=["GET"])
@_require_auth
def get_customer_profile_by_id(customer_id):
    """
    Get full customer profile by customer ID.

    Returns profile with plan details, opt-out/DND status,
    engagement score, and tags.
    """
    service = _get_service()
    try:
        profile = service.get_customer_profile(customer_id=customer_id)
        return jsonify(profile), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        logger.exception("Failed to get customer profile by ID %d", customer_id)
        return jsonify({"error": "Internal server error"}), 500


@crm_bp.route("/customers/by-mobile/<mobile>/profile", methods=["GET"])
@_require_auth
def get_customer_profile_by_mobile(mobile):
    """
    Get full customer profile by mobile number.

    Returns profile with plan details, opt-out/DND status,
    engagement score, and tags.
    """
    service = _get_service()
    try:
        profile = service.get_customer_profile(mobile=mobile)
        return jsonify(profile), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        logger.exception("Failed to get customer profile for mobile %s", mobile)
        return jsonify({"error": "Internal server error"}), 500


# ------------------------------------------------------------------
# Interaction Timeline (Requirement 7.2, 7.6)
# ------------------------------------------------------------------

@crm_bp.route("/customers/<int:customer_id>/timeline", methods=["GET"])
@_require_auth
def get_timeline_by_id(customer_id):
    """
    Get merged interaction timeline for a customer (by ID).

    Query params: page (int, default 1), per_page (int, default 50)
    """
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 50, type=int)

    service = _get_service()
    try:
        result = service.get_interaction_timeline(
            customer_id=customer_id, page=page, per_page=per_page
        )
        return jsonify(result), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        logger.exception("Failed to get timeline for customer %d", customer_id)
        return jsonify({"error": "Internal server error"}), 500


@crm_bp.route("/customers/by-mobile/<mobile>/timeline", methods=["GET"])
@_require_auth
def get_timeline_by_mobile(mobile):
    """
    Get merged interaction timeline for a customer (by mobile).

    Query params: page (int, default 1), per_page (int, default 50)
    """
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 50, type=int)

    service = _get_service()
    try:
        result = service.get_interaction_timeline(
            mobile=mobile, page=page, per_page=per_page
        )
        return jsonify(result), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        logger.exception("Failed to get timeline for mobile %s", mobile)
        return jsonify({"error": "Internal server error"}), 500


# ------------------------------------------------------------------
# Notes (Requirement 7.4)
# ------------------------------------------------------------------

@crm_bp.route("/customers/<int:customer_id>/notes", methods=["POST"])
@_require_auth
def add_note_by_id(customer_id):
    """
    Add a note to a customer profile (by ID).

    Expects JSON: {"note": "text"}
    Operator name taken from session.
    """
    data = request.get_json(force=True)
    if not data or not data.get("note"):
        return jsonify({"error": "Note text is required."}), 400

    operator = session.get("user_name", session.get("username", "unknown"))
    service = _get_service()

    try:
        note = service.add_note(
            customer_id=customer_id,
            note=data["note"],
            operator=operator,
        )
        return jsonify(note), 201
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.exception("Failed to add note for customer %d", customer_id)
        return jsonify({"error": "Internal server error"}), 500


@crm_bp.route("/customers/by-mobile/<mobile>/notes", methods=["POST"])
@_require_auth
def add_note_by_mobile(mobile):
    """
    Add a note to a customer profile (by mobile).

    Expects JSON: {"note": "text"}
    Operator name taken from session.
    """
    data = request.get_json(force=True)
    if not data or not data.get("note"):
        return jsonify({"error": "Note text is required."}), 400

    operator = session.get("user_name", session.get("username", "unknown"))
    service = _get_service()

    try:
        note = service.add_note(
            mobile=mobile,
            note=data["note"],
            operator=operator,
        )
        return jsonify(note), 201
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.exception("Failed to add note for mobile %s", mobile)
        return jsonify({"error": "Internal server error"}), 500


# ------------------------------------------------------------------
# Tags (Requirement 7.5)
# ------------------------------------------------------------------

@crm_bp.route("/customers/<int:customer_id>/tags", methods=["POST"])
@_require_auth
def add_tags_by_id(customer_id):
    """
    Add tags to a customer (by ID).

    Expects JSON: {"tags": ["VIP", "complaint_pending"]}
    Operator name taken from session.
    """
    data = request.get_json(force=True)
    if not data or not data.get("tags"):
        return jsonify({"error": "Tags list is required."}), 400

    operator = session.get("user_name", session.get("username", "unknown"))
    service = _get_service()

    try:
        added = service.add_tags(
            customer_id=customer_id,
            tags=data["tags"],
            operator=operator,
        )
        return jsonify({"added_tags": added}), 201
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.exception("Failed to add tags for customer %d", customer_id)
        return jsonify({"error": "Internal server error"}), 500


@crm_bp.route("/customers/by-mobile/<mobile>/tags", methods=["POST"])
@_require_auth
def add_tags_by_mobile(mobile):
    """
    Add tags to a customer (by mobile).

    Expects JSON: {"tags": ["VIP", "complaint_pending"]}
    Operator name taken from session.
    """
    data = request.get_json(force=True)
    if not data or not data.get("tags"):
        return jsonify({"error": "Tags list is required."}), 400

    operator = session.get("user_name", session.get("username", "unknown"))
    service = _get_service()

    try:
        added = service.add_tags(
            mobile=mobile,
            tags=data["tags"],
            operator=operator,
        )
        return jsonify({"added_tags": added}), 201
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.exception("Failed to add tags for mobile %s", mobile)
        return jsonify({"error": "Internal server error"}), 500


@crm_bp.route("/customers/<int:customer_id>/tags/<tag_name>", methods=["DELETE"])
@_require_auth
def remove_tag_by_id(customer_id, tag_name):
    """
    Remove a tag from a customer (by ID).

    Operator name taken from session.
    """
    operator = session.get("user_name", session.get("username", "unknown"))
    service = _get_service()

    try:
        removed = service.remove_tag(
            customer_id=customer_id,
            tag=tag_name,
            operator=operator,
        )
        if removed:
            return jsonify({"removed": True, "tag": tag_name}), 200
        else:
            return jsonify({"removed": False, "tag": tag_name, "message": "Tag not found on customer"}), 404
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.exception("Failed to remove tag for customer %d", customer_id)
        return jsonify({"error": "Internal server error"}), 500


@crm_bp.route("/customers/by-mobile/<mobile>/tags/<tag_name>", methods=["DELETE"])
@_require_auth
def remove_tag_by_mobile(mobile, tag_name):
    """
    Remove a tag from a customer (by mobile).

    Operator name taken from session.
    """
    operator = session.get("user_name", session.get("username", "unknown"))
    service = _get_service()

    try:
        removed = service.remove_tag(
            mobile=mobile,
            tag=tag_name,
            operator=operator,
        )
        if removed:
            return jsonify({"removed": True, "tag": tag_name}), 200
        else:
            return jsonify({"removed": False, "tag": tag_name, "message": "Tag not found on customer"}), 404
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.exception("Failed to remove tag for mobile %s", mobile)
        return jsonify({"error": "Internal server error"}), 500


# ------------------------------------------------------------------
# Campaign History (Requirement 7.3)
# ------------------------------------------------------------------

@crm_bp.route("/customers/<int:customer_id>/campaigns", methods=["GET"])
@_require_auth
def get_campaign_history_by_id(customer_id):
    """
    Get campaign delivery history for a customer (by ID).

    Returns list of campaigns targeting this customer with delivery status.
    """
    service = _get_service()
    try:
        history = service.get_campaign_history(customer_id=customer_id)
        return jsonify({"campaigns": history}), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        logger.exception("Failed to get campaign history for customer %d", customer_id)
        return jsonify({"error": "Internal server error"}), 500


@crm_bp.route("/customers/by-mobile/<mobile>/campaigns", methods=["GET"])
@_require_auth
def get_campaign_history_by_mobile(mobile):
    """
    Get campaign delivery history for a customer (by mobile).

    Returns list of campaigns targeting this customer with delivery status.
    """
    service = _get_service()
    try:
        history = service.get_campaign_history(mobile=mobile)
        return jsonify({"campaigns": history}), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        logger.exception("Failed to get campaign history for mobile %s", mobile)
        return jsonify({"error": "Internal server error"}), 500
