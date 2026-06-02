"""
Segmentation Engine Blueprint — audience builder, filter evaluation, saved segments.

Flask Blueprint registered at /api/segments/
Routes use existing Flask session authentication from auth.py.
"""

import logging
from functools import wraps

from flask import Blueprint, jsonify, request, session

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Flask Blueprint
# ---------------------------------------------------------------------------
segment_bp = Blueprint("segments", __name__, url_prefix="/api/segments")


def _require_auth(f):
    """Decorator: require Flask session authentication."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("user_id"):
            return jsonify({"error": "Authentication required."}), 401
        return f(*args, **kwargs)
    return decorated


def _get_service():
    """Get SegmentationService instance using app's MySQL connection factory."""
    from app import get_mysql_connection
    from services.segmentation import SegmentationService
    return SegmentationService(get_mysql_connection)


# ------------------------------------------------------------------
# Estimate count (real-time)
# ------------------------------------------------------------------

@segment_bp.route("/estimate", methods=["POST"])
@_require_auth
def estimate_count():
    """
    Estimate the number of customers matching filter criteria.

    Expects JSON body with filter criteria dict.
    Returns {"count": <int>, "warning": <str|null>}
    """
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "Filter criteria required in request body."}), 400

    filters = data.get("filters", data)
    service = _get_service()

    try:
        count = service.estimate_count(filters)
        result = {"count": count}
        if count == 0:
            result["warning"] = (
                "No customers match the specified filter criteria. "
                "Please adjust your filters."
            )
        return jsonify(result), 200
    except Exception as exc:
        logger.exception("Failed to estimate segment count")
        return jsonify({"error": str(exc)}), 500


# ------------------------------------------------------------------
# Evaluate segment (paginated results)
# ------------------------------------------------------------------

@segment_bp.route("/evaluate", methods=["POST"])
@_require_auth
def evaluate_segment():
    """
    Evaluate filter criteria and return paginated customer list.

    Expects JSON body: {"filters": {...}, "page": 1, "per_page": 50}
    """
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "Filter criteria required in request body."}), 400

    filters = data.get("filters", {})
    page = data.get("page", 1)
    per_page = data.get("per_page", 50)

    service = _get_service()

    try:
        result = service.evaluate_segment(filters, page=page, per_page=per_page)
        return jsonify(result), 200
    except Exception as exc:
        logger.exception("Failed to evaluate segment")
        return jsonify({"error": str(exc)}), 500


# ------------------------------------------------------------------
# Save segment
# ------------------------------------------------------------------

@segment_bp.route("/", methods=["POST"])
@_require_auth
def save_segment():
    """
    Save a named segment definition.

    Expects JSON body: {"name": "...", "filters": {...}, "description": "..."}
    """
    data = request.get_json(force=True)
    if not data or not data.get("name"):
        return jsonify({"error": "Segment name is required."}), 400
    if not data.get("filters"):
        return jsonify({"error": "Filter criteria are required."}), 400

    service = _get_service()
    operator = session.get("user_name", "system")

    try:
        segment = service.save_segment(
            name=data["name"],
            filters=data["filters"],
            description=data.get("description", ""),
            created_by=operator,
        )
        return jsonify(segment), 201
    except Exception as exc:
        logger.exception("Failed to save segment")
        return jsonify({"error": str(exc)}), 500


# ------------------------------------------------------------------
# Load segment by ID
# ------------------------------------------------------------------

@segment_bp.route("/<int:segment_id>", methods=["GET"])
@_require_auth
def get_segment(segment_id):
    """Load a saved segment definition by ID."""
    service = _get_service()
    try:
        segment = service.load_segment(segment_id)
        return jsonify(segment), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        logger.exception("Failed to load segment")
        return jsonify({"error": str(exc)}), 500


# ------------------------------------------------------------------
# List segments (paginated)
# ------------------------------------------------------------------

@segment_bp.route("/", methods=["GET"])
@_require_auth
def list_segments():
    """List saved segments with pagination."""
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 20, type=int)

    service = _get_service()
    try:
        result = service.list_segments(page=page, per_page=per_page)
        return jsonify(result), 200
    except Exception as exc:
        logger.exception("Failed to list segments")
        return jsonify({"error": str(exc)}), 500


# ------------------------------------------------------------------
# Re-evaluate a saved segment (dynamic refresh)
# ------------------------------------------------------------------

@segment_bp.route("/<int:segment_id>/evaluate", methods=["GET"])
@_require_auth
def evaluate_saved_segment(segment_id):
    """
    Re-evaluate a saved segment against current data.

    Query params: page (default 1), per_page (default 50)
    """
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 50, type=int)

    service = _get_service()
    try:
        segment = service.load_segment(segment_id)
        filters = segment.get("filter_criteria", {})
        result = service.evaluate_segment(filters, page=page, per_page=per_page)
        result["segment_id"] = segment_id
        result["segment_name"] = segment.get("name")
        return jsonify(result), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        logger.exception("Failed to evaluate saved segment")
        return jsonify({"error": str(exc)}), 500
