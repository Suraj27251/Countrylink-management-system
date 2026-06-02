"""
Media Library Blueprint — file upload, media grid, WhatsApp media type validation.

Flask Blueprint registered at /api/media/
Handles media asset uploads with size/type validation per WhatsApp Business API limits,
grid view with search and type filtering, and usage count tracking.

Requirements: 14.1, 14.2, 14.3, 14.4, 14.5
"""

import logging
import os
import uuid
from datetime import datetime
from functools import wraps
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request, session
from werkzeug.utils import secure_filename

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Flask Blueprint
# ---------------------------------------------------------------------------
media_bp = Blueprint("media", __name__, url_prefix="/api/media")

# ---------------------------------------------------------------------------
# Media type configuration — WhatsApp Business API limits
# ---------------------------------------------------------------------------
MEDIA_TYPE_CONFIG = {
    "image": {
        "max_size_bytes": 5 * 1024 * 1024,  # 5 MB
        "max_size_label": "5MB",
        "allowed_mimes": {
            "image/jpeg", "image/png", "image/webp", "image/gif",
        },
    },
    "video": {
        "max_size_bytes": 16 * 1024 * 1024,  # 16 MB
        "max_size_label": "16MB",
        "allowed_mimes": {
            "video/mp4", "video/3gpp",
        },
    },
    "document": {
        "max_size_bytes": 100 * 1024 * 1024,  # 100 MB
        "max_size_label": "100MB",
        "allowed_mimes": {
            "application/pdf",
            "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/vnd.ms-excel",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.ms-powerpoint",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "text/plain",
            "text/csv",
        },
    },
}

# Build a flat set of all supported mimes for quick lookup
ALL_SUPPORTED_MIMES = set()
for cfg in MEDIA_TYPE_CONFIG.values():
    ALL_SUPPORTED_MIMES.update(cfg["allowed_mimes"])

# Media storage directory (relative to app root)
MEDIA_UPLOAD_DIR = Path("static") / "uploads" / "media"
MEDIA_THUMBNAIL_DIR = Path("static") / "uploads" / "media" / "thumbnails"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_auth(f):
    """Decorator: require Flask session authentication."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("user_id"):
            return jsonify({"error": "Authentication required."}), 401
        return f(*args, **kwargs)
    return decorated


def _get_connection():
    """Get MySQL connection from app context."""
    from app import get_mysql_connection
    return get_mysql_connection()


def _ensure_upload_dirs():
    """Ensure media upload directories exist."""
    base = Path(current_app.root_path)
    (base / MEDIA_UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    (base / MEDIA_THUMBNAIL_DIR).mkdir(parents=True, exist_ok=True)


def _detect_media_type(mime_type: str) -> str | None:
    """Determine media_type category from MIME type. Returns None if unsupported."""
    for media_type, config in MEDIA_TYPE_CONFIG.items():
        if mime_type in config["allowed_mimes"]:
            return media_type
    return None


def _validate_file(file_storage, content_type: str | None = None):
    """
    Validate uploaded file against size and MIME type constraints.

    Returns:
        tuple: (media_type, error_message) — error_message is None on success.
    """
    # Determine MIME type from file or content-type header
    mime_type = content_type or file_storage.content_type or ""
    mime_type = mime_type.strip().lower()

    if not mime_type:
        return None, "MIME type could not be determined. Please specify a valid content type."

    # Check if MIME type is supported at all
    if mime_type not in ALL_SUPPORTED_MIMES:
        return None, (
            f"Unsupported file type: '{mime_type}'. "
            f"Supported types: images (JPEG, PNG, WebP, GIF), "
            f"videos (MP4, 3GPP), documents (PDF, Word, Excel, PowerPoint, TXT, CSV)."
        )

    # Determine category
    media_type = _detect_media_type(mime_type)
    if media_type is None:
        return None, f"Unsupported file type: '{mime_type}'."

    # Check file size — read to determine actual size
    file_storage.seek(0, os.SEEK_END)
    file_size = file_storage.tell()
    file_storage.seek(0)

    config = MEDIA_TYPE_CONFIG[media_type]
    if file_size > config["max_size_bytes"]:
        return None, (
            f"File size ({file_size / (1024 * 1024):.2f}MB) exceeds the "
            f"maximum allowed size for {media_type} files ({config['max_size_label']}). "
            f"WhatsApp Business API limit: {media_type} ≤ {config['max_size_label']}."
        )

    return media_type, None


# ---------------------------------------------------------------------------
# Upload endpoint (Requirements 14.1, 14.2, 14.5)
# ---------------------------------------------------------------------------

@media_bp.route("/upload", methods=["POST"])
@_require_auth
def upload_media():
    """
    Upload a media file to the library.

    Expects multipart/form-data with:
        - file: The media file to upload
        - content_type (optional): Override MIME type detection

    Returns:
        201: Media asset record on success.
        400: Validation error (size, type).
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided. Include a 'file' field in the upload."}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No filename specified."}), 400

    # Allow explicit content_type override from form data
    content_type = request.form.get("content_type", None)

    # Validate file
    media_type, error = _validate_file(file, content_type)
    if error:
        return jsonify({"error": error}), 400

    # Determine actual mime and size
    mime_type = (content_type or file.content_type or "").strip().lower()
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    # Generate unique filename
    original_filename = secure_filename(file.filename) or "upload"
    ext = os.path.splitext(original_filename)[1] or ""
    unique_name = f"{uuid.uuid4().hex}{ext}"

    # Save file to disk
    _ensure_upload_dirs()
    base = Path(current_app.root_path)
    storage_path = str(MEDIA_UPLOAD_DIR / unique_name)
    full_path = base / storage_path
    file.save(str(full_path))

    # Generate thumbnail path (placeholder — actual thumbnail generation
    # would be done by a background task or on-the-fly)
    thumbnail_path = None
    if media_type == "image":
        thumbnail_path = str(MEDIA_THUMBNAIL_DIR / f"thumb_{unique_name}")

    # Get operator name from session
    uploaded_by = session.get("user_name", session.get("username", "unknown"))
    organization_id = session.get("organization_id", 1)

    # Insert into database
    conn = None
    try:
        conn = _get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO media_assets
                (organization_id, filename, original_filename, mime_type,
                 file_size_bytes, media_type, storage_path, thumbnail_path,
                 usage_count, uploaded_by, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 0, %s, NOW())
            """,
            (
                organization_id,
                unique_name,
                original_filename,
                mime_type,
                file_size,
                media_type,
                storage_path,
                thumbnail_path,
                uploaded_by,
            ),
        )
        conn.commit()
        asset_id = cursor.lastrowid

        asset = {
            "id": asset_id,
            "organization_id": organization_id,
            "filename": unique_name,
            "original_filename": original_filename,
            "mime_type": mime_type,
            "file_size_bytes": file_size,
            "media_type": media_type,
            "storage_path": storage_path,
            "thumbnail_path": thumbnail_path,
            "usage_count": 0,
            "uploaded_by": uploaded_by,
            "created_at": datetime.utcnow().isoformat(),
        }

        logger.info(
            "Media uploaded: %s (%s, %d bytes) by %s",
            original_filename, media_type, file_size, uploaded_by,
        )
        return jsonify(asset), 201

    except Exception as exc:
        # Clean up file if DB insert fails
        if full_path.exists():
            full_path.unlink(missing_ok=True)
        logger.exception("Failed to save media asset to database")
        return jsonify({"error": "Internal server error"}), 500
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Grid view endpoint (Requirements 14.3, 14.4)
# ---------------------------------------------------------------------------

@media_bp.route("/", methods=["GET"])
@_require_auth
def list_media():
    """
    List media assets in a paginated grid view with search and type filtering.

    Query params:
        page (int): Page number (default 1)
        per_page (int): Items per page (default 20, max 100)
        search (str): Search by original_filename (partial match)
        media_type (str): Filter by type: image, video, document
        sort (str): Sort field: created_at, filename, file_size_bytes, usage_count (default: created_at)
        order (str): Sort order: asc, desc (default: desc)

    Returns:
        200: Paginated list of media assets with metadata.
    """
    page = request.args.get("page", 1, type=int)
    per_page = min(request.args.get("per_page", 20, type=int), 100)
    search = request.args.get("search", "", type=str).strip()
    media_type_filter = request.args.get("media_type", "", type=str).strip().lower()
    sort_field = request.args.get("sort", "created_at", type=str).strip()
    sort_order = request.args.get("order", "desc", type=str).strip().lower()

    # Validate sort fields to prevent injection
    allowed_sort_fields = {"created_at", "filename", "file_size_bytes", "usage_count", "original_filename"}
    if sort_field not in allowed_sort_fields:
        sort_field = "created_at"
    if sort_order not in ("asc", "desc"):
        sort_order = "desc"

    # Validate media_type filter
    if media_type_filter and media_type_filter not in MEDIA_TYPE_CONFIG:
        return jsonify({"error": f"Invalid media_type filter: '{media_type_filter}'. Must be one of: image, video, document."}), 400

    organization_id = session.get("organization_id", 1)
    offset = (page - 1) * per_page

    conn = None
    try:
        conn = _get_connection()
        cursor = conn.cursor(dictionary=True)

        # Build query with optional filters
        where_clauses = ["organization_id = %s"]
        params = [organization_id]

        if search:
            where_clauses.append("original_filename LIKE %s")
            params.append(f"%{search}%")

        if media_type_filter:
            where_clauses.append("media_type = %s")
            params.append(media_type_filter)

        where_sql = " AND ".join(where_clauses)

        # Get total count
        count_sql = f"SELECT COUNT(*) as total FROM media_assets WHERE {where_sql}"
        cursor.execute(count_sql, params)
        total = cursor.fetchone()["total"]

        # Get paginated results
        query_sql = f"""
            SELECT id, organization_id, filename, original_filename, mime_type,
                   file_size_bytes, media_type, storage_path, thumbnail_path,
                   usage_count, uploaded_by, created_at
            FROM media_assets
            WHERE {where_sql}
            ORDER BY {sort_field} {sort_order}
            LIMIT %s OFFSET %s
        """
        cursor.execute(query_sql, params + [per_page, offset])
        assets = cursor.fetchall()

        # Serialize datetime fields
        for asset in assets:
            if asset.get("created_at") and hasattr(asset["created_at"], "isoformat"):
                asset["created_at"] = asset["created_at"].isoformat()

        return jsonify({
            "assets": assets,
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total": total,
                "total_pages": (total + per_page - 1) // per_page if per_page > 0 else 0,
            },
        }), 200

    except Exception as exc:
        logger.exception("Failed to list media assets")
        return jsonify({"error": "Internal server error"}), 500
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Get single media asset (Requirement 14.3)
# ---------------------------------------------------------------------------

@media_bp.route("/<int:asset_id>", methods=["GET"])
@_require_auth
def get_media(asset_id):
    """
    Get a single media asset by ID.

    Returns:
        200: Media asset record.
        404: Asset not found.
    """
    organization_id = session.get("organization_id", 1)
    conn = None
    try:
        conn = _get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT id, organization_id, filename, original_filename, mime_type,
                   file_size_bytes, media_type, storage_path, thumbnail_path,
                   usage_count, uploaded_by, created_at
            FROM media_assets
            WHERE id = %s AND organization_id = %s
            """,
            (asset_id, organization_id),
        )
        asset = cursor.fetchone()
        if not asset:
            return jsonify({"error": "Media asset not found."}), 404

        if asset.get("created_at") and hasattr(asset["created_at"], "isoformat"):
            asset["created_at"] = asset["created_at"].isoformat()

        return jsonify(asset), 200

    except Exception as exc:
        logger.exception("Failed to get media asset %d", asset_id)
        return jsonify({"error": "Internal server error"}), 500
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Filter by type for template media selection (Requirement 14.3)
# ---------------------------------------------------------------------------

@media_bp.route("/by-type/<media_type>", methods=["GET"])
@_require_auth
def list_media_by_type(media_type):
    """
    List media assets filtered by type for template media selection.

    Path params:
        media_type (str): One of: image, video, document

    Query params:
        page (int): Page number (default 1)
        per_page (int): Items per page (default 20)

    Returns:
        200: Paginated list of assets of the specified type.
        400: Invalid media_type.
    """
    media_type = media_type.strip().lower()
    if media_type not in MEDIA_TYPE_CONFIG:
        return jsonify({
            "error": f"Invalid media type: '{media_type}'. Must be one of: image, video, document."
        }), 400

    page = request.args.get("page", 1, type=int)
    per_page = min(request.args.get("per_page", 20, type=int), 100)
    organization_id = session.get("organization_id", 1)
    offset = (page - 1) * per_page

    conn = None
    try:
        conn = _get_connection()
        cursor = conn.cursor(dictionary=True)

        # Count
        cursor.execute(
            "SELECT COUNT(*) as total FROM media_assets WHERE organization_id = %s AND media_type = %s",
            (organization_id, media_type),
        )
        total = cursor.fetchone()["total"]

        # Fetch
        cursor.execute(
            """
            SELECT id, organization_id, filename, original_filename, mime_type,
                   file_size_bytes, media_type, storage_path, thumbnail_path,
                   usage_count, uploaded_by, created_at
            FROM media_assets
            WHERE organization_id = %s AND media_type = %s
            ORDER BY created_at DESC
            LIMIT %s OFFSET %s
            """,
            (organization_id, media_type, per_page, offset),
        )
        assets = cursor.fetchall()

        for asset in assets:
            if asset.get("created_at") and hasattr(asset["created_at"], "isoformat"):
                asset["created_at"] = asset["created_at"].isoformat()

        return jsonify({
            "assets": assets,
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total": total,
                "total_pages": (total + per_page - 1) // per_page if per_page > 0 else 0,
            },
        }), 200

    except Exception as exc:
        logger.exception("Failed to list media by type '%s'", media_type)
        return jsonify({"error": "Internal server error"}), 500
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Usage count tracking (Requirement 14.4)
# ---------------------------------------------------------------------------

@media_bp.route("/<int:asset_id>/increment-usage", methods=["POST"])
@_require_auth
def increment_usage(asset_id):
    """
    Increment the usage count for a media asset.

    Called when a media asset is attached to a campaign template.

    Returns:
        200: Updated usage count.
        404: Asset not found.
    """
    organization_id = session.get("organization_id", 1)
    conn = None
    try:
        conn = _get_connection()
        cursor = conn.cursor(dictionary=True)

        # Verify asset exists
        cursor.execute(
            "SELECT id, usage_count FROM media_assets WHERE id = %s AND organization_id = %s",
            (asset_id, organization_id),
        )
        asset = cursor.fetchone()
        if not asset:
            return jsonify({"error": "Media asset not found."}), 404

        # Increment
        cursor.execute(
            "UPDATE media_assets SET usage_count = usage_count + 1 WHERE id = %s",
            (asset_id,),
        )
        conn.commit()

        return jsonify({
            "id": asset_id,
            "usage_count": asset["usage_count"] + 1,
        }), 200

    except Exception as exc:
        logger.exception("Failed to increment usage for asset %d", asset_id)
        return jsonify({"error": "Internal server error"}), 500
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass


@media_bp.route("/<int:asset_id>/decrement-usage", methods=["POST"])
@_require_auth
def decrement_usage(asset_id):
    """
    Decrement the usage count for a media asset.

    Called when a media asset is detached from a campaign template.
    Usage count will not go below 0.

    Returns:
        200: Updated usage count.
        404: Asset not found.
    """
    organization_id = session.get("organization_id", 1)
    conn = None
    try:
        conn = _get_connection()
        cursor = conn.cursor(dictionary=True)

        # Verify asset exists
        cursor.execute(
            "SELECT id, usage_count FROM media_assets WHERE id = %s AND organization_id = %s",
            (asset_id, organization_id),
        )
        asset = cursor.fetchone()
        if not asset:
            return jsonify({"error": "Media asset not found."}), 404

        new_count = max(0, asset["usage_count"] - 1)
        cursor.execute(
            "UPDATE media_assets SET usage_count = %s WHERE id = %s",
            (new_count, asset_id),
        )
        conn.commit()

        return jsonify({
            "id": asset_id,
            "usage_count": new_count,
        }), 200

    except Exception as exc:
        logger.exception("Failed to decrement usage for asset %d", asset_id)
        return jsonify({"error": "Internal server error"}), 500
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Delete media asset
# ---------------------------------------------------------------------------

@media_bp.route("/<int:asset_id>", methods=["DELETE"])
@_require_auth
def delete_media(asset_id):
    """
    Delete a media asset by ID.

    Removes the record from the database and the file from disk.
    Will not delete assets with usage_count > 0.

    Returns:
        200: Deletion confirmed.
        404: Asset not found.
        409: Asset is in use — cannot delete.
    """
    organization_id = session.get("organization_id", 1)
    conn = None
    try:
        conn = _get_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute(
            "SELECT id, storage_path, usage_count FROM media_assets WHERE id = %s AND organization_id = %s",
            (asset_id, organization_id),
        )
        asset = cursor.fetchone()
        if not asset:
            return jsonify({"error": "Media asset not found."}), 404

        if asset["usage_count"] > 0:
            return jsonify({
                "error": f"Cannot delete media asset — it is currently used by {asset['usage_count']} template(s). Remove it from all templates first."
            }), 409

        # Delete from disk
        base = Path(current_app.root_path)
        file_path = base / asset["storage_path"]
        if file_path.exists():
            file_path.unlink(missing_ok=True)

        # Delete from database
        cursor.execute("DELETE FROM media_assets WHERE id = %s", (asset_id,))
        conn.commit()

        logger.info("Media asset %d deleted", asset_id)
        return jsonify({"deleted": True, "id": asset_id}), 200

    except Exception as exc:
        logger.exception("Failed to delete media asset %d", asset_id)
        return jsonify({"error": "Internal server error"}), 500
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass
