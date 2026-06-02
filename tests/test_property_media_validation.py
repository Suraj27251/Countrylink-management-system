"""
Property-based tests for media file validation using Hypothesis.

**Validates: Requirements 14.2**

Property 19: Media file size validation
- image ≤ 5MB accepted, > 5MB rejected
- video ≤ 16MB accepted, > 16MB rejected
- document ≤ 100MB accepted, > 100MB rejected
- Unsupported MIME types always rejected with specific constraint
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import io
from unittest.mock import MagicMock

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from blueprints.media_bp import _validate_file, MEDIA_TYPE_CONFIG, ALL_SUPPORTED_MIMES


# --- Constants matching WhatsApp Business API limits ---

IMAGE_MAX_BYTES = 5 * 1024 * 1024       # 5 MB
VIDEO_MAX_BYTES = 16 * 1024 * 1024      # 16 MB
DOCUMENT_MAX_BYTES = 100 * 1024 * 1024  # 100 MB

IMAGE_MIMES = sorted(MEDIA_TYPE_CONFIG["image"]["allowed_mimes"])
VIDEO_MIMES = sorted(MEDIA_TYPE_CONFIG["video"]["allowed_mimes"])
DOCUMENT_MIMES = sorted(MEDIA_TYPE_CONFIG["document"]["allowed_mimes"])


# --- Strategies ---

st_image_mime = st.sampled_from(IMAGE_MIMES)
st_video_mime = st.sampled_from(VIDEO_MIMES)
st_document_mime = sorted(MEDIA_TYPE_CONFIG["document"]["allowed_mimes"])
st_document_mime = st.sampled_from(DOCUMENT_MIMES)

# File sizes within accepted bounds
st_valid_image_size = st.integers(min_value=1, max_value=IMAGE_MAX_BYTES)
st_valid_video_size = st.integers(min_value=1, max_value=VIDEO_MAX_BYTES)
st_valid_document_size = st.integers(min_value=1, max_value=DOCUMENT_MAX_BYTES)

# File sizes exceeding limits (just over to well over)
st_oversized_image = st.integers(min_value=IMAGE_MAX_BYTES + 1, max_value=IMAGE_MAX_BYTES + 10 * 1024 * 1024)
st_oversized_video = st.integers(min_value=VIDEO_MAX_BYTES + 1, max_value=VIDEO_MAX_BYTES + 20 * 1024 * 1024)
st_oversized_document = st.integers(min_value=DOCUMENT_MAX_BYTES + 1, max_value=DOCUMENT_MAX_BYTES + 50 * 1024 * 1024)

# Unsupported MIME types
st_unsupported_mime = st.sampled_from([
    "application/zip",
    "application/x-tar",
    "audio/mpeg",
    "audio/wav",
    "application/octet-stream",
    "text/html",
    "application/javascript",
    "image/svg+xml",
    "video/webm",
    "application/x-executable",
])


# --- Helpers ---

def make_file_storage(size_bytes: int, content_type: str) -> MagicMock:
    """
    Create a mock file storage object that simulates a file of the given size
    with the specified content type. Uses an in-memory BytesIO buffer to
    support seek/tell operations used by _validate_file().
    """
    buf = io.BytesIO(b"\x00" * size_bytes)
    mock = MagicMock()
    mock.content_type = content_type
    mock.seek = buf.seek
    mock.tell = buf.tell
    return mock


# --- Property Tests ---

class TestImageSizeValidation:
    """
    Property 19 — Image file acceptance: media_type=image AND size ≤ 5MB.

    For any file with an image MIME type, the Media_Library SHALL accept it
    if and only if the file size is ≤ 5MB (5,242,880 bytes).

    **Validates: Requirements 14.2**
    """

    @given(mime=st_image_mime, size=st_valid_image_size)
    @settings(max_examples=200)
    def test_image_accepted_within_size_limit(self, mime, size):
        """
        Property: An image file with size ≤ 5MB SHALL be accepted and
        classified as media_type 'image'.

        **Validates: Requirements 14.2**
        """
        file_storage = make_file_storage(size, mime)
        media_type, error = _validate_file(file_storage, content_type=mime)

        assert media_type == "image"
        assert error is None

    @given(mime=st_image_mime, size=st_oversized_image)
    @settings(max_examples=200)
    def test_image_rejected_over_size_limit(self, mime, size):
        """
        Property: An image file with size > 5MB SHALL be rejected with an
        error message specifying the size constraint violated.

        **Validates: Requirements 14.2**
        """
        file_storage = make_file_storage(size, mime)
        media_type, error = _validate_file(file_storage, content_type=mime)

        assert media_type is None
        assert error is not None
        assert "5MB" in error or "5mb" in error.lower()
        assert "image" in error.lower()


class TestVideoSizeValidation:
    """
    Property 19 — Video file acceptance: media_type=video AND size ≤ 16MB.

    For any file with a video MIME type, the Media_Library SHALL accept it
    if and only if the file size is ≤ 16MB (16,777,216 bytes).

    **Validates: Requirements 14.2**
    """

    @given(mime=st_video_mime, size=st_valid_video_size)
    @settings(max_examples=200)
    def test_video_accepted_within_size_limit(self, mime, size):
        """
        Property: A video file with size ≤ 16MB SHALL be accepted and
        classified as media_type 'video'.

        **Validates: Requirements 14.2**
        """
        file_storage = make_file_storage(size, mime)
        media_type, error = _validate_file(file_storage, content_type=mime)

        assert media_type == "video"
        assert error is None

    @given(mime=st_video_mime, size=st_oversized_video)
    @settings(max_examples=200)
    def test_video_rejected_over_size_limit(self, mime, size):
        """
        Property: A video file with size > 16MB SHALL be rejected with an
        error message specifying the size constraint violated.

        **Validates: Requirements 14.2**
        """
        file_storage = make_file_storage(size, mime)
        media_type, error = _validate_file(file_storage, content_type=mime)

        assert media_type is None
        assert error is not None
        assert "16MB" in error or "16mb" in error.lower()
        assert "video" in error.lower()


class TestDocumentSizeValidation:
    """
    Property 19 — Document file acceptance: media_type=document AND size ≤ 100MB.

    For any file with a document MIME type, the Media_Library SHALL accept it
    if and only if the file size is ≤ 100MB (104,857,600 bytes).

    **Validates: Requirements 14.2**
    """

    @given(mime=st_document_mime, size=st_valid_document_size)
    @settings(max_examples=200)
    def test_document_accepted_within_size_limit(self, mime, size):
        """
        Property: A document file with size ≤ 100MB SHALL be accepted and
        classified as media_type 'document'.

        **Validates: Requirements 14.2**
        """
        file_storage = make_file_storage(size, mime)
        media_type, error = _validate_file(file_storage, content_type=mime)

        assert media_type == "document"
        assert error is None

    @given(mime=st_document_mime, size=st_oversized_document)
    @settings(max_examples=200)
    def test_document_rejected_over_size_limit(self, mime, size):
        """
        Property: A document file with size > 100MB SHALL be rejected with an
        error message specifying the size constraint violated.

        **Validates: Requirements 14.2**
        """
        file_storage = make_file_storage(size, mime)
        media_type, error = _validate_file(file_storage, content_type=mime)

        assert media_type is None
        assert error is not None
        assert "100MB" in error or "100mb" in error.lower()
        assert "document" in error.lower()


class TestUnsupportedMimeRejection:
    """
    Property 19 — Unsupported MIME types are always rejected.

    For any file with a MIME type not in the supported set (image, video,
    document types defined in MEDIA_TYPE_CONFIG), the Media_Library SHALL
    reject the upload with a specific error message listing supported types.

    **Validates: Requirements 14.2**
    """

    @given(
        mime=st_unsupported_mime,
        size=st.integers(min_value=1, max_value=50 * 1024 * 1024),
    )
    @settings(max_examples=200)
    def test_unsupported_mime_always_rejected(self, mime, size):
        """
        Property: Any file with an unsupported MIME type SHALL be rejected
        regardless of file size, with an error message stating the type
        is unsupported.

        **Validates: Requirements 14.2**
        """
        file_storage = make_file_storage(size, mime)
        media_type, error = _validate_file(file_storage, content_type=mime)

        assert media_type is None
        assert error is not None
        assert "unsupported" in error.lower() or "Unsupported" in error

    @given(size=st.integers(min_value=1, max_value=50 * 1024 * 1024))
    @settings(max_examples=100)
    def test_empty_mime_rejected(self, size):
        """
        Property: A file with an empty or undetermined MIME type SHALL be
        rejected with a specific error.

        **Validates: Requirements 14.2**
        """
        file_storage = make_file_storage(size, "")
        # Override content_type on mock to be empty as well
        file_storage.content_type = ""
        media_type, error = _validate_file(file_storage, content_type="")

        assert media_type is None
        assert error is not None


class TestAcceptanceMatrixBoundary:
    """
    Property 19 — Boundary validation at exact size limits.

    Files at exactly the size limit SHALL be accepted. Files at limit + 1 byte
    SHALL be rejected. This tests the boundary condition of the acceptance matrix.

    **Validates: Requirements 14.2**
    """

    @given(mime=st_image_mime)
    @settings(max_examples=50)
    def test_image_at_exact_limit_accepted(self, mime):
        """
        Property: An image file at exactly 5MB (5,242,880 bytes) SHALL be accepted.

        **Validates: Requirements 14.2**
        """
        file_storage = make_file_storage(IMAGE_MAX_BYTES, mime)
        media_type, error = _validate_file(file_storage, content_type=mime)

        assert media_type == "image"
        assert error is None

    @given(mime=st_image_mime)
    @settings(max_examples=50)
    def test_image_at_one_over_limit_rejected(self, mime):
        """
        Property: An image file at 5MB + 1 byte SHALL be rejected.

        **Validates: Requirements 14.2**
        """
        file_storage = make_file_storage(IMAGE_MAX_BYTES + 1, mime)
        media_type, error = _validate_file(file_storage, content_type=mime)

        assert media_type is None
        assert error is not None

    @given(mime=st_video_mime)
    @settings(max_examples=50)
    def test_video_at_exact_limit_accepted(self, mime):
        """
        Property: A video file at exactly 16MB (16,777,216 bytes) SHALL be accepted.

        **Validates: Requirements 14.2**
        """
        file_storage = make_file_storage(VIDEO_MAX_BYTES, mime)
        media_type, error = _validate_file(file_storage, content_type=mime)

        assert media_type == "video"
        assert error is None

    @given(mime=st_video_mime)
    @settings(max_examples=50)
    def test_video_at_one_over_limit_rejected(self, mime):
        """
        Property: A video file at 16MB + 1 byte SHALL be rejected.

        **Validates: Requirements 14.2**
        """
        file_storage = make_file_storage(VIDEO_MAX_BYTES + 1, mime)
        media_type, error = _validate_file(file_storage, content_type=mime)

        assert media_type is None
        assert error is not None

    @given(mime=st_document_mime)
    @settings(max_examples=50)
    def test_document_at_exact_limit_accepted(self, mime):
        """
        Property: A document file at exactly 100MB (104,857,600 bytes) SHALL be accepted.

        **Validates: Requirements 14.2**
        """
        file_storage = make_file_storage(DOCUMENT_MAX_BYTES, mime)
        media_type, error = _validate_file(file_storage, content_type=mime)

        assert media_type == "document"
        assert error is None

    @given(mime=st_document_mime)
    @settings(max_examples=50)
    def test_document_at_one_over_limit_rejected(self, mime):
        """
        Property: A document file at 100MB + 1 byte SHALL be rejected.

        **Validates: Requirements 14.2**
        """
        file_storage = make_file_storage(DOCUMENT_MAX_BYTES + 1, mime)
        media_type, error = _validate_file(file_storage, content_type=mime)

        assert media_type is None
        assert error is not None
