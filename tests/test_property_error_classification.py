"""
Property-based tests for error code classification determinism using Hypothesis.

**Validates: Requirements 21.1**

Property 25: Error code classification determinism
- For any WhatsApp API error code present in the error_classifications lookup table,
  the Retry_Categorizer SHALL return exactly one category (transient, permanent, or
  suppression), and the classification SHALL be deterministic (same code always maps
  to same category).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from migrations.seed_error_classifications import ERROR_CLASSIFICATIONS
from services.retry_categorizer import RetryCategorizerService, FailureCategoryEnum


# ---------------------------------------------------------------------------
# Valid categories (from the FailureCategoryEnum)
# ---------------------------------------------------------------------------
VALID_CATEGORIES = {"transient", "permanent", "suppression"}

# Extract known error codes with their expected classifications
KNOWN_ERROR_CODES = [
    (entry[0], entry[2]) for entry in ERROR_CLASSIFICATIONS
]

# Hypothesis strategies
known_error_entries = st.sampled_from(ERROR_CLASSIFICATIONS)
known_error_codes = st.sampled_from(KNOWN_ERROR_CODES)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------
class MockCursor:
    """Mock MySQL cursor that simulates error_classifications table lookups."""

    def __init__(self, error_code_to_return=None):
        self._result = error_code_to_return
        self._closed = False

    def execute(self, sql, params=None):
        if params and self._result:
            # Only return result if queried code matches
            queried_code = params[0]
            if queried_code == self._result["error_code"]:
                self._fetchone_result = self._result
            else:
                self._fetchone_result = None
        elif self._result:
            self._fetchone_result = self._result
        else:
            self._fetchone_result = None

    def fetchone(self):
        return self._fetchone_result

    def close(self):
        self._closed = True


class MockConnection:
    """Mock MySQL connection."""

    def __init__(self, cursor_instance=None):
        self._cursor = cursor_instance or MockCursor()

    def cursor(self, dictionary=False):
        return self._cursor

    def is_connected(self):
        return True

    def commit(self):
        pass

    def rollback(self):
        pass


def _build_mock_row(entry):
    """Build a mock DB row from an ERROR_CLASSIFICATIONS entry."""
    error_code, error_pattern, category, description, should_retry = entry
    return {
        "error_code": error_code,
        "category": category,
        "description": description,
        "should_retry": should_retry,
    }


# ---------------------------------------------------------------------------
# Property Tests
# ---------------------------------------------------------------------------
class TestErrorClassificationDeterminism:
    """Property-based tests for error code classification — Property 25."""

    @given(entry=known_error_entries)
    @settings(max_examples=200)
    def test_each_error_code_maps_to_exactly_one_valid_category(self, entry: tuple):
        """
        Property: For any error code in the error_classifications table,
        classify_error() returns exactly one category from the valid set
        {transient, permanent, suppression}.

        **Validates: Requirements 21.1**
        """
        error_code, error_pattern, expected_category, description, should_retry = entry

        mock_row = _build_mock_row(entry)
        cursor = MockCursor(mock_row)
        conn = MockConnection(cursor)
        service = RetryCategorizerService(lambda: conn)

        result = service.classify_error(error_code, "test error message")

        # Must return exactly one category
        assert result.category in VALID_CATEGORIES, (
            f"Error code {error_code} returned category '{result.category}' "
            f"which is not in valid set {VALID_CATEGORIES}"
        )

        # Must match the expected category from the lookup table
        assert result.category == expected_category, (
            f"Error code {error_code} expected category '{expected_category}', "
            f"got '{result.category}'"
        )

    @given(entry=known_error_entries, data=st.data())
    @settings(max_examples=200)
    def test_classification_is_deterministic(self, entry: tuple, data):
        """
        Property: Classifying the same error code multiple times always
        produces the same category — the mapping is deterministic.

        **Validates: Requirements 21.1**
        """
        error_code, error_pattern, expected_category, description, should_retry = entry

        # Generate a random number of classification calls (2–5)
        num_calls = data.draw(st.integers(min_value=2, max_value=5))

        mock_row = _build_mock_row(entry)
        results = []

        for _ in range(num_calls):
            cursor = MockCursor(mock_row)
            conn = MockConnection(cursor)
            service = RetryCategorizerService(lambda: conn)
            result = service.classify_error(error_code, "any error message")
            results.append(result.category)

        # All calls must produce the same category
        assert len(set(results)) == 1, (
            f"Error code {error_code} produced non-deterministic classifications: "
            f"{results}"
        )

        # And that category must be the expected one
        assert results[0] == expected_category, (
            f"Error code {error_code} consistently returned '{results[0]}' "
            f"but expected '{expected_category}'"
        )

    @given(entry=known_error_entries)
    @settings(max_examples=200)
    def test_should_retry_matches_category(self, entry: tuple):
        """
        Property: The should_retry flag returned by classify_error() is
        consistent with the lookup table data — each code has a
        deterministic retry decision.

        **Validates: Requirements 21.1**
        """
        error_code, error_pattern, expected_category, description, expected_retry = entry

        mock_row = _build_mock_row(entry)
        cursor = MockCursor(mock_row)
        conn = MockConnection(cursor)
        service = RetryCategorizerService(lambda: conn)

        result = service.classify_error(error_code, "test message")

        assert result.should_retry == bool(expected_retry), (
            f"Error code {error_code} (category={expected_category}): "
            f"expected should_retry={bool(expected_retry)}, got {result.should_retry}"
        )

    @given(entry=known_error_entries)
    @settings(max_examples=200)
    def test_error_code_in_result_matches_input(self, entry: tuple):
        """
        Property: The error_code in the returned FailureCategory always
        matches the input error_code — no cross-contamination.

        **Validates: Requirements 21.1**
        """
        error_code, error_pattern, expected_category, description, should_retry = entry

        mock_row = _build_mock_row(entry)
        cursor = MockCursor(mock_row)
        conn = MockConnection(cursor)
        service = RetryCategorizerService(lambda: conn)

        result = service.classify_error(error_code, "test message")

        assert result.error_code == error_code, (
            f"Input error_code={error_code} but result.error_code={result.error_code}"
        )
