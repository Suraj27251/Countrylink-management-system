"""
Tests for the error_classifications seed function.

Validates that seed_error_classifications() correctly inserts all known
WhatsApp Business API error codes and is idempotent (safe to run repeatedly).
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from migrations.seed_error_classifications import (
    ERROR_CLASSIFICATIONS,
    seed_error_classifications,
)


def _make_mock_connection(rowcount_sequence=None):
    """Create a mock MySQL connection with cursor behavior."""
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value = cursor
    conn.is_connected.return_value = True

    if rowcount_sequence is not None:
        cursor.rowcount = rowcount_sequence.pop(0) if rowcount_sequence else 1
        type(cursor).rowcount = property(
            lambda self, seq=rowcount_sequence: seq.pop(0) if seq else 1
        )
    else:
        # Default: all inserts succeed (rowcount=1)
        cursor.rowcount = 1

    return conn, cursor


class TestErrorClassificationsData:
    """Tests for the ERROR_CLASSIFICATIONS data structure."""

    def test_all_entries_have_five_fields(self):
        for entry in ERROR_CLASSIFICATIONS:
            assert len(entry) == 5, f"Entry has wrong number of fields: {entry}"

    def test_all_error_codes_are_integers(self):
        for error_code, _, _, _, _ in ERROR_CLASSIFICATIONS:
            assert isinstance(error_code, int), f"Expected int, got {type(error_code)} for {error_code}"

    def test_all_categories_are_valid(self):
        valid_categories = {'transient', 'permanent', 'suppression'}
        for _, _, category, _, _ in ERROR_CLASSIFICATIONS:
            assert category in valid_categories, f"Invalid category: {category}"

    def test_should_retry_is_0_or_1(self):
        for error_code, _, _, _, should_retry in ERROR_CLASSIFICATIONS:
            assert should_retry in (0, 1), f"Invalid should_retry={should_retry} for code {error_code}"

    def test_error_codes_are_unique(self):
        codes = [code for code, _, _, _, _ in ERROR_CLASSIFICATIONS]
        assert len(codes) == len(set(codes)), "Duplicate error codes found"

    def test_contains_required_error_codes(self):
        """Verify all error codes specified in the task are present."""
        required_codes = {131047, 131026, 131056, 131053, 131031, 131021,
                         131045, 131049, 131051, 130472, 368, 131000}
        actual_codes = {code for code, _, _, _, _ in ERROR_CLASSIFICATIONS}
        missing = required_codes - actual_codes
        assert not missing, f"Missing required error codes: {missing}"

    def test_minimum_count(self):
        """At least 12 error codes as specified in the task."""
        assert len(ERROR_CLASSIFICATIONS) >= 12

    def test_specific_classifications(self):
        """Verify specific codes map to expected categories and retry flags."""
        lookup = {code: (cat, retry) for code, _, cat, _, retry in ERROR_CLASSIFICATIONS}

        # Transient errors that should retry
        assert lookup[131047] == ('transient', 1), "131047 should be transient with retry"
        assert lookup[131053] == ('transient', 1), "131053 should be transient with retry"
        assert lookup[131045] == ('transient', 1), "131045 should be transient with retry"
        assert lookup[130472] == ('transient', 1), "130472 should be transient with retry"
        assert lookup[131000] == ('transient', 1), "131000 should be transient with retry"

        # Transient but no retry (expired message)
        assert lookup[131049] == ('transient', 0), "131049 should be transient without retry"

        # Permanent errors
        assert lookup[131026] == ('permanent', 0), "131026 should be permanent"
        assert lookup[131031] == ('permanent', 0), "131031 should be permanent"
        assert lookup[131021] == ('permanent', 0), "131021 should be permanent"
        assert lookup[131051] == ('permanent', 0), "131051 should be permanent"

        # Suppression errors
        assert lookup[131056] == ('suppression', 0), "131056 should be suppression"
        assert lookup[368] == ('suppression', 0), "368 should be suppression"


class TestSeedFunction:
    """Tests for the seed_error_classifications() function."""

    def test_seed_inserts_all_classifications(self):
        """Seed function should call INSERT IGNORE for each classification."""
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value = cursor
        conn.is_connected.return_value = True
        cursor.rowcount = 1

        result = seed_error_classifications(conn)

        assert cursor.execute.call_count == len(ERROR_CLASSIFICATIONS)
        conn.commit.assert_called_once()
        assert result == len(ERROR_CLASSIFICATIONS)

    def test_seed_uses_insert_ignore(self):
        """The SQL should use INSERT IGNORE for idempotency."""
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value = cursor
        conn.is_connected.return_value = True
        cursor.rowcount = 1

        seed_error_classifications(conn)

        # Check that the SQL contains INSERT IGNORE
        first_call_args = cursor.execute.call_args_list[0]
        sql = first_call_args[0][0]
        assert 'INSERT IGNORE' in sql

    def test_seed_passes_correct_parameters(self):
        """Verify the correct values are passed for each error code."""
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value = cursor
        conn.is_connected.return_value = True
        cursor.rowcount = 1

        seed_error_classifications(conn)

        # Check first insert call passes first error classification tuple
        first_call_args = cursor.execute.call_args_list[0]
        params = first_call_args[0][1]
        expected_first = ERROR_CLASSIFICATIONS[0]
        assert params == expected_first

    def test_seed_is_idempotent_when_rows_already_exist(self):
        """When INSERT IGNORE finds duplicates, rowcount=0, function still succeeds."""
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value = cursor
        conn.is_connected.return_value = True
        # All rows already exist (rowcount=0 for INSERT IGNORE duplicates)
        cursor.rowcount = 0

        result = seed_error_classifications(conn)

        # Should return 0 new rows inserted
        assert result == 0
        conn.commit.assert_called_once()

    def test_seed_rolls_back_on_error(self):
        """If an error occurs, the transaction should be rolled back."""
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value = cursor
        conn.is_connected.return_value = True
        cursor.execute.side_effect = Exception("DB error")

        try:
            seed_error_classifications(conn)
        except Exception:
            pass

        conn.rollback.assert_called_once()

    def test_seed_closes_cursor_on_success(self):
        """Cursor should always be closed after execution."""
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value = cursor
        conn.is_connected.return_value = True
        cursor.rowcount = 1

        seed_error_classifications(conn)

        cursor.close.assert_called_once()

    def test_seed_closes_cursor_on_error(self):
        """Cursor should be closed even if an error occurs."""
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value = cursor
        conn.is_connected.return_value = True
        cursor.execute.side_effect = Exception("DB error")

        try:
            seed_error_classifications(conn)
        except Exception:
            pass

        cursor.close.assert_called_once()
