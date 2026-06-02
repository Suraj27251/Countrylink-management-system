"""
Unit tests for the RetryCategorizerService.

Tests cover:
- Error classification lookup (21.1, 21.5)
- Retry decision logic (21.2)
- Exponential backoff computation (21.2)
- Permanent failure handling (21.3)
- Suppression handling (21.4)
- Campaign pause on suppression threshold (21.7)
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.retry_categorizer import (
    RetryCategorizerService,
    FailureCategory,
    FailureCategoryEnum,
    RetryDecision,
)


def _make_mock_connection():
    """Create a mock MySQL connection with cursor behavior."""
    conn = MagicMock()
    cursor = MagicMock()
    cursor_dict = MagicMock()
    conn.cursor.return_value = cursor_dict
    conn.is_connected.return_value = True
    return conn, cursor_dict


class TestComputeBackoff:
    """Tests for exponential backoff computation."""

    def test_attempt_1_returns_5_seconds(self):
        assert RetryCategorizerService.compute_backoff(1) == 5

    def test_attempt_2_returns_15_seconds(self):
        assert RetryCategorizerService.compute_backoff(2) == 15

    def test_attempt_3_returns_45_seconds(self):
        assert RetryCategorizerService.compute_backoff(3) == 45

    def test_formula_is_5_times_3_power_n_minus_1(self):
        """Verify formula: 5 * 3^(N-1)"""
        for n in range(1, 6):
            expected = 5 * (3 ** (n - 1))
            assert RetryCategorizerService.compute_backoff(n) == expected


class TestClassifyError:
    """Tests for classify_error()."""

    def test_known_error_code_returns_classification(self):
        conn, cursor = _make_mock_connection()
        cursor.fetchone.return_value = {
            "error_code": 131047,
            "category": "transient",
            "description": "Rate limit hit",
            "should_retry": 1,
        }

        service = RetryCategorizerService(get_connection=lambda: conn)
        result = service.classify_error(131047, "Rate limit hit")

        assert result.category == "transient"
        assert result.error_code == 131047
        assert result.description == "Rate limit hit"
        assert result.should_retry is True

    def test_permanent_error_code(self):
        conn, cursor = _make_mock_connection()
        cursor.fetchone.return_value = {
            "error_code": 131026,
            "category": "permanent",
            "description": "Invalid phone number",
            "should_retry": 0,
        }

        service = RetryCategorizerService(get_connection=lambda: conn)
        result = service.classify_error(131026, "Invalid phone number")

        assert result.category == "permanent"
        assert result.should_retry is False

    def test_suppression_error_code(self):
        conn, cursor = _make_mock_connection()
        cursor.fetchone.return_value = {
            "error_code": 131056,
            "category": "suppression",
            "description": "Blocked by user",
            "should_retry": 0,
        }

        service = RetryCategorizerService(get_connection=lambda: conn)
        result = service.classify_error(131056, "Blocked by user")

        assert result.category == "suppression"
        assert result.should_retry is False

    def test_unknown_error_code_defaults_to_transient(self):
        conn, cursor = _make_mock_connection()
        cursor.fetchone.return_value = None  # Not found

        service = RetryCategorizerService(get_connection=lambda: conn)
        result = service.classify_error(99999, "Unknown error")

        assert result.category == "transient"
        assert result.error_code == 99999
        assert result.should_retry is True

    def test_cursor_is_closed_after_query(self):
        conn, cursor = _make_mock_connection()
        cursor.fetchone.return_value = None

        service = RetryCategorizerService(get_connection=lambda: conn)
        service.classify_error(131000, "test")

        cursor.close.assert_called_once()


class TestShouldRetry:
    """Tests for should_retry()."""

    def test_transient_error_within_max_retries(self):
        conn, cursor = _make_mock_connection()
        cursor.fetchone.return_value = {
            "id": 1,
            "campaign_id": 10,
            "retry_count": 1,
            "max_retries": 3,
            "error_code": 131047,
            "error_category": "transient",
            "status": "failed",
        }

        service = RetryCategorizerService(get_connection=lambda: conn)
        result = service.should_retry(1)

        assert result.should_retry is True
        assert result.retry_count == 1
        assert result.max_retries == 3
        assert result.backoff_seconds == 15  # attempt 2: 5 * 3^(2-1) = 15

    def test_retry_count_zero_first_attempt(self):
        conn, cursor = _make_mock_connection()
        cursor.fetchone.return_value = {
            "id": 1,
            "campaign_id": 10,
            "retry_count": 0,
            "max_retries": 3,
            "error_code": 131047,
            "error_category": "transient",
            "status": "failed",
        }

        service = RetryCategorizerService(get_connection=lambda: conn)
        result = service.should_retry(1)

        assert result.should_retry is True
        assert result.backoff_seconds == 5  # attempt 1: 5 * 3^(1-1) = 5

    def test_max_retries_exceeded(self):
        conn, cursor = _make_mock_connection()
        cursor.fetchone.return_value = {
            "id": 1,
            "campaign_id": 10,
            "retry_count": 3,
            "max_retries": 3,
            "error_code": 131047,
            "error_category": "transient",
            "status": "failed",
        }

        service = RetryCategorizerService(get_connection=lambda: conn)
        result = service.should_retry(1)

        assert result.should_retry is False
        assert "Max retries exceeded" in result.reason

    def test_permanent_error_not_retryable(self):
        conn, cursor = _make_mock_connection()
        cursor.fetchone.return_value = {
            "id": 1,
            "campaign_id": 10,
            "retry_count": 0,
            "max_retries": 3,
            "error_code": 131026,
            "error_category": "permanent",
            "status": "failed",
        }

        service = RetryCategorizerService(get_connection=lambda: conn)
        result = service.should_retry(1)

        assert result.should_retry is False
        assert "permanent" in result.reason

    def test_suppression_error_not_retryable(self):
        conn, cursor = _make_mock_connection()
        cursor.fetchone.return_value = {
            "id": 1,
            "campaign_id": 10,
            "retry_count": 0,
            "max_retries": 3,
            "error_code": 131056,
            "error_category": "suppression",
            "status": "failed",
        }

        service = RetryCategorizerService(get_connection=lambda: conn)
        result = service.should_retry(1)

        assert result.should_retry is False
        assert "suppression" in result.reason

    def test_message_not_found(self):
        conn, cursor = _make_mock_connection()
        cursor.fetchone.return_value = None

        service = RetryCategorizerService(get_connection=lambda: conn)
        result = service.should_retry(999)

        assert result.should_retry is False
        assert "not found" in result.reason

    def test_next_retry_at_is_in_future(self):
        conn, cursor = _make_mock_connection()
        cursor.fetchone.return_value = {
            "id": 1,
            "campaign_id": 10,
            "retry_count": 0,
            "max_retries": 3,
            "error_code": 131047,
            "error_category": "transient",
            "status": "failed",
        }

        service = RetryCategorizerService(get_connection=lambda: conn)
        before = datetime.now(timezone.utc)
        result = service.should_retry(1)
        after = datetime.now(timezone.utc)

        assert result.next_retry_at is not None
        # next_retry_at should be ~5s in future
        assert result.next_retry_at >= before + timedelta(seconds=5)
        assert result.next_retry_at <= after + timedelta(seconds=5)


class TestHandlePermanentFailure:
    """Tests for handle_permanent_failure()."""

    def test_marks_message_permanently_failed(self):
        conn, cursor = _make_mock_connection()
        cursor.fetchone.return_value = {
            "id": 1,
            "campaign_id": 10,
            "customer_mobile": "919876543210",
            "error_code": 131026,
            "error_message": "Invalid number",
        }

        service = RetryCategorizerService(get_connection=lambda: conn)
        service.handle_permanent_failure(1)

        # Should have executed: SELECT, UPDATE (message), INSERT (tag), INSERT (activity)
        assert cursor.execute.call_count == 4
        conn.commit.assert_called_once()

    def test_flags_customer_with_tag(self):
        conn, cursor = _make_mock_connection()
        cursor.fetchone.return_value = {
            "id": 1,
            "campaign_id": 10,
            "customer_mobile": "919876543210",
            "error_code": 131026,
            "error_message": "Invalid number",
        }

        service = RetryCategorizerService(get_connection=lambda: conn)
        service.handle_permanent_failure(1)

        # Check that one of the execute calls includes customer_tags INSERT
        calls = cursor.execute.call_args_list
        tag_insert_found = False
        for c in calls:
            sql = c[0][0]
            if "customer_tags" in sql and "invalid_whatsapp_number" in sql:
                tag_insert_found = True
                break
        assert tag_insert_found, "Should insert 'invalid_whatsapp_number' tag"

    def test_message_not_found_no_error(self):
        conn, cursor = _make_mock_connection()
        cursor.fetchone.return_value = None

        service = RetryCategorizerService(get_connection=lambda: conn)
        # Should not raise
        service.handle_permanent_failure(999)

    def test_rollback_on_error(self):
        conn, cursor = _make_mock_connection()
        cursor.fetchone.return_value = {
            "id": 1,
            "campaign_id": 10,
            "customer_mobile": "919876543210",
            "error_code": 131026,
            "error_message": "Invalid number",
        }
        # Fail on the UPDATE
        cursor.execute.side_effect = [None, Exception("DB error")]

        service = RetryCategorizerService(get_connection=lambda: conn)
        try:
            service.handle_permanent_failure(1)
        except Exception:
            pass

        conn.rollback.assert_called_once()


class TestHandleSuppression:
    """Tests for handle_suppression()."""

    def test_adds_customer_to_suppression_list(self):
        conn, cursor = _make_mock_connection()
        cursor.fetchone.side_effect = [
            # First fetchone: message lookup
            {
                "id": 1,
                "campaign_id": 10,
                "customer_mobile": "919876543210",
                "error_code": 131056,
                "error_message": "Blocked by user",
            },
            # Second fetchone: suppression rate check
            {
                "total_messages": 100,
                "suppression_count": 5,
            },
        ]

        service = RetryCategorizerService(get_connection=lambda: conn)
        service.handle_suppression(1)

        # Should have executed: SELECT, UPDATE, INSERT (suppression), INSERT (activity)
        # Then _check_suppression_rate: SELECT
        calls = cursor.execute.call_args_list
        suppression_insert_found = False
        for c in calls:
            sql = c[0][0]
            if "suppression_list" in sql and "INSERT" in sql:
                suppression_insert_found = True
                break
        assert suppression_insert_found

    def test_message_not_found_no_error(self):
        conn, cursor = _make_mock_connection()
        cursor.fetchone.return_value = None

        service = RetryCategorizerService(get_connection=lambda: conn)
        service.handle_suppression(999)


class TestCheckSuppressionRate:
    """Tests for _check_suppression_rate() — auto campaign pause at > 20%."""

    def test_no_pause_below_threshold(self):
        conn, cursor = _make_mock_connection()
        cursor.fetchone.return_value = {
            "total_messages": 100,
            "suppression_count": 15,  # 15% — below 20%
        }

        service = RetryCategorizerService(get_connection=lambda: conn)
        service._check_suppression_rate(10)

        # Should only SELECT, no UPDATE or INSERT for pause
        assert cursor.execute.call_count == 1

    def test_pause_above_threshold(self):
        conn, cursor = _make_mock_connection()
        cursor.fetchone.return_value = {
            "total_messages": 100,
            "suppression_count": 25,  # 25% — above 20%
        }
        cursor.rowcount = 1  # Campaign was in 'sending' state

        service = RetryCategorizerService(get_connection=lambda: conn)
        service._check_suppression_rate(10)

        # Should: SELECT, UPDATE (pause campaign), INSERT (notification)
        assert cursor.execute.call_count == 3
        conn.commit.assert_called_once()

    def test_no_pause_when_campaign_not_sending(self):
        conn, cursor = _make_mock_connection()
        cursor.fetchone.return_value = {
            "total_messages": 100,
            "suppression_count": 25,  # 25% — above 20%
        }
        cursor.rowcount = 0  # Campaign NOT in 'sending' state

        service = RetryCategorizerService(get_connection=lambda: conn)
        service._check_suppression_rate(10)

        # Should: SELECT, UPDATE (no-op), but NOT insert notification
        assert cursor.execute.call_count == 2

    def test_zero_messages_no_action(self):
        conn, cursor = _make_mock_connection()
        cursor.fetchone.return_value = {
            "total_messages": 0,
            "suppression_count": 0,
        }

        service = RetryCategorizerService(get_connection=lambda: conn)
        service._check_suppression_rate(10)

        # Only the SELECT query
        assert cursor.execute.call_count == 1

    def test_exact_20_percent_no_pause(self):
        """20% exactly should NOT trigger pause (threshold is > 20%)."""
        conn, cursor = _make_mock_connection()
        cursor.fetchone.return_value = {
            "total_messages": 100,
            "suppression_count": 20,  # exactly 20%
        }

        service = RetryCategorizerService(get_connection=lambda: conn)
        service._check_suppression_rate(10)

        # Should only SELECT, no pause
        assert cursor.execute.call_count == 1


class TestScheduleRetry:
    """Tests for schedule_retry()."""

    def test_schedules_retry_for_eligible_message(self):
        conn, cursor = _make_mock_connection()
        cursor.fetchone.return_value = {
            "id": 1,
            "campaign_id": 10,
            "retry_count": 0,
            "max_retries": 3,
            "error_code": 131047,
            "error_category": "transient",
            "status": "failed",
        }

        service = RetryCategorizerService(get_connection=lambda: conn)
        result = service.schedule_retry(1)

        assert result.should_retry is True
        # Should have: SELECT (from should_retry) + UPDATE (schedule)
        conn.commit.assert_called()

    def test_does_not_schedule_for_ineligible_message(self):
        conn, cursor = _make_mock_connection()
        cursor.fetchone.return_value = {
            "id": 1,
            "campaign_id": 10,
            "retry_count": 3,
            "max_retries": 3,
            "error_code": 131047,
            "error_category": "transient",
            "status": "failed",
        }

        service = RetryCategorizerService(get_connection=lambda: conn)
        result = service.schedule_retry(1)

        assert result.should_retry is False


class TestProcessFailure:
    """Tests for process_failure() end-to-end routing."""

    def test_transient_error_triggers_retry(self):
        conn, cursor = _make_mock_connection()
        # classify_error lookup
        cursor.fetchone.side_effect = [
            # classify_error: SELECT
            {
                "error_code": 131047,
                "category": "transient",
                "description": "Rate limit",
                "should_retry": 1,
            },
            # should_retry: SELECT
            {
                "id": 1,
                "campaign_id": 10,
                "retry_count": 0,
                "max_retries": 3,
                "error_code": 131047,
                "error_category": "transient",
                "status": "failed",
            },
        ]

        service = RetryCategorizerService(get_connection=lambda: conn)
        service.process_failure(1, 131047, "Rate limit hit")

        # verify commit was called (for updating error details + scheduling retry)
        assert conn.commit.call_count >= 2

    def test_permanent_error_triggers_permanent_handling(self):
        conn, cursor = _make_mock_connection()
        cursor.fetchone.side_effect = [
            # classify_error: SELECT
            {
                "error_code": 131026,
                "category": "permanent",
                "description": "Invalid number",
                "should_retry": 0,
            },
            # handle_permanent_failure: SELECT
            {
                "id": 1,
                "campaign_id": 10,
                "customer_mobile": "919876543210",
                "error_code": 131026,
                "error_message": "Invalid number",
            },
        ]

        service = RetryCategorizerService(get_connection=lambda: conn)
        service.process_failure(1, 131026, "Invalid phone number")

        # Check that customer_tags insert was called
        calls = cursor.execute.call_args_list
        tag_found = any("customer_tags" in str(c) for c in calls)
        assert tag_found

    def test_suppression_error_triggers_suppression_handling(self):
        conn, cursor = _make_mock_connection()
        cursor.fetchone.side_effect = [
            # classify_error: SELECT
            {
                "error_code": 131056,
                "category": "suppression",
                "description": "Blocked by user",
                "should_retry": 0,
            },
            # handle_suppression: SELECT message
            {
                "id": 1,
                "campaign_id": 10,
                "customer_mobile": "919876543210",
                "error_code": 131056,
                "error_message": "Blocked by user",
            },
            # _check_suppression_rate: SELECT counts
            {
                "total_messages": 100,
                "suppression_count": 5,
            },
        ]

        service = RetryCategorizerService(get_connection=lambda: conn)
        service.process_failure(1, 131056, "Blocked by user")

        # Check suppression_list insert
        calls = cursor.execute.call_args_list
        suppression_found = any("suppression_list" in str(c) for c in calls)
        assert suppression_found
