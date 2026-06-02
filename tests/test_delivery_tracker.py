"""
Unit tests for the DeliveryTracker service.

Tests cover:
- Campaign message detection (5.1)
- Status update processing (5.2)
- Valid status transitions / regression prevention (5.2)
- Campaign aggregate count updates (5.3)
- Permanent failure handling via RetryCategorizerService (5.4)
- Timestamp parsing
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.delivery_tracker import DeliveryTracker


def _make_mock_connection():
    """Create a mock MySQL connection with cursor behavior."""
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value = cursor
    conn.is_connected.return_value = True
    return conn, cursor


class TestIsCampaignMessage:
    """Tests for is_campaign_message()."""

    def test_returns_true_when_message_exists(self):
        conn, cursor = _make_mock_connection()
        cursor.fetchone.return_value = {"id": 42}
        tracker = DeliveryTracker(get_connection=lambda: conn)

        assert tracker.is_campaign_message("wamid.abc123") is True
        cursor.execute.assert_called_once()

    def test_returns_false_when_message_not_found(self):
        conn, cursor = _make_mock_connection()
        cursor.fetchone.return_value = None
        tracker = DeliveryTracker(get_connection=lambda: conn)

        assert tracker.is_campaign_message("wamid.unknown") is False

    def test_returns_false_for_empty_message_id(self):
        conn, cursor = _make_mock_connection()
        tracker = DeliveryTracker(get_connection=lambda: conn)

        assert tracker.is_campaign_message("") is False
        assert tracker.is_campaign_message(None) is False

    def test_returns_false_on_db_error(self):
        conn, cursor = _make_mock_connection()
        cursor.execute.side_effect = Exception("DB error")
        tracker = DeliveryTracker(get_connection=lambda: conn)

        assert tracker.is_campaign_message("wamid.abc123") is False


class TestProcessStatusUpdate:
    """Tests for process_status_update()."""

    def test_updates_delivered_status(self):
        conn, cursor = _make_mock_connection()
        cursor.fetchone.return_value = {
            "id": 1,
            "campaign_id": 10,
            "current_status": "sent",
            "customer_mobile": "919876543210",
        }
        tracker = DeliveryTracker(get_connection=lambda: conn)

        with patch.object(tracker, '_update_campaign_counts'):
            result = tracker.process_status_update(
                whatsapp_message_id="wamid.abc123",
                status="delivered",
                timestamp="1700000000",
            )

        assert result is True

    def test_updates_read_status(self):
        conn, cursor = _make_mock_connection()
        cursor.fetchone.return_value = {
            "id": 2,
            "campaign_id": 10,
            "current_status": "delivered",
            "customer_mobile": "919876543210",
        }
        tracker = DeliveryTracker(get_connection=lambda: conn)

        with patch.object(tracker, '_update_campaign_counts'):
            result = tracker.process_status_update(
                whatsapp_message_id="wamid.abc456",
                status="read",
                timestamp="1700000100",
            )

        assert result is True

    def test_updates_failed_status_with_error_details(self):
        conn, cursor = _make_mock_connection()
        cursor.fetchone.return_value = {
            "id": 3,
            "campaign_id": 10,
            "current_status": "sent",
            "customer_mobile": "919876543210",
        }
        tracker = DeliveryTracker(get_connection=lambda: conn)

        with patch.object(tracker, '_update_campaign_counts'), \
             patch.object(tracker, '_handle_failure') as mock_handle:
            result = tracker.process_status_update(
                whatsapp_message_id="wamid.fail1",
                status="failed",
                timestamp="1700000200",
                error_code=131026,
                error_message="Invalid number",
            )

        assert result is True
        mock_handle.assert_called_once_with(3, 131026, "Invalid number")

    def test_returns_false_for_non_campaign_message(self):
        conn, cursor = _make_mock_connection()
        cursor.fetchone.return_value = None  # No campaign message found
        tracker = DeliveryTracker(get_connection=lambda: conn)

        result = tracker.process_status_update(
            whatsapp_message_id="wamid.notcampaign",
            status="delivered",
            timestamp="1700000000",
        )

        assert result is False

    def test_returns_false_for_empty_message_id(self):
        conn, cursor = _make_mock_connection()
        tracker = DeliveryTracker(get_connection=lambda: conn)

        assert tracker.process_status_update("", "delivered") is False
        assert tracker.process_status_update(None, "delivered") is False

    def test_returns_false_for_unsupported_status(self):
        conn, cursor = _make_mock_connection()
        tracker = DeliveryTracker(get_connection=lambda: conn)

        result = tracker.process_status_update(
            whatsapp_message_id="wamid.abc",
            status="unknown",
        )
        assert result is False

    def test_prevents_status_regression(self):
        conn, cursor = _make_mock_connection()
        # Message is already "read", trying to set to "delivered" should be rejected
        cursor.fetchone.return_value = {
            "id": 5,
            "campaign_id": 10,
            "current_status": "read",
            "customer_mobile": "919876543210",
        }
        tracker = DeliveryTracker(get_connection=lambda: conn)

        result = tracker.process_status_update(
            whatsapp_message_id="wamid.regression",
            status="delivered",
        )
        assert result is False

    def test_does_not_override_permanently_failed(self):
        conn, cursor = _make_mock_connection()
        cursor.fetchone.return_value = {
            "id": 6,
            "campaign_id": 10,
            "current_status": "permanently_failed",
            "customer_mobile": "919876543210",
        }
        tracker = DeliveryTracker(get_connection=lambda: conn)

        result = tracker.process_status_update(
            whatsapp_message_id="wamid.permfail",
            status="delivered",
        )
        assert result is False


class TestIsValidStatusTransition:
    """Tests for _is_valid_status_transition()."""

    def test_sent_to_delivered_is_valid(self):
        tracker = DeliveryTracker(get_connection=MagicMock())
        assert tracker._is_valid_status_transition("sent", "delivered") is True

    def test_delivered_to_read_is_valid(self):
        tracker = DeliveryTracker(get_connection=MagicMock())
        assert tracker._is_valid_status_transition("delivered", "read") is True

    def test_sent_to_failed_is_valid(self):
        tracker = DeliveryTracker(get_connection=MagicMock())
        assert tracker._is_valid_status_transition("sent", "failed") is True

    def test_queued_to_sent_is_valid(self):
        tracker = DeliveryTracker(get_connection=MagicMock())
        assert tracker._is_valid_status_transition("queued", "sent") is True

    def test_read_to_delivered_is_invalid(self):
        tracker = DeliveryTracker(get_connection=MagicMock())
        assert tracker._is_valid_status_transition("read", "delivered") is False

    def test_permanently_failed_to_anything_is_invalid(self):
        tracker = DeliveryTracker(get_connection=MagicMock())
        assert tracker._is_valid_status_transition("permanently_failed", "sent") is False
        assert tracker._is_valid_status_transition("permanently_failed", "delivered") is False

    def test_failed_from_delivered_is_valid(self):
        tracker = DeliveryTracker(get_connection=MagicMock())
        # Failed can come after delivered (e.g., recipient blocks after receiving)
        assert tracker._is_valid_status_transition("delivered", "failed") is True

    def test_sending_to_failed_is_valid(self):
        tracker = DeliveryTracker(get_connection=MagicMock())
        assert tracker._is_valid_status_transition("sending", "failed") is True


class TestUpdateCampaignCounts:
    """Tests for _update_campaign_counts()."""

    def test_updates_all_aggregate_counts(self):
        conn, cursor = _make_mock_connection()
        cursor.fetchone.return_value = {
            "sent_count": 5,
            "delivered_count": 3,
            "read_count": 1,
            "failed_count": 2,
        }
        tracker = DeliveryTracker(get_connection=lambda: conn)

        tracker._update_campaign_counts(10)

        # Verify UPDATE was called with correct values
        update_call = cursor.execute.call_args_list[-1]
        args = update_call[0]
        assert "UPDATE campaigns" in args[0]
        assert args[1] == (5, 3, 1, 2, 10)

    def test_handles_null_counts_as_zero(self):
        conn, cursor = _make_mock_connection()
        cursor.fetchone.return_value = {
            "sent_count": None,
            "delivered_count": None,
            "read_count": None,
            "failed_count": None,
        }
        tracker = DeliveryTracker(get_connection=lambda: conn)

        tracker._update_campaign_counts(10)

        update_call = cursor.execute.call_args_list[-1]
        args = update_call[0]
        assert args[1] == (0, 0, 0, 0, 10)


class TestParseTimestamp:
    """Tests for _parse_timestamp()."""

    def test_parses_valid_unix_timestamp(self):
        result = DeliveryTracker._parse_timestamp("1700000000")
        # Should produce a valid datetime string
        assert len(result) == 19  # 'YYYY-MM-DD HH:MM:SS'
        datetime.strptime(result, "%Y-%m-%d %H:%M:%S")  # Validates format

    def test_returns_current_time_for_none(self):
        result = DeliveryTracker._parse_timestamp(None)
        assert len(result) == 19
        datetime.strptime(result, "%Y-%m-%d %H:%M:%S")

    def test_returns_current_time_for_invalid_timestamp(self):
        result = DeliveryTracker._parse_timestamp("not-a-number")
        assert len(result) == 19
        datetime.strptime(result, "%Y-%m-%d %H:%M:%S")


class TestHandleFailure:
    """Tests for _handle_failure delegation to RetryCategorizerService."""

    def test_delegates_to_retry_categorizer(self):
        conn, cursor = _make_mock_connection()
        tracker = DeliveryTracker(get_connection=lambda: conn)

        with patch.object(tracker._retry_categorizer, 'process_failure') as mock_pf:
            tracker._handle_failure(message_id=42, error_code=131026, error_message="Invalid number")

        mock_pf.assert_called_once_with(42, 131026, "Invalid number")

    def test_handles_retry_categorizer_exception_gracefully(self):
        conn, cursor = _make_mock_connection()
        tracker = DeliveryTracker(get_connection=lambda: conn)

        with patch.object(tracker._retry_categorizer, 'process_failure', side_effect=Exception("DB error")):
            # Should not raise
            tracker._handle_failure(message_id=42, error_code=131047, error_message="Rate limit")
