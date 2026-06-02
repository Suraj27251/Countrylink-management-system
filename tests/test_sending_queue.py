"""
Unit tests for SendingQueue — enqueue, process_batch, pause/resume/cancel, progress.

Uses mock MySQL connections and a mock dispatcher to test business logic
without database or network dependencies.
"""

import json
import unittest
from unittest.mock import MagicMock, patch

from services.sending_queue import SendingQueue, BatchResult, QueueProgress
from services.channel import DispatchResult


class MockCursor:
    """Mock MySQL cursor with dictionary=True support."""

    def __init__(self):
        self.executed = []
        self.fetchone_result = None
        self.fetchall_result = []
        self.lastrowid = 1
        self.rowcount = 1
        self._closed = False

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchone(self):
        return self.fetchone_result

    def fetchall(self):
        return self.fetchall_result

    def close(self):
        self._closed = True


class MockConnection:
    """Mock MySQL connection."""

    def __init__(self, cursor_instance=None):
        self._cursor = cursor_instance or MockCursor()
        self._committed = False
        self._rolled_back = False
        self._closed = False

    def cursor(self, dictionary=False):
        return self._cursor

    def start_transaction(self):
        pass

    def commit(self):
        self._committed = True

    def rollback(self):
        self._rolled_back = True

    def close(self):
        self._closed = True


class TestEnqueueCampaign(unittest.TestCase):
    """Test enqueue_campaign creates messages with idempotency keys."""

    def test_enqueue_creates_messages_with_idempotency_key(self):
        """Each recipient gets a message with key = campaign_id_mobile_template_id."""
        call_count = [0]
        cursors = []

        def get_conn():
            call_count[0] += 1
            cursor = MockCursor()
            cursors.append(cursor)
            # First call fetches campaign details
            cursor.fetchone_result = {"template_id": 10, "segment_id": 1}
            # For template fetch - will return on second fetchone call
            if call_count[0] == 1:
                # We need to handle multiple fetchone calls on same cursor
                results = [
                    {"template_id": 10, "segment_id": 1},  # campaign
                    {"template_name": "test_tmpl", "body_text": "Hello {{1}}", "placeholder_mappings": None},  # template
                ]
                cursor._fetch_idx = 0
                original_fetchone = cursor.fetchone

                def multi_fetchone():
                    idx = cursor._fetch_idx
                    cursor._fetch_idx += 1
                    if idx < len(results):
                        return results[idx]
                    return None

                cursor.fetchone = multi_fetchone
            return MockConnection(cursor)

        queue = SendingQueue(
            get_connection=get_conn,
            dispatcher=MagicMock(),
            throttle_rate=80,
        )

        recipients = [
            {"mobile": "919876543210", "customer_name": "Rajesh"},
            {"mobile": "919876543211", "customer_name": "Priya"},
        ]

        count = queue.enqueue_campaign(campaign_id=5, recipients=recipients)

        # Verify INSERT IGNORE statements were executed
        insert_calls = [
            (sql, params) for sql, params in cursors[0].executed
            if "INSERT IGNORE" in (sql or "")
        ]
        self.assertEqual(len(insert_calls), 2)

        # Verify idempotency keys
        first_params = insert_calls[0][1]
        # idempotency_key is last param
        self.assertEqual(first_params[-1], "5_919876543210_10")

        second_params = insert_calls[1][1]
        self.assertEqual(second_params[-1], "5_919876543211_10")

    def test_enqueue_skips_empty_mobile(self):
        """Recipients with empty mobile are skipped."""
        cursor = MockCursor()
        results = [
            {"template_id": 10, "segment_id": 1},
            {"template_name": "tmpl", "body_text": "Hi", "placeholder_mappings": None},
        ]
        cursor._fetch_idx = 0
        def multi_fetchone():
            idx = cursor._fetch_idx
            cursor._fetch_idx += 1
            return results[idx] if idx < len(results) else None
        cursor.fetchone = multi_fetchone

        conn = MockConnection(cursor)
        queue = SendingQueue(get_connection=lambda: conn, dispatcher=MagicMock())

        recipients = [
            {"mobile": "", "customer_name": "Empty"},
            {"mobile": "  ", "customer_name": "Whitespace"},
        ]
        count = queue.enqueue_campaign(campaign_id=1, recipients=recipients)
        # No INSERT IGNORE should have been called (only campaign + template SELECTs)
        insert_calls = [s for s, p in cursor.executed if "INSERT IGNORE" in (s or "")]
        self.assertEqual(len(insert_calls), 0)


class TestProcessBatch(unittest.TestCase):
    """Test process_batch dispatches messages and tracks results."""

    def test_process_batch_sends_messages(self):
        """Messages are dispatched and marked as sent on success."""
        # Mock dispatcher that always succeeds
        mock_dispatcher = MagicMock()
        mock_dispatcher.send_template.return_value = DispatchResult(
            success=True, message_id="wamid_123"
        )

        call_count = [0]

        def get_conn():
            call_count[0] += 1
            cursor = MockCursor()
            conn = MockConnection(cursor)

            if call_count[0] == 1:
                # Main batch fetch returns one message
                cursor.fetchall_result = [
                    {
                        "id": 1, "campaign_id": 5, "customer_mobile": "919876543210",
                        "customer_name": "Rajesh", "template_id": 10,
                        "template_params": json.dumps({"1": "Rajesh"}),
                    }
                ]
            elif call_count[0] == 2:
                # Template name lookup
                cursor.fetchone_result = {"template_name": "welcome_msg"}
            # Further calls are for marking sent / updating progress
            return conn

        queue = SendingQueue(
            get_connection=get_conn,
            dispatcher=mock_dispatcher,
            throttle_rate=1000,  # High rate to avoid sleep in test
        )

        result = queue.process_batch(batch_size=10)
        self.assertEqual(result.sent_count, 1)
        self.assertEqual(result.failed_count, 0)

    def test_process_batch_handles_dispatch_failure(self):
        """Failed dispatches are tracked in result."""
        mock_dispatcher = MagicMock()
        mock_dispatcher.send_template.return_value = DispatchResult(
            success=False, error_code=131047, error_message="Rate limited"
        )

        call_count = [0]

        def get_conn():
            call_count[0] += 1
            cursor = MockCursor()
            conn = MockConnection(cursor)

            if call_count[0] == 1:
                cursor.fetchall_result = [
                    {
                        "id": 2, "campaign_id": 5, "customer_mobile": "919000000001",
                        "customer_name": "Test", "template_id": 10,
                        "template_params": json.dumps({"1": "Test"}),
                    }
                ]
            elif call_count[0] == 2:
                cursor.fetchone_result = {"template_name": "renewal_reminder"}
            return conn

        queue = SendingQueue(
            get_connection=get_conn,
            dispatcher=mock_dispatcher,
            throttle_rate=1000,
        )

        result = queue.process_batch(batch_size=10)
        self.assertEqual(result.sent_count, 0)
        self.assertEqual(result.failed_count, 1)
        self.assertEqual(result.errors[0]["error_code"], 131047)

    def test_process_batch_skips_invalid_params(self):
        """Messages with invalid template params are marked as failed."""
        mock_dispatcher = MagicMock()
        call_count = [0]

        def get_conn():
            call_count[0] += 1
            cursor = MockCursor()
            conn = MockConnection(cursor)

            if call_count[0] == 1:
                # Message with empty param value
                cursor.fetchall_result = [
                    {
                        "id": 3, "campaign_id": 5, "customer_mobile": "919000000002",
                        "customer_name": "Bad", "template_id": 10,
                        "template_params": json.dumps({"1": ""}),
                    }
                ]
            return conn

        queue = SendingQueue(
            get_connection=get_conn,
            dispatcher=mock_dispatcher,
            throttle_rate=1000,
        )

        result = queue.process_batch(batch_size=10)
        self.assertEqual(result.failed_count, 1)
        # Dispatcher should NOT have been called
        mock_dispatcher.send_template.assert_not_called()


class TestCampaignControl(unittest.TestCase):
    """Test pause, resume, and cancel campaign operations."""

    def test_pause_campaign(self):
        """Pausing sets campaign status and adds to exclusion set."""
        cursor = MockCursor()
        conn = MockConnection(cursor)
        queue = SendingQueue(get_connection=lambda: conn, dispatcher=MagicMock())

        queue.pause_campaign(campaign_id=7)

        # Verify SQL update
        update_calls = [s for s, p in cursor.executed if "UPDATE campaigns" in (s or "")]
        self.assertTrue(any("paused" in s for s in update_calls))

        # Verify internal state
        self.assertIn(7, queue._paused_campaigns)

    def test_resume_campaign(self):
        """Resuming removes from paused set and updates status to sending."""
        cursor = MockCursor()
        conn = MockConnection(cursor)
        queue = SendingQueue(get_connection=lambda: conn, dispatcher=MagicMock())

        queue._paused_campaigns.add(7)
        queue.resume_campaign(campaign_id=7)

        # Verify internal state
        self.assertNotIn(7, queue._paused_campaigns)

        # Verify SQL update
        update_calls = [s for s, p in cursor.executed if "UPDATE campaigns" in (s or "")]
        self.assertTrue(any("sending" in s for s in update_calls))

    def test_cancel_campaign(self):
        """Cancelling marks queued messages as skipped and updates campaign status."""
        cursor = MockCursor()
        conn = MockConnection(cursor)
        queue = SendingQueue(get_connection=lambda: conn, dispatcher=MagicMock())

        queue.cancel_campaign(campaign_id=9)

        # Verify campaign_messages update
        msg_update_calls = [
            s for s, p in cursor.executed
            if "UPDATE campaign_messages" in (s or "")
        ]
        self.assertTrue(any("skipped" in s for s in msg_update_calls))

        # Verify campaign status update
        campaign_update_calls = [
            s for s, p in cursor.executed
            if "UPDATE campaigns" in (s or "")
        ]
        self.assertTrue(any("cancelled" in s for s in campaign_update_calls))

        # Verify internal state
        self.assertIn(9, queue._cancelled_campaigns)

    def test_paused_campaign_excluded_from_batch(self):
        """Paused campaign messages are not fetched in process_batch."""
        cursor = MockCursor()
        cursor.fetchall_result = []  # No messages returned
        conn = MockConnection(cursor)
        queue = SendingQueue(get_connection=lambda: conn, dispatcher=MagicMock())

        queue._paused_campaigns.add(5)
        result = queue.process_batch(batch_size=10)

        # The SQL should exclude campaign 5
        sql_calls = [s for s, p in cursor.executed if s and "NOT IN" in s]
        self.assertTrue(len(sql_calls) > 0)


class TestGetProgress(unittest.TestCase):
    """Test real-time progress tracking."""

    def test_get_progress_returns_correct_counts(self):
        """Progress returns aggregated counts from campaign_messages."""
        call_count = [0]

        def get_conn():
            call_count[0] += 1
            cursor = MockCursor()
            conn = MockConnection(cursor)

            if call_count[0] == 1:
                # Mock aggregate query result
                results = [
                    {"total": 100, "sent_count": 60, "failed_count": 5, "remaining_count": 35},
                    {"status": "sending"},
                ]
                cursor._fetch_idx = 0
                def multi_fetchone():
                    idx = cursor._fetch_idx
                    cursor._fetch_idx += 1
                    return results[idx] if idx < len(results) else None
                cursor.fetchone = multi_fetchone

            return conn

        queue = SendingQueue(get_connection=get_conn, dispatcher=MagicMock())
        progress = queue.get_progress(campaign_id=5)

        self.assertEqual(progress.campaign_id, 5)
        self.assertEqual(progress.total, 100)
        self.assertEqual(progress.sent_count, 60)
        self.assertEqual(progress.failed_count, 5)
        self.assertEqual(progress.remaining_count, 35)
        self.assertEqual(progress.status, "sending")


class TestIdempotencyKey(unittest.TestCase):
    """Test that idempotency key format is correct."""

    def test_idempotency_key_format(self):
        """Key should be campaign_id_mobile_template_id."""
        cursor = MockCursor()
        results = [
            {"template_id": 42, "segment_id": 1},
            {"template_name": "tmpl", "body_text": "Hi {{1}}", "placeholder_mappings": None},
        ]
        cursor._fetch_idx = 0
        def multi_fetchone():
            idx = cursor._fetch_idx
            cursor._fetch_idx += 1
            return results[idx] if idx < len(results) else None
        cursor.fetchone = multi_fetchone

        conn = MockConnection(cursor)
        queue = SendingQueue(get_connection=lambda: conn, dispatcher=MagicMock())

        recipients = [{"mobile": "918055782345", "customer_name": "Test"}]
        queue.enqueue_campaign(campaign_id=99, recipients=recipients)

        insert_calls = [
            (sql, params) for sql, params in cursor.executed
            if "INSERT IGNORE" in (sql or "")
        ]
        self.assertEqual(len(insert_calls), 1)
        # idempotency_key is the last parameter
        self.assertEqual(insert_calls[0][1][-1], "99_918055782345_42")


class TestValidateMessageParams(unittest.TestCase):
    """Test template param validation before dispatch."""

    def test_valid_params_pass(self):
        """Non-empty, non-null params under 1024 chars pass."""
        queue = SendingQueue(get_connection=MagicMock(), dispatcher=MagicMock())
        msg = {"template_params": json.dumps({"1": "Rajesh", "2": "Eclipse 100"})}
        self.assertTrue(queue._validate_message_params(msg))

    def test_null_param_fails(self):
        """Null param value fails validation."""
        queue = SendingQueue(get_connection=MagicMock(), dispatcher=MagicMock())
        msg = {"template_params": json.dumps({"1": None})}
        self.assertFalse(queue._validate_message_params(msg))

    def test_empty_param_fails(self):
        """Empty string param value fails validation."""
        queue = SendingQueue(get_connection=MagicMock(), dispatcher=MagicMock())
        msg = {"template_params": json.dumps({"1": ""})}
        self.assertFalse(queue._validate_message_params(msg))

    def test_oversized_param_fails(self):
        """Param exceeding 1024 chars fails validation."""
        queue = SendingQueue(get_connection=MagicMock(), dispatcher=MagicMock())
        long_value = "x" * 1025
        msg = {"template_params": json.dumps({"1": long_value})}
        self.assertFalse(queue._validate_message_params(msg))

    def test_no_params_passes(self):
        """Message with no template_params is valid (no params needed)."""
        queue = SendingQueue(get_connection=MagicMock(), dispatcher=MagicMock())
        msg = {"template_params": None}
        self.assertTrue(queue._validate_message_params(msg))


if __name__ == "__main__":
    unittest.main()
