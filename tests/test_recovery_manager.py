"""
Unit tests for RecoveryManager — startup recovery, stale message detection,
idempotency-based deduplication, and campaign resumption.

Requirements: 26.1, 26.2, 26.3, 26.4, 26.5, 26.6, 26.7
"""

import unittest
from unittest.mock import MagicMock, patch

from services.recovery_manager import RecoveryManager, RecoveryReport


class MockCursor:
    """Mock MySQL cursor with dictionary=True support."""

    def __init__(self):
        self.executed = []
        self.fetchall_results = []
        self.fetchone_results = []
        self._fetchall_idx = 0
        self._fetchone_idx = 0
        self.rowcount = 0

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchall(self):
        if self._fetchall_idx < len(self.fetchall_results):
            result = self.fetchall_results[self._fetchall_idx]
            self._fetchall_idx += 1
            return result
        return []

    def fetchone(self):
        if self._fetchone_idx < len(self.fetchone_results):
            result = self.fetchone_results[self._fetchone_idx]
            self._fetchone_idx += 1
            return result
        return None

    def close(self):
        pass


class MockConnection:
    """Mock MySQL connection."""

    def __init__(self, cursor_instance=None):
        self._cursor = cursor_instance or MockCursor()
        self._committed = False
        self._rolled_back = False

    def cursor(self, dictionary=False):
        return self._cursor

    def commit(self):
        self._committed = True

    def rollback(self):
        self._rolled_back = True

    def close(self):
        pass


class TestRecoveryManagerInit(unittest.TestCase):
    """Tests for RecoveryManager initialization."""

    def test_default_stale_minutes(self):
        """Default stale_minutes should be 5."""
        mgr = RecoveryManager(lambda: MockConnection())
        self.assertEqual(mgr._stale_minutes, 5)

    def test_custom_stale_minutes(self):
        """stale_minutes can be customized."""
        mgr = RecoveryManager(lambda: MockConnection(), stale_minutes=10)
        self.assertEqual(mgr._stale_minutes, 10)


class TestIdentifyStaleMessages(unittest.TestCase):
    """Tests for identify_stale_messages method."""

    def test_returns_stale_message_ids(self):
        """Should return IDs of messages in 'sending' with old updated_at."""
        cursor = MockCursor()
        cursor.fetchall_results = [
            [{"id": 101}, {"id": 102}, {"id": 103}]
        ]
        conn = MockConnection(cursor)
        mgr = RecoveryManager(lambda: conn)

        result = mgr.identify_stale_messages(5)

        self.assertEqual(result, [101, 102, 103])
        # Verify the SQL queries 'sending' status and uses the interval
        sql_executed = cursor.executed[0][0]
        self.assertIn("sending", sql_executed)
        self.assertIn("INTERVAL", sql_executed)

    def test_returns_empty_when_no_stale(self):
        """Should return empty list when no stale messages exist."""
        cursor = MockCursor()
        cursor.fetchall_results = [[]]
        conn = MockConnection(cursor)
        mgr = RecoveryManager(lambda: conn)

        result = mgr.identify_stale_messages(5)

        self.assertEqual(result, [])

    def test_uses_custom_stale_minutes(self):
        """Should use the provided stale_minutes value in query."""
        cursor = MockCursor()
        cursor.fetchall_results = [[]]
        conn = MockConnection(cursor)
        mgr = RecoveryManager(lambda: conn)

        mgr.identify_stale_messages(10)

        params = cursor.executed[0][1]
        self.assertEqual(params, (10,))


class TestDeduplicateAndRequeue(unittest.TestCase):
    """Tests for deduplicate_and_requeue method."""

    def test_empty_list_returns_zero(self):
        """Should return 0 for empty message_ids list."""
        mgr = RecoveryManager(lambda: MockConnection())
        result = mgr.deduplicate_and_requeue([])
        self.assertEqual(result, 0)

    def test_requeues_messages_without_duplicates(self):
        """Should requeue messages when no duplicate delivery exists."""
        cursor = MockCursor()
        # fetchall: return messages with their idempotency keys
        cursor.fetchall_results = [
            [
                {"id": 1, "idempotency_key": "10_9876543210_5", "campaign_id": 10},
                {"id": 2, "idempotency_key": "10_9876543211_5", "campaign_id": 10},
            ]
        ]
        # fetchone: for each message, check if duplicate exists
        # None means no duplicate found
        cursor.fetchone_results = [None, None]

        conn = MockConnection(cursor)
        mgr = RecoveryManager(lambda: conn)

        result = mgr.deduplicate_and_requeue([1, 2])

        self.assertEqual(result, 2)

    def test_skips_messages_with_existing_delivery(self):
        """Should skip messages whose idempotency_key already has a delivery."""
        cursor = MockCursor()
        cursor.fetchall_results = [
            [
                {"id": 1, "idempotency_key": "10_9876543210_5", "campaign_id": 10},
            ]
        ]
        # fetchone: return a row indicating duplicate exists
        cursor.fetchone_results = [{"id": 99}]

        conn = MockConnection(cursor)
        mgr = RecoveryManager(lambda: conn)

        result = mgr.deduplicate_and_requeue([1])

        self.assertEqual(result, 0)
        # Verify the message was marked as 'skipped'
        skipped_sql = [sql for sql, _ in cursor.executed if "skipped" in sql]
        self.assertTrue(len(skipped_sql) > 0)


class TestRecoverOnStartup(unittest.TestCase):
    """Tests for the full recover_on_startup flow."""

    def test_returns_recovery_report(self):
        """Should return a RecoveryReport with all fields populated."""
        mgr = RecoveryManager(lambda: MockConnection())

        # Mock all internal methods
        mgr.identify_stale_messages = MagicMock(return_value=[])
        mgr._reset_stale_messages = MagicMock(return_value=0)
        mgr._deduplicate_pending_messages = MagicMock(return_value=0)
        mgr._resume_interrupted_campaigns = MagicMock(return_value=0)
        mgr._count_pending_messages = MagicMock(return_value=0)

        report = mgr.recover_on_startup()

        self.assertIsInstance(report, RecoveryReport)
        self.assertEqual(report.messages_requeued, 0)
        self.assertEqual(report.duplicates_prevented, 0)
        self.assertEqual(report.campaigns_resumed, 0)
        self.assertGreaterEqual(report.total_recovery_time_seconds, 0)
        self.assertEqual(report.errors, [])

    def test_handles_stale_messages(self):
        """Should reset stale messages and report count."""
        mgr = RecoveryManager(lambda: MockConnection())

        mgr.identify_stale_messages = MagicMock(return_value=[101, 102, 103])
        mgr._reset_stale_messages = MagicMock(return_value=3)
        mgr._deduplicate_pending_messages = MagicMock(return_value=0)
        mgr._resume_interrupted_campaigns = MagicMock(return_value=0)
        mgr._count_pending_messages = MagicMock(return_value=3)

        report = mgr.recover_on_startup()

        self.assertEqual(report.stale_messages_reset, 3)
        self.assertEqual(report.messages_requeued, 3)
        mgr._reset_stale_messages.assert_called_once_with([101, 102, 103])

    def test_handles_duplicates(self):
        """Should report duplicates prevented during recovery."""
        mgr = RecoveryManager(lambda: MockConnection())

        mgr.identify_stale_messages = MagicMock(return_value=[])
        mgr._deduplicate_pending_messages = MagicMock(return_value=5)
        mgr._resume_interrupted_campaigns = MagicMock(return_value=0)
        mgr._count_pending_messages = MagicMock(return_value=10)

        report = mgr.recover_on_startup()

        self.assertEqual(report.duplicates_prevented, 5)

    def test_resumes_campaigns(self):
        """Should report campaigns resumed."""
        mgr = RecoveryManager(lambda: MockConnection())

        mgr.identify_stale_messages = MagicMock(return_value=[])
        mgr._deduplicate_pending_messages = MagicMock(return_value=0)
        mgr._resume_interrupted_campaigns = MagicMock(return_value=2)
        mgr._count_pending_messages = MagicMock(return_value=50)

        report = mgr.recover_on_startup()

        self.assertEqual(report.campaigns_resumed, 2)

    def test_handles_exception_gracefully(self):
        """Should catch exceptions and include in report errors."""
        mgr = RecoveryManager(lambda: MockConnection())

        mgr.identify_stale_messages = MagicMock(
            side_effect=Exception("DB connection lost")
        )

        report = mgr.recover_on_startup()

        self.assertTrue(len(report.errors) > 0)
        self.assertIn("DB connection lost", report.errors[0])

    def test_recovery_time_is_tracked(self):
        """Should track total recovery time in seconds."""
        mgr = RecoveryManager(lambda: MockConnection())

        mgr.identify_stale_messages = MagicMock(return_value=[])
        mgr._deduplicate_pending_messages = MagicMock(return_value=0)
        mgr._resume_interrupted_campaigns = MagicMock(return_value=0)
        mgr._count_pending_messages = MagicMock(return_value=0)

        report = mgr.recover_on_startup()

        self.assertGreaterEqual(report.total_recovery_time_seconds, 0)


class TestResumeInterruptedCampaigns(unittest.TestCase):
    """Tests for _resume_interrupted_campaigns."""

    def test_resumes_sending_campaigns_with_queued_messages(self):
        """Should resume campaigns in 'sending' state with queued messages."""
        cursor = MockCursor()
        # First fetchall: campaigns in 'sending' with queued messages
        cursor.fetchall_results = [
            [{"id": 10}, {"id": 20}],
            # Second fetchall: approved campaigns with queued messages
            [],
        ]
        conn = MockConnection(cursor)
        mgr = RecoveryManager(lambda: conn)

        result = mgr._resume_interrupted_campaigns()

        self.assertEqual(result, 2)

    def test_resumes_approved_campaigns_with_queued_messages(self):
        """Should transition approved campaigns with queued messages to sending."""
        cursor = MockCursor()
        cursor.fetchall_results = [
            # No sending campaigns
            [],
            # Approved campaigns with queued messages
            [{"id": 30}],
        ]
        conn = MockConnection(cursor)
        mgr = RecoveryManager(lambda: conn)

        result = mgr._resume_interrupted_campaigns()

        self.assertEqual(result, 1)

    def test_returns_zero_when_nothing_to_resume(self):
        """Should return 0 when no campaigns need resuming."""
        cursor = MockCursor()
        cursor.fetchall_results = [[], []]
        conn = MockConnection(cursor)
        mgr = RecoveryManager(lambda: conn)

        result = mgr._resume_interrupted_campaigns()

        self.assertEqual(result, 0)


class TestDeduplicatePendingMessages(unittest.TestCase):
    """Tests for _deduplicate_pending_messages."""

    def test_marks_duplicates_as_skipped(self):
        """Should skip queued messages that already have a successful delivery."""
        cursor = MockCursor()
        cursor.fetchall_results = [
            [{"id": 5}, {"id": 6}]
        ]
        cursor.rowcount = 2
        conn = MockConnection(cursor)
        mgr = RecoveryManager(lambda: conn)

        result = mgr._deduplicate_pending_messages()

        # rowcount is 2, so 2 duplicates prevented
        self.assertEqual(result, 2)

    def test_no_duplicates_returns_zero(self):
        """Should return 0 when no duplicates exist."""
        cursor = MockCursor()
        cursor.fetchall_results = [[]]
        conn = MockConnection(cursor)
        mgr = RecoveryManager(lambda: conn)

        result = mgr._deduplicate_pending_messages()

        self.assertEqual(result, 0)


if __name__ == "__main__":
    unittest.main()
