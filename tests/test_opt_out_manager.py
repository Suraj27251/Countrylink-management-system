"""
Unit tests for OptOutManager — keyword recognition, suppression list management,
opt-out/opt-in processing, and DND functionality.
"""

import unittest
from unittest.mock import MagicMock, patch, call

from services.opt_out_manager import (
    OptOutManager,
    OPT_OUT_KEYWORDS,
    OPT_IN_KEYWORDS,
)


class MockCursor:
    """Mock MySQL cursor with dictionary=True support."""

    def __init__(self):
        self.executed = []
        self.fetchone_results = []
        self._call_idx = 0
        self._closed = False
        self.rowcount = 0

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchone(self):
        if self._call_idx < len(self.fetchone_results):
            result = self.fetchone_results[self._call_idx]
            self._call_idx += 1
            return result
        return None

    def close(self):
        self._closed = True


class MockConnection:
    """Mock MySQL connection."""

    def __init__(self, cursor_instance=None):
        self._cursor = cursor_instance or MockCursor()
        self._committed = False
        self._commit_count = 0
        self._rolled_back = False
        self._connected = True

    def cursor(self, dictionary=False):
        return self._cursor

    def commit(self):
        self._committed = True
        self._commit_count += 1

    def rollback(self):
        self._rolled_back = True

    def start_transaction(self):
        pass

    def close(self):
        self._connected = False

    def is_connected(self):
        return self._connected


class TestKeywordRecognition(unittest.TestCase):
    """Tests for opt-out and opt-in keyword detection."""

    def setUp(self):
        self.mgr = OptOutManager(lambda: MockConnection())

    def test_opt_out_keywords_all_recognized(self):
        """All defined opt-out keywords should be recognized."""
        for keyword in OPT_OUT_KEYWORDS:
            with self.subTest(keyword=keyword):
                self.assertTrue(self.mgr.is_opt_out_keyword(keyword))

    def test_opt_out_keywords_case_insensitive(self):
        """Opt-out keywords should match regardless of case."""
        self.assertTrue(self.mgr.is_opt_out_keyword("STOP"))
        self.assertTrue(self.mgr.is_opt_out_keyword("Stop"))
        self.assertTrue(self.mgr.is_opt_out_keyword("stop"))
        self.assertTrue(self.mgr.is_opt_out_keyword("UNSUBSCRIBE"))
        self.assertTrue(self.mgr.is_opt_out_keyword("Unsubscribe"))
        self.assertTrue(self.mgr.is_opt_out_keyword("OPT OUT"))
        self.assertTrue(self.mgr.is_opt_out_keyword("Opt Out"))
        self.assertTrue(self.mgr.is_opt_out_keyword("opt out"))
        self.assertTrue(self.mgr.is_opt_out_keyword("CANCEL"))
        self.assertTrue(self.mgr.is_opt_out_keyword("Cancel"))
        self.assertTrue(self.mgr.is_opt_out_keyword("DND"))
        self.assertTrue(self.mgr.is_opt_out_keyword("dnd"))

    def test_opt_out_keywords_with_whitespace(self):
        """Opt-out keywords with leading/trailing whitespace should match."""
        self.assertTrue(self.mgr.is_opt_out_keyword("  STOP  "))
        self.assertTrue(self.mgr.is_opt_out_keyword("\tstop\n"))

    def test_non_opt_out_keywords_not_recognized(self):
        """Non-opt-out text should not match."""
        self.assertFalse(self.mgr.is_opt_out_keyword("hello"))
        self.assertFalse(self.mgr.is_opt_out_keyword("please stop sending"))
        self.assertFalse(self.mgr.is_opt_out_keyword("stopped"))
        self.assertFalse(self.mgr.is_opt_out_keyword(""))
        self.assertFalse(self.mgr.is_opt_out_keyword(None))

    def test_opt_in_keywords_all_recognized(self):
        """All defined opt-in keywords should be recognized."""
        for keyword in OPT_IN_KEYWORDS:
            with self.subTest(keyword=keyword):
                self.assertTrue(self.mgr.is_opt_in_keyword(keyword))

    def test_opt_in_keywords_case_insensitive(self):
        """Opt-in keywords should match regardless of case."""
        self.assertTrue(self.mgr.is_opt_in_keyword("START"))
        self.assertTrue(self.mgr.is_opt_in_keyword("start"))
        self.assertTrue(self.mgr.is_opt_in_keyword("Start"))
        self.assertTrue(self.mgr.is_opt_in_keyword("SUBSCRIBE"))
        self.assertTrue(self.mgr.is_opt_in_keyword("subscribe"))
        self.assertTrue(self.mgr.is_opt_in_keyword("Subscribe"))

    def test_opt_in_keywords_with_whitespace(self):
        """Opt-in keywords with whitespace should match."""
        self.assertTrue(self.mgr.is_opt_in_keyword("  START  "))
        self.assertTrue(self.mgr.is_opt_in_keyword("\tsubscribe\n"))

    def test_non_opt_in_keywords_not_recognized(self):
        """Non-opt-in text should not match."""
        self.assertFalse(self.mgr.is_opt_in_keyword("starting"))
        self.assertFalse(self.mgr.is_opt_in_keyword("please subscribe me"))
        self.assertFalse(self.mgr.is_opt_in_keyword(""))
        self.assertFalse(self.mgr.is_opt_in_keyword(None))

    def test_opt_out_and_opt_in_no_overlap(self):
        """Opt-out and opt-in keyword sets should not overlap."""
        self.assertEqual(OPT_OUT_KEYWORDS & OPT_IN_KEYWORDS, set())


class TestProcessOptOut(unittest.TestCase):
    """Tests for process_opt_out functionality."""

    def test_new_opt_out_inserts_suppression_record(self):
        """First opt-out should INSERT into suppression_list."""
        cursor = MockCursor()
        # First fetchone: no existing record
        cursor.fetchone_results = [None]
        conn = MockConnection(cursor)
        send_msg = MagicMock()
        mgr = OptOutManager(lambda: conn, send_message_fn=send_msg)

        mgr.process_opt_out("919876543210", "STOP")

        # Should have an INSERT statement
        insert_calls = [
            (sql, params)
            for sql, params in cursor.executed
            if "INSERT INTO suppression_list" in sql
        ]
        self.assertEqual(len(insert_calls), 1)
        sql, params = insert_calls[0]
        self.assertEqual(params[0], "919876543210")
        self.assertEqual(params[1], "stop")

    def test_opt_out_sends_confirmation_message(self):
        """Opt-out should trigger a confirmation message to the customer."""
        cursor = MockCursor()
        cursor.fetchone_results = [None]
        conn = MockConnection(cursor)
        send_msg = MagicMock()
        mgr = OptOutManager(lambda: conn, send_message_fn=send_msg)

        mgr.process_opt_out("919876543210", "STOP")

        send_msg.assert_called_once()
        args = send_msg.call_args
        self.assertEqual(args[0][0], "919876543210")
        self.assertEqual(args[0][1], "text")
        self.assertIn("unsubscribed", args[1]["text"].lower())

    def test_opt_out_no_confirmation_when_no_send_fn(self):
        """Opt-out should not fail if no send_message_fn is configured."""
        cursor = MockCursor()
        cursor.fetchone_results = [None]
        conn = MockConnection(cursor)
        mgr = OptOutManager(lambda: conn, send_message_fn=None)

        # Should not raise
        mgr.process_opt_out("919876543210", "STOP")

    def test_duplicate_opt_out_no_duplicate_record(self):
        """If customer is already opted-out, no new record should be created."""
        cursor = MockCursor()
        # Already active
        cursor.fetchone_results = [{"id": 1, "is_active": 1}]
        conn = MockConnection(cursor)
        send_msg = MagicMock()
        mgr = OptOutManager(lambda: conn, send_message_fn=send_msg)

        mgr.process_opt_out("919876543210", "STOP")

        # Should NOT have an INSERT statement
        insert_calls = [
            (sql, params)
            for sql, params in cursor.executed
            if "INSERT INTO suppression_list" in sql
        ]
        self.assertEqual(len(insert_calls), 0)

    def test_reactivation_of_inactive_record(self):
        """Opt-out on inactive record should reactivate it."""
        cursor = MockCursor()
        # Existing but inactive
        cursor.fetchone_results = [{"id": 5, "is_active": 0}]
        conn = MockConnection(cursor)
        send_msg = MagicMock()
        mgr = OptOutManager(lambda: conn, send_message_fn=send_msg)

        mgr.process_opt_out("919876543210", "UNSUBSCRIBE")

        # Should have an UPDATE statement to reactivate
        update_calls = [
            (sql, params)
            for sql, params in cursor.executed
            if "UPDATE suppression_list" in sql and "is_active = 1" in sql
        ]
        self.assertEqual(len(update_calls), 1)

    def test_opt_out_records_customer_activity(self):
        """Opt-out should record activity in customer_activity table."""
        cursor = MockCursor()
        cursor.fetchone_results = [None]
        conn = MockConnection(cursor)
        mgr = OptOutManager(lambda: conn)

        mgr.process_opt_out("919876543210", "STOP")

        activity_calls = [
            (sql, params)
            for sql, params in cursor.executed
            if "INSERT INTO customer_activity" in sql
        ]
        self.assertEqual(len(activity_calls), 1)
        sql, params = activity_calls[0]
        self.assertEqual(params[0], "919876543210")
        self.assertIn("opt_out", sql)


class TestProcessOptIn(unittest.TestCase):
    """Tests for process_opt_in functionality."""

    def test_opt_in_deactivates_suppression_record(self):
        """Opt-in should set is_active=0 on suppression records."""
        cursor = MockCursor()
        cursor.rowcount = 1
        conn = MockConnection(cursor)
        send_msg = MagicMock()
        mgr = OptOutManager(lambda: conn, send_message_fn=send_msg)

        mgr.process_opt_in("919876543210", "START")

        # Should have an UPDATE with is_active = 0
        update_calls = [
            (sql, params)
            for sql, params in cursor.executed
            if "UPDATE suppression_list" in sql and "is_active = 0" in sql
        ]
        self.assertEqual(len(update_calls), 1)
        sql, params = update_calls[0]
        self.assertEqual(params[0], "919876543210")

    def test_opt_in_sends_confirmation_message(self):
        """Opt-in should trigger a welcome-back confirmation message."""
        cursor = MockCursor()
        cursor.rowcount = 1
        conn = MockConnection(cursor)
        send_msg = MagicMock()
        mgr = OptOutManager(lambda: conn, send_message_fn=send_msg)

        mgr.process_opt_in("919876543210", "START")

        send_msg.assert_called_once()
        args = send_msg.call_args
        self.assertEqual(args[0][0], "919876543210")
        self.assertEqual(args[0][1], "text")
        self.assertIn("re-subscribed", args[1]["text"].lower())

    def test_opt_in_no_error_when_not_suppressed(self):
        """Opt-in for a customer not on suppression list should not error."""
        cursor = MockCursor()
        cursor.rowcount = 0
        conn = MockConnection(cursor)
        send_msg = MagicMock()
        mgr = OptOutManager(lambda: conn, send_message_fn=send_msg)

        # Should not raise
        mgr.process_opt_in("919876543210", "SUBSCRIBE")

    def test_opt_in_records_customer_activity(self):
        """Opt-in should record activity in customer_activity table."""
        cursor = MockCursor()
        cursor.rowcount = 1
        conn = MockConnection(cursor)
        mgr = OptOutManager(lambda: conn)

        mgr.process_opt_in("919876543210", "START")

        activity_calls = [
            (sql, params)
            for sql, params in cursor.executed
            if "INSERT INTO customer_activity" in sql
        ]
        self.assertEqual(len(activity_calls), 1)
        sql, params = activity_calls[0]
        self.assertEqual(params[0], "919876543210")


class TestIsSuppressed(unittest.TestCase):
    """Tests for is_suppressed check."""

    def test_suppressed_when_active_record_exists(self):
        """Should return True when an active suppression record exists."""
        cursor = MockCursor()
        cursor.fetchone_results = [{"1": 1}]
        conn = MockConnection(cursor)
        mgr = OptOutManager(lambda: conn)

        self.assertTrue(mgr.is_suppressed("919876543210"))

    def test_not_suppressed_when_no_record(self):
        """Should return False when no suppression record exists."""
        cursor = MockCursor()
        cursor.fetchone_results = [None]
        conn = MockConnection(cursor)
        mgr = OptOutManager(lambda: conn)

        self.assertFalse(mgr.is_suppressed("919876543210"))

    def test_queries_for_is_active_1(self):
        """Query should check is_active = 1."""
        cursor = MockCursor()
        cursor.fetchone_results = [None]
        conn = MockConnection(cursor)
        mgr = OptOutManager(lambda: conn)

        mgr.is_suppressed("919876543210")

        self.assertEqual(len(cursor.executed), 1)
        sql, params = cursor.executed[0]
        self.assertIn("is_active = 1", sql)
        self.assertEqual(params[0], "919876543210")

    def test_returns_false_on_db_error(self):
        """Should return False (fail-safe) on database errors."""

        def failing_connection():
            raise Exception("DB connection failed")

        mgr = OptOutManager(failing_connection)
        # Should not raise, returns False as fail-safe
        self.assertFalse(mgr.is_suppressed("919876543210"))


class TestAddToDnd(unittest.TestCase):
    """Tests for manual DND addition by operators."""

    def test_new_dnd_inserts_record(self):
        """Adding DND for new customer should INSERT record."""
        cursor = MockCursor()
        cursor.fetchone_results = [None]
        conn = MockConnection(cursor)
        mgr = OptOutManager(lambda: conn)

        mgr.add_to_dnd("919876543210", "Customer requested DND", "operator1")

        insert_calls = [
            (sql, params)
            for sql, params in cursor.executed
            if "INSERT INTO suppression_list" in sql
        ]
        self.assertEqual(len(insert_calls), 1)
        sql, params = insert_calls[0]
        self.assertEqual(params[0], "919876543210")
        self.assertIn("manual_dnd", sql)
        self.assertEqual(params[1], "Customer requested DND")
        self.assertEqual(params[2], "operator1")

    def test_dnd_already_active_no_duplicate(self):
        """If customer already has active DND, no new record should be created."""
        cursor = MockCursor()
        cursor.fetchone_results = [{"id": 3, "is_active": 1}]
        conn = MockConnection(cursor)
        mgr = OptOutManager(lambda: conn)

        mgr.add_to_dnd("919876543210", "test reason", "operator1")

        # Should NOT have INSERT or UPDATE
        insert_calls = [
            (sql, params)
            for sql, params in cursor.executed
            if "INSERT INTO suppression_list" in sql
        ]
        update_calls = [
            (sql, params)
            for sql, params in cursor.executed
            if "UPDATE suppression_list" in sql
        ]
        self.assertEqual(len(insert_calls), 0)
        self.assertEqual(len(update_calls), 0)

    def test_dnd_reactivation_of_inactive_record(self):
        """Inactive DND record should be reactivated."""
        cursor = MockCursor()
        cursor.fetchone_results = [{"id": 7, "is_active": 0}]
        conn = MockConnection(cursor)
        mgr = OptOutManager(lambda: conn)

        mgr.add_to_dnd("919876543210", "new reason", "operator2")

        update_calls = [
            (sql, params)
            for sql, params in cursor.executed
            if "UPDATE suppression_list" in sql and "is_active = 1" in sql
        ]
        self.assertEqual(len(update_calls), 1)
        _, params = update_calls[0]
        self.assertEqual(params[0], "operator2")
        self.assertEqual(params[1], "new reason")
        self.assertEqual(params[2], 7)

    def test_dnd_records_customer_activity(self):
        """DND addition should record activity in customer_activity."""
        cursor = MockCursor()
        cursor.fetchone_results = [None]
        conn = MockConnection(cursor)
        mgr = OptOutManager(lambda: conn)

        mgr.add_to_dnd("919876543210", "Spam complaints", "operator1")

        activity_calls = [
            (sql, params)
            for sql, params in cursor.executed
            if "INSERT INTO customer_activity" in sql
        ]
        self.assertEqual(len(activity_calls), 1)


class TestOptOutManagerConstants(unittest.TestCase):
    """Tests for module-level constants."""

    def test_opt_out_keywords_set(self):
        """OPT_OUT_KEYWORDS should contain expected keywords."""
        expected = {"stop", "unsubscribe", "opt out", "cancel", "dnd"}
        self.assertEqual(OPT_OUT_KEYWORDS, expected)

    def test_opt_in_keywords_set(self):
        """OPT_IN_KEYWORDS should contain expected keywords."""
        expected = {"start", "subscribe"}
        self.assertEqual(OPT_IN_KEYWORDS, expected)

    def test_all_keywords_lowercase(self):
        """All keyword constants should be stored in lowercase."""
        for kw in OPT_OUT_KEYWORDS:
            self.assertEqual(kw, kw.lower())
        for kw in OPT_IN_KEYWORDS:
            self.assertEqual(kw, kw.lower())


if __name__ == "__main__":
    unittest.main()
