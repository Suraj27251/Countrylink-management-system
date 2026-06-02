"""
Unit tests for CampaignService.test_send — test send functionality.

Tests that the /api/campaigns/<id>/test-send endpoint:
- Accepts 1-5 test mobile numbers (Req 16.1)
- Dispatches immediately bypassing regular queue (Req 16.2)
- Marks test messages distinctly with is_test_send=1 (Req 16.3)
- Returns delivery status for each test number (Req 16.4)
"""

import json
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch, call

from blueprints.campaign_bp import CampaignService


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


class TestCampaignTestSendValidation(unittest.TestCase):
    """Test input validation for test_send."""

    def _make_service(self, cursor=None):
        cursor = cursor or MockCursor()
        conn = MockConnection(cursor)
        return CampaignService(lambda: conn)

    def test_rejects_empty_test_numbers(self):
        """test_send should reject empty test_numbers list."""
        service = self._make_service()
        with self.assertRaises(ValueError) as ctx:
            service.test_send(1, [], "operator")
        self.assertIn("At least 1", str(ctx.exception))

    def test_rejects_more_than_5_numbers(self):
        """test_send should reject more than 5 test numbers."""
        service = self._make_service()
        numbers = ["91987654321" + str(i) for i in range(6)]
        with self.assertRaises(ValueError) as ctx:
            service.test_send(1, numbers, "operator")
        self.assertIn("Maximum 5", str(ctx.exception))

    def test_rejects_empty_string_numbers(self):
        """test_send should reject empty string numbers."""
        service = self._make_service()
        with self.assertRaises(ValueError) as ctx:
            service.test_send(1, ["", "919876543210"], "operator")
        self.assertIn("non-empty", str(ctx.exception))

    def test_rejects_whitespace_only_numbers(self):
        """test_send should reject whitespace-only numbers."""
        service = self._make_service()
        with self.assertRaises(ValueError) as ctx:
            service.test_send(1, ["   "], "operator")
        self.assertIn("non-empty", str(ctx.exception))


class TestCampaignTestSendCampaignValidation(unittest.TestCase):
    """Test campaign and template validation for test_send."""

    def test_raises_if_campaign_not_found(self):
        """test_send should raise ValueError when campaign doesn't exist."""
        call_count = [0]

        def get_conn():
            call_count[0] += 1
            c = MockCursor()
            c.fetchone_result = None  # Campaign not found
            return MockConnection(c)

        service = CampaignService(get_conn)
        with self.assertRaises(ValueError) as ctx:
            service.test_send(999, ["919876543210"], "operator")
        self.assertIn("not found", str(ctx.exception))

    def test_raises_if_no_template_assigned(self):
        """test_send should raise ValueError when campaign has no template_id."""
        call_count = [0]

        def get_conn():
            call_count[0] += 1
            c = MockCursor()
            # Campaign exists but no template
            c.fetchone_result = {
                "id": 1, "name": "Test", "template_id": None, "status": "draft"
            }
            return MockConnection(c)

        service = CampaignService(get_conn)
        with self.assertRaises(ValueError) as ctx:
            service.test_send(1, ["919876543210"], "operator")
        self.assertIn("no template", str(ctx.exception))

    def test_raises_if_template_not_found(self):
        """test_send should raise ValueError when template doesn't exist in DB."""
        fetch_results = iter([
            # First fetchone: campaign with template_id
            {"id": 1, "name": "Test", "template_id": 5, "status": "draft"},
            # Second fetchone: template not found
            None,
        ])

        def get_conn():
            class SequentialCursor(MockCursor):
                def fetchone(self_cursor):
                    return next(fetch_results)

            conn = MockConnection(SequentialCursor())
            return conn

        service = CampaignService(get_conn)
        with self.assertRaises(ValueError) as ctx:
            service.test_send(1, ["919876543210"], "operator")
        self.assertIn("not found", str(ctx.exception))


class TestCampaignTestSendDispatch(unittest.TestCase):
    """Test that test_send dispatches immediately and returns results."""

    @patch("services.channel.WhatsAppDispatcher")
    def test_dispatches_to_all_numbers_and_returns_results(self, mock_dispatcher_cls):
        """test_send should dispatch to each number and return status for each."""
        from services.channel import DispatchResult

        # Mock the dispatcher instance
        mock_dispatcher = MagicMock()
        mock_dispatcher.send_template.return_value = DispatchResult(
            success=True, message_id="wamid_test_123"
        )
        mock_dispatcher_cls.return_value = mock_dispatcher

        call_count = [0]

        def get_conn():
            call_count[0] += 1
            if call_count[0] <= 1:
                class SeqCursor(MockCursor):
                    def __init__(self_cursor):
                        super().__init__()
                        self_cursor._results = iter([
                            {"id": 1, "name": "Test Campaign", "template_id": 5, "status": "draft"},
                            {"id": 5, "template_name": "hello_template", "body_text": "Hi {{1}}",
                             "placeholder_mappings": json.dumps({"1": "customer_name"})},
                        ])

                    def fetchone(self_cursor):
                        return next(self_cursor._results)

                return MockConnection(SeqCursor())
            else:
                # For audit log connection
                return MockConnection(MockCursor())

        service = CampaignService(get_conn)
        test_numbers = ["919876543210", "919876543211"]
        results = service.test_send(1, test_numbers, "operator1")

        # Should have results for each number
        self.assertEqual(len(results), 2)

        # Each result should have correct structure
        for i, result in enumerate(results):
            self.assertEqual(result["mobile"], test_numbers[i])
            self.assertEqual(result["status"], "sent")
            self.assertEqual(result["message_id"], "wamid_test_123")
            self.assertIsNone(result["error_code"])
            self.assertIsNone(result["error_message"])

        # Dispatcher should be called for each number
        self.assertEqual(mock_dispatcher.send_template.call_count, 2)

    @patch("services.channel.WhatsAppDispatcher")
    def test_handles_failed_dispatch(self, mock_dispatcher_cls):
        """test_send should report failures correctly for each number."""
        from services.channel import DispatchResult

        mock_dispatcher = MagicMock()
        # First number succeeds, second fails
        mock_dispatcher.send_template.side_effect = [
            DispatchResult(success=True, message_id="wamid_ok"),
            DispatchResult(success=False, error_code=131026, error_message="Invalid number"),
        ]
        mock_dispatcher_cls.return_value = mock_dispatcher

        call_count = [0]

        def get_conn():
            call_count[0] += 1
            if call_count[0] <= 1:
                class SeqCursor(MockCursor):
                    def __init__(self_cursor):
                        super().__init__()
                        self_cursor._results = iter([
                            {"id": 1, "name": "Test", "template_id": 5, "status": "draft"},
                            {"id": 5, "template_name": "test_tmpl", "body_text": "Hello",
                             "placeholder_mappings": None},
                        ])

                    def fetchone(self_cursor):
                        return next(self_cursor._results)

                return MockConnection(SeqCursor())
            else:
                return MockConnection(MockCursor())

        service = CampaignService(get_conn)
        results = service.test_send(1, ["919876543210", "919876543211"], "op")

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["status"], "sent")
        self.assertEqual(results[0]["message_id"], "wamid_ok")
        self.assertEqual(results[1]["status"], "failed")
        self.assertEqual(results[1]["error_code"], 131026)
        self.assertEqual(results[1]["error_message"], "Invalid number")

    @patch("services.channel.WhatsAppDispatcher")
    def test_inserts_delivery_log_with_is_test_send(self, mock_dispatcher_cls):
        """test_send should insert into campaign_messages with is_test_send=1."""
        from services.channel import DispatchResult

        mock_dispatcher = MagicMock()
        mock_dispatcher.send_template.return_value = DispatchResult(
            success=True, message_id="wamid_test"
        )
        mock_dispatcher_cls.return_value = mock_dispatcher

        call_count = [0]
        captured_cursor = [None]

        def get_conn():
            call_count[0] += 1
            if call_count[0] <= 1:
                class SeqCursor(MockCursor):
                    def __init__(self_cursor):
                        super().__init__()
                        self_cursor._results = iter([
                            {"id": 1, "name": "Test", "template_id": 5, "status": "draft"},
                            {"id": 5, "template_name": "test_tmpl", "body_text": "Hi",
                             "placeholder_mappings": None},
                        ])

                    def fetchone(self_cursor):
                        return next(self_cursor._results)

                cursor = SeqCursor()
                captured_cursor[0] = cursor
                return MockConnection(cursor)
            else:
                return MockConnection(MockCursor())

        service = CampaignService(get_conn)
        service.test_send(1, ["919876543210"], "op")

        # Verify that the INSERT SQL contains is_test_send
        insert_calls = [
            (sql, params) for sql, params in captured_cursor[0].executed
            if "INSERT INTO campaign_messages" in sql
        ]
        self.assertEqual(len(insert_calls), 1)

        sql, params = insert_calls[0]
        self.assertIn("is_test_send", sql)
        # The is_test_send param should be 1 (True)
        # It's the 11th parameter in the INSERT
        self.assertEqual(params[10], 1)  # is_test_send = 1


class TestCampaignTestSendBypassesQueue(unittest.TestCase):
    """Test that test_send bypasses the regular sending queue."""

    @patch("services.channel.WhatsAppDispatcher")
    def test_does_not_use_sending_queue(self, mock_dispatcher_cls):
        """test_send should dispatch directly via WhatsAppDispatcher, not the queue."""
        from services.channel import DispatchResult

        mock_dispatcher = MagicMock()
        mock_dispatcher.send_template.return_value = DispatchResult(
            success=True, message_id="wamid_direct"
        )
        mock_dispatcher_cls.return_value = mock_dispatcher

        call_count = [0]

        def get_conn():
            call_count[0] += 1
            if call_count[0] <= 1:
                class SeqCursor(MockCursor):
                    def __init__(self_cursor):
                        super().__init__()
                        self_cursor._results = iter([
                            {"id": 1, "name": "Test", "template_id": 5, "status": "draft"},
                            {"id": 5, "template_name": "tmpl", "body_text": "Msg",
                             "placeholder_mappings": None},
                        ])

                    def fetchone(self_cursor):
                        return next(self_cursor._results)

                return MockConnection(SeqCursor())
            else:
                return MockConnection(MockCursor())

        service = CampaignService(get_conn)
        results = service.test_send(1, ["919876543210"], "op")

        # Dispatcher was called directly (not through SendingQueue)
        mock_dispatcher.send_template.assert_called_once_with(
            recipient="919876543210",
            template_name="tmpl",
            params=[],
        )

        # Result confirms direct dispatch
        self.assertEqual(results[0]["status"], "sent")


class TestCampaignTestSendTemplateParams(unittest.TestCase):
    """Test template parameter resolution for test send."""

    @patch("services.channel.WhatsAppDispatcher")
    def test_resolves_placeholder_mappings_as_sample_values(self, mock_dispatcher_cls):
        """test_send should resolve template params using field names as sample values."""
        from services.channel import DispatchResult

        mock_dispatcher = MagicMock()
        mock_dispatcher.send_template.return_value = DispatchResult(
            success=True, message_id="wamid_x"
        )
        mock_dispatcher_cls.return_value = mock_dispatcher

        call_count = [0]

        def get_conn():
            call_count[0] += 1
            if call_count[0] <= 1:
                class SeqCursor(MockCursor):
                    def __init__(self_cursor):
                        super().__init__()
                        self_cursor._results = iter([
                            {"id": 1, "name": "Test", "template_id": 5, "status": "draft"},
                            {"id": 5, "template_name": "hello_tmpl",
                             "body_text": "Hi {{1}}, your plan {{2}} expires soon",
                             "placeholder_mappings": json.dumps({"1": "customer_name", "2": "plan_name"})},
                        ])

                    def fetchone(self_cursor):
                        return next(self_cursor._results)

                return MockConnection(SeqCursor())
            else:
                return MockConnection(MockCursor())

        service = CampaignService(get_conn)
        service.test_send(1, ["919876543210"], "op")

        # Verify dispatcher was called with sample params
        mock_dispatcher.send_template.assert_called_once_with(
            recipient="919876543210",
            template_name="hello_tmpl",
            params=["[customer_name]", "[plan_name]"],
        )

    @patch("services.channel.WhatsAppDispatcher")
    def test_handles_template_without_placeholders(self, mock_dispatcher_cls):
        """test_send should work for templates with no placeholders."""
        from services.channel import DispatchResult

        mock_dispatcher = MagicMock()
        mock_dispatcher.send_template.return_value = DispatchResult(
            success=True, message_id="wamid_y"
        )
        mock_dispatcher_cls.return_value = mock_dispatcher

        call_count = [0]

        def get_conn():
            call_count[0] += 1
            if call_count[0] <= 1:
                class SeqCursor(MockCursor):
                    def __init__(self_cursor):
                        super().__init__()
                        self_cursor._results = iter([
                            {"id": 1, "name": "Test", "template_id": 5, "status": "draft"},
                            {"id": 5, "template_name": "simple_tmpl",
                             "body_text": "Hello there!",
                             "placeholder_mappings": None},
                        ])

                    def fetchone(self_cursor):
                        return next(self_cursor._results)

                return MockConnection(SeqCursor())
            else:
                return MockConnection(MockCursor())

        service = CampaignService(get_conn)
        service.test_send(1, ["919876543210"], "op")

        # Verify dispatcher called with empty params
        mock_dispatcher.send_template.assert_called_once_with(
            recipient="919876543210",
            template_name="simple_tmpl",
            params=[],
        )


class TestCampaignTestSendAccepts1To5Numbers(unittest.TestCase):
    """Test that exactly 1-5 numbers are accepted."""

    @patch("services.channel.WhatsAppDispatcher")
    def test_accepts_1_number(self, mock_dispatcher_cls):
        """test_send should accept exactly 1 number."""
        from services.channel import DispatchResult

        mock_dispatcher = MagicMock()
        mock_dispatcher.send_template.return_value = DispatchResult(
            success=True, message_id="wamid_1"
        )
        mock_dispatcher_cls.return_value = mock_dispatcher

        call_count = [0]

        def get_conn():
            call_count[0] += 1
            if call_count[0] <= 1:
                class SeqCursor(MockCursor):
                    def __init__(self_cursor):
                        super().__init__()
                        self_cursor._results = iter([
                            {"id": 1, "name": "T", "template_id": 5, "status": "draft"},
                            {"id": 5, "template_name": "t", "body_text": "x",
                             "placeholder_mappings": None},
                        ])

                    def fetchone(self_cursor):
                        return next(self_cursor._results)

                return MockConnection(SeqCursor())
            else:
                return MockConnection(MockCursor())

        service = CampaignService(get_conn)
        results = service.test_send(1, ["919876543210"], "op")
        self.assertEqual(len(results), 1)

    @patch("services.channel.WhatsAppDispatcher")
    def test_accepts_5_numbers(self, mock_dispatcher_cls):
        """test_send should accept exactly 5 numbers."""
        from services.channel import DispatchResult

        mock_dispatcher = MagicMock()
        mock_dispatcher.send_template.return_value = DispatchResult(
            success=True, message_id="wamid_5"
        )
        mock_dispatcher_cls.return_value = mock_dispatcher

        call_count = [0]

        def get_conn():
            call_count[0] += 1
            if call_count[0] <= 1:
                class SeqCursor(MockCursor):
                    def __init__(self_cursor):
                        super().__init__()
                        self_cursor._results = iter([
                            {"id": 1, "name": "T", "template_id": 5, "status": "draft"},
                            {"id": 5, "template_name": "t", "body_text": "x",
                             "placeholder_mappings": None},
                        ])

                    def fetchone(self_cursor):
                        return next(self_cursor._results)

                return MockConnection(SeqCursor())
            else:
                return MockConnection(MockCursor())

        service = CampaignService(get_conn)
        numbers = [f"91987654321{i}" for i in range(5)]
        results = service.test_send(1, numbers, "op")
        self.assertEqual(len(results), 5)


if __name__ == "__main__":
    unittest.main()
