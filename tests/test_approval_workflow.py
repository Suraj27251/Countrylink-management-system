"""
Unit tests for Operator Approval Workflow — approve, reject, preview,
and submit-for-approval operations.

Uses mock MySQL connections to test business logic without database dependency.
Validates Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6
"""

import json
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

from blueprints.campaign_bp import CampaignService


class MockCursor:
    """Mock MySQL cursor with dictionary=True support."""

    def __init__(self):
        self.executed = []
        self.fetchone_result = None
        self.fetchall_result = []
        self.lastrowid = 1
        self._closed = False
        self.rowcount = 1

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


class TestApproveCampaign(unittest.TestCase):
    """Test campaign approval workflow."""

    def test_approve_pending_campaign_succeeds(self):
        """Approving a pending_approval campaign should transition to sending."""
        call_count = [0]

        def get_conn():
            call_count[0] += 1
            c = MockCursor()
            if call_count[0] == 1:
                # For approve_campaign: SELECT returns pending_approval state
                c.fetchone_result = {
                    "id": 1, "status": "pending_approval",
                    "segment_id": 10, "template_id": 20,
                }
                return MockConnection(c)
            else:
                # For get_campaign after approval
                c.fetchone_result = {
                    "id": 1, "status": "sending", "name": "Test Campaign",
                    "approved_at": datetime.now(), "approved_by": "operator1",
                }
                return MockConnection(c)

        service = CampaignService(get_conn)
        # Patch _enqueue_campaign_recipients to avoid DB calls
        with patch.object(service, '_enqueue_campaign_recipients'):
            result = service.approve_campaign(1, "operator1")

        self.assertEqual(result["status"], "sending")
        self.assertEqual(result["approved_by"], "operator1")

    def test_approve_non_pending_campaign_raises(self):
        """Approving a campaign not in pending_approval should raise ValueError."""
        invalid_states = ["draft", "scheduled", "approved", "sending", "completed", "cancelled"]

        for state in invalid_states:
            cursor = MockCursor()
            cursor.fetchone_result = {
                "id": 1, "status": state,
                "segment_id": 10, "template_id": 20,
            }
            conn = MockConnection(cursor)
            service = CampaignService(lambda: conn)

            with self.assertRaises(ValueError, msg=f"Should reject approve from '{state}'"):
                service.approve_campaign(1, "operator1")

    def test_approve_not_found_raises(self):
        """Approving a non-existent campaign should raise ValueError."""
        cursor = MockCursor()
        cursor.fetchone_result = None
        conn = MockConnection(cursor)
        service = CampaignService(lambda: conn)

        with self.assertRaises(ValueError):
            service.approve_campaign(999, "operator1")

    def test_approve_triggers_enqueue(self):
        """Approving should call _enqueue_campaign_recipients with segment_id."""
        call_count = [0]

        def get_conn():
            call_count[0] += 1
            c = MockCursor()
            if call_count[0] == 1:
                c.fetchone_result = {
                    "id": 1, "status": "pending_approval",
                    "segment_id": 10, "template_id": 20,
                }
                return MockConnection(c)
            else:
                c.fetchone_result = {"id": 1, "status": "sending", "name": "Test"}
                return MockConnection(c)

        service = CampaignService(get_conn)
        with patch.object(service, '_enqueue_campaign_recipients') as mock_enqueue:
            service.approve_campaign(1, "operator1")
            mock_enqueue.assert_called_once_with(1, 10)

    def test_approve_logs_audit_entries(self):
        """Approving should create audit log entries for approval and sending."""
        cursors_used = []

        def get_conn():
            c = MockCursor()
            c.fetchone_result = {
                "id": 1, "status": "pending_approval",
                "segment_id": 10, "template_id": 20,
            }
            cursors_used.append(c)
            return MockConnection(c)

        service = CampaignService(get_conn)
        with patch.object(service, '_enqueue_campaign_recipients'):
            try:
                service.approve_campaign(1, "operator1")
            except Exception:
                pass  # May fail on get_campaign due to mock, but audit logs should be called

        # Check that operator_actions INSERT was called on the first cursor
        first_cursor = cursors_used[0]
        audit_inserts = [
            sql for sql, _ in first_cursor.executed
            if "operator_actions" in sql
        ]
        # Should have 2 audit entries: approve_campaign and start_sending
        self.assertEqual(len(audit_inserts), 2)


class TestRejectCampaign(unittest.TestCase):
    """Test campaign rejection workflow."""

    def test_reject_pending_campaign_succeeds(self):
        """Rejecting a pending_approval campaign should transition to cancelled."""
        call_count = [0]

        def get_conn():
            call_count[0] += 1
            c = MockCursor()
            if call_count[0] == 1:
                c.fetchone_result = {"id": 1, "status": "pending_approval"}
                return MockConnection(c)
            else:
                c.fetchone_result = {
                    "id": 1, "status": "cancelled", "name": "Test Campaign",
                }
                return MockConnection(c)

        service = CampaignService(get_conn)
        result = service.reject_campaign(1, "operator1", "Poor audience targeting")
        self.assertEqual(result["status"], "cancelled")

    def test_reject_non_pending_campaign_raises(self):
        """Rejecting a campaign not in pending_approval should raise ValueError."""
        invalid_states = ["draft", "scheduled", "approved", "sending", "completed"]

        for state in invalid_states:
            cursor = MockCursor()
            cursor.fetchone_result = {"id": 1, "status": state}
            conn = MockConnection(cursor)
            service = CampaignService(lambda: conn)

            with self.assertRaises(ValueError, msg=f"Should reject reject from '{state}'"):
                service.reject_campaign(1, "operator1", "reason")

    def test_reject_not_found_raises(self):
        """Rejecting a non-existent campaign should raise ValueError."""
        cursor = MockCursor()
        cursor.fetchone_result = None
        conn = MockConnection(cursor)
        service = CampaignService(lambda: conn)

        with self.assertRaises(ValueError):
            service.reject_campaign(999, "operator1", "reason")

    def test_reject_logs_reason_in_audit(self):
        """Rejection audit log should include the rejection reason."""
        cursors_used = []

        def get_conn():
            c = MockCursor()
            c.fetchone_result = {"id": 1, "status": "pending_approval"}
            cursors_used.append(c)
            return MockConnection(c)

        service = CampaignService(get_conn)
        try:
            service.reject_campaign(1, "operator1", "Bad targeting")
        except Exception:
            pass

        first_cursor = cursors_used[0]
        # Find the audit INSERT call
        audit_calls = [
            (sql, params) for sql, params in first_cursor.executed
            if "operator_actions" in sql
        ]
        self.assertTrue(len(audit_calls) >= 1)
        # The details JSON should contain the reason
        details_json = audit_calls[0][1][-1]  # Last param is the JSON details
        details = json.loads(details_json)
        self.assertEqual(details["reason"], "Bad targeting")


class TestApprovalPreview(unittest.TestCase):
    """Test approval preview endpoint logic."""

    def test_preview_returns_campaign_info(self):
        """Preview should return campaign name, status, and type."""
        call_count = [0]

        def get_conn():
            call_count[0] += 1
            c = MockCursor()
            # Campaign fetch
            c.fetchone_result = {
                "id": 1, "name": "Test Campaign", "status": "pending_approval",
                "segment_id": 10, "template_id": 20, "campaign_type": "promotional",
            }
            return MockConnection(c)

        service = CampaignService(get_conn)
        # The method will try multiple fetchone calls on the same cursor
        # Mock accordingly with a sequence
        cursor = MockCursor()
        responses = [
            # Campaign fetch
            {"id": 1, "name": "Test Campaign", "status": "pending_approval",
             "segment_id": 10, "template_id": 20, "campaign_type": "promotional"},
            # Segment fetch
            {"name": "Active Users", "filter_criteria": json.dumps({"status": "active"})},
            # Recipient count
            {"cnt": 150},
            # Template fetch
            {"template_name": "promo_offer", "body_text": "Hello {{1}}!",
             "header_type": "none", "footer_text": "Reply STOP",
             "placeholder_count": 1, "placeholder_mappings": json.dumps({"1": "customer_name"})},
        ]
        fetch_index = [0]

        def mock_fetchone():
            if fetch_index[0] < len(responses):
                result = responses[fetch_index[0]]
                fetch_index[0] += 1
                return result
            return None

        cursor.fetchone = mock_fetchone
        conn = MockConnection(cursor)
        service = CampaignService(lambda: conn)

        preview = service.get_approval_preview(1)

        self.assertEqual(preview["campaign_id"], 1)
        self.assertEqual(preview["campaign_name"], "Test Campaign")
        self.assertEqual(preview["recipient_count"], 150)
        self.assertEqual(preview["template_name"], "promo_offer")
        self.assertIn("body_text", preview["template_content"])
        self.assertGreater(preview["estimated_time_seconds"], 0)

    def test_preview_not_found_raises(self):
        """Preview should raise ValueError for non-existent campaign."""
        cursor = MockCursor()
        cursor.fetchone_result = None
        conn = MockConnection(cursor)
        service = CampaignService(lambda: conn)

        with self.assertRaises(ValueError):
            service.get_approval_preview(999)

    def test_preview_no_segment_returns_zero_count(self):
        """Preview with no segment_id should return recipient_count=0."""
        cursor = MockCursor()
        cursor.fetchone_result = {
            "id": 1, "name": "Test", "status": "draft",
            "segment_id": None, "template_id": 20, "campaign_type": "promotional",
        }
        # Override fetchone to return None for template lookup
        responses = [
            {"id": 1, "name": "Test", "status": "draft",
             "segment_id": None, "template_id": 20, "campaign_type": "promotional"},
            # Template fetch
            {"template_name": "test", "body_text": "Hi",
             "header_type": "none", "footer_text": "",
             "placeholder_count": 0, "placeholder_mappings": None},
        ]
        fetch_index = [0]

        def mock_fetchone():
            if fetch_index[0] < len(responses):
                result = responses[fetch_index[0]]
                fetch_index[0] += 1
                return result
            return None

        cursor.fetchone = mock_fetchone
        conn = MockConnection(cursor)
        service = CampaignService(lambda: conn)

        preview = service.get_approval_preview(1)
        self.assertEqual(preview["recipient_count"], 0)


class TestSubmitForApproval(unittest.TestCase):
    """Test submit_for_approval workflow."""

    def test_submit_draft_succeeds(self):
        """Submitting a draft campaign for approval should transition to pending_approval."""
        call_count = [0]

        def get_conn():
            call_count[0] += 1
            c = MockCursor()
            if call_count[0] == 1:
                c.fetchone_result = {"id": 1, "status": "draft"}
                return MockConnection(c)
            else:
                c.fetchone_result = {"id": 1, "status": "pending_approval", "name": "Test"}
                return MockConnection(c)

        service = CampaignService(get_conn)
        result = service.submit_for_approval(1, "operator1")
        self.assertEqual(result["status"], "pending_approval")

    def test_submit_scheduled_succeeds(self):
        """Submitting a scheduled campaign for approval should transition to pending_approval."""
        call_count = [0]

        def get_conn():
            call_count[0] += 1
            c = MockCursor()
            if call_count[0] == 1:
                c.fetchone_result = {"id": 1, "status": "scheduled"}
                return MockConnection(c)
            else:
                c.fetchone_result = {"id": 1, "status": "pending_approval", "name": "Test"}
                return MockConnection(c)

        service = CampaignService(get_conn)
        result = service.submit_for_approval(1, "operator1")
        self.assertEqual(result["status"], "pending_approval")

    def test_submit_from_invalid_state_raises(self):
        """Submitting from an invalid state should raise ValueError."""
        invalid_states = ["pending_approval", "approved", "sending", "completed", "cancelled"]

        for state in invalid_states:
            cursor = MockCursor()
            cursor.fetchone_result = {"id": 1, "status": state}
            conn = MockConnection(cursor)
            service = CampaignService(lambda: conn)

            with self.assertRaises(ValueError, msg=f"Should reject submit from '{state}'"):
                service.submit_for_approval(1, "operator1")

    def test_submit_not_found_raises(self):
        """Submitting a non-existent campaign should raise ValueError."""
        cursor = MockCursor()
        cursor.fetchone_result = None
        conn = MockConnection(cursor)
        service = CampaignService(lambda: conn)

        with self.assertRaises(ValueError):
            service.submit_for_approval(999, "operator1")

    def test_submit_logs_audit(self):
        """Submit for approval should log to operator_actions."""
        cursors_used = []

        def get_conn():
            c = MockCursor()
            c.fetchone_result = {"id": 1, "status": "draft"}
            cursors_used.append(c)
            return MockConnection(c)

        service = CampaignService(get_conn)
        try:
            service.submit_for_approval(1, "operator1")
        except Exception:
            pass

        first_cursor = cursors_used[0]
        audit_calls = [
            sql for sql, _ in first_cursor.executed
            if "operator_actions" in sql
        ]
        self.assertTrue(len(audit_calls) >= 1)


class TestAutomationRuleApprovalGate(unittest.TestCase):
    """
    Test that automation_rule-generated drafts require approval.

    Validates Requirement 2.5: Automation_Rule drafts require operator
    approval before any messages are dispatched.
    """

    def test_automation_draft_cannot_skip_to_sending(self):
        """
        An automation-generated draft cannot transition directly to 'sending'
        or 'approved' — it must go through 'pending_approval' first.
        """
        # Draft → sending is not a valid transition
        cursor = MockCursor()
        cursor.fetchone_result = {"id": 1, "status": "draft"}
        conn = MockConnection(cursor)
        service = CampaignService(lambda: conn)

        with self.assertRaises(ValueError):
            service.transition_state(1, "sending", "automation_bot")

        # Draft → approved is also not valid
        cursor2 = MockCursor()
        cursor2.fetchone_result = {"id": 1, "status": "draft"}
        conn2 = MockConnection(cursor2)
        service2 = CampaignService(lambda: conn2)

        with self.assertRaises(ValueError):
            service2.transition_state(1, "approved", "automation_bot")

    def test_automation_draft_must_go_through_pending_approval(self):
        """
        An automation-generated draft must transition to pending_approval
        before it can be approved and sent.
        """
        # Step 1: draft → pending_approval (valid)
        call_count = [0]

        def get_conn():
            call_count[0] += 1
            c = MockCursor()
            if call_count[0] == 1:
                c.fetchone_result = {"id": 1, "status": "draft"}
                return MockConnection(c)
            else:
                c.fetchone_result = {"id": 1, "status": "pending_approval", "name": "Auto"}
                return MockConnection(c)

        service = CampaignService(get_conn)
        result = service.submit_for_approval(1, "automation_bot")
        self.assertEqual(result["status"], "pending_approval")


if __name__ == "__main__":
    unittest.main()
