"""
Unit tests for ReactivationService — reactivation workflows and automation rules.

Tests:
- Workflow templates listing and retrieval
- Campaign preparation from workflows (pre-populated segment + template)
- Automation rules CRUD (create, read, update, delete)
- Automation rule execution (create_campaign_draft, notify_operator)
- Reactivation success tracking
- All automation-generated campaigns require operator approval (draft status)

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5
"""

import json
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from services.reactivation import (
    ReactivationService,
    ReactivationWorkflow,
    REACTIVATION_WORKFLOWS,
)


class MockCursor:
    """Mock MySQL cursor with dictionary=True support."""

    def __init__(self):
        self.executed = []
        self.fetchone_results = []
        self.fetchall_results = []
        self.lastrowid = 1
        self._fetchone_index = 0
        self._fetchall_index = 0
        self._closed = False

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchone(self):
        if self._fetchone_index < len(self.fetchone_results):
            result = self.fetchone_results[self._fetchone_index]
            self._fetchone_index += 1
            return result
        return None

    def fetchall(self):
        if self._fetchall_index < len(self.fetchall_results):
            result = self.fetchall_results[self._fetchall_index]
            self._fetchall_index += 1
            return result
        return []

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


class TestReactivationWorkflows(unittest.TestCase):
    """Test reactivation workflow template listing and retrieval."""

    def setUp(self):
        self.cursor = MockCursor()
        self.conn = MockConnection(self.cursor)
        self.service = ReactivationService(lambda: self.conn)

    def test_list_workflows_returns_five_templates(self):
        """Requirement 6.1: 5 workflow templates available."""
        workflows = self.service.list_workflows()
        self.assertEqual(len(workflows), 5)

    def test_list_workflows_contains_expected_ids(self):
        """Requirement 6.1: All expected workflow types present."""
        workflows = self.service.list_workflows()
        ids = [w["workflow_id"] for w in workflows]
        self.assertIn("expired_recovery", ids)
        self.assertIn("inactive_comeback", ids)
        self.assertIn("disconnected_reengagement", ids)
        self.assertIn("speed_upgrade", ids)
        self.assertIn("festive_promotions", ids)

    def test_get_workflow_valid_id(self):
        """Valid workflow ID returns workflow details."""
        wf = self.service.get_workflow("expired_recovery")
        self.assertEqual(wf["workflow_id"], "expired_recovery")
        self.assertEqual(wf["name"], "Expired Plan Recovery")
        self.assertEqual(wf["campaign_type"], "reactivation")
        self.assertIn("status", wf["suggested_segment_filters"])
        self.assertEqual(wf["suggested_segment_filters"]["status"], "expired")

    def test_get_workflow_invalid_id_raises(self):
        """Invalid workflow ID raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            self.service.get_workflow("nonexistent")
        self.assertIn("Unknown workflow_id", str(ctx.exception))

    def test_workflow_has_cooldown_and_success_window(self):
        """Requirement 6.4, 6.5: Each workflow has cooldown and success window."""
        for wf_id in REACTIVATION_WORKFLOWS:
            wf = self.service.get_workflow(wf_id)
            self.assertEqual(wf["cooldown_days"], 7)
            self.assertEqual(wf["success_window_days"], 30)

    def test_workflow_has_suggested_template(self):
        """Requirement 6.2: Each workflow has a suggested template name."""
        for wf_id in REACTIVATION_WORKFLOWS:
            wf = self.service.get_workflow(wf_id)
            self.assertIsNotNone(wf["suggested_template_name"])
            self.assertTrue(len(wf["suggested_template_name"]) > 0)


class TestPrepareWorkflowCampaign(unittest.TestCase):
    """Test pre-populating campaigns from workflows (Requirement 6.2)."""

    def setUp(self):
        self.cursor = MockCursor()
        self.conn = MockConnection(self.cursor)
        self.service = ReactivationService(lambda: self.conn)

    def test_prepare_campaign_creates_draft(self):
        """Automation-generated campaigns are created in draft status."""
        # Mock: segment not found → create new
        self.cursor.fetchone_results = [
            None,  # SELECT segment
            None,  # SELECT template
        ]
        self.cursor.lastrowid = 10

        result = self.service.prepare_campaign_from_workflow(
            "expired_recovery", "operator1"
        )

        self.assertEqual(result["status"], "draft")
        self.assertEqual(result["workflow_id"], "expired_recovery")
        self.assertIn("campaign_id", result)

    def test_prepare_campaign_invalid_workflow_raises(self):
        """Invalid workflow ID raises ValueError."""
        with self.assertRaises(ValueError):
            self.service.prepare_campaign_from_workflow("invalid", "op")


class TestAutomationRulesCRUD(unittest.TestCase):
    """Test automation rules create, read, update, delete."""

    def setUp(self):
        self.cursor = MockCursor()
        self.conn = MockConnection(self.cursor)
        self.service = ReactivationService(lambda: self.conn)

    def test_create_rule_valid_data(self):
        """Creating a rule with valid data succeeds."""
        # Mock the get_automation_rule after create
        self.cursor.fetchone_results = [
            {
                "id": 1,
                "organization_id": 1,
                "tenant_id": 1,
                "name": "Test Rule",
                "trigger_type": "schedule",
                "trigger_config": '{"cron": "0 9 * * 1"}',
                "condition_config": None,
                "action_type": "create_campaign_draft",
                "action_config": '{"campaign_type": "reactivation"}',
                "is_active": 1,
                "last_triggered_at": None,
                "created_by": "operator1",
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
            }
        ]

        data = {
            "name": "Test Rule",
            "trigger_type": "schedule",
            "trigger_config": {"cron": "0 9 * * 1"},
            "action_type": "create_campaign_draft",
            "action_config": {"campaign_type": "reactivation"},
        }
        result = self.service.create_automation_rule(data, "operator1")
        self.assertEqual(result["name"], "Test Rule")
        self.assertEqual(result["trigger_type"], "schedule")
        self.assertEqual(result["action_type"], "create_campaign_draft")

    def test_create_rule_missing_required_field_raises(self):
        """Missing required field raises ValueError."""
        data = {
            "name": "Test",
            "trigger_type": "schedule",
            # Missing trigger_config, action_type, action_config
        }
        with self.assertRaises(ValueError) as ctx:
            self.service.create_automation_rule(data, "op")
        self.assertIn("Missing required field", str(ctx.exception))

    def test_create_rule_invalid_trigger_type_raises(self):
        """Invalid trigger_type raises ValueError."""
        data = {
            "name": "Test",
            "trigger_type": "invalid_type",
            "trigger_config": {},
            "action_type": "create_campaign_draft",
            "action_config": {},
        }
        with self.assertRaises(ValueError) as ctx:
            self.service.create_automation_rule(data, "op")
        self.assertIn("Invalid trigger_type", str(ctx.exception))

    def test_create_rule_invalid_action_type_raises(self):
        """Invalid action_type raises ValueError."""
        data = {
            "name": "Test",
            "trigger_type": "schedule",
            "trigger_config": {},
            "action_type": "send_immediately",
            "action_config": {},
        }
        with self.assertRaises(ValueError) as ctx:
            self.service.create_automation_rule(data, "op")
        self.assertIn("Invalid action_type", str(ctx.exception))

    def test_get_rule_not_found_raises(self):
        """Getting a non-existent rule raises ValueError."""
        self.cursor.fetchone_results = [None]
        with self.assertRaises(ValueError) as ctx:
            self.service.get_automation_rule(999)
        self.assertIn("not found", str(ctx.exception))

    def test_list_rules_pagination(self):
        """List rules returns paginated results."""
        self.cursor.fetchone_results = [{"total": 2}]
        self.cursor.fetchall_results = [
            [
                {
                    "id": 1, "organization_id": 1, "tenant_id": 1,
                    "name": "Rule 1", "trigger_type": "schedule",
                    "trigger_config": '{}', "condition_config": None,
                    "action_type": "create_campaign_draft", "action_config": '{}',
                    "is_active": 1, "last_triggered_at": None,
                    "created_by": "op", "created_at": datetime.now(),
                    "updated_at": datetime.now(),
                },
                {
                    "id": 2, "organization_id": 1, "tenant_id": 1,
                    "name": "Rule 2", "trigger_type": "event",
                    "trigger_config": '{}', "condition_config": None,
                    "action_type": "notify_operator", "action_config": '{}',
                    "is_active": 1, "last_triggered_at": None,
                    "created_by": "op", "created_at": datetime.now(),
                    "updated_at": datetime.now(),
                },
            ]
        ]

        result = self.service.list_automation_rules(page=1, per_page=10)
        self.assertEqual(result["total"], 2)
        self.assertEqual(len(result["rules"]), 2)
        self.assertEqual(result["page"], 1)

    def test_update_rule_not_found_raises(self):
        """Updating a non-existent rule raises ValueError."""
        self.cursor.fetchone_results = [None]
        with self.assertRaises(ValueError):
            self.service.update_automation_rule(999, {"name": "new"}, "op")

    def test_delete_rule_not_found_raises(self):
        """Deleting a non-existent rule raises ValueError."""
        self.cursor.fetchone_results = [None]
        with self.assertRaises(ValueError):
            self.service.delete_automation_rule(999)

    def test_delete_rule_success(self):
        """Deleting an existing rule returns True."""
        self.cursor.fetchone_results = [{"id": 1}]
        result = self.service.delete_automation_rule(1)
        self.assertTrue(result)


class TestAutomationRuleExecution(unittest.TestCase):
    """Test automation rule execution creates drafts requiring approval."""

    def setUp(self):
        self.cursor = MockCursor()
        self.conn = MockConnection(self.cursor)
        self.service = ReactivationService(lambda: self.conn)

    def test_execute_create_campaign_draft_requires_approval(self):
        """
        Requirement 2.5 / Property 4: Automation-generated campaigns
        require operator approval — created as draft.
        """
        # Mock get_automation_rule
        self.cursor.fetchone_results = [
            {
                "id": 1,
                "organization_id": 1,
                "tenant_id": 1,
                "name": "Auto Recovery",
                "trigger_type": "schedule",
                "trigger_config": '{"cron": "0 9 * * 1"}',
                "condition_config": None,
                "action_type": "create_campaign_draft",
                "action_config": json.dumps({
                    "campaign_name": "Weekly Recovery",
                    "campaign_type": "reactivation",
                    "segment_id": 5,
                    "template_id": 3,
                }),
                "is_active": 1,
                "last_triggered_at": None,
                "created_by": "admin",
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
            },
        ]
        self.cursor.lastrowid = 42

        result = self.service.execute_rule(1)

        self.assertEqual(result["action"], "create_campaign_draft")
        self.assertEqual(result["status"], "draft")
        self.assertTrue(result["requires_approval"])
        self.assertEqual(result["campaign_id"], 42)

    def test_execute_notify_operator(self):
        """Executing a notify_operator rule creates a notification."""
        self.cursor.fetchone_results = [
            {
                "id": 2,
                "organization_id": 1,
                "tenant_id": 1,
                "name": "Alert Rule",
                "trigger_type": "threshold",
                "trigger_config": '{"metric": "churn_rate", "threshold": 0.1}',
                "condition_config": None,
                "action_type": "notify_operator",
                "action_config": json.dumps({
                    "title": "High Churn Alert",
                    "severity": "warning",
                    "target_operators": ["admin"],
                }),
                "is_active": 1,
                "last_triggered_at": None,
                "created_by": "admin",
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
            },
        ]
        self.cursor.lastrowid = 7

        result = self.service.execute_rule(2)

        self.assertEqual(result["action"], "notify_operator")
        self.assertEqual(result["notification_id"], 7)
        self.assertEqual(result["severity"], "warning")


class TestReactivationSuccessTracking(unittest.TestCase):
    """Test reactivation success tracking (Requirement 6.5)."""

    def setUp(self):
        self.cursor = MockCursor()
        self.conn = MockConnection(self.cursor)
        self.service = ReactivationService(lambda: self.conn)

    def test_no_campaigns_returns_zero_metrics(self):
        """No reactivation messages returns zero metrics."""
        self.cursor.fetchall_results = [[]]  # No campaign messages

        result = self.service.track_reactivation_success()

        self.assertEqual(result["total_targeted"], 0)
        self.assertEqual(result["reactivated_count"], 0)
        self.assertEqual(result["reactivation_rate"], 0.0)

    def test_tracks_status_change_within_window(self):
        """Requirement 6.5: Track status change from expired/inactive to active."""
        # Mock: 3 reactivation messages sent
        self.cursor.fetchall_results = [
            [
                {"campaign_id": 1, "customer_mobile": "9876543210",
                 "sent_at": datetime.now() - timedelta(days=10),
                 "campaign_name": "Recovery", "created_by": "op"},
                {"campaign_id": 1, "customer_mobile": "9876543211",
                 "sent_at": datetime.now() - timedelta(days=15),
                 "campaign_name": "Recovery", "created_by": "op"},
                {"campaign_id": 1, "customer_mobile": "9876543212",
                 "sent_at": datetime.now() - timedelta(days=20),
                 "campaign_name": "Recovery", "created_by": "op"},
            ],
            # Second fetchall: active status check
            [
                {"mobile": "9876543210", "status": "active"},
                {"mobile": "9876543212", "status": "active"},
            ],
        ]

        result = self.service.track_reactivation_success(days_window=30)

        self.assertEqual(result["total_targeted"], 3)
        self.assertEqual(result["reactivated_count"], 2)
        self.assertAlmostEqual(result["reactivation_rate"], 0.6667, places=3)


class TestValidTriggerTypes(unittest.TestCase):
    """Test that all three trigger types are accepted."""

    def setUp(self):
        self.cursor = MockCursor()
        self.conn = MockConnection(self.cursor)
        self.service = ReactivationService(lambda: self.conn)

    def _make_rule_data(self, trigger_type):
        return {
            "name": f"Rule - {trigger_type}",
            "trigger_type": trigger_type,
            "trigger_config": {"key": "value"},
            "action_type": "create_campaign_draft",
            "action_config": {"campaign_type": "reactivation"},
        }

    def test_schedule_trigger_accepted(self):
        """Schedule trigger type is valid."""
        self.cursor.fetchone_results = [
            {"id": 1, "organization_id": 1, "tenant_id": 1,
             "name": "Rule - schedule", "trigger_type": "schedule",
             "trigger_config": '{"key": "value"}', "condition_config": None,
             "action_type": "create_campaign_draft",
             "action_config": '{"campaign_type": "reactivation"}',
             "is_active": 1, "last_triggered_at": None,
             "created_by": "op", "created_at": datetime.now(),
             "updated_at": datetime.now()}
        ]
        result = self.service.create_automation_rule(
            self._make_rule_data("schedule"), "op"
        )
        self.assertEqual(result["trigger_type"], "schedule")

    def test_event_trigger_accepted(self):
        """Event trigger type is valid."""
        self.cursor.fetchone_results = [
            {"id": 2, "organization_id": 1, "tenant_id": 1,
             "name": "Rule - event", "trigger_type": "event",
             "trigger_config": '{"key": "value"}', "condition_config": None,
             "action_type": "create_campaign_draft",
             "action_config": '{"campaign_type": "reactivation"}',
             "is_active": 1, "last_triggered_at": None,
             "created_by": "op", "created_at": datetime.now(),
             "updated_at": datetime.now()}
        ]
        result = self.service.create_automation_rule(
            self._make_rule_data("event"), "op"
        )
        self.assertEqual(result["trigger_type"], "event")

    def test_threshold_trigger_accepted(self):
        """Threshold trigger type is valid."""
        self.cursor.fetchone_results = [
            {"id": 3, "organization_id": 1, "tenant_id": 1,
             "name": "Rule - threshold", "trigger_type": "threshold",
             "trigger_config": '{"key": "value"}', "condition_config": None,
             "action_type": "create_campaign_draft",
             "action_config": '{"campaign_type": "reactivation"}',
             "is_active": 1, "last_triggered_at": None,
             "created_by": "op", "created_at": datetime.now(),
             "updated_at": datetime.now()}
        ]
        result = self.service.create_automation_rule(
            self._make_rule_data("threshold"), "op"
        )
        self.assertEqual(result["trigger_type"], "threshold")


if __name__ == "__main__":
    unittest.main()
