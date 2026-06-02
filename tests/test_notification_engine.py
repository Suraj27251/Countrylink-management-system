"""
Unit tests for NotificationEngine — alert creation, retrieval,
acknowledgment, and alert generators.

Requirements: 25.1, 25.2, 25.3, 25.4, 25.5, 25.6, 25.7
"""

import json
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from services.notification_engine import (
    ALERT_CAMPAIGN_DEGRADED,
    ALERT_QUEUE_OVERLOADED,
    ALERT_QUALITY_DROP,
    ALERT_TEMPLATE_REJECTED,
    ALERT_WEBHOOK_CONNECTIVITY,
    CAMPAIGN_DEGRADED_THRESHOLD,
    QUEUE_OVERLOADED_THRESHOLD,
    SEVERITY_CRITICAL,
    SEVERITY_INFO,
    SEVERITY_WARNING,
    NotificationEngine,
)


class MockCursor:
    """Mock MySQL cursor with dictionary=True support."""

    def __init__(self):
        self.executed = []
        self.fetchone_results = []
        self.fetchall_results = []
        self._fetchone_idx = 0
        self._fetchall_idx = 0
        self._closed = False
        self.rowcount = 0
        self.lastrowid = 1

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchone(self):
        if self._fetchone_idx < len(self.fetchone_results):
            result = self.fetchone_results[self._fetchone_idx]
            self._fetchone_idx += 1
            return result
        return None

    def fetchall(self):
        if self._fetchall_idx < len(self.fetchall_results):
            result = self.fetchall_results[self._fetchall_idx]
            self._fetchall_idx += 1
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
        self._connected = True

    def cursor(self, dictionary=False):
        return self._cursor

    def commit(self):
        self._committed = True

    def rollback(self):
        self._rolled_back = True

    def close(self):
        self._connected = False


class TestSendAlert(unittest.TestCase):
    """Tests for NotificationEngine.send_alert()."""

    def setUp(self):
        self.cursor = MockCursor()
        self.cursor.lastrowid = 42
        self.conn = MockConnection(self.cursor)
        self.engine = NotificationEngine(lambda: self.conn)

    def test_send_alert_creates_record(self):
        """send_alert inserts a record into system_notifications."""
        result = self.engine.send_alert(
            alert_type=ALERT_CAMPAIGN_DEGRADED,
            severity=SEVERITY_CRITICAL,
            title="Campaign degraded",
            details={"campaign_id": 1, "failure_rate": 0.15},
            target_operators=["admin", "operator1"],
        )

        self.assertEqual(result, 42)
        self.assertTrue(self.conn._committed)
        self.assertEqual(len(self.cursor.executed), 1)

        sql, params = self.cursor.executed[0]
        self.assertIn("INSERT INTO system_notifications", sql)
        self.assertEqual(params[0], 1)  # organization_id
        self.assertEqual(params[1], ALERT_CAMPAIGN_DEGRADED)
        self.assertEqual(params[2], SEVERITY_CRITICAL)
        self.assertEqual(params[3], "Campaign degraded")
        # details JSON
        self.assertEqual(json.loads(params[4]), {"campaign_id": 1, "failure_rate": 0.15})
        # target_operators JSON
        self.assertEqual(json.loads(params[5]), ["admin", "operator1"])

    def test_send_alert_with_none_details(self):
        """send_alert handles None details and target_operators."""
        result = self.engine.send_alert(
            alert_type=ALERT_TEMPLATE_REJECTED,
            severity=SEVERITY_WARNING,
            title="Template rejected",
        )

        self.assertEqual(result, 42)
        sql, params = self.cursor.executed[0]
        self.assertIsNone(params[4])  # details
        self.assertIsNone(params[5])  # target_operators

    def test_send_alert_custom_org_id(self):
        """send_alert uses provided organization_id."""
        self.engine.send_alert(
            alert_type=ALERT_QUALITY_DROP,
            severity=SEVERITY_WARNING,
            title="Quality drop",
            organization_id=5,
        )

        sql, params = self.cursor.executed[0]
        self.assertEqual(params[0], 5)


class TestGetUnacknowledged(unittest.TestCase):
    """Tests for NotificationEngine.get_unacknowledged()."""

    def setUp(self):
        self.cursor = MockCursor()
        self.conn = MockConnection(self.cursor)
        self.engine = NotificationEngine(lambda: self.conn)

    def test_get_unacknowledged_returns_all_when_no_operator(self):
        """get_unacknowledged returns all unacked alerts without operator filter."""
        self.cursor.fetchall_results = [[
            {
                "id": 1,
                "organization_id": 1,
                "alert_type": ALERT_CAMPAIGN_DEGRADED,
                "severity": SEVERITY_CRITICAL,
                "title": "Campaign degraded",
                "details": json.dumps({"campaign_id": 1}),
                "target_operators": None,
                "created_at": datetime(2024, 1, 15, 10, 0, 0),
                "delivered_push": 0,
                "delivered_whatsapp": 0,
            },
        ]]

        results = self.engine.get_unacknowledged()

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], 1)
        self.assertEqual(results[0]["details"], {"campaign_id": 1})

    def test_get_unacknowledged_filters_by_operator(self):
        """get_unacknowledged filters by operator_name."""
        self.cursor.fetchall_results = [[
            {
                "id": 1,
                "organization_id": 1,
                "alert_type": ALERT_CAMPAIGN_DEGRADED,
                "severity": SEVERITY_CRITICAL,
                "title": "Alert 1",
                "details": None,
                "target_operators": json.dumps(["admin", "op1"]),
                "created_at": datetime(2024, 1, 15, 10, 0, 0),
                "delivered_push": 0,
                "delivered_whatsapp": 0,
            },
            {
                "id": 2,
                "organization_id": 1,
                "alert_type": ALERT_QUEUE_OVERLOADED,
                "severity": SEVERITY_WARNING,
                "title": "Alert 2",
                "details": None,
                "target_operators": json.dumps(["op2"]),
                "created_at": datetime(2024, 1, 15, 11, 0, 0),
                "delivered_push": 0,
                "delivered_whatsapp": 0,
            },
        ]]

        results = self.engine.get_unacknowledged(operator_name="admin")

        # Only alert 1 targets "admin"
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], 1)

    def test_get_unacknowledged_includes_broadcast(self):
        """get_unacknowledged includes alerts with NULL target_operators (broadcast)."""
        self.cursor.fetchall_results = [[
            {
                "id": 1,
                "organization_id": 1,
                "alert_type": ALERT_WEBHOOK_CONNECTIVITY,
                "severity": SEVERITY_WARNING,
                "title": "Broadcast alert",
                "details": None,
                "target_operators": None,
                "created_at": datetime(2024, 1, 15, 10, 0, 0),
                "delivered_push": 0,
                "delivered_whatsapp": 0,
            },
        ]]

        results = self.engine.get_unacknowledged(operator_name="anyone")

        # Broadcast alerts (NULL target_operators) should be included
        self.assertEqual(len(results), 1)


class TestAcknowledge(unittest.TestCase):
    """Tests for NotificationEngine.acknowledge()."""

    def setUp(self):
        self.cursor = MockCursor()
        self.conn = MockConnection(self.cursor)
        self.engine = NotificationEngine(lambda: self.conn)

    def test_acknowledge_success(self):
        """acknowledge returns True and commits when row updated."""
        self.cursor.rowcount = 1

        result = self.engine.acknowledge(42, "admin")

        self.assertTrue(result)
        self.assertTrue(self.conn._committed)
        sql, params = self.cursor.executed[0]
        self.assertIn("UPDATE system_notifications", sql)
        self.assertEqual(params, ("admin", 42))

    def test_acknowledge_not_found(self):
        """acknowledge returns False when no row affected."""
        self.cursor.rowcount = 0

        result = self.engine.acknowledge(999, "admin")

        self.assertFalse(result)


class TestCheckCampaignDegraded(unittest.TestCase):
    """Tests for campaign_degraded alert generation (Requirement 25.1)."""

    def setUp(self):
        self.cursor = MockCursor()
        self.cursor.lastrowid = 10
        self.conn = MockConnection(self.cursor)
        self.engine = NotificationEngine(lambda: self.conn)

    def test_alert_generated_when_failure_exceeds_threshold(self):
        """Alert generated when failure rate > 10%."""
        # First call: check_campaign_degraded reads campaign
        # Second call: send_alert inserts notification
        self.cursor.fetchone_results = [
            {"campaign_name": "Test Campaign", "total_recipients": 100, "sent_count": 100, "failed_count": 15},
        ]

        result = self.engine.check_campaign_degraded(1)

        self.assertIsNotNone(result)
        # The send_alert INSERT should have been executed
        self.assertTrue(len(self.cursor.executed) >= 2)

    def test_no_alert_when_below_threshold(self):
        """No alert when failure rate <= 10%."""
        self.cursor.fetchone_results = [
            {"campaign_name": "OK Campaign", "total_recipients": 100, "sent_count": 100, "failed_count": 5},
        ]

        result = self.engine.check_campaign_degraded(1)

        self.assertIsNone(result)

    def test_no_alert_for_nonexistent_campaign(self):
        """No alert when campaign not found."""
        self.cursor.fetchone_results = [None]

        result = self.engine.check_campaign_degraded(999)

        self.assertIsNone(result)

    def test_no_alert_for_zero_recipients(self):
        """No alert when total_recipients is 0 (avoid division by zero)."""
        self.cursor.fetchone_results = [
            {"campaign_name": "Empty", "total_recipients": 0, "sent_count": 0, "failed_count": 0},
        ]

        result = self.engine.check_campaign_degraded(1)

        self.assertIsNone(result)


class TestCheckQueueOverloaded(unittest.TestCase):
    """Tests for queue_overloaded alert generation (Requirement 25.2)."""

    def setUp(self):
        self.cursor = MockCursor()
        self.cursor.lastrowid = 20
        self.conn = MockConnection(self.cursor)
        self.engine = NotificationEngine(lambda: self.conn)

    def test_alert_generated_when_backlog_and_low_throughput(self):
        """Alert when backlog > 10k AND throughput < 50%."""
        self.cursor.fetchone_results = [
            {"backlog": 15000},   # > 10k
            {"sent_last_minute": 20},  # 20/60 = 0.33/sec, ratio = 0.33/80 ≈ 0.004 < 0.5
        ]

        result = self.engine.check_queue_overloaded(configured_throughput=80)

        self.assertIsNotNone(result)

    def test_no_alert_when_backlog_below_threshold(self):
        """No alert when backlog <= 10k."""
        self.cursor.fetchone_results = [
            {"backlog": 5000},    # <= 10k
            {"sent_last_minute": 10},
        ]

        result = self.engine.check_queue_overloaded()

        self.assertIsNone(result)

    def test_no_alert_when_throughput_healthy(self):
        """No alert when throughput >= 50% even with high backlog."""
        self.cursor.fetchone_results = [
            {"backlog": 15000},   # > 10k
            {"sent_last_minute": 3000},  # 3000/60 = 50/sec, ratio = 50/80 = 0.625 > 0.5
        ]

        result = self.engine.check_queue_overloaded(configured_throughput=80)

        self.assertIsNone(result)


class TestCheckWebhookConnectivity(unittest.TestCase):
    """Tests for webhook_connectivity alert generation (Requirement 25.3)."""

    def setUp(self):
        self.cursor = MockCursor()
        self.cursor.lastrowid = 30
        self.conn = MockConnection(self.cursor)
        self.engine = NotificationEngine(lambda: self.conn)

    def test_no_alert_when_no_active_campaigns(self):
        """No alert when no campaigns in sending state."""
        self.cursor.fetchone_results = [
            {"active_count": 0},
        ]

        result = self.engine.check_webhook_connectivity()

        self.assertIsNone(result)

    def test_alert_when_no_callbacks_and_active_campaign(self):
        """Alert when no callbacks in last hour with active campaigns."""
        self.cursor.fetchone_results = [
            {"active_count": 2},
            {"last_callback": None},
        ]

        result = self.engine.check_webhook_connectivity()

        self.assertIsNotNone(result)

    def test_alert_when_gap_exceeds_threshold(self):
        """Alert when last callback was >5 minutes ago."""
        old_time = datetime.now() - timedelta(minutes=10)
        self.cursor.fetchone_results = [
            {"active_count": 1},
            {"last_callback": old_time},
        ]

        result = self.engine.check_webhook_connectivity()

        self.assertIsNotNone(result)

    def test_no_alert_when_recent_callback(self):
        """No alert when last callback was recent (< 5 minutes)."""
        recent_time = datetime.now() - timedelta(minutes=2)
        self.cursor.fetchone_results = [
            {"active_count": 1},
            {"last_callback": recent_time},
        ]

        result = self.engine.check_webhook_connectivity()

        self.assertIsNone(result)


class TestAlertTemplateRejected(unittest.TestCase):
    """Tests for template_rejected alert generation (Requirement 25.4)."""

    def setUp(self):
        self.cursor = MockCursor()
        self.cursor.lastrowid = 40
        self.conn = MockConnection(self.cursor)
        self.engine = NotificationEngine(lambda: self.conn)

    def test_generates_template_rejected_alert(self):
        """alert_template_rejected creates correct notification."""
        result = self.engine.alert_template_rejected(
            template_name="renewal_reminder",
            rejection_reason="Promotional content in utility template",
        )

        self.assertEqual(result, 40)
        sql, params = self.cursor.executed[0]
        self.assertIn("INSERT INTO system_notifications", sql)
        self.assertEqual(params[1], ALERT_TEMPLATE_REJECTED)
        self.assertEqual(params[2], SEVERITY_WARNING)
        details = json.loads(params[4])
        self.assertEqual(details["template_name"], "renewal_reminder")
        self.assertEqual(details["rejection_reason"], "Promotional content in utility template")


class TestAlertQualityDrop(unittest.TestCase):
    """Tests for quality_drop alert generation (Requirement 25.5)."""

    def setUp(self):
        self.cursor = MockCursor()
        self.cursor.lastrowid = 50
        self.conn = MockConnection(self.cursor)
        self.engine = NotificationEngine(lambda: self.conn)

    def test_yellow_tier_drop(self):
        """quality_drop alert for Green→Yellow has WARNING severity."""
        result = self.engine.alert_quality_drop(
            previous_tier="green",
            current_tier="yellow",
            metrics={"failure_rate": 0.06},
        )

        self.assertEqual(result, 50)
        sql, params = self.cursor.executed[0]
        self.assertEqual(params[2], SEVERITY_WARNING)
        details = json.loads(params[4])
        self.assertEqual(details["previous_tier"], "green")
        self.assertEqual(details["current_tier"], "yellow")
        self.assertIn("recommendations", details)
        self.assertTrue(len(details["recommendations"]) > 0)

    def test_red_tier_drop(self):
        """quality_drop alert for Yellow→Red has CRITICAL severity."""
        result = self.engine.alert_quality_drop(
            previous_tier="yellow",
            current_tier="red",
            metrics={"failure_rate": 0.12, "blocked_count": 25},
        )

        self.assertEqual(result, 50)
        sql, params = self.cursor.executed[0]
        self.assertEqual(params[2], SEVERITY_CRITICAL)
        details = json.loads(params[4])
        self.assertEqual(details["current_tier"], "red")
        self.assertIn("IMMEDIATE ACTION", details["recommendations"][0])


class TestGetUnacknowledgedCount(unittest.TestCase):
    """Tests for get_unacknowledged_count utility."""

    def setUp(self):
        self.cursor = MockCursor()
        self.conn = MockConnection(self.cursor)
        self.engine = NotificationEngine(lambda: self.conn)

    def test_returns_count(self):
        """get_unacknowledged_count returns correct count."""
        self.cursor.fetchone_results = [{"cnt": 7}]

        count = self.engine.get_unacknowledged_count()

        self.assertEqual(count, 7)

    def test_returns_zero_when_none(self):
        """get_unacknowledged_count returns 0 when no results."""
        self.cursor.fetchone_results = [None]

        count = self.engine.get_unacknowledged_count()

        self.assertEqual(count, 0)


if __name__ == "__main__":
    unittest.main()
