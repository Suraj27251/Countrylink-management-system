"""
Unit tests for QualityMonitor — quality tier determination, metric aggregation,
alert generation, block recording, and adaptive cooldown integration.
"""

import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from services.quality_monitor import (
    Alert,
    QualityDashboard,
    QualityMetrics,
    QualityMonitor,
    QualityTier,
)


class MockCursor:
    """Mock MySQL cursor with dictionary=True support."""

    def __init__(self):
        self.executed = []
        self.fetchone_results = []
        self._call_idx = 0
        self._closed = False
        self.rowcount = 1

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
        self._rolled_back = False

    def cursor(self, dictionary=False):
        return self._cursor

    def commit(self):
        self._committed = True

    def rollback(self):
        self._rolled_back = True

    def close(self):
        pass

    def is_connected(self):
        return True


class TestQualityMonitorComputeMetrics(unittest.TestCase):
    """Tests for compute_metrics() aggregation logic."""

    def _make_monitor_with_metrics(
        self, total_sent=100, total_failed=5, total_read=60,
        blocked_count=3, opt_out_count=2
    ):
        """Helper to create a QualityMonitor with specific metrics returned."""
        cursor = MockCursor()
        # Query 1: campaign_messages aggregation
        cursor.fetchone_results.append({
            "total_sent": total_sent,
            "total_failed": total_failed,
            "total_read": total_read,
        })
        # Query 2: blocked count
        cursor.fetchone_results.append({"blocked_count": blocked_count})
        # Query 3: opt-out count
        cursor.fetchone_results.append({"opt_out_count": opt_out_count})

        conn = MockConnection(cursor)
        monitor = QualityMonitor(lambda: conn)
        return monitor, cursor

    def test_compute_metrics_calculates_rates(self):
        """compute_metrics should calculate failure, read, opt-out rates."""
        monitor, _ = self._make_monitor_with_metrics(
            total_sent=200, total_failed=10, total_read=120,
            blocked_count=5, opt_out_count=4
        )
        metrics = monitor.compute_metrics(period_hours=24)

        self.assertEqual(metrics.period_hours, 24)
        self.assertEqual(metrics.total_sent, 200)
        self.assertEqual(metrics.total_failed, 10)
        self.assertEqual(metrics.total_read, 120)
        self.assertEqual(metrics.blocked_count, 5)
        self.assertEqual(metrics.total_opt_outs, 4)
        self.assertAlmostEqual(metrics.failure_rate, 0.05, places=4)
        self.assertAlmostEqual(metrics.read_rate, 0.60, places=4)
        self.assertAlmostEqual(metrics.opt_out_rate, 0.02, places=4)

    def test_compute_metrics_zero_sent(self):
        """compute_metrics should handle zero messages sent gracefully."""
        monitor, _ = self._make_monitor_with_metrics(
            total_sent=0, total_failed=0, total_read=0,
            blocked_count=0, opt_out_count=0
        )
        metrics = monitor.compute_metrics(period_hours=24)

        self.assertEqual(metrics.failure_rate, 0.0)
        self.assertEqual(metrics.read_rate, 0.0)
        self.assertEqual(metrics.opt_out_rate, 0.0)

    def test_compute_metrics_null_values(self):
        """compute_metrics should handle None/null database values."""
        cursor = MockCursor()
        cursor.fetchone_results = [
            {"total_sent": None, "total_failed": None, "total_read": None},
            {"blocked_count": None},
            {"opt_out_count": None},
        ]
        conn = MockConnection(cursor)
        monitor = QualityMonitor(lambda: conn)

        metrics = monitor.compute_metrics(period_hours=24)
        self.assertEqual(metrics.total_sent, 0)
        self.assertEqual(metrics.failure_rate, 0.0)

    def test_compute_metrics_7d_window(self):
        """compute_metrics should support 7-day (168h) window."""
        monitor, cursor = self._make_monitor_with_metrics(
            total_sent=1000, total_failed=50, total_read=700,
            blocked_count=15, opt_out_count=10
        )
        metrics = monitor.compute_metrics(period_hours=168)

        self.assertEqual(metrics.period_hours, 168)
        # Verify query uses the 168h period
        sql, params = cursor.executed[0]
        self.assertIn("campaign_messages", sql)
        # period_start should be ~168h ago
        period_start = params[0]
        expected_start = datetime.now() - timedelta(hours=168)
        self.assertAlmostEqual(
            period_start.timestamp(), expected_start.timestamp(), delta=2
        )

    def test_compute_metrics_closes_cursor(self):
        """compute_metrics should close cursor after query."""
        monitor, cursor = self._make_monitor_with_metrics()
        monitor.compute_metrics(period_hours=24)
        self.assertTrue(cursor._closed)


class TestQualityTierDetermination(unittest.TestCase):
    """Tests for get_quality_tier() tier classification."""

    def _make_monitor_for_tier(
        self, failure_rate=0.0, blocked_count=0, opt_out_rate=0.0
    ):
        """Helper to create monitor that returns specific metric values."""
        # We need to calculate total_sent and total_failed from failure_rate
        total_sent = 1000
        total_failed = int(failure_rate * total_sent)
        total_read = int(0.5 * total_sent)
        opt_out_count = int(opt_out_rate * total_sent)

        cursor = MockCursor()
        cursor.fetchone_results = [
            {"total_sent": total_sent, "total_failed": total_failed,
             "total_read": total_read},
            {"blocked_count": blocked_count},
            {"opt_out_count": opt_out_count},
        ]
        conn = MockConnection(cursor)
        return QualityMonitor(lambda: conn)

    def test_green_tier_all_good(self):
        """GREEN when all metrics within thresholds."""
        monitor = self._make_monitor_for_tier(
            failure_rate=0.02, blocked_count=3, opt_out_rate=0.01
        )
        self.assertEqual(monitor.get_quality_tier(), QualityTier.GREEN)

    def test_yellow_tier_high_failure_rate(self):
        """YELLOW when failure_rate > 5% but <= 10%."""
        monitor = self._make_monitor_for_tier(
            failure_rate=0.06, blocked_count=3, opt_out_rate=0.01
        )
        self.assertEqual(monitor.get_quality_tier(), QualityTier.YELLOW)

    def test_yellow_tier_high_block_count(self):
        """YELLOW when blocked_count > 10 but <= 20."""
        monitor = self._make_monitor_for_tier(
            failure_rate=0.02, blocked_count=15, opt_out_rate=0.01
        )
        self.assertEqual(monitor.get_quality_tier(), QualityTier.YELLOW)

    def test_yellow_tier_high_opt_out_rate(self):
        """YELLOW when opt_out_rate > 3%."""
        monitor = self._make_monitor_for_tier(
            failure_rate=0.02, blocked_count=3, opt_out_rate=0.04
        )
        self.assertEqual(monitor.get_quality_tier(), QualityTier.YELLOW)

    def test_red_tier_very_high_failure_rate(self):
        """RED when failure_rate > 10%."""
        monitor = self._make_monitor_for_tier(
            failure_rate=0.12, blocked_count=3, opt_out_rate=0.01
        )
        self.assertEqual(monitor.get_quality_tier(), QualityTier.RED)

    def test_red_tier_very_high_block_count(self):
        """RED when blocked_count > 20."""
        monitor = self._make_monitor_for_tier(
            failure_rate=0.02, blocked_count=25, opt_out_rate=0.01
        )
        self.assertEqual(monitor.get_quality_tier(), QualityTier.RED)

    def test_boundary_exactly_at_yellow_failure(self):
        """Exactly 5% failure_rate should NOT trigger YELLOW (> not >=)."""
        monitor = self._make_monitor_for_tier(
            failure_rate=0.05, blocked_count=3, opt_out_rate=0.01
        )
        self.assertEqual(monitor.get_quality_tier(), QualityTier.GREEN)

    def test_boundary_exactly_at_red_failure(self):
        """Exactly 10% failure_rate should NOT trigger RED (> not >=)."""
        monitor = self._make_monitor_for_tier(
            failure_rate=0.10, blocked_count=3, opt_out_rate=0.01
        )
        # 10% is not > 10%, so should be YELLOW (since > 5%)
        self.assertEqual(monitor.get_quality_tier(), QualityTier.YELLOW)


class TestCheckAlerts(unittest.TestCase):
    """Tests for check_alerts() alert generation."""

    def _make_monitor_for_alerts(
        self, failure_rate=0.0, blocked_count=0, total_sent=1000
    ):
        """Helper to create monitor with specific metric values for alert testing."""
        total_failed = int(failure_rate * total_sent)
        cursor = MockCursor()
        # compute_metrics queries (called by check_alerts)
        cursor.fetchone_results = [
            {"total_sent": total_sent, "total_failed": total_failed,
             "total_read": 500},
            {"blocked_count": blocked_count},
            {"opt_out_count": 5},
        ]
        conn = MockConnection(cursor)
        return QualityMonitor(lambda: conn)

    def test_no_alerts_when_below_thresholds(self):
        """No alerts generated when all metrics are within thresholds."""
        monitor = self._make_monitor_for_alerts(
            failure_rate=0.03, blocked_count=5
        )
        alerts = monitor.check_alerts()
        self.assertEqual(len(alerts), 0)

    def test_quality_warning_when_failure_exceeds_5pct(self):
        """quality_warning alert generated when failure_rate > 5%."""
        monitor = self._make_monitor_for_alerts(
            failure_rate=0.06, blocked_count=5
        )
        alerts = monitor.check_alerts()

        warning_alerts = [a for a in alerts if a.alert_type == "quality_warning"]
        self.assertEqual(len(warning_alerts), 1)
        self.assertEqual(warning_alerts[0].severity, "warning")
        self.assertIn("failure rate", warning_alerts[0].title.lower())

    def test_block_spike_when_blocks_exceed_10(self):
        """block_spike alert generated when blocked_count > 10."""
        monitor = self._make_monitor_for_alerts(
            failure_rate=0.02, blocked_count=15
        )
        alerts = monitor.check_alerts()

        spike_alerts = [a for a in alerts if a.alert_type == "block_spike"]
        self.assertEqual(len(spike_alerts), 1)
        self.assertEqual(spike_alerts[0].severity, "critical")
        self.assertIn("blocked", spike_alerts[0].title.lower())

    def test_both_alerts_when_both_thresholds_exceeded(self):
        """Both alerts generated when both thresholds exceeded."""
        monitor = self._make_monitor_for_alerts(
            failure_rate=0.08, blocked_count=12
        )
        alerts = monitor.check_alerts()

        alert_types = {a.alert_type for a in alerts}
        self.assertIn("quality_warning", alert_types)
        self.assertIn("block_spike", alert_types)

    def test_alert_details_contain_metrics(self):
        """Alert details should contain relevant metric values."""
        monitor = self._make_monitor_for_alerts(
            failure_rate=0.07, blocked_count=5
        )
        alerts = monitor.check_alerts()

        warning = [a for a in alerts if a.alert_type == "quality_warning"][0]
        self.assertIn("failure_rate", warning.details)
        self.assertIn("total_sent", warning.details)
        self.assertIn("recommendation", warning.details)

    def test_boundary_exactly_5pct_no_alert(self):
        """Exactly 5% failure_rate should NOT trigger alert (> not >=)."""
        monitor = self._make_monitor_for_alerts(
            failure_rate=0.05, blocked_count=5
        )
        alerts = monitor.check_alerts()
        warning_alerts = [a for a in alerts if a.alert_type == "quality_warning"]
        self.assertEqual(len(warning_alerts), 0)

    def test_boundary_exactly_10_blocks_no_alert(self):
        """Exactly 10 blocked should NOT trigger alert (> not >=)."""
        monitor = self._make_monitor_for_alerts(
            failure_rate=0.02, blocked_count=10
        )
        alerts = monitor.check_alerts()
        spike_alerts = [a for a in alerts if a.alert_type == "block_spike"]
        self.assertEqual(len(spike_alerts), 0)


class TestRecordBlock(unittest.TestCase):
    """Tests for record_block() suppression list management."""

    def test_record_block_inserts_new_suppression(self):
        """record_block should INSERT into suppression_list for new numbers."""
        cursor = MockCursor()
        # First query: check existing — not found
        cursor.fetchone_results = [None]
        conn = MockConnection(cursor)
        monitor = QualityMonitor(lambda: conn)

        monitor.record_block("919876543210")

        # Should have queries: SELECT check, INSERT, commit, INSERT activity
        insert_queries = [
            q for q, _ in cursor.executed if "INSERT INTO suppression_list" in q
        ]
        self.assertEqual(len(insert_queries), 1)
        self.assertIn("user_blocked", insert_queries[0])

    def test_record_block_skips_already_active(self):
        """record_block should skip if already blocked (active)."""
        cursor = MockCursor()
        # Already active blocked entry
        cursor.fetchone_results = [{"id": 1, "is_active": 1}]
        conn = MockConnection(cursor)
        monitor = QualityMonitor(lambda: conn)

        monitor.record_block("919876543210")

        # Should only have the SELECT query (no INSERT/UPDATE)
        insert_queries = [
            q for q, _ in cursor.executed if "INSERT INTO suppression_list" in q
        ]
        update_queries = [
            q for q, _ in cursor.executed if "UPDATE suppression_list" in q
        ]
        self.assertEqual(len(insert_queries), 0)
        self.assertEqual(len(update_queries), 0)

    def test_record_block_reactivates_inactive(self):
        """record_block should reactivate an inactive blocked entry."""
        cursor = MockCursor()
        # Existing but inactive
        cursor.fetchone_results = [{"id": 42, "is_active": 0}]
        conn = MockConnection(cursor)
        monitor = QualityMonitor(lambda: conn)

        monitor.record_block("919876543210")

        update_queries = [
            (q, p) for q, p in cursor.executed if "UPDATE suppression_list" in q
        ]
        self.assertEqual(len(update_queries), 1)
        self.assertIn("is_active = 1", update_queries[0][0])

    def test_record_block_commits(self):
        """record_block should commit the transaction."""
        cursor = MockCursor()
        cursor.fetchone_results = [None]
        conn = MockConnection(cursor)
        monitor = QualityMonitor(lambda: conn)

        monitor.record_block("919876543210")
        self.assertTrue(conn._committed)


class TestLogHourlyMetrics(unittest.TestCase):
    """Tests for log_hourly_metrics() periodic logging."""

    def test_log_hourly_inserts_to_quality_metrics(self):
        """log_hourly_metrics should INSERT into quality_metrics table."""
        cursor = MockCursor()
        # compute_metrics queries (called internally)
        cursor.fetchone_results = [
            {"total_sent": 500, "total_failed": 20, "total_read": 300},
            {"blocked_count": 5},
            {"opt_out_count": 3},
            # get_quality_tier also calls compute_metrics
            {"total_sent": 500, "total_failed": 20, "total_read": 300},
            {"blocked_count": 5},
            {"opt_out_count": 3},
        ]
        conn = MockConnection(cursor)
        monitor = QualityMonitor(lambda: conn)

        monitor.log_hourly_metrics()

        insert_queries = [
            q for q, _ in cursor.executed if "INSERT INTO quality_metrics" in q
        ]
        self.assertEqual(len(insert_queries), 1)
        self.assertTrue(conn._committed)

    def test_log_hourly_includes_tier(self):
        """log_hourly_metrics should include the current quality tier."""
        cursor = MockCursor()
        # High failure rate → YELLOW tier
        cursor.fetchone_results = [
            {"total_sent": 100, "total_failed": 7, "total_read": 50},
            {"blocked_count": 3},
            {"opt_out_count": 1},
            # Second call for get_quality_tier
            {"total_sent": 100, "total_failed": 7, "total_read": 50},
            {"blocked_count": 3},
            {"opt_out_count": 1},
        ]
        conn = MockConnection(cursor)
        monitor = QualityMonitor(lambda: conn)

        monitor.log_hourly_metrics()

        insert_queries = [
            (q, p) for q, p in cursor.executed
            if "INSERT INTO quality_metrics" in q
        ]
        self.assertEqual(len(insert_queries), 1)
        _, params = insert_queries[0]
        # quality_tier is the 7th param (index 6)
        self.assertEqual(params[6], "yellow")


class TestAdaptiveCooldown(unittest.TestCase):
    """Tests for get_adaptive_cooldown_hours() integration."""

    def _make_monitor_with_tier(self, tier_metrics):
        """Helper to create monitor returning specific tier."""
        cursor = MockCursor()
        cursor.fetchone_results = [
            {"total_sent": tier_metrics["total_sent"],
             "total_failed": tier_metrics["total_failed"],
             "total_read": tier_metrics.get("total_read", 500)},
            {"blocked_count": tier_metrics.get("blocked_count", 0)},
            {"opt_out_count": tier_metrics.get("opt_out_count", 0)},
        ]
        conn = MockConnection(cursor)
        return QualityMonitor(lambda: conn)

    def test_green_tier_returns_72h(self):
        """GREEN tier should return standard 72h cooldown."""
        monitor = self._make_monitor_with_tier({
            "total_sent": 1000, "total_failed": 20,  # 2% failure
            "blocked_count": 3, "opt_out_count": 5,
        })
        self.assertEqual(monitor.get_adaptive_cooldown_hours(), 72)

    def test_yellow_tier_returns_120h(self):
        """YELLOW tier should return extended 120h cooldown."""
        monitor = self._make_monitor_with_tier({
            "total_sent": 1000, "total_failed": 60,  # 6% failure → YELLOW
            "blocked_count": 3, "opt_out_count": 5,
        })
        self.assertEqual(monitor.get_adaptive_cooldown_hours(), 120)

    def test_red_tier_returns_120h(self):
        """RED tier should also return 120h cooldown."""
        monitor = self._make_monitor_with_tier({
            "total_sent": 1000, "total_failed": 120,  # 12% failure → RED
            "blocked_count": 3, "opt_out_count": 5,
        })
        self.assertEqual(monitor.get_adaptive_cooldown_hours(), 120)


class TestGetDashboardData(unittest.TestCase):
    """Tests for get_dashboard_data() dashboard compilation."""

    def test_dashboard_returns_complete_data(self):
        """get_dashboard_data should return tier, 24h/7d metrics, and alerts."""
        cursor = MockCursor()
        # get_quality_tier → compute_metrics(24h)
        cursor.fetchone_results = [
            {"total_sent": 1000, "total_failed": 20, "total_read": 600},
            {"blocked_count": 3},
            {"opt_out_count": 5},
            # compute_metrics(24h) for dashboard
            {"total_sent": 1000, "total_failed": 20, "total_read": 600},
            {"blocked_count": 3},
            {"opt_out_count": 5},
            # compute_metrics(168h) for 7d
            {"total_sent": 5000, "total_failed": 100, "total_read": 3000},
            {"blocked_count": 15},
            {"opt_out_count": 20},
            # check_alerts → compute_metrics(24h)
            {"total_sent": 1000, "total_failed": 20, "total_read": 600},
            {"blocked_count": 3},
            {"opt_out_count": 5},
        ]
        conn = MockConnection(cursor)
        monitor = QualityMonitor(lambda: conn)

        dashboard = monitor.get_dashboard_data()

        self.assertIsInstance(dashboard, QualityDashboard)
        self.assertEqual(dashboard.current_tier, QualityTier.GREEN)
        self.assertIsInstance(dashboard.metrics_24h, QualityMetrics)
        self.assertIsInstance(dashboard.metrics_7d, QualityMetrics)
        self.assertIsInstance(dashboard.active_alerts, list)


class TestQualityMetricsDataclass(unittest.TestCase):
    """Tests for QualityMetrics dataclass."""

    def test_defaults(self):
        """QualityMetrics should have sensible defaults."""
        metrics = QualityMetrics(
            period_hours=24,
            period_start=datetime.now(),
            period_end=datetime.now(),
        )
        self.assertEqual(metrics.blocked_count, 0)
        self.assertEqual(metrics.failure_rate, 0.0)
        self.assertEqual(metrics.opt_out_rate, 0.0)
        self.assertEqual(metrics.read_rate, 0.0)

    def test_all_fields(self):
        """QualityMetrics should store all field values."""
        now = datetime.now()
        metrics = QualityMetrics(
            period_hours=24,
            period_start=now - timedelta(hours=24),
            period_end=now,
            blocked_count=5,
            failure_rate=0.03,
            opt_out_rate=0.01,
            read_rate=0.65,
            total_sent=1000,
            total_failed=30,
            total_opt_outs=10,
            total_read=650,
        )
        self.assertEqual(metrics.blocked_count, 5)
        self.assertAlmostEqual(metrics.failure_rate, 0.03)


class TestQualityTierEnum(unittest.TestCase):
    """Tests for QualityTier enum."""

    def test_enum_values(self):
        """QualityTier should have green, yellow, red values."""
        self.assertEqual(QualityTier.GREEN.value, "green")
        self.assertEqual(QualityTier.YELLOW.value, "yellow")
        self.assertEqual(QualityTier.RED.value, "red")

    def test_enum_string_comparison(self):
        """QualityTier should be comparable to strings."""
        self.assertEqual(QualityTier.GREEN, "green")
        self.assertEqual(QualityTier.YELLOW, "yellow")


if __name__ == "__main__":
    unittest.main()
