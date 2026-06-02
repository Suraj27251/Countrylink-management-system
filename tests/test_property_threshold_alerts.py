"""
Property-based tests for threshold-triggered alert correctness using Hypothesis.

**Validates: Requirements 18.2, 18.3, 21.7, 25.1**

Property 26: Threshold-triggered alert correctness
- For any campaign where failure_rate > 10%, the Notification_Engine SHALL
  generate a "campaign_degraded" alert.
- For any rolling 24-hour window where blocked_count increase > 10, a
  "block_spike" alert SHALL be generated.
- For any active campaign where suppression_rate > 20%, an automatic pause
  SHALL be triggered.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from services.notification_engine import (
    NotificationEngine,
    CAMPAIGN_DEGRADED_THRESHOLD,
    ALERT_CAMPAIGN_DEGRADED,
)
from services.quality_monitor import QualityMonitor, QualityMetrics, Alert
from services.retry_categorizer import RetryCategorizerService


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------
# Failure rate > 10% (just above threshold)
failure_rates_above_threshold = st.floats(
    min_value=0.101, max_value=1.0, allow_nan=False, allow_infinity=False
)
# Failure rate <= 10% (at or below threshold)
failure_rates_below_threshold = st.floats(
    min_value=0.0, max_value=0.10, allow_nan=False, allow_infinity=False
)

# Blocked count > 10 (above block_spike threshold)
blocked_counts_above_threshold = st.integers(min_value=11, max_value=1000)
# Blocked count <= 10 (at or below threshold)
blocked_counts_at_or_below_threshold = st.integers(min_value=0, max_value=10)

# Suppression rate > 20% (above pause threshold)
suppression_rates_above_threshold = st.floats(
    min_value=0.201, max_value=1.0, allow_nan=False, allow_infinity=False
)
# Suppression rate <= 20% (at or below threshold)
suppression_rates_below_threshold = st.floats(
    min_value=0.0, max_value=0.20, allow_nan=False, allow_infinity=False
)

# Campaign IDs and total recipients
campaign_ids = st.integers(min_value=1, max_value=100000)
total_recipients = st.integers(min_value=10, max_value=100000)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------
class MockCursor:
    """Mock MySQL cursor with dictionary=True support."""

    def __init__(self):
        self.executed = []
        self.fetchone_result = None
        self.fetchall_result = []
        self.lastrowid = 1
        self._closed = False
        self.rowcount = 0
        self._fetchone_queue = []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        # If there's a queued result, pop it for the next fetchone
        if self._fetchone_queue:
            self.fetchone_result = self._fetchone_queue.pop(0)

    def fetchone(self):
        return self.fetchone_result

    def fetchall(self):
        return self.fetchall_result

    def close(self):
        self._closed = True

    @property
    def description(self):
        return None


class MockConnection:
    """Mock MySQL connection."""

    def __init__(self, cursor_instance=None):
        self._cursor = cursor_instance or MockCursor()
        self._committed = False
        self._rolled_back = False
        self._closed = False

    def cursor(self, dictionary=False):
        return self._cursor

    def commit(self):
        self._committed = True

    def rollback(self):
        self._rolled_back = True

    def close(self):
        self._closed = True

    def is_connected(self):
        return True


# ---------------------------------------------------------------------------
# Property Tests
# ---------------------------------------------------------------------------
class TestCampaignDegradedAlert:
    """
    Property tests for campaign_degraded alert generation.

    Requirement 25.1: WHEN a campaign send fails for more than 10% of
    recipients, THE Notification_Engine SHALL generate a "campaign_degraded"
    alert.
    """

    @given(
        campaign_id=campaign_ids,
        total=total_recipients,
        failure_rate=failure_rates_above_threshold,
    )
    @settings(max_examples=200)
    def test_failure_rate_above_10_percent_triggers_campaign_degraded(
        self, campaign_id: int, total: int, failure_rate: float
    ):
        """
        Property: For any campaign where failure_rate > 10%, the
        Notification_Engine generates a "campaign_degraded" alert.

        **Validates: Requirements 25.1**
        """
        failed = int(total * failure_rate)
        # Ensure failed count produces a rate > threshold
        assume(failed > 0)
        assume(failed / total > CAMPAIGN_DEGRADED_THRESHOLD)

        # Set up mock that returns campaign data with high failure rate
        cursor = MockCursor()
        cursor.fetchone_result = {
            "campaign_name": f"Test Campaign {campaign_id}",
            "total_recipients": total,
            "sent_count": total,
            "failed_count": failed,
        }

        # Track if send_alert is called with the right type
        alerts_generated = []

        class TrackingNotificationEngine(NotificationEngine):
            def send_alert(self, alert_type, severity, title, details=None,
                           target_operators=None, organization_id=1):
                alerts_generated.append({
                    "alert_type": alert_type,
                    "severity": severity,
                    "title": title,
                    "details": details,
                })
                return 1

        conn = MockConnection(cursor)
        engine = TrackingNotificationEngine(lambda: conn)

        result = engine.check_campaign_degraded(campaign_id)

        # Alert must be generated
        assert result is not None, (
            f"Expected campaign_degraded alert for failure_rate={failure_rate:.2%} "
            f"(failed={failed}, total={total}), but no alert was generated"
        )
        assert len(alerts_generated) == 1
        assert alerts_generated[0]["alert_type"] == ALERT_CAMPAIGN_DEGRADED
        assert alerts_generated[0]["severity"] == "critical"

    @given(
        campaign_id=campaign_ids,
        total=total_recipients,
        failure_rate=failure_rates_below_threshold,
    )
    @settings(max_examples=200)
    def test_failure_rate_at_or_below_10_percent_does_not_trigger_alert(
        self, campaign_id: int, total: int, failure_rate: float
    ):
        """
        Property: For any campaign where failure_rate <= 10%, the
        Notification_Engine does NOT generate a "campaign_degraded" alert.

        **Validates: Requirements 25.1**
        """
        failed = int(total * failure_rate)
        # Make sure computed rate is at or below threshold
        actual_rate = failed / total if total > 0 else 0
        assume(actual_rate <= CAMPAIGN_DEGRADED_THRESHOLD)

        cursor = MockCursor()
        cursor.fetchone_result = {
            "campaign_name": f"Test Campaign {campaign_id}",
            "total_recipients": total,
            "sent_count": total,
            "failed_count": failed,
        }

        alerts_generated = []

        class TrackingNotificationEngine(NotificationEngine):
            def send_alert(self, alert_type, severity, title, details=None,
                           target_operators=None, organization_id=1):
                alerts_generated.append({"alert_type": alert_type})
                return 1

        conn = MockConnection(cursor)
        engine = TrackingNotificationEngine(lambda: conn)

        result = engine.check_campaign_degraded(campaign_id)

        # No alert should be generated
        assert result is None, (
            f"Unexpected campaign_degraded alert for failure_rate={actual_rate:.2%} "
            f"(failed={failed}, total={total})"
        )
        assert len(alerts_generated) == 0


class TestBlockSpikeAlert:
    """
    Property tests for block_spike alert generation.

    Requirement 18.3: WHEN the blocked user count increases by more than 10
    within a 24-hour period, THE Quality_Monitor SHALL generate a "block_spike"
    alert.
    """

    @given(blocked_count=blocked_counts_above_threshold)
    @settings(max_examples=200)
    def test_blocked_count_above_10_triggers_block_spike(self, blocked_count: int):
        """
        Property: For any 24-hour window where blocked_count > 10, a
        "block_spike" alert is generated by check_alerts().

        **Validates: Requirements 18.3**
        """
        # Mock the compute_metrics to return metrics with high blocked_count
        # but failure_rate below its threshold to isolate block_spike behavior
        from unittest.mock import patch, MagicMock
        from datetime import datetime, timedelta

        now = datetime.now()
        mock_metrics = QualityMetrics(
            period_hours=24,
            period_start=now - timedelta(hours=24),
            period_end=now,
            blocked_count=blocked_count,
            failure_rate=0.01,  # Below quality_warning threshold
            opt_out_rate=0.01,
            read_rate=0.5,
            total_sent=1000,
            total_failed=10,
            total_opt_outs=10,
            total_read=500,
        )

        conn = MockConnection()
        monitor = QualityMonitor(lambda: conn)

        with patch.object(monitor, 'compute_metrics', return_value=mock_metrics):
            with patch.object(monitor, '_persist_alerts'):
                alerts = monitor.check_alerts()

        # Must have at least one block_spike alert
        block_spike_alerts = [a for a in alerts if a.alert_type == "block_spike"]
        assert len(block_spike_alerts) >= 1, (
            f"Expected block_spike alert for blocked_count={blocked_count} (>10), "
            f"but got alerts: {[a.alert_type for a in alerts]}"
        )
        # Verify severity is critical
        assert block_spike_alerts[0].severity == "critical"

    @given(blocked_count=blocked_counts_at_or_below_threshold)
    @settings(max_examples=200)
    def test_blocked_count_at_or_below_10_does_not_trigger_block_spike(
        self, blocked_count: int
    ):
        """
        Property: For any 24-hour window where blocked_count <= 10, no
        "block_spike" alert is generated.

        **Validates: Requirements 18.3**
        """
        from unittest.mock import patch
        from datetime import datetime, timedelta

        now = datetime.now()
        mock_metrics = QualityMetrics(
            period_hours=24,
            period_start=now - timedelta(hours=24),
            period_end=now,
            blocked_count=blocked_count,
            failure_rate=0.01,  # Below quality_warning threshold
            opt_out_rate=0.01,
            read_rate=0.5,
            total_sent=1000,
            total_failed=10,
            total_opt_outs=10,
            total_read=500,
        )

        conn = MockConnection()
        monitor = QualityMonitor(lambda: conn)

        with patch.object(monitor, 'compute_metrics', return_value=mock_metrics):
            with patch.object(monitor, '_persist_alerts'):
                alerts = monitor.check_alerts()

        # Must NOT have any block_spike alert
        block_spike_alerts = [a for a in alerts if a.alert_type == "block_spike"]
        assert len(block_spike_alerts) == 0, (
            f"Unexpected block_spike alert for blocked_count={blocked_count} (<=10)"
        )


class TestSuppressionRatePause:
    """
    Property tests for automatic campaign pause on suppression rate > 20%.

    Requirement 21.7: WHEN more than 20% of messages in an active campaign
    fail with "suppression" classification, THE Retry_Categorizer SHALL
    trigger an automatic campaign pause and generate a critical alert.
    """

    @given(
        campaign_id=campaign_ids,
        total_messages=st.integers(min_value=10, max_value=10000),
        suppression_rate=suppression_rates_above_threshold,
    )
    @settings(max_examples=200)
    def test_suppression_rate_above_20_percent_triggers_pause(
        self, campaign_id: int, total_messages: int, suppression_rate: float
    ):
        """
        Property: For any active campaign where suppression_rate > 20%,
        the campaign is paused and a critical alert is generated.

        **Validates: Requirements 21.7**
        """
        suppression_count = int(total_messages * suppression_rate)
        # Ensure the integer division still yields > 20%
        assume(suppression_count > 0)
        assume(suppression_count / total_messages > 0.20)

        # Mock cursor that returns message counts showing high suppression
        cursor = MockCursor()
        cursor._fetchone_queue = [
            {
                "total_messages": total_messages,
                "suppression_count": suppression_count,
            }
        ]
        # rowcount > 0 indicates campaign was in 'sending' state and got paused
        cursor.rowcount = 1

        conn = MockConnection(cursor)
        service = RetryCategorizerService(lambda: conn)

        # Call the internal suppression rate check
        service._check_suppression_rate(campaign_id)

        # Verify that UPDATE campaigns SET status = 'paused' was executed
        update_queries = [
            (sql, params) for sql, params in cursor.executed
            if "UPDATE campaigns" in sql and "'paused'" in sql
        ]
        assert len(update_queries) >= 1, (
            f"Expected campaign pause for suppression_rate="
            f"{suppression_count}/{total_messages}={suppression_rate:.2%} (>20%), "
            f"but no UPDATE campaigns to 'paused' was executed. "
            f"Queries: {[sql[:80] for sql, _ in cursor.executed]}"
        )

        # Verify that a critical alert was inserted into system_notifications
        alert_queries = [
            (sql, params) for sql, params in cursor.executed
            if "INSERT INTO system_notifications" in sql
        ]
        assert len(alert_queries) >= 1, (
            f"Expected critical alert for suppression_rate={suppression_rate:.2%}, "
            f"but no INSERT INTO system_notifications was executed"
        )

    @given(
        campaign_id=campaign_ids,
        total_messages=st.integers(min_value=10, max_value=10000),
        suppression_rate=suppression_rates_below_threshold,
    )
    @settings(max_examples=200)
    def test_suppression_rate_at_or_below_20_percent_does_not_pause(
        self, campaign_id: int, total_messages: int, suppression_rate: float
    ):
        """
        Property: For any campaign where suppression_rate <= 20%, no
        automatic pause is triggered.

        **Validates: Requirements 21.7**
        """
        suppression_count = int(total_messages * suppression_rate)
        actual_rate = suppression_count / total_messages if total_messages > 0 else 0
        assume(actual_rate <= 0.20)

        cursor = MockCursor()
        cursor._fetchone_queue = [
            {
                "total_messages": total_messages,
                "suppression_count": suppression_count,
            }
        ]

        conn = MockConnection(cursor)
        service = RetryCategorizerService(lambda: conn)

        service._check_suppression_rate(campaign_id)

        # Verify that NO campaign pause was executed
        update_queries = [
            (sql, params) for sql, params in cursor.executed
            if "UPDATE campaigns" in sql and "'paused'" in sql
        ]
        assert len(update_queries) == 0, (
            f"Unexpected campaign pause for suppression_rate="
            f"{suppression_count}/{total_messages}={actual_rate:.2%} (<=20%)"
        )


class TestQualityWarningAlert:
    """
    Property tests for quality_warning alert on failure_rate > 5%.

    Requirement 18.2: WHEN the message failure rate exceeds 5% over a rolling
    24-hour period, THE Quality_Monitor SHALL generate a "quality_warning"
    alert.
    """

    @given(
        failure_rate=st.floats(
            min_value=0.051, max_value=1.0,
            allow_nan=False, allow_infinity=False
        )
    )
    @settings(max_examples=200)
    def test_failure_rate_above_5_percent_triggers_quality_warning(
        self, failure_rate: float
    ):
        """
        Property: For any 24-hour window where failure_rate > 5%, a
        "quality_warning" alert is generated by check_alerts().

        **Validates: Requirements 18.2**
        """
        from unittest.mock import patch
        from datetime import datetime, timedelta

        total_sent = 1000
        total_failed = int(total_sent * failure_rate)
        assume(total_failed > 0)
        assume(total_failed / total_sent > 0.05)

        now = datetime.now()
        mock_metrics = QualityMetrics(
            period_hours=24,
            period_start=now - timedelta(hours=24),
            period_end=now,
            blocked_count=0,
            failure_rate=failure_rate,
            opt_out_rate=0.01,
            read_rate=0.5,
            total_sent=total_sent,
            total_failed=total_failed,
            total_opt_outs=10,
            total_read=500,
        )

        conn = MockConnection()
        monitor = QualityMonitor(lambda: conn)

        with patch.object(monitor, 'compute_metrics', return_value=mock_metrics):
            with patch.object(monitor, '_persist_alerts'):
                alerts = monitor.check_alerts()

        # Must have a quality_warning alert
        quality_warning_alerts = [a for a in alerts if a.alert_type == "quality_warning"]
        assert len(quality_warning_alerts) >= 1, (
            f"Expected quality_warning alert for failure_rate={failure_rate:.2%} (>5%), "
            f"but got alerts: {[a.alert_type for a in alerts]}"
        )
