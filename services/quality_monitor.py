"""
Quality Monitor for WhatsApp Business API health tracking.

Tracks and aggregates quality indicators including blocked users, message
failure rate, opt-out rate, and read rate. Determines quality tier (GREEN,
YELLOW, RED) and generates alerts when thresholds are exceeded. Integrates
with the CooldownManager to enforce adaptive cooldown periods.

Requirements: 18.1, 18.2, 18.3, 18.4, 18.5, 18.6, 18.7
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)


class QualityTier(str, Enum):
    """WhatsApp Business API quality tier levels."""
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


@dataclass
class QualityMetrics:
    """Aggregated quality metrics for a given time window."""
    period_hours: int
    period_start: datetime
    period_end: datetime
    blocked_count: int = 0
    failure_rate: float = 0.0
    opt_out_rate: float = 0.0
    read_rate: float = 0.0
    total_sent: int = 0
    total_failed: int = 0
    total_opt_outs: int = 0
    total_read: int = 0


@dataclass
class Alert:
    """A quality alert generated when thresholds are exceeded."""
    alert_type: str
    severity: str
    title: str
    details: dict = field(default_factory=dict)
    created_at: Optional[datetime] = None


@dataclass
class QualityDashboard:
    """Data for the quality monitoring dashboard."""
    current_tier: QualityTier
    metrics_24h: QualityMetrics
    metrics_7d: QualityMetrics
    active_alerts: List[Alert] = field(default_factory=list)


class QualityMonitor:
    """
    Monitors WhatsApp Business API quality indicators and manages quality tier.

    Aggregates metrics per rolling 24-hour and 7-day periods:
    - Blocked user count
    - Message send failure rate
    - Opt-out rate
    - Read rate (engagement proxy)

    Generates alerts when thresholds are exceeded:
    - failure_rate > 5% over 24h → quality_warning alert
    - blocked_count increase > 10 in 24h → block_spike alert

    Determines quality tier (GREEN/YELLOW/RED) based on combined metrics.
    When tier drops to YELLOW, the CooldownManager minimum interval increases
    from 72h to 120h.

    Args:
        get_connection: Callable that returns a MySQL connection.
    """

    # Threshold constants
    FAILURE_RATE_WARNING_THRESHOLD = 0.05  # 5% failure rate triggers warning
    BLOCK_SPIKE_THRESHOLD = 10  # >10 blocks in 24h triggers block_spike

    # Quality tier thresholds
    # RED: failure_rate > 10% OR blocked_count > 20/day
    # YELLOW: failure_rate > 5% OR blocked_count > 10/day OR opt_out_rate > 3%
    # GREEN: all metrics within acceptable bounds
    RED_FAILURE_RATE = 0.10
    RED_BLOCK_COUNT = 20
    YELLOW_FAILURE_RATE = 0.05
    YELLOW_BLOCK_COUNT = 10
    YELLOW_OPT_OUT_RATE = 0.03

    def __init__(self, get_connection: Callable):
        """
        Initialize the Quality Monitor.

        Args:
            get_connection: Callable returning an active mysql.connector connection.
        """
        self._get_connection = get_connection

    def compute_metrics(self, period_hours: int = 24) -> QualityMetrics:
        """
        Aggregate quality metrics for a rolling time window.

        Computes blocked_count, failure_rate, opt_out_rate, and read_rate
        by querying campaign_messages and suppression_list tables.

        Args:
            period_hours: Number of hours to look back (default 24h).

        Returns:
            QualityMetrics with aggregated values for the period.
        """
        now = datetime.now()
        period_start = now - timedelta(hours=period_hours)

        conn = self._get_connection()
        cursor = conn.cursor(dictionary=True)
        try:
            # Get message counts for the period
            cursor.execute(
                """
                SELECT
                    COUNT(*) AS total_sent,
                    SUM(CASE WHEN status = 'failed' OR status = 'permanently_failed'
                        THEN 1 ELSE 0 END) AS total_failed,
                    SUM(CASE WHEN status = 'read' THEN 1 ELSE 0 END) AS total_read
                FROM campaign_messages
                WHERE sent_at >= %s
                  AND sent_at <= %s
                  AND status != 'queued'
                  AND status != 'skipped'
                """,
                (period_start, now),
            )
            msg_row = cursor.fetchone()

            total_sent = msg_row["total_sent"] if msg_row and msg_row["total_sent"] else 0
            total_failed = msg_row["total_failed"] if msg_row and msg_row["total_failed"] else 0
            total_read = msg_row["total_read"] if msg_row and msg_row["total_read"] else 0

            # Get blocked count (suppression entries with reason 'user_blocked'
            # created in this period)
            cursor.execute(
                """
                SELECT COUNT(*) AS blocked_count
                FROM suppression_list
                WHERE reason = 'user_blocked'
                  AND is_active = 1
                  AND created_at >= %s
                  AND created_at <= %s
                """,
                (period_start, now),
            )
            block_row = cursor.fetchone()
            blocked_count = block_row["blocked_count"] if block_row and block_row["blocked_count"] else 0

            # Get opt-out count for the period
            cursor.execute(
                """
                SELECT COUNT(*) AS opt_out_count
                FROM suppression_list
                WHERE reason = 'opt_out_keyword'
                  AND is_active = 1
                  AND created_at >= %s
                  AND created_at <= %s
                """,
                (period_start, now),
            )
            opt_row = cursor.fetchone()
            total_opt_outs = opt_row["opt_out_count"] if opt_row and opt_row["opt_out_count"] else 0

            # Compute rates (avoid division by zero)
            failure_rate = total_failed / total_sent if total_sent > 0 else 0.0
            read_rate = total_read / total_sent if total_sent > 0 else 0.0
            opt_out_rate = total_opt_outs / total_sent if total_sent > 0 else 0.0

            return QualityMetrics(
                period_hours=period_hours,
                period_start=period_start,
                period_end=now,
                blocked_count=blocked_count,
                failure_rate=failure_rate,
                opt_out_rate=opt_out_rate,
                read_rate=read_rate,
                total_sent=total_sent,
                total_failed=total_failed,
                total_opt_outs=total_opt_outs,
                total_read=total_read,
            )
        finally:
            cursor.close()

    def get_quality_tier(self) -> QualityTier:
        """
        Determine the current quality tier based on 24-hour metrics.

        Tier determination:
        - RED: failure_rate > 10% OR blocked_count > 20/day
        - YELLOW: failure_rate > 5% OR blocked_count > 10/day OR opt_out_rate > 3%
        - GREEN: all metrics within acceptable bounds

        Returns:
            QualityTier enum value (GREEN, YELLOW, or RED).
        """
        metrics = self.compute_metrics(period_hours=24)

        # Check RED thresholds first (most severe)
        if metrics.failure_rate > self.RED_FAILURE_RATE:
            return QualityTier.RED
        if metrics.blocked_count > self.RED_BLOCK_COUNT:
            return QualityTier.RED

        # Check YELLOW thresholds
        if metrics.failure_rate > self.YELLOW_FAILURE_RATE:
            return QualityTier.YELLOW
        if metrics.blocked_count > self.YELLOW_BLOCK_COUNT:
            return QualityTier.YELLOW
        if metrics.opt_out_rate > self.YELLOW_OPT_OUT_RATE:
            return QualityTier.YELLOW

        # All metrics within acceptable bounds
        return QualityTier.GREEN

    def check_alerts(self) -> List[Alert]:
        """
        Check current metrics and generate alerts when thresholds are exceeded.

        Generates:
        - "quality_warning" when failure_rate > 5% over 24h (Req 18.2)
        - "block_spike" when blocked_count > 10 in 24h (Req 18.3)

        Returns:
            List of Alert objects for any exceeded thresholds.
        """
        metrics = self.compute_metrics(period_hours=24)
        alerts: List[Alert] = []
        now = datetime.now()

        # Check failure rate threshold (Req 18.2)
        if metrics.failure_rate > self.FAILURE_RATE_WARNING_THRESHOLD:
            alerts.append(Alert(
                alert_type="quality_warning",
                severity="warning",
                title=(
                    f"Message failure rate {metrics.failure_rate:.1%} "
                    f"exceeds 5% threshold"
                ),
                details={
                    "failure_rate": round(metrics.failure_rate, 4),
                    "total_sent": metrics.total_sent,
                    "total_failed": metrics.total_failed,
                    "period_hours": 24,
                    "recommendation": "Review recent campaign errors and consider pausing active campaigns",
                },
                created_at=now,
            ))

        # Check block spike threshold (Req 18.3)
        if metrics.blocked_count > self.BLOCK_SPIKE_THRESHOLD:
            alerts.append(Alert(
                alert_type="block_spike",
                severity="critical",
                title=(
                    f"Blocked user count {metrics.blocked_count} "
                    f"exceeds 10/day threshold"
                ),
                details={
                    "blocked_count": metrics.blocked_count,
                    "period_hours": 24,
                    "recommendation": "Pause active campaigns and review messaging content",
                },
                created_at=now,
            ))

        # Persist alerts to system_notifications if any generated
        if alerts:
            self._persist_alerts(alerts)

        return alerts

    def record_block(self, mobile: str) -> None:
        """
        Record that a customer has blocked the WhatsApp Business number.

        Adds the customer to a permanent suppression list excluded from
        all future campaigns (Req 18.7).

        Args:
            mobile: Customer mobile number that blocked the business number.
        """
        conn = self._get_connection()
        cursor = conn.cursor(dictionary=True)
        try:
            # Check if already in suppression list with user_blocked reason
            cursor.execute(
                """
                SELECT id, is_active FROM suppression_list
                WHERE customer_mobile = %s AND reason = 'user_blocked'
                LIMIT 1
                """,
                (mobile,),
            )
            existing = cursor.fetchone()

            if existing and existing.get("is_active"):
                # Already blocked — no action needed
                logger.info(
                    "Customer %s already in blocked suppression list, skipping.",
                    mobile,
                )
                return
            elif existing and not existing.get("is_active"):
                # Re-activate existing blocked record
                cursor.execute(
                    """
                    UPDATE suppression_list
                    SET is_active = 1, removed_at = NULL,
                        created_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                    """,
                    (existing["id"],),
                )
            else:
                # Insert new suppression record
                cursor.execute(
                    """
                    INSERT INTO suppression_list
                        (organization_id, customer_mobile, reason, added_by,
                         is_active, created_at)
                    VALUES (1, %s, 'user_blocked', 'quality_monitor', 1,
                            CURRENT_TIMESTAMP)
                    """,
                    (mobile,),
                )

            conn.commit()
            logger.info(
                "Customer %s added to permanent suppression (user_blocked).",
                mobile,
            )

            # Record activity in customer_activity
            try:
                cursor.execute(
                    """
                    INSERT INTO customer_activity
                        (customer_mobile, activity_type, channel, details,
                         created_at)
                    VALUES (%s, 'status_change', 'whatsapp',
                            '{"action": "blocked_business_number"}',
                            CURRENT_TIMESTAMP)
                    """,
                    (mobile,),
                )
                conn.commit()
            except Exception:
                logger.warning(
                    "Failed to record block activity for %s", mobile
                )

        except Exception:
            if conn:
                try:
                    conn.rollback()
                except Exception:
                    pass
            logger.exception(
                "Failed to record block for mobile=%s", mobile
            )
            raise
        finally:
            cursor.close()

    def log_hourly_metrics(self) -> None:
        """
        Log quality metric snapshots to the quality_metrics table.

        Called hourly to record metrics for historical trend analysis (Req 18.6).
        Computes metrics for both 24h and 7d windows and stores them.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            # Compute and log 24h metrics
            metrics_24h = self.compute_metrics(period_hours=24)
            tier = self.get_quality_tier()

            cursor.execute(
                """
                INSERT INTO quality_metrics
                    (period_start, period_end, blocked_count, failure_rate,
                     opt_out_rate, read_rate, quality_tier, computed_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                """,
                (
                    metrics_24h.period_start,
                    metrics_24h.period_end,
                    metrics_24h.blocked_count,
                    round(metrics_24h.failure_rate, 4),
                    round(metrics_24h.opt_out_rate, 4),
                    round(metrics_24h.read_rate, 4),
                    tier.value,
                ),
            )
            conn.commit()
            logger.info(
                "Logged hourly quality metrics: tier=%s, failure_rate=%.2f%%, "
                "blocked=%d, opt_out_rate=%.2f%%, read_rate=%.2f%%",
                tier.value,
                metrics_24h.failure_rate * 100,
                metrics_24h.blocked_count,
                metrics_24h.opt_out_rate * 100,
                metrics_24h.read_rate * 100,
            )
        except Exception:
            if conn:
                try:
                    conn.rollback()
                except Exception:
                    pass
            logger.exception("Failed to log hourly quality metrics")
            raise
        finally:
            cursor.close()

    def get_adaptive_cooldown_hours(self) -> int:
        """
        Return the current cooldown window based on quality tier.

        When quality tier is YELLOW, increase cooldown from 72h to 120h (Req 18.5).
        When tier is RED, also use 120h.
        When tier is GREEN, use the standard 72h.

        Returns:
            Cooldown window in hours (72 or 120).
        """
        tier = self.get_quality_tier()
        if tier in (QualityTier.YELLOW, QualityTier.RED):
            return 120
        return 72

    def get_dashboard_data(self) -> QualityDashboard:
        """
        Compile all quality monitoring data for dashboard display (Req 18.4).

        Returns:
            QualityDashboard with current tier, 24h/7d metrics, and alerts.
        """
        tier = self.get_quality_tier()
        metrics_24h = self.compute_metrics(period_hours=24)
        metrics_7d = self.compute_metrics(period_hours=168)  # 7 days
        alerts = self.check_alerts()

        return QualityDashboard(
            current_tier=tier,
            metrics_24h=metrics_24h,
            metrics_7d=metrics_7d,
            active_alerts=alerts,
        )

    def _persist_alerts(self, alerts: List[Alert]) -> None:
        """
        Persist generated alerts to the system_notifications table.

        Args:
            alerts: List of alerts to persist.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            for alert in alerts:
                import json
                details_json = json.dumps(alert.details)
                cursor.execute(
                    """
                    INSERT INTO system_notifications
                        (organization_id, alert_type, severity, title, details,
                         created_at)
                    VALUES (1, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                    """,
                    (
                        alert.alert_type,
                        alert.severity,
                        alert.title,
                        details_json,
                    ),
                )
            conn.commit()
            logger.info("Persisted %d quality alerts to system_notifications", len(alerts))
        except Exception:
            if conn:
                try:
                    conn.rollback()
                except Exception:
                    pass
            logger.warning("Failed to persist quality alerts")
        finally:
            cursor.close()
