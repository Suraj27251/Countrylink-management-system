"""
Notification Engine — internal alerting for critical system events.

Provides NotificationEngine class that creates system_notifications records,
delivers in-app persistent notifications, and manages acknowledgment state.

Generates alerts for: campaign_degraded (>10% failures), queue_overloaded
(>10k backlog), webhook_connectivity (5min gap), template_rejected, quality_drop.

Requirements: 25.1, 25.2, 25.3, 25.4, 25.5, 25.6, 25.7
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Alert type constants
# ---------------------------------------------------------------------------
ALERT_CAMPAIGN_DEGRADED = "campaign_degraded"
ALERT_QUEUE_OVERLOADED = "queue_overloaded"
ALERT_WEBHOOK_CONNECTIVITY = "webhook_connectivity"
ALERT_TEMPLATE_REJECTED = "template_rejected"
ALERT_QUALITY_DROP = "quality_drop"

# Severity levels
SEVERITY_INFO = "info"
SEVERITY_WARNING = "warning"
SEVERITY_CRITICAL = "critical"

# Thresholds
CAMPAIGN_DEGRADED_THRESHOLD = 0.10  # >10% failures
QUEUE_OVERLOADED_THRESHOLD = 10000  # >10k backlog
WEBHOOK_GAP_MINUTES = 5  # 5 min gap
QUEUE_THROUGHPUT_DROP_RATIO = 0.50  # below 50% of configured throughput


class NotificationEngine:
    """
    Internal notification system for operator alerting.

    Creates persistent in-app notifications in system_notifications table.
    Supports alert generation, retrieval of unacknowledged alerts, and
    acknowledgment by operators.

    Parameters
    ----------
    get_connection : callable
        A zero-argument function that returns a MySQL connection object.
    """

    def __init__(self, get_connection: Callable):
        self._get_conn = get_connection

    # ------------------------------------------------------------------
    # Core alert creation (Requirement 25.6, 25.7)
    # ------------------------------------------------------------------

    def send_alert(
        self,
        alert_type: str,
        severity: str,
        title: str,
        details: Optional[dict] = None,
        target_operators: Optional[List[str]] = None,
        organization_id: int = 1,
    ) -> int:
        """
        Create a system_notifications record with type, severity, and details.

        Delivers an in-app persistent notification. Browser push and WhatsApp
        delivery are optional and tracked via delivered_push / delivered_whatsapp
        columns.

        Parameters
        ----------
        alert_type : str
            One of: campaign_degraded, queue_overloaded, webhook_connectivity,
            template_rejected, quality_drop
        severity : str
            One of: info, warning, critical
        title : str
            Short human-readable alert title.
        details : dict, optional
            JSON-serializable detail payload (metrics, thresholds, etc.)
        target_operators : list of str, optional
            Operator names to notify. If None, targets all operators.
        organization_id : int
            Tenant scoping (default 1).

        Returns
        -------
        int
            The ID of the newly created notification record.
        """
        conn = self._get_conn()
        try:
            cursor = conn.cursor(dictionary=True)
            sql = """
                INSERT INTO system_notifications
                    (organization_id, alert_type, severity, title, details,
                     target_operators, delivered_push, delivered_whatsapp, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, 0, 0, NOW())
            """
            params = (
                organization_id,
                alert_type,
                severity,
                title,
                json.dumps(details) if details else None,
                json.dumps(target_operators) if target_operators else None,
            )
            cursor.execute(sql, params)
            conn.commit()
            notification_id = cursor.lastrowid
            cursor.close()

            logger.info(
                "Alert created: id=%s type=%s severity=%s title=%s",
                notification_id, alert_type, severity, title,
            )
            return notification_id
        except Exception as exc:
            conn.rollback()
            logger.error("Failed to create alert: %s", exc)
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Retrieve unacknowledged notifications (Requirement 25.6, 25.7)
    # ------------------------------------------------------------------

    def get_unacknowledged(
        self, operator_id: Optional[int] = None, operator_name: Optional[str] = None,
        organization_id: int = 1, limit: int = 50,
    ) -> List[dict]:
        """
        Get all unacknowledged notifications for an operator.

        Filters by target_operators JSON array (if the operator is listed) or
        returns all unacknowledged alerts if target_operators is NULL (broadcast).

        Parameters
        ----------
        operator_id : int, optional
            Legacy operator ID (unused in current schema but reserved).
        operator_name : str, optional
            Operator username to match against target_operators JSON.
        organization_id : int
            Tenant scoping (default 1).
        limit : int
            Maximum records to return (default 50).

        Returns
        -------
        list of dict
            Notification records with id, alert_type, severity, title, details,
            target_operators, created_at.
        """
        conn = self._get_conn()
        try:
            cursor = conn.cursor(dictionary=True)
            sql = """
                SELECT id, organization_id, alert_type, severity, title,
                       details, target_operators, created_at,
                       delivered_push, delivered_whatsapp
                FROM system_notifications
                WHERE organization_id = %s
                  AND acknowledged_at IS NULL
                ORDER BY
                    FIELD(severity, 'critical', 'warning', 'info'),
                    created_at DESC
                LIMIT %s
            """
            cursor.execute(sql, (organization_id, limit))
            rows = cursor.fetchall()
            cursor.close()

            results = []
            for row in rows:
                # Decode JSON fields
                if row.get("details") and isinstance(row["details"], str):
                    row["details"] = json.loads(row["details"])
                if row.get("target_operators") and isinstance(row["target_operators"], str):
                    row["target_operators"] = json.loads(row["target_operators"])

                # Filter by operator if specified
                if operator_name and row.get("target_operators"):
                    if operator_name not in row["target_operators"]:
                        continue

                results.append(row)

            return results
        except Exception as exc:
            logger.error("Failed to get unacknowledged notifications: %s", exc)
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Acknowledge notification (Requirement 25.6)
    # ------------------------------------------------------------------

    def acknowledge(
        self, notification_id: int, operator_name: str
    ) -> bool:
        """
        Mark a notification as acknowledged by an operator.

        Parameters
        ----------
        notification_id : int
            The system_notifications.id to acknowledge.
        operator_name : str
            The operator acknowledging the notification.

        Returns
        -------
        bool
            True if the notification was successfully acknowledged, False if
            already acknowledged or not found.
        """
        conn = self._get_conn()
        try:
            cursor = conn.cursor(dictionary=True)
            # Only acknowledge if not already acknowledged
            sql = """
                UPDATE system_notifications
                SET acknowledged_by = %s,
                    acknowledged_at = NOW()
                WHERE id = %s
                  AND acknowledged_at IS NULL
            """
            cursor.execute(sql, (operator_name, notification_id))
            conn.commit()
            affected = cursor.rowcount
            cursor.close()

            if affected > 0:
                logger.info(
                    "Notification %s acknowledged by %s",
                    notification_id, operator_name,
                )
                return True
            return False
        except Exception as exc:
            conn.rollback()
            logger.error("Failed to acknowledge notification %s: %s", notification_id, exc)
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Alert generators (Requirements 25.1, 25.2, 25.3, 25.4, 25.5)
    # ------------------------------------------------------------------

    def check_campaign_degraded(self, campaign_id: int) -> Optional[int]:
        """
        Check if a campaign has >10% failures and generate alert if so.

        Requirement 25.1: WHEN a campaign send fails for more than 10% of
        recipients, generate a "campaign_degraded" alert.

        Returns notification_id if alert generated, else None.
        """
        conn = self._get_conn()
        try:
            cursor = conn.cursor(dictionary=True)
            sql = """
                SELECT c.name AS campaign_name,
                       c.total_recipients,
                       c.sent_count,
                       c.failed_count
                FROM campaigns c
                WHERE c.id = %s
            """
            cursor.execute(sql, (campaign_id,))
            campaign = cursor.fetchone()
            cursor.close()

            if not campaign:
                return None

            total = campaign["total_recipients"] or 0
            failed = campaign["failed_count"] or 0

            if total == 0:
                return None

            failure_rate = failed / total
            if failure_rate > CAMPAIGN_DEGRADED_THRESHOLD:
                return self.send_alert(
                    alert_type=ALERT_CAMPAIGN_DEGRADED,
                    severity=SEVERITY_CRITICAL,
                    title=f"Campaign '{campaign['campaign_name']}' degraded — {failure_rate:.1%} failure rate",
                    details={
                        "campaign_id": campaign_id,
                        "campaign_name": campaign["campaign_name"],
                        "total_recipients": total,
                        "failed_count": failed,
                        "failure_rate": round(failure_rate, 4),
                        "threshold": CAMPAIGN_DEGRADED_THRESHOLD,
                    },
                )
            return None
        except Exception as exc:
            logger.error("Error checking campaign degraded: %s", exc)
            raise
        finally:
            conn.close()

    def check_queue_overloaded(self, configured_throughput: int = 80) -> Optional[int]:
        """
        Check if queue backlog exceeds 10k pending messages and processing
        rate drops below 50% of configured throughput.

        Requirement 25.2: WHEN the Sending_Queue backlog exceeds 10,000
        pending messages and processing rate drops below 50%, generate
        "queue_overloaded" alert.

        Returns notification_id if alert generated, else None.
        """
        conn = self._get_conn()
        try:
            cursor = conn.cursor(dictionary=True)

            # Count pending messages
            sql_backlog = """
                SELECT COUNT(*) AS backlog
                FROM campaign_messages
                WHERE status IN ('queued', 'sending')
            """
            cursor.execute(sql_backlog)
            backlog_row = cursor.fetchone()
            backlog = backlog_row["backlog"] if backlog_row else 0

            # Calculate recent processing rate (messages sent in last 60 seconds)
            sql_rate = """
                SELECT COUNT(*) AS sent_last_minute
                FROM campaign_messages
                WHERE status IN ('sent', 'delivered', 'read')
                  AND sent_at >= NOW() - INTERVAL 60 SECOND
            """
            cursor.execute(sql_rate)
            rate_row = cursor.fetchone()
            sent_last_minute = rate_row["sent_last_minute"] if rate_row else 0
            cursor.close()

            # Processing rate per second
            current_rate = sent_last_minute / 60.0
            expected_rate = configured_throughput
            rate_ratio = current_rate / expected_rate if expected_rate > 0 else 0

            if backlog > QUEUE_OVERLOADED_THRESHOLD and rate_ratio < QUEUE_THROUGHPUT_DROP_RATIO:
                return self.send_alert(
                    alert_type=ALERT_QUEUE_OVERLOADED,
                    severity=SEVERITY_CRITICAL,
                    title=f"Queue overloaded — {backlog:,} pending messages, throughput at {rate_ratio:.0%}",
                    details={
                        "backlog": backlog,
                        "current_rate_per_sec": round(current_rate, 2),
                        "configured_throughput": configured_throughput,
                        "throughput_ratio": round(rate_ratio, 4),
                        "threshold_backlog": QUEUE_OVERLOADED_THRESHOLD,
                        "threshold_ratio": QUEUE_THROUGHPUT_DROP_RATIO,
                    },
                )
            return None
        except Exception as exc:
            logger.error("Error checking queue overloaded: %s", exc)
            raise
        finally:
            conn.close()

    def check_webhook_connectivity(self) -> Optional[int]:
        """
        Check if no webhook callbacks received for >5 minutes during an
        active campaign send.

        Requirement 25.3: WHEN the system fails to receive Meta webhook
        callbacks for more than 5 minutes during an active campaign send,
        generate "webhook_connectivity" alert.

        Returns notification_id if alert generated, else None.
        """
        conn = self._get_conn()
        try:
            cursor = conn.cursor(dictionary=True)

            # Check if there are active sending campaigns
            sql_active = """
                SELECT COUNT(*) AS active_count
                FROM campaigns
                WHERE status = 'sending'
            """
            cursor.execute(sql_active)
            active_row = cursor.fetchone()
            active_count = active_row["active_count"] if active_row else 0

            if active_count == 0:
                cursor.close()
                return None

            # Check last webhook callback timestamp
            sql_last_webhook = """
                SELECT MAX(updated_at) AS last_callback
                FROM campaign_messages
                WHERE status IN ('delivered', 'read', 'failed')
                  AND updated_at >= NOW() - INTERVAL 1 HOUR
            """
            cursor.execute(sql_last_webhook)
            webhook_row = cursor.fetchone()
            cursor.close()

            last_callback = webhook_row["last_callback"] if webhook_row else None

            if last_callback is None:
                # No callbacks at all in the last hour while campaigns active
                return self.send_alert(
                    alert_type=ALERT_WEBHOOK_CONNECTIVITY,
                    severity=SEVERITY_WARNING,
                    title="Webhook connectivity lost — no callbacks received during active campaign",
                    details={
                        "active_campaigns": active_count,
                        "last_callback": None,
                        "gap_minutes": "unknown (no recent callbacks)",
                        "threshold_minutes": WEBHOOK_GAP_MINUTES,
                    },
                )

            # Check if gap exceeds threshold
            now = datetime.now()
            if isinstance(last_callback, datetime):
                gap = now - last_callback
            else:
                gap = timedelta(minutes=WEBHOOK_GAP_MINUTES + 1)

            gap_minutes = gap.total_seconds() / 60.0

            if gap_minutes > WEBHOOK_GAP_MINUTES:
                return self.send_alert(
                    alert_type=ALERT_WEBHOOK_CONNECTIVITY,
                    severity=SEVERITY_WARNING,
                    title=f"Webhook connectivity gap — {gap_minutes:.1f} min since last callback",
                    details={
                        "active_campaigns": active_count,
                        "last_callback": last_callback.isoformat() if isinstance(last_callback, datetime) else str(last_callback),
                        "gap_minutes": round(gap_minutes, 1),
                        "threshold_minutes": WEBHOOK_GAP_MINUTES,
                    },
                )
            return None
        except Exception as exc:
            logger.error("Error checking webhook connectivity: %s", exc)
            raise
        finally:
            conn.close()

    def alert_template_rejected(
        self, template_name: str, rejection_reason: str
    ) -> int:
        """
        Generate a "template_rejected" alert when Meta rejects a template.

        Requirement 25.4: WHEN a Template submission is rejected by Meta's
        template review process, generate "template_rejected" alert.

        Returns notification_id.
        """
        return self.send_alert(
            alert_type=ALERT_TEMPLATE_REJECTED,
            severity=SEVERITY_WARNING,
            title=f"Template '{template_name}' rejected by Meta",
            details={
                "template_name": template_name,
                "rejection_reason": rejection_reason,
            },
        )

    def alert_quality_drop(
        self, previous_tier: str, current_tier: str, metrics: Optional[dict] = None
    ) -> int:
        """
        Generate a "quality_drop" alert when quality tier drops.

        Requirement 25.5: WHEN the Quality_Monitor detects a quality tier drop,
        generate "quality_drop" alert with metrics and recommended actions.

        Returns notification_id.
        """
        recommendations = []
        if current_tier == "yellow":
            recommendations = [
                "Cooldown interval increased to 120 hours automatically",
                "Review recent campaign failure rates",
                "Consider pausing non-critical campaigns",
            ]
        elif current_tier == "red":
            recommendations = [
                "IMMEDIATE ACTION: Pause all active campaigns",
                "Review blocked user reports",
                "Contact Meta Business support if persistent",
                "Audit recent message content for policy violations",
            ]

        return self.send_alert(
            alert_type=ALERT_QUALITY_DROP,
            severity=SEVERITY_CRITICAL if current_tier == "red" else SEVERITY_WARNING,
            title=f"Quality tier dropped: {previous_tier.upper()} → {current_tier.upper()}",
            details={
                "previous_tier": previous_tier,
                "current_tier": current_tier,
                "metrics": metrics or {},
                "recommendations": recommendations,
            },
        )

    # ------------------------------------------------------------------
    # Notification count (utility)
    # ------------------------------------------------------------------

    def get_unacknowledged_count(self, organization_id: int = 1) -> int:
        """
        Get count of unacknowledged notifications for badge display.

        Returns
        -------
        int
            Number of unacknowledged notifications.
        """
        conn = self._get_conn()
        try:
            cursor = conn.cursor(dictionary=True)
            sql = """
                SELECT COUNT(*) AS cnt
                FROM system_notifications
                WHERE organization_id = %s
                  AND acknowledged_at IS NULL
            """
            cursor.execute(sql, (organization_id,))
            row = cursor.fetchone()
            cursor.close()
            return row["cnt"] if row else 0
        except Exception as exc:
            logger.error("Failed to get notification count: %s", exc)
            raise
        finally:
            conn.close()
