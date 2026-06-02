"""
Recovery Manager Service for Enterprise WhatsApp CRM.

Responsible for persisting queue state and automatically resuming pending
campaign messages after server restart or crash recovery. Uses idempotency
keys to prevent duplicate sends during recovery.

Requirements: 26.1, 26.2, 26.3, 26.4, 26.5, 26.6, 26.7
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, List

logger = logging.getLogger(__name__)


@dataclass
class RecoveryReport:
    """Report of a recovery operation."""
    messages_requeued: int = 0
    duplicates_prevented: int = 0
    campaigns_resumed: int = 0
    stale_messages_reset: int = 0
    total_recovery_time_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)


class RecoveryManager:
    """
    Manages campaign queue recovery on application startup.

    Queries campaign_messages for records in 'queued' or 'sending' status,
    identifies stale messages, uses idempotency keys to prevent duplicates,
    and re-enqueues messages for processing.

    Parameters
    ----------
    get_connection : callable
        A zero-argument function that returns a MySQL connection object.
    stale_minutes : int, optional
        Number of minutes after which a 'sending' message is considered stale
        (default 5).
    """

    def __init__(self, get_connection: Callable, stale_minutes: int = 5):
        self._get_conn = get_connection
        self._stale_minutes = stale_minutes

    def recover_on_startup(self) -> RecoveryReport:
        """
        Perform full recovery on application startup.

        1. Identify stale messages (in 'sending' status for > stale_minutes)
           and reset them to 'queued'.
        2. Resume campaigns that were in 'sending' state at shutdown.
        3. Count pending messages to confirm recovery readiness.
        4. Log all recovery actions.

        This must complete within 60 seconds for up to 100,000 pending messages
        (Requirement 26.7).

        Returns
        -------
        RecoveryReport
            Summary of all recovery actions taken.
        """
        start_time = time.time()
        report = RecoveryReport()

        try:
            # Step 1: Reset stale messages (sending for > stale_minutes)
            stale_ids = self.identify_stale_messages(self._stale_minutes)
            if stale_ids:
                reset_count = self._reset_stale_messages(stale_ids)
                report.stale_messages_reset = reset_count
                report.messages_requeued += reset_count
                logger.info(
                    "Recovery: Reset %d stale messages to 'queued'", reset_count
                )

            # Step 2: Deduplicate and confirm queued messages are safe to dispatch
            duplicates_prevented = self._deduplicate_pending_messages()
            report.duplicates_prevented = duplicates_prevented
            if duplicates_prevented > 0:
                logger.info(
                    "Recovery: Prevented %d duplicate messages", duplicates_prevented
                )

            # Step 3: Resume campaigns that were in 'sending' state
            campaigns_resumed = self._resume_interrupted_campaigns()
            report.campaigns_resumed = campaigns_resumed
            if campaigns_resumed > 0:
                logger.info(
                    "Recovery: Resumed %d interrupted campaigns", campaigns_resumed
                )

            # Step 4: Count total pending messages for logging
            pending_count = self._count_pending_messages()
            logger.info(
                "Recovery: %d messages pending for dispatch after recovery",
                pending_count,
            )

        except Exception as exc:
            error_msg = f"Recovery error: {exc}"
            report.errors.append(error_msg)
            logger.exception("Recovery failed: %s", exc)

        report.total_recovery_time_seconds = time.time() - start_time

        logger.info(
            "Recovery completed in %.2fs — requeued=%d, duplicates_prevented=%d, "
            "campaigns_resumed=%d, stale_reset=%d",
            report.total_recovery_time_seconds,
            report.messages_requeued,
            report.duplicates_prevented,
            report.campaigns_resumed,
            report.stale_messages_reset,
        )

        return report

    def identify_stale_messages(self, stale_minutes: int = 5) -> List[int]:
        """
        Find messages in 'sending' status with updated_at older than stale_minutes.

        These are messages that were being processed when the server went down
        and never received a delivery confirmation.

        Parameters
        ----------
        stale_minutes : int
            Number of minutes without update after which a 'sending' message
            is considered stale (default 5).

        Returns
        -------
        List[int]
            List of campaign_message IDs that are stale.
        """
        conn = self._get_conn()
        cursor = None
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                """
                SELECT id
                FROM campaign_messages
                WHERE status = 'sending'
                  AND updated_at < DATE_SUB(NOW(), INTERVAL %s MINUTE)
                """,
                (stale_minutes,)
            )
            rows = cursor.fetchall()
            return [row["id"] for row in rows]
        finally:
            if cursor:
                cursor.close()
            conn.close()

    def deduplicate_and_requeue(self, message_ids: List[int]) -> int:
        """
        Reset specified messages to 'queued' after verifying idempotency.

        For each message, checks if a successful delivery (status 'sent',
        'delivered', or 'read') already exists with the same idempotency_key.
        If so, marks the duplicate as 'skipped'. Otherwise, resets to 'queued'.

        Parameters
        ----------
        message_ids : List[int]
            IDs of campaign_messages to check and potentially re-queue.

        Returns
        -------
        int
            Number of messages actually re-queued (excludes duplicates).
        """
        if not message_ids:
            return 0

        conn = self._get_conn()
        cursor = None
        requeued = 0

        try:
            cursor = conn.cursor(dictionary=True)

            # Process in batches of 1000 for efficiency
            batch_size = 1000
            for i in range(0, len(message_ids), batch_size):
                batch = message_ids[i:i + batch_size]
                placeholders = ", ".join(["%s"] * len(batch))

                # Fetch idempotency keys for these messages
                cursor.execute(
                    f"""
                    SELECT id, idempotency_key, campaign_id
                    FROM campaign_messages
                    WHERE id IN ({placeholders})
                    """,
                    batch
                )
                messages = cursor.fetchall()

                for msg in messages:
                    # Check if another record with same idempotency_key
                    # already succeeded (sent/delivered/read)
                    cursor.execute(
                        """
                        SELECT id FROM campaign_messages
                        WHERE idempotency_key = %s
                          AND status IN ('sent', 'delivered', 'read')
                          AND id != %s
                        LIMIT 1
                        """,
                        (msg["idempotency_key"], msg["id"])
                    )
                    already_sent = cursor.fetchone()

                    if already_sent:
                        # Duplicate detected — skip this message
                        cursor.execute(
                            "UPDATE campaign_messages SET status = 'skipped' "
                            "WHERE id = %s",
                            (msg["id"],)
                        )
                    else:
                        # Safe to re-queue
                        cursor.execute(
                            "UPDATE campaign_messages SET status = 'queued', "
                            "retry_count = retry_count WHERE id = %s",
                            (msg["id"],)
                        )
                        requeued += 1

                conn.commit()

            return requeued

        except Exception:
            if conn:
                conn.rollback()
            raise
        finally:
            if cursor:
                cursor.close()
            conn.close()

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _reset_stale_messages(self, message_ids: List[int]) -> int:
        """
        Reset stale messages to 'queued' using idempotency check.

        Uses deduplicate_and_requeue internally to ensure no duplicates
        are dispatched.
        """
        return self.deduplicate_and_requeue(message_ids)

    def _deduplicate_pending_messages(self) -> int:
        """
        Scan all 'queued' messages and detect any whose idempotency_key
        already has a successful delivery. Mark duplicates as 'skipped'.

        Returns the number of duplicates prevented.
        """
        conn = self._get_conn()
        cursor = None
        duplicates_prevented = 0

        try:
            cursor = conn.cursor(dictionary=True)

            # Find queued messages that have a sibling with same idempotency_key
            # already in a delivered state
            cursor.execute(
                """
                SELECT cm1.id
                FROM campaign_messages cm1
                INNER JOIN campaign_messages cm2
                    ON cm1.idempotency_key = cm2.idempotency_key
                    AND cm2.status IN ('sent', 'delivered', 'read')
                    AND cm2.id != cm1.id
                WHERE cm1.status = 'queued'
                """
            )
            duplicate_rows = cursor.fetchall()

            if duplicate_rows:
                duplicate_ids = [row["id"] for row in duplicate_rows]
                # Mark duplicates as skipped in batches
                batch_size = 1000
                for i in range(0, len(duplicate_ids), batch_size):
                    batch = duplicate_ids[i:i + batch_size]
                    placeholders = ", ".join(["%s"] * len(batch))
                    cursor.execute(
                        f"UPDATE campaign_messages SET status = 'skipped' "
                        f"WHERE id IN ({placeholders})",
                        batch
                    )
                    duplicates_prevented += cursor.rowcount

                conn.commit()

            return duplicates_prevented

        except Exception:
            if conn:
                conn.rollback()
            raise
        finally:
            if cursor:
                cursor.close()
            conn.close()

    def _resume_interrupted_campaigns(self) -> int:
        """
        Resume campaigns that were in 'sending' state at shutdown.

        Ensures campaigns with remaining queued messages are still marked
        as 'sending' so the worker picks them up.

        Returns the number of campaigns resumed.
        """
        conn = self._get_conn()
        cursor = None
        resumed = 0

        try:
            cursor = conn.cursor(dictionary=True)

            # Find campaigns in 'sending' status that have queued messages
            cursor.execute(
                """
                SELECT DISTINCT c.id
                FROM campaigns c
                INNER JOIN campaign_messages cm
                    ON cm.campaign_id = c.id
                WHERE c.status = 'sending'
                  AND cm.status = 'queued'
                """
            )
            campaigns = cursor.fetchall()

            for campaign in campaigns:
                # Campaign is already in 'sending' with queued messages —
                # confirm it stays in 'sending' (touch updated_at)
                cursor.execute(
                    "UPDATE campaigns SET status = 'sending', "
                    "updated_at = NOW() WHERE id = %s",
                    (campaign["id"],)
                )
                resumed += 1

            # Also handle edge case: campaigns that crashed between
            # 'approved' → 'sending' transition. If approved with queued
            # messages, transition to 'sending'.
            cursor.execute(
                """
                SELECT DISTINCT c.id
                FROM campaigns c
                INNER JOIN campaign_messages cm
                    ON cm.campaign_id = c.id
                WHERE c.status = 'approved'
                  AND cm.status = 'queued'
                """
            )
            approved_with_queue = cursor.fetchall()

            for campaign in approved_with_queue:
                cursor.execute(
                    "UPDATE campaigns SET status = 'sending', "
                    "started_at = COALESCE(started_at, NOW()), "
                    "updated_at = NOW() WHERE id = %s",
                    (campaign["id"],)
                )
                resumed += 1

            if resumed > 0:
                conn.commit()

            return resumed

        except Exception:
            if conn:
                conn.rollback()
            raise
        finally:
            if cursor:
                cursor.close()
            conn.close()

    def _count_pending_messages(self) -> int:
        """Count all messages currently in 'queued' status."""
        conn = self._get_conn()
        cursor = None
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                "SELECT COUNT(*) as cnt FROM campaign_messages "
                "WHERE status = 'queued'"
            )
            row = cursor.fetchone()
            return row["cnt"] if row else 0
        finally:
            if cursor:
                cursor.close()
            conn.close()
