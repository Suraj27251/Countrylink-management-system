"""
Sending Queue Service for Enterprise WhatsApp CRM.

Provides a database-backed message queue with background workers for campaign
message dispatch, throttling, and real-time progress tracking.

Accepts a get_connection callable for testability and uses parameterized SQL
throughout. Background processing uses threading + concurrent.futures to match
the existing WEBHOOK_ASYNC_PROCESSING pattern.

Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 12.3
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from services.channel import DispatchResult, WhatsAppDispatcher
from services.template_validator import TemplateValidator

logger = logging.getLogger(__name__)


@dataclass
class BatchResult:
    """Result of processing a single batch of messages."""
    sent_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    errors: List[Dict] = field(default_factory=list)


@dataclass
class QueueProgress:
    """Real-time progress snapshot for a campaign."""
    campaign_id: int = 0
    total: int = 0
    sent_count: int = 0
    failed_count: int = 0
    remaining_count: int = 0
    status: str = ""


class SendingQueue:
    """
    Database-backed sending queue with background workers.

    Parameters
    ----------
    get_connection : callable
        A zero-argument function that returns a MySQL connection object.
    dispatcher : WhatsAppDispatcher, optional
        Channel dispatcher for sending messages. Defaults to WhatsAppDispatcher().
    template_validator : TemplateValidator, optional
        Validator for per-message parameter validation. Defaults to TemplateValidator().
    throttle_rate : int, optional
        Maximum messages per second (default 80).
    max_workers : int, optional
        Maximum concurrent dispatch threads (default 4).
    """

    def __init__(
        self,
        get_connection: Callable,
        dispatcher: Optional[WhatsAppDispatcher] = None,
        template_validator: Optional[TemplateValidator] = None,
        throttle_rate: int = 80,
        max_workers: int = 4,
    ):
        self._get_conn = get_connection
        self._dispatcher = dispatcher or WhatsAppDispatcher()
        self._template_validator = template_validator or TemplateValidator()
        self._throttle_rate = throttle_rate
        self._max_workers = max_workers

        # Campaign pause/cancel signaling
        self._paused_campaigns: set = set()
        self._cancelled_campaigns: set = set()
        self._lock = threading.Lock()

        # Background worker state
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False

    # ------------------------------------------------------------------
    # Enqueue
    # ------------------------------------------------------------------

    def enqueue_campaign(self, campaign_id: int, recipients: List[dict]) -> int:
        """
        Create campaign_messages records for each recipient with idempotency.

        Each message gets an idempotency_key = f"{campaign_id}_{mobile}_{template_id}".
        Uses INSERT IGNORE to prevent duplicates on the UNIQUE idempotency_key.

        Parameters
        ----------
        campaign_id : int
            The campaign to enqueue messages for.
        recipients : list of dict
            Each dict must have 'mobile' key, and optionally 'customer_name'.
            Template information is pulled from the campaign record.

        Returns
        -------
        int
            Number of messages successfully enqueued (new inserts).
        """
        conn = self._get_conn()
        cursor = None
        try:
            cursor = conn.cursor(dictionary=True)

            # Fetch campaign details for template_id and template params
            cursor.execute(
                "SELECT template_id, segment_id FROM campaigns WHERE id = %s",
                (campaign_id,)
            )
            campaign = cursor.fetchone()
            if not campaign:
                raise ValueError(f"Campaign {campaign_id} not found")

            template_id = campaign["template_id"]
            if not template_id:
                raise ValueError(f"Campaign {campaign_id} has no template_id assigned")

            # Fetch template details for param mappings
            cursor.execute(
                "SELECT template_name, body_text, placeholder_mappings "
                "FROM campaign_templates WHERE id = %s",
                (template_id,)
            )
            template = cursor.fetchone()

            enqueued_count = 0
            batch_size = 500

            for i in range(0, len(recipients), batch_size):
                batch = recipients[i:i + batch_size]
                conn.start_transaction()

                for recipient in batch:
                    mobile = recipient.get("mobile", "").strip()
                    if not mobile:
                        continue

                    customer_name = recipient.get("customer_name", "")
                    idempotency_key = f"{campaign_id}_{mobile}_{template_id}"

                    # Resolve template params for this recipient
                    template_params = self._resolve_template_params(
                        template, recipient
                    )

                    sql = """
                        INSERT IGNORE INTO campaign_messages
                            (campaign_id, customer_mobile, customer_name,
                             template_id, template_params, channel, status,
                             idempotency_key)
                        VALUES
                            (%s, %s, %s, %s, %s, %s, 'queued', %s)
                    """
                    params = (
                        campaign_id,
                        mobile,
                        customer_name,
                        template_id,
                        template_params,
                        "whatsapp",
                        idempotency_key,
                    )
                    cursor.execute(sql, params)
                    if cursor.rowcount > 0:
                        enqueued_count += 1

                conn.commit()

            # Update campaign total_recipients count
            cursor.execute(
                "UPDATE campaigns SET total_recipients = %s WHERE id = %s",
                (enqueued_count, campaign_id)
            )
            conn.commit()

            logger.info(
                "Enqueued %d messages for campaign %d", enqueued_count, campaign_id
            )
            return enqueued_count

        except Exception:
            conn.rollback()
            raise
        finally:
            if cursor:
                cursor.close()
            conn.close()

    # ------------------------------------------------------------------
    # Process Batch
    # ------------------------------------------------------------------

    def process_batch(self, batch_size: int = 80) -> BatchResult:
        """
        Fetch and dispatch a batch of queued messages.

        Respects throttle rate (default 80/sec), skips paused/cancelled campaigns,
        and validates template params before dispatch.

        Parameters
        ----------
        batch_size : int
            Number of messages to process in this batch (default 80).

        Returns
        -------
        BatchResult with sent_count, failed_count, skipped_count.
        """
        result = BatchResult()
        conn = self._get_conn()
        cursor = None

        try:
            cursor = conn.cursor(dictionary=True)

            # Fetch a batch of queued messages, excluding paused/cancelled campaigns
            excluded_campaigns = self._get_excluded_campaign_ids()
            if excluded_campaigns:
                placeholders = ", ".join(["%s"] * len(excluded_campaigns))
                sql = f"""
                    SELECT cm.id, cm.campaign_id, cm.customer_mobile,
                           cm.customer_name, cm.template_id, cm.template_params
                    FROM campaign_messages cm
                    WHERE cm.status = 'queued'
                      AND cm.campaign_id NOT IN ({placeholders})
                    ORDER BY cm.id ASC
                    LIMIT %s
                    FOR UPDATE SKIP LOCKED
                """
                params = list(excluded_campaigns) + [batch_size]
            else:
                sql = """
                    SELECT cm.id, cm.campaign_id, cm.customer_mobile,
                           cm.customer_name, cm.template_id, cm.template_params
                    FROM campaign_messages cm
                    WHERE cm.status = 'queued'
                    ORDER BY cm.id ASC
                    LIMIT %s
                    FOR UPDATE SKIP LOCKED
                """
                params = [batch_size]

            cursor.execute(sql, params)
            messages = cursor.fetchall()

            if not messages:
                return result

            # Mark messages as 'sending'
            message_ids = [m["id"] for m in messages]
            id_placeholders = ", ".join(["%s"] * len(message_ids))
            cursor.execute(
                f"UPDATE campaign_messages SET status = 'sending' "
                f"WHERE id IN ({id_placeholders})",
                message_ids
            )
            conn.commit()

            # Dispatch messages with throttling
            interval = 1.0 / self._throttle_rate if self._throttle_rate > 0 else 0

            for msg in messages:
                campaign_id = msg["campaign_id"]

                # Check if campaign was paused or cancelled mid-batch
                if self._is_campaign_excluded(campaign_id):
                    self._revert_message_status(msg["id"], "queued")
                    result.skipped_count += 1
                    continue

                # Validate template params before dispatch
                validation_ok = self._validate_message_params(msg)
                if not validation_ok:
                    self._mark_message_failed(
                        msg["id"], campaign_id, None,
                        "Template parameter validation failed"
                    )
                    result.failed_count += 1
                    continue

                # Dispatch via channel
                dispatch_result = self._dispatch_message(msg)

                if dispatch_result.success:
                    self._mark_message_sent(
                        msg["id"], campaign_id, dispatch_result.message_id
                    )
                    result.sent_count += 1
                else:
                    self._mark_message_failed(
                        msg["id"], campaign_id,
                        dispatch_result.error_code,
                        dispatch_result.error_message
                    )
                    result.failed_count += 1
                    result.errors.append({
                        "message_id": msg["id"],
                        "error_code": dispatch_result.error_code,
                        "error_message": dispatch_result.error_message,
                    })

                # Throttle
                if interval > 0:
                    time.sleep(interval)

            # Update campaign progress counters
            self._update_campaign_progress(messages)

            return result

        except Exception as exc:
            logger.exception("Error processing batch: %s", exc)
            if conn:
                conn.rollback()
            return result
        finally:
            if cursor:
                cursor.close()
            conn.close()

    # ------------------------------------------------------------------
    # Campaign Control
    # ------------------------------------------------------------------

    def pause_campaign(self, campaign_id: int) -> None:
        """
        Pause a campaign — stop dispatching new messages for it.

        Updates campaign status to 'paused' and signals the worker to skip
        messages belonging to this campaign.
        """
        with self._lock:
            self._paused_campaigns.add(campaign_id)

        conn = self._get_conn()
        cursor = None
        try:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE campaigns SET status = 'paused' WHERE id = %s",
                (campaign_id,)
            )
            conn.commit()
            logger.info("Campaign %d paused", campaign_id)
        finally:
            if cursor:
                cursor.close()
            conn.close()

    def resume_campaign(self, campaign_id: int) -> None:
        """
        Resume a paused campaign — allow 'queued' messages to be dispatched.

        Updates campaign status to 'sending' and removes from paused set.
        """
        with self._lock:
            self._paused_campaigns.discard(campaign_id)

        conn = self._get_conn()
        cursor = None
        try:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE campaigns SET status = 'sending' WHERE id = %s",
                (campaign_id,)
            )
            conn.commit()
            logger.info("Campaign %d resumed", campaign_id)
        finally:
            if cursor:
                cursor.close()
            conn.close()

    def cancel_campaign(self, campaign_id: int) -> None:
        """
        Cancel a campaign — transition all remaining 'queued' messages to 'cancelled'.

        Updates campaign status to 'cancelled' and marks all queued messages
        as cancelled so they are never dispatched.
        """
        with self._lock:
            self._cancelled_campaigns.add(campaign_id)

        conn = self._get_conn()
        cursor = None
        try:
            cursor = conn.cursor()

            # Cancel all queued messages for this campaign
            cursor.execute(
                "UPDATE campaign_messages SET status = 'skipped' "
                "WHERE campaign_id = %s AND status = 'queued'",
                (campaign_id,)
            )

            # Update campaign status
            cursor.execute(
                "UPDATE campaigns SET status = 'cancelled' WHERE id = %s",
                (campaign_id,)
            )
            conn.commit()
            logger.info("Campaign %d cancelled", campaign_id)
        finally:
            if cursor:
                cursor.close()
            conn.close()

    # ------------------------------------------------------------------
    # Progress Tracking
    # ------------------------------------------------------------------

    def get_progress(self, campaign_id: int) -> QueueProgress:
        """
        Get real-time progress for a campaign.

        Returns sent_count, failed_count, remaining_count from the DB.
        """
        conn = self._get_conn()
        cursor = None
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                """
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN status = 'sent' OR status = 'delivered'
                             OR status = 'read' THEN 1 ELSE 0 END) as sent_count,
                    SUM(CASE WHEN status = 'failed' OR status = 'permanently_failed'
                             THEN 1 ELSE 0 END) as failed_count,
                    SUM(CASE WHEN status = 'queued' OR status = 'sending'
                             THEN 1 ELSE 0 END) as remaining_count
                FROM campaign_messages
                WHERE campaign_id = %s
                """,
                (campaign_id,)
            )
            row = cursor.fetchone()

            # Fetch campaign status
            cursor.execute(
                "SELECT status FROM campaigns WHERE id = %s",
                (campaign_id,)
            )
            campaign_row = cursor.fetchone()

            return QueueProgress(
                campaign_id=campaign_id,
                total=row["total"] or 0,
                sent_count=row["sent_count"] or 0,
                failed_count=row["failed_count"] or 0,
                remaining_count=row["remaining_count"] or 0,
                status=campaign_row["status"] if campaign_row else "",
            )
        finally:
            if cursor:
                cursor.close()
            conn.close()

    # ------------------------------------------------------------------
    # Background Worker
    # ------------------------------------------------------------------

    def start_worker(self, poll_interval: float = 1.0) -> None:
        """
        Start the background worker thread that continuously processes batches.

        Uses a daemon thread that polls for queued messages at the configured
        interval. Matches the existing threading pattern used for webhook
        async processing.

        Parameters
        ----------
        poll_interval : float
            Seconds to wait between batch processing attempts (default 1.0).
        """
        if self._worker_thread and self._worker_thread.is_alive():
            logger.warning("Worker already running")
            return

        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            args=(poll_interval,),
            daemon=True,
        )
        self._worker_thread.start()
        logger.info("Sending queue worker started")

    def stop_worker(self) -> None:
        """Stop the background worker thread gracefully."""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=10)
            logger.info("Sending queue worker stopped")

    def _worker_loop(self, poll_interval: float) -> None:
        """Main loop for background worker — processes batches until stopped."""
        while self._running:
            try:
                result = self.process_batch(batch_size=self._throttle_rate)
                # If nothing was processed, wait before polling again
                if result.sent_count == 0 and result.failed_count == 0:
                    time.sleep(poll_interval)

                # Check for campaign completion
                self._check_campaign_completions()

            except Exception as exc:
                logger.exception("Worker loop error: %s", exc)
                time.sleep(poll_interval * 2)

    def _check_campaign_completions(self) -> None:
        """Check if any active campaigns have completed all messages."""
        conn = self._get_conn()
        cursor = None
        try:
            cursor = conn.cursor(dictionary=True)
            # Find campaigns in 'sending' status with no remaining queued messages
            cursor.execute(
                """
                SELECT c.id
                FROM campaigns c
                WHERE c.status = 'sending'
                  AND c.total_recipients > 0
                  AND EXISTS (
                      SELECT 1 FROM campaign_messages cm
                      WHERE cm.campaign_id = c.id
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM campaign_messages cm
                      WHERE cm.campaign_id = c.id
                        AND cm.status IN ('queued', 'sending')
                  )
                """
            )
            completed = cursor.fetchall()
            for row in completed:
                cursor.execute(
                    "UPDATE campaigns SET status = 'completed', "
                    "completed_at = NOW() WHERE id = %s",
                    (row["id"],)
                )
            if completed:
                conn.commit()
        except Exception as exc:
            logger.exception("Error checking campaign completions: %s", exc)
        finally:
            if cursor:
                cursor.close()
            conn.close()

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _resolve_template_params(self, template: Optional[dict], recipient: dict) -> Optional[str]:
        """Resolve template placeholder mappings for a recipient."""
        if not template:
            return None

        import json
        mappings_raw = template.get("placeholder_mappings")
        if not mappings_raw:
            return None

        if isinstance(mappings_raw, str):
            mappings = json.loads(mappings_raw)
        else:
            mappings = mappings_raw

        # Resolve each mapping from recipient data
        resolved = {}
        for placeholder, field_name in mappings.items():
            value = recipient.get(field_name, "")
            if value is not None:
                resolved[placeholder] = self._template_validator.sanitize_param(str(value))
            else:
                resolved[placeholder] = ""

        return json.dumps(resolved)

    def _validate_message_params(self, msg: dict) -> bool:
        """Validate template params for a message before dispatch."""
        import json

        template_params = msg.get("template_params")
        if not template_params:
            # No params needed — allow dispatch
            return True

        if isinstance(template_params, str):
            try:
                params = json.loads(template_params)
            except (json.JSONDecodeError, TypeError):
                return False
        else:
            params = template_params

        # Validate each param value is non-null, non-empty, <= 1024 chars
        for key, value in params.items():
            if value is None or str(value).strip() == "":
                return False
            if len(str(value)) > 1024:
                return False

        return True

    def _dispatch_message(self, msg: dict) -> DispatchResult:
        """Dispatch a single message via the channel dispatcher."""
        import json

        mobile = msg["customer_mobile"]

        # Get template name from DB
        conn = self._get_conn()
        cursor = None
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                "SELECT template_name FROM campaign_templates WHERE id = %s",
                (msg["template_id"],)
            )
            template_row = cursor.fetchone()
            template_name = template_row["template_name"] if template_row else ""
        finally:
            if cursor:
                cursor.close()
            conn.close()

        # Parse params as ordered list
        template_params = msg.get("template_params")
        params_list = []
        if template_params:
            if isinstance(template_params, str):
                try:
                    params_dict = json.loads(template_params)
                except (json.JSONDecodeError, TypeError):
                    params_dict = {}
            else:
                params_dict = template_params

            # Sort by key to maintain order (numeric placeholders: 1, 2, 3...)
            for key in sorted(params_dict.keys(), key=lambda k: (k.isdigit(), int(k) if k.isdigit() else k)):
                params_list.append(str(params_dict[key]))

        return self._dispatcher.send_template(
            recipient=mobile,
            template_name=template_name,
            params=params_list,
        )

    def _mark_message_sent(self, message_id: int, campaign_id: int, whatsapp_message_id: Optional[str]) -> None:
        """Mark a message as sent and update campaign counters."""
        conn = self._get_conn()
        cursor = None
        try:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE campaign_messages SET status = 'sent', "
                "whatsapp_message_id = %s, sent_at = NOW() "
                "WHERE id = %s",
                (whatsapp_message_id, message_id)
            )
            cursor.execute(
                "UPDATE campaigns SET sent_count = sent_count + 1 WHERE id = %s",
                (campaign_id,)
            )
            conn.commit()
        finally:
            if cursor:
                cursor.close()
            conn.close()

    def _mark_message_failed(
        self, message_id: int, campaign_id: int,
        error_code: Optional[int], error_message: Optional[str]
    ) -> None:
        """Mark a message as failed and update campaign counters."""
        conn = self._get_conn()
        cursor = None
        try:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE campaign_messages SET status = 'failed', "
                "error_code = %s, error_message = %s, failed_at = NOW() "
                "WHERE id = %s",
                (error_code, error_message, message_id)
            )
            cursor.execute(
                "UPDATE campaigns SET failed_count = failed_count + 1 WHERE id = %s",
                (campaign_id,)
            )
            conn.commit()
        finally:
            if cursor:
                cursor.close()
            conn.close()

    def _revert_message_status(self, message_id: int, status: str) -> None:
        """Revert a message back to a given status (e.g., 'queued')."""
        conn = self._get_conn()
        cursor = None
        try:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE campaign_messages SET status = %s WHERE id = %s",
                (status, message_id)
            )
            conn.commit()
        finally:
            if cursor:
                cursor.close()
            conn.close()

    def _get_excluded_campaign_ids(self) -> list:
        """Get campaign IDs that should be excluded from dispatch."""
        with self._lock:
            return list(self._paused_campaigns | self._cancelled_campaigns)

    def _is_campaign_excluded(self, campaign_id: int) -> bool:
        """Check if a campaign is currently paused or cancelled."""
        with self._lock:
            return (
                campaign_id in self._paused_campaigns
                or campaign_id in self._cancelled_campaigns
            )

    def _update_campaign_progress(self, messages: List[dict]) -> None:
        """Update campaign progress counters based on processed messages."""
        # Group by campaign_id and update aggregate counts from DB
        campaign_ids = set(m["campaign_id"] for m in messages)
        conn = self._get_conn()
        cursor = None
        try:
            cursor = conn.cursor(dictionary=True)
            for cid in campaign_ids:
                cursor.execute(
                    """
                    UPDATE campaigns SET
                        sent_count = (
                            SELECT COUNT(*) FROM campaign_messages
                            WHERE campaign_id = %s AND status IN ('sent', 'delivered', 'read')
                        ),
                        failed_count = (
                            SELECT COUNT(*) FROM campaign_messages
                            WHERE campaign_id = %s AND status IN ('failed', 'permanently_failed')
                        )
                    WHERE id = %s
                    """,
                    (cid, cid, cid)
                )
            conn.commit()
        except Exception as exc:
            logger.exception("Error updating campaign progress: %s", exc)
        finally:
            if cursor:
                cursor.close()
            conn.close()
