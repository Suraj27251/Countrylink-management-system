"""
Delivery Tracker Service for campaign message status updates.

Processes webhook status callbacks from the Meta WhatsApp Business API and
updates campaign_messages records with delivery statuses (delivered, read, failed).
Integrates with RetryCategorizerService for permanent failure handling and
suppression list management. Updates campaign aggregate counts on each status update.

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5
"""

import logging
from datetime import datetime
from typing import Optional

from services.retry_categorizer import RetryCategorizerService, FailureCategoryEnum

logger = logging.getLogger(__name__)


class DeliveryTracker:
    """
    Tracks delivery status updates for campaign messages.

    Detects campaign message status callbacks from Meta webhooks, updates
    individual message records, handles permanent failures via RetryCategorizerService,
    and maintains campaign-level aggregate counts.

    Args:
        get_connection: Callable that returns an active MySQL connection.
    """

    # Valid campaign message statuses that can be processed
    TRACKABLE_STATUSES = {"sent", "delivered", "read", "failed"}

    def __init__(self, get_connection):
        """
        Initialize the delivery tracker.

        Args:
            get_connection: Callable returning an active mysql.connector connection.
        """
        self._get_connection = get_connection
        self._retry_categorizer = RetryCategorizerService(get_connection)

    def is_campaign_message(self, whatsapp_message_id: str) -> bool:
        """
        Check if a whatsapp_message_id belongs to a campaign message.

        Args:
            whatsapp_message_id: The Meta message ID from the webhook.

        Returns:
            True if the message ID matches a campaign_messages record.
        """
        if not whatsapp_message_id:
            return False

        connection = self._get_connection()
        cursor = None
        try:
            cursor = connection.cursor(dictionary=True)
            cursor.execute(
                """
                SELECT id FROM campaign_messages
                WHERE whatsapp_message_id = %s
                LIMIT 1
                """,
                (whatsapp_message_id,),
            )
            row = cursor.fetchone()
            return row is not None
        except Exception as exc:
            logger.error(
                "Error checking campaign message for whatsapp_message_id=%s: %s",
                whatsapp_message_id,
                exc,
            )
            return False
        finally:
            if cursor:
                cursor.close()
            if connection and connection.is_connected():
                connection.close()

    def process_status_update(
        self,
        whatsapp_message_id: str,
        status: str,
        timestamp: Optional[str] = None,
        error_code: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> bool:
        """
        Process a webhook status update for a campaign message.

        Updates the campaign_messages record with the new status and timestamp.
        For failed messages, delegates to RetryCategorizerService for classification
        and appropriate handling (retry, permanent failure, or suppression).
        Updates campaign aggregate counts after each status change.

        Args:
            whatsapp_message_id: Meta WhatsApp message ID from the webhook.
            status: The delivery status ('sent', 'delivered', 'read', 'failed').
            timestamp: Unix timestamp string from the webhook.
            error_code: WhatsApp API error code (for failed messages).
            error_message: Human-readable error description (for failed messages).

        Returns:
            True if the status was successfully processed, False otherwise.
        """
        if not whatsapp_message_id:
            logger.warning("process_status_update called with empty message_id")
            return False

        if status not in self.TRACKABLE_STATUSES:
            logger.info(
                "Ignoring non-trackable status '%s' for message_id=%s",
                status,
                whatsapp_message_id,
            )
            return False

        # Parse timestamp
        status_time = self._parse_timestamp(timestamp)

        connection = self._get_connection()
        cursor = None
        try:
            cursor = connection.cursor(dictionary=True)

            # Lookup the campaign message
            cursor.execute(
                """
                SELECT id, campaign_id, status AS current_status, customer_mobile
                FROM campaign_messages
                WHERE whatsapp_message_id = %s
                LIMIT 1
                """,
                (whatsapp_message_id,),
            )
            message = cursor.fetchone()

            if not message:
                logger.debug(
                    "No campaign message found for whatsapp_message_id=%s",
                    whatsapp_message_id,
                )
                return False

            message_id = message["id"]
            campaign_id = message["campaign_id"]
            current_status = message["current_status"]

            # Prevent status regression (don't go backwards in delivery pipeline)
            if not self._is_valid_status_transition(current_status, status):
                logger.info(
                    "Ignoring status regression for message %d: %s -> %s",
                    message_id,
                    current_status,
                    status,
                )
                return False

            # Update the message record based on status
            if status == "delivered":
                cursor.execute(
                    """
                    UPDATE campaign_messages
                    SET status = 'delivered',
                        delivered_at = %s,
                        updated_at = NOW()
                    WHERE id = %s
                    """,
                    (status_time, message_id),
                )
            elif status == "read":
                cursor.execute(
                    """
                    UPDATE campaign_messages
                    SET status = 'read',
                        read_at = %s,
                        updated_at = NOW()
                    WHERE id = %s
                    """,
                    (status_time, message_id),
                )
            elif status == "sent":
                cursor.execute(
                    """
                    UPDATE campaign_messages
                    SET status = 'sent',
                        sent_at = %s,
                        updated_at = NOW()
                    WHERE id = %s
                    """,
                    (status_time, message_id),
                )
            elif status == "failed":
                cursor.execute(
                    """
                    UPDATE campaign_messages
                    SET status = 'failed',
                        failed_at = %s,
                        error_code = %s,
                        error_message = %s,
                        updated_at = NOW()
                    WHERE id = %s
                    """,
                    (status_time, error_code, error_message, message_id),
                )

            connection.commit()

            logger.info(
                "Campaign message %d status updated: %s -> %s (campaign=%d)",
                message_id,
                current_status,
                status,
                campaign_id,
            )

            # Update campaign aggregate counts
            self._update_campaign_counts(campaign_id)

            # Handle failures via RetryCategorizerService
            if status == "failed" and error_code is not None:
                self._handle_failure(message_id, error_code, error_message or "")

            return True

        except Exception as exc:
            if connection and connection.is_connected():
                connection.rollback()
            logger.error(
                "Error processing status update for whatsapp_message_id=%s: %s",
                whatsapp_message_id,
                exc,
            )
            return False
        finally:
            if cursor:
                cursor.close()
            if connection and connection.is_connected():
                connection.close()

    def _is_valid_status_transition(self, current_status: str, new_status: str) -> bool:
        """
        Check if a status transition is valid (no regression).

        Status progression: queued -> sending -> sent -> delivered -> read
        Failed can come from sent or delivered.
        permanently_failed is terminal.

        Args:
            current_status: Current status of the message.
            new_status: Proposed new status.

        Returns:
            True if the transition is valid.
        """
        # Status priority (higher number = later in pipeline)
        priority = {
            "queued": 0,
            "sending": 1,
            "sent": 2,
            "delivered": 3,
            "read": 4,
            "failed": 3,  # Same level as delivered (can fail after sent)
            "permanently_failed": 5,  # Terminal
            "skipped": 5,  # Terminal
        }

        # Terminal statuses cannot be changed
        if current_status in ("permanently_failed", "skipped"):
            return False

        current_priority = priority.get(current_status, 0)
        new_priority = priority.get(new_status, 0)

        # Allow the transition if new status is at equal or higher priority
        # Special case: 'failed' can override 'sent' or 'delivered'
        if new_status == "failed":
            return current_status in ("sent", "delivered", "sending")

        return new_priority > current_priority

    def _update_campaign_counts(self, campaign_id: int) -> None:
        """
        Update campaign aggregate counts (delivered_count, read_count, failed_count).

        Recomputes counts from campaign_messages for accuracy.

        Args:
            campaign_id: The campaign to update counts for.
        """
        connection = self._get_connection()
        cursor = None
        try:
            cursor = connection.cursor(dictionary=True)
            cursor.execute(
                """
                SELECT
                    SUM(CASE WHEN status = 'sent' THEN 1 ELSE 0 END) AS sent_count,
                    SUM(CASE WHEN status IN ('delivered', 'read') THEN 1 ELSE 0 END) AS delivered_count,
                    SUM(CASE WHEN status = 'read' THEN 1 ELSE 0 END) AS read_count,
                    SUM(CASE WHEN status IN ('failed', 'permanently_failed') THEN 1 ELSE 0 END) AS failed_count
                FROM campaign_messages
                WHERE campaign_id = %s
                """,
                (campaign_id,),
            )
            counts = cursor.fetchone()

            if not counts:
                return

            cursor.execute(
                """
                UPDATE campaigns
                SET sent_count = %s,
                    delivered_count = %s,
                    read_count = %s,
                    failed_count = %s,
                    updated_at = NOW()
                WHERE id = %s
                """,
                (
                    counts["sent_count"] or 0,
                    counts["delivered_count"] or 0,
                    counts["read_count"] or 0,
                    counts["failed_count"] or 0,
                    campaign_id,
                ),
            )
            connection.commit()

            logger.debug(
                "Campaign %d counts updated: sent=%d delivered=%d read=%d failed=%d",
                campaign_id,
                counts["sent_count"] or 0,
                counts["delivered_count"] or 0,
                counts["read_count"] or 0,
                counts["failed_count"] or 0,
            )
        except Exception as exc:
            if connection and connection.is_connected():
                connection.rollback()
            logger.error(
                "Error updating campaign %d counts: %s", campaign_id, exc
            )
        finally:
            if cursor:
                cursor.close()
            if connection and connection.is_connected():
                connection.close()

    def _handle_failure(self, message_id: int, error_code: int, error_message: str) -> None:
        """
        Handle a failed message by classifying the error and taking appropriate action.

        Delegates to RetryCategorizerService.process_failure() which handles:
        - Transient errors: schedule retry with exponential backoff
        - Permanent errors: mark permanently_failed, flag customer
        - Suppression errors: add to suppression list, check campaign pause threshold

        Args:
            message_id: ID of the failed campaign_messages record.
            error_code: WhatsApp API error code.
            error_message: Human-readable error description.
        """
        try:
            self._retry_categorizer.process_failure(message_id, error_code, error_message)
        except Exception as exc:
            logger.error(
                "Error handling failure for message %d (error_code=%d): %s",
                message_id,
                error_code,
                exc,
            )

    @staticmethod
    def compute_delivery_rates(sent_count: int, delivered_count: int, read_count: int, failed_count: int) -> dict:
        """
        Compute delivery rate metrics from aggregate campaign counts.

        All rates are bounded in [0.0, 1.0]. If sent_count is 0, all rates
        are 0.0 (division by zero protection). Test-send messages must be
        excluded from the input counts before calling this function.

        Args:
            sent_count: Total messages sent (excludes test-sends).
            delivered_count: Messages that reached the recipient.
            read_count: Messages that were opened/read.
            failed_count: Messages that failed permanently or transiently.

        Returns:
            dict with keys: delivery_rate, read_rate, failure_rate (all float in [0.0, 1.0]).
        """
        if sent_count <= 0:
            return {
                "delivery_rate": 0.0,
                "read_rate": 0.0,
                "failure_rate": 0.0,
            }

        delivery_rate = min(1.0, max(0.0, delivered_count / sent_count))
        read_rate = min(1.0, max(0.0, read_count / sent_count))
        failure_rate = min(1.0, max(0.0, failed_count / sent_count))

        return {
            "delivery_rate": delivery_rate,
            "read_rate": read_rate,
            "failure_rate": failure_rate,
        }

    @staticmethod
    def _parse_timestamp(timestamp_str: Optional[str]) -> str:
        """
        Parse a Unix timestamp string into a MySQL datetime string.

        Args:
            timestamp_str: Unix timestamp from the webhook (e.g., '1700000000').

        Returns:
            Formatted datetime string 'YYYY-MM-DD HH:MM:SS'.
        """
        if timestamp_str:
            try:
                dt = datetime.fromtimestamp(int(timestamp_str))
                return dt.strftime("%Y-%m-%d %H:%M:%S")
            except (ValueError, OSError, OverflowError):
                pass
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
