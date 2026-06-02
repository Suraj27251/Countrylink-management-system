"""
Retry Categorizer Service for smart message failure handling.

Classifies WhatsApp Business API errors by type and applies differentiated
retry strategies:
- Transient errors: retry with exponential backoff (5s, 15s, 45s)
- Permanent errors: mark permanently_failed, flag customer
- Suppression errors: add to suppression list, trigger campaign pause if rate > 20%

Requirements: 21.1, 21.2, 21.3, 21.4, 21.5, 21.6, 21.7
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class FailureCategoryEnum(str, Enum):
    """Error classification categories."""
    TRANSIENT = "transient"
    PERMANENT = "permanent"
    SUPPRESSION = "suppression"


@dataclass
class FailureCategory:
    """Result of classifying a message send error."""
    category: str  # 'transient', 'permanent', or 'suppression'
    error_code: int
    description: Optional[str] = None
    should_retry: bool = False


@dataclass
class RetryDecision:
    """Decision on whether and when to retry a failed message."""
    should_retry: bool
    retry_count: int
    max_retries: int
    next_retry_at: Optional[datetime] = None
    backoff_seconds: Optional[int] = None
    reason: str = ""


class RetryCategorizerService:
    """
    Classifies message send failures and determines retry strategy.

    Uses the error_classifications lookup table to map WhatsApp Business API
    error codes to failure categories. Applies exponential backoff for transient
    errors, permanent failure handling, and suppression list management.

    Args:
        get_connection: Callable that returns an active MySQL connection.
    """

    # Exponential backoff: 5 * 3^(N-1) seconds for attempt N
    BASE_BACKOFF_SECONDS = 5
    BACKOFF_MULTIPLIER = 3
    DEFAULT_MAX_RETRIES = 3

    # Campaign pause threshold: suppression rate > 20%
    SUPPRESSION_PAUSE_THRESHOLD = 0.20

    def __init__(self, get_connection):
        """
        Initialize the retry categorizer.

        Args:
            get_connection: Callable returning an active mysql.connector connection.
        """
        self._get_connection = get_connection

    def classify_error(self, error_code: int, error_message: str) -> FailureCategory:
        """
        Classify a message send error by looking up the error code in the
        error_classifications table.

        If the error code is not found in the lookup table, defaults to
        'transient' category to allow retry attempts.

        Args:
            error_code: WhatsApp Business API error code.
            error_message: Human-readable error description from the API.

        Returns:
            FailureCategory with the classification result.
        """
        connection = self._get_connection()
        cursor = None
        try:
            cursor = connection.cursor(dictionary=True)
            cursor.execute(
                """
                SELECT error_code, category, description, should_retry
                FROM error_classifications
                WHERE error_code = %s
                LIMIT 1
                """,
                (error_code,),
            )
            row = cursor.fetchone()

            if row:
                return FailureCategory(
                    category=row["category"],
                    error_code=error_code,
                    description=row["description"],
                    should_retry=bool(row["should_retry"]),
                )

            # Unknown error code — default to transient with retry allowed
            logger.warning(
                "Unknown error code %d ('%s') — defaulting to transient",
                error_code,
                error_message,
            )
            return FailureCategory(
                category=FailureCategoryEnum.TRANSIENT.value,
                error_code=error_code,
                description=error_message,
                should_retry=True,
            )
        finally:
            if cursor:
                cursor.close()

    def should_retry(self, message_id: int) -> RetryDecision:
        """
        Determine if a failed message should be retried.

        Checks the message's current retry_count against max_retries and
        verifies the error category is 'transient'. Computes next retry time
        using exponential backoff: 5 * 3^(N-1) seconds.

        Args:
            message_id: ID of the campaign_messages record to evaluate.

        Returns:
            RetryDecision with retry recommendation and timing.
        """
        connection = self._get_connection()
        cursor = None
        try:
            cursor = connection.cursor(dictionary=True)
            cursor.execute(
                """
                SELECT id, campaign_id, retry_count, max_retries,
                       error_code, error_category, status
                FROM campaign_messages
                WHERE id = %s
                LIMIT 1
                """,
                (message_id,),
            )
            message = cursor.fetchone()

            if not message:
                return RetryDecision(
                    should_retry=False,
                    retry_count=0,
                    max_retries=self.DEFAULT_MAX_RETRIES,
                    reason="Message not found",
                )

            retry_count = message["retry_count"] or 0
            max_retries = message["max_retries"] or self.DEFAULT_MAX_RETRIES
            error_category = message["error_category"]

            # Only retry transient errors
            if error_category != FailureCategoryEnum.TRANSIENT.value:
                return RetryDecision(
                    should_retry=False,
                    retry_count=retry_count,
                    max_retries=max_retries,
                    reason=f"Error category '{error_category}' is not retryable",
                )

            # Check if max retries exceeded
            if retry_count >= max_retries:
                return RetryDecision(
                    should_retry=False,
                    retry_count=retry_count,
                    max_retries=max_retries,
                    reason=f"Max retries exceeded ({retry_count}/{max_retries})",
                )

            # Compute exponential backoff: 5 * 3^(N-1) where N is next attempt
            next_attempt = retry_count + 1
            backoff_seconds = self.compute_backoff(next_attempt)
            next_retry_at = datetime.now(timezone.utc) + timedelta(seconds=backoff_seconds)

            return RetryDecision(
                should_retry=True,
                retry_count=retry_count,
                max_retries=max_retries,
                next_retry_at=next_retry_at,
                backoff_seconds=backoff_seconds,
                reason=f"Transient error, attempt {next_attempt}/{max_retries}",
            )
        finally:
            if cursor:
                cursor.close()

    @staticmethod
    def compute_backoff(attempt_number: int) -> int:
        """
        Compute exponential backoff delay for the given attempt number.

        Formula: 5 * 3^(N-1) seconds
          - Attempt 1: 5s
          - Attempt 2: 15s
          - Attempt 3: 45s

        Args:
            attempt_number: The retry attempt number (1-based).

        Returns:
            Backoff delay in seconds.
        """
        return RetryCategorizerService.BASE_BACKOFF_SECONDS * (
            RetryCategorizerService.BACKOFF_MULTIPLIER ** (attempt_number - 1)
        )

    def handle_permanent_failure(self, message_id: int) -> None:
        """
        Handle a permanent failure: mark message as permanently_failed and
        flag the customer record.

        Requirement 21.3: Mark message as 'permanently_failed', flag customer
        with 'invalid_whatsapp_number', exclude from future WhatsApp campaigns.

        Args:
            message_id: ID of the failed campaign_messages record.
        """
        connection = self._get_connection()
        cursor = None
        try:
            cursor = connection.cursor(dictionary=True)

            # Get message details
            cursor.execute(
                """
                SELECT id, campaign_id, customer_mobile, error_code, error_message
                FROM campaign_messages
                WHERE id = %s
                LIMIT 1
                """,
                (message_id,),
            )
            message = cursor.fetchone()
            if not message:
                logger.warning("handle_permanent_failure: message %d not found", message_id)
                return

            # Mark message as permanently_failed
            cursor.execute(
                """
                UPDATE campaign_messages
                SET status = 'permanently_failed',
                    error_category = 'permanent',
                    failed_at = NOW(),
                    updated_at = NOW()
                WHERE id = %s
                """,
                (message_id,),
            )

            # Flag customer by adding a tag (invalid_whatsapp_number)
            customer_mobile = message["customer_mobile"]
            cursor.execute(
                """
                INSERT IGNORE INTO customer_tags
                    (organization_id, branch_id, customer_mobile, tag_name, added_by)
                VALUES
                    (1, 1, %s, 'invalid_whatsapp_number', 'system_retry_categorizer')
                """,
                (customer_mobile,),
            )

            # Log the activity
            cursor.execute(
                """
                INSERT INTO customer_activity
                    (customer_mobile, activity_type, channel, details)
                VALUES
                    (%s, 'status_change', 'whatsapp',
                     JSON_OBJECT('action', 'permanent_failure_flagged',
                                 'error_code', %s,
                                 'message_id', %s))
                """,
                (customer_mobile, message["error_code"], message_id),
            )

            connection.commit()
            logger.info(
                "Permanently failed message %d, flagged customer %s",
                message_id,
                customer_mobile,
            )
        except Exception as exc:
            if connection.is_connected():
                connection.rollback()
            logger.error("Error handling permanent failure for message %d: %s", message_id, exc)
            raise
        finally:
            if cursor:
                cursor.close()

    def handle_suppression(self, message_id: int) -> None:
        """
        Handle a suppression failure: add customer to suppression list.

        Requirement 21.4: Add customer to permanent suppression list maintained
        by Quality_Monitor, exclude from all future campaign sends.

        Also checks if suppression rate > 20% for the campaign and triggers
        automatic campaign pause (Requirement 21.7).

        Args:
            message_id: ID of the failed campaign_messages record.
        """
        connection = self._get_connection()
        cursor = None
        try:
            cursor = connection.cursor(dictionary=True)

            # Get message details
            cursor.execute(
                """
                SELECT id, campaign_id, customer_mobile, error_code, error_message
                FROM campaign_messages
                WHERE id = %s
                LIMIT 1
                """,
                (message_id,),
            )
            message = cursor.fetchone()
            if not message:
                logger.warning("handle_suppression: message %d not found", message_id)
                return

            customer_mobile = message["customer_mobile"]
            campaign_id = message["campaign_id"]

            # Mark message with suppression category
            cursor.execute(
                """
                UPDATE campaign_messages
                SET status = 'permanently_failed',
                    error_category = 'suppression',
                    failed_at = NOW(),
                    updated_at = NOW()
                WHERE id = %s
                """,
                (message_id,),
            )

            # Add to suppression list
            cursor.execute(
                """
                INSERT IGNORE INTO suppression_list
                    (organization_id, customer_mobile, reason, added_by, is_active)
                VALUES
                    (1, %s, 'user_blocked', 'system_retry_categorizer', 1)
                """,
                (customer_mobile,),
            )

            # Log activity
            cursor.execute(
                """
                INSERT INTO customer_activity
                    (customer_mobile, activity_type, channel, details)
                VALUES
                    (%s, 'status_change', 'whatsapp',
                     JSON_OBJECT('action', 'added_to_suppression',
                                 'reason', 'user_blocked',
                                 'error_code', %s,
                                 'message_id', %s))
                """,
                (customer_mobile, message["error_code"], message_id),
            )

            connection.commit()
            logger.info(
                "Added customer %s to suppression list (message %d, campaign %d)",
                customer_mobile,
                message_id,
                campaign_id,
            )

            # Check suppression rate for campaign and trigger pause if > 20%
            self._check_suppression_rate(campaign_id)

        except Exception as exc:
            if connection.is_connected():
                connection.rollback()
            logger.error("Error handling suppression for message %d: %s", message_id, exc)
            raise
        finally:
            if cursor:
                cursor.close()

    def _check_suppression_rate(self, campaign_id: int) -> None:
        """
        Check if the campaign's suppression rate exceeds 20% and trigger
        an automatic pause if so.

        Requirement 21.7: When more than 20% of messages in an active campaign
        fail with 'suppression' classification, trigger automatic campaign pause
        and generate a critical alert.

        Args:
            campaign_id: The campaign to check.
        """
        connection = self._get_connection()
        cursor = None
        try:
            cursor = connection.cursor(dictionary=True)

            # Get campaign message counts
            cursor.execute(
                """
                SELECT
                    COUNT(*) AS total_messages,
                    SUM(CASE WHEN error_category = 'suppression' THEN 1 ELSE 0 END) AS suppression_count
                FROM campaign_messages
                WHERE campaign_id = %s
                  AND status NOT IN ('queued', 'skipped')
                """,
                (campaign_id,),
            )
            counts = cursor.fetchone()

            if not counts or counts["total_messages"] == 0:
                return

            total = counts["total_messages"]
            suppression_count = counts["suppression_count"] or 0
            suppression_rate = suppression_count / total

            if suppression_rate > self.SUPPRESSION_PAUSE_THRESHOLD:
                # Pause the campaign
                cursor.execute(
                    """
                    UPDATE campaigns
                    SET status = 'paused', updated_at = NOW()
                    WHERE id = %s AND status = 'sending'
                    """,
                    (campaign_id,),
                )

                if cursor.rowcount > 0:
                    # Generate critical alert
                    cursor.execute(
                        """
                        INSERT INTO system_notifications
                            (organization_id, alert_type, severity, title, details)
                        VALUES
                            (1, 'campaign_degraded', 'critical',
                             %s,
                             JSON_OBJECT(
                                 'campaign_id', %s,
                                 'suppression_rate', %s,
                                 'suppression_count', %s,
                                 'total_processed', %s,
                                 'action_taken', 'campaign_paused'
                             ))
                        """,
                        (
                            f"Campaign {campaign_id} auto-paused: suppression rate {suppression_rate:.1%}",
                            campaign_id,
                            round(suppression_rate, 4),
                            suppression_count,
                            total,
                        ),
                    )

                    connection.commit()
                    logger.critical(
                        "Campaign %d auto-paused: suppression rate %.1f%% (%d/%d)",
                        campaign_id,
                        suppression_rate * 100,
                        suppression_count,
                        total,
                    )
                else:
                    # Campaign not in 'sending' state — no action needed
                    logger.info(
                        "Campaign %d suppression rate %.1f%% but not in 'sending' state",
                        campaign_id,
                        suppression_rate * 100,
                    )
        except Exception as exc:
            if connection.is_connected():
                connection.rollback()
            logger.error(
                "Error checking suppression rate for campaign %d: %s",
                campaign_id,
                exc,
            )
            raise
        finally:
            if cursor:
                cursor.close()

    def schedule_retry(self, message_id: int) -> Optional[RetryDecision]:
        """
        Schedule a retry for a failed message if eligible.

        Updates the campaign_messages record with incremented retry_count
        and computed next_retry_at timestamp.

        Args:
            message_id: ID of the message to retry.

        Returns:
            RetryDecision if retry was scheduled, or None if not eligible.
        """
        decision = self.should_retry(message_id)
        if not decision.should_retry:
            return decision

        connection = self._get_connection()
        cursor = None
        try:
            cursor = connection.cursor()
            cursor.execute(
                """
                UPDATE campaign_messages
                SET retry_count = retry_count + 1,
                    next_retry_at = %s,
                    status = 'queued',
                    updated_at = NOW()
                WHERE id = %s
                """,
                (decision.next_retry_at, message_id),
            )
            connection.commit()
            logger.info(
                "Scheduled retry for message %d: attempt %d, backoff %ds, next at %s",
                message_id,
                decision.retry_count + 1,
                decision.backoff_seconds,
                decision.next_retry_at,
            )
            return decision
        except Exception as exc:
            if connection.is_connected():
                connection.rollback()
            logger.error("Error scheduling retry for message %d: %s", message_id, exc)
            raise
        finally:
            if cursor:
                cursor.close()

    def process_failure(self, message_id: int, error_code: int, error_message: str) -> None:
        """
        Process a message send failure end-to-end:
        1. Classify the error
        2. Handle based on category (retry, permanent fail, or suppress)

        This is the main entry point called by the Sending Queue when a
        message dispatch fails.

        Args:
            message_id: ID of the failed campaign_messages record.
            error_code: WhatsApp Business API error code.
            error_message: Human-readable error description.
        """
        # Classify the error
        classification = self.classify_error(error_code, error_message)

        # Update the message with error details
        connection = self._get_connection()
        cursor = None
        try:
            cursor = connection.cursor()
            cursor.execute(
                """
                UPDATE campaign_messages
                SET error_code = %s,
                    error_message = %s,
                    error_category = %s,
                    updated_at = NOW()
                WHERE id = %s
                """,
                (error_code, error_message, classification.category, message_id),
            )
            connection.commit()
        except Exception as exc:
            if connection.is_connected():
                connection.rollback()
            logger.error("Error updating message %d error details: %s", message_id, exc)
            raise
        finally:
            if cursor:
                cursor.close()

        # Handle based on category
        if classification.category == FailureCategoryEnum.TRANSIENT.value:
            if classification.should_retry:
                decision = self.schedule_retry(message_id)
                if decision and not decision.should_retry:
                    # Max retries exhausted — mark as failed
                    self._mark_failed(message_id)
            else:
                # Transient but should_retry=False (e.g., expired message)
                self._mark_failed(message_id)

        elif classification.category == FailureCategoryEnum.PERMANENT.value:
            self.handle_permanent_failure(message_id)

        elif classification.category == FailureCategoryEnum.SUPPRESSION.value:
            self.handle_suppression(message_id)

    def _mark_failed(self, message_id: int) -> None:
        """Mark a message as failed (all retries exhausted or non-retryable transient)."""
        connection = self._get_connection()
        cursor = None
        try:
            cursor = connection.cursor()
            cursor.execute(
                """
                UPDATE campaign_messages
                SET status = 'failed',
                    failed_at = NOW(),
                    updated_at = NOW()
                WHERE id = %s
                """,
                (message_id,),
            )
            connection.commit()
            logger.info("Message %d marked as failed (retries exhausted)", message_id)
        except Exception as exc:
            if connection.is_connected():
                connection.rollback()
            logger.error("Error marking message %d as failed: %s", message_id, exc)
            raise
        finally:
            if cursor:
                cursor.close()
