"""
Opt-Out Manager for WhatsApp CRM.

Handles opt-out/opt-in keyword recognition, suppression list management,
and DND (Do Not Disturb) enforcement for all outbound campaigns.

Requirements: 19.1, 19.2, 19.3, 19.4, 19.5, 19.6
"""

import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# Opt-out keywords (case-insensitive matching)
OPT_OUT_KEYWORDS = {"stop", "unsubscribe", "opt out", "cancel", "dnd"}

# Opt-in keywords (case-insensitive matching)
OPT_IN_KEYWORDS = {"start", "subscribe"}


class OptOutManager:
    """
    Manages customer opt-out/opt-in and suppression list.

    Processes unsubscribe requests via keyword detection, maintains the
    suppression_list table, and provides suppression checks for dispatch.
    """

    def __init__(self, get_connection, send_message_fn=None):
        """
        Initialize the OptOutManager.

        Args:
            get_connection: Callable that returns a MySQL connection.
            send_message_fn: Optional callable to send WhatsApp messages.
                             Signature: send_message_fn(to_number, message_type, text=None)
                             If None, confirmation messages are skipped.
        """
        self._get_connection = get_connection
        self._send_message = send_message_fn

    def is_opt_out_keyword(self, text: str) -> bool:
        """Check if text matches an opt-out keyword (case-insensitive)."""
        if not text:
            return False
        return text.strip().lower() in OPT_OUT_KEYWORDS

    def is_opt_in_keyword(self, text: str) -> bool:
        """Check if text matches an opt-in keyword (case-insensitive)."""
        if not text:
            return False
        return text.strip().lower() in OPT_IN_KEYWORDS

    def process_opt_out(self, mobile: str, keyword: str) -> None:
        """
        Process an opt-out request from a customer.

        Adds the customer to the suppression_list with reason 'opt_out_keyword',
        records the source keyword, and sends a confirmation message.

        Args:
            mobile: Customer mobile number.
            keyword: The opt-out keyword that was detected.
        """
        conn = None
        cursor = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor(dictionary=True)
            conn.start_transaction()

            # Check if already suppressed with opt_out_keyword reason
            cursor.execute(
                """
                SELECT id, is_active FROM suppression_list
                WHERE customer_mobile = %s AND reason = 'opt_out_keyword'
                LIMIT 1
                """,
                (mobile,),
            )
            existing = cursor.fetchone()

            if existing and existing.get("is_active"):
                # Already opted out — no action needed
                conn.commit()
                logger.info(
                    "Customer %s already opted out, skipping duplicate.",
                    mobile,
                )
            elif existing and not existing.get("is_active"):
                # Re-activate existing suppression record
                cursor.execute(
                    """
                    UPDATE suppression_list
                    SET is_active = 1, source_keyword = %s, removed_at = NULL,
                        created_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                    """,
                    (keyword.strip().lower(), existing["id"]),
                )
                conn.commit()
                logger.info(
                    "Customer %s re-opted out (reactivated suppression). Keyword: %s",
                    mobile,
                    keyword,
                )
            else:
                # Insert new suppression record
                cursor.execute(
                    """
                    INSERT INTO suppression_list
                        (customer_mobile, reason, source_keyword, added_by, is_active, created_at)
                    VALUES (%s, 'opt_out_keyword', %s, 'system', 1, CURRENT_TIMESTAMP)
                    """,
                    (mobile, keyword.strip().lower()),
                )
                conn.commit()
                logger.info(
                    "Customer %s added to suppression list. Keyword: %s",
                    mobile,
                    keyword,
                )

            # Record activity in customer_activity
            try:
                cursor.execute(
                    """
                    INSERT INTO customer_activity
                        (customer_mobile, activity_type, channel, details, created_at)
                    VALUES (%s, 'opt_out', 'whatsapp', %s, CURRENT_TIMESTAMP)
                    """,
                    (mobile, f'{{"keyword": "{keyword.strip().lower()}"}}'),
                )
                conn.commit()
            except Exception:
                logger.warning(
                    "Failed to record opt-out activity for %s", mobile
                )

            # Send confirmation message
            self._send_opt_out_confirmation(mobile)

        except Exception:
            if conn:
                try:
                    conn.rollback()
                except Exception:
                    pass
            logger.exception("Failed to process opt-out for mobile=%s", mobile)
            raise
        finally:
            if cursor:
                cursor.close()
            if conn and conn.is_connected():
                conn.close()

    def process_opt_in(self, mobile: str, keyword: str) -> None:
        """
        Process an opt-in (re-subscribe) request from a customer.

        Removes the customer from the suppression_list by setting is_active=0
        and recording the removal timestamp.

        Args:
            mobile: Customer mobile number.
            keyword: The opt-in keyword that was detected (START or SUBSCRIBE).
        """
        conn = None
        cursor = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor(dictionary=True)
            conn.start_transaction()

            # Deactivate all active suppression records for this mobile
            # that were added via opt_out_keyword
            cursor.execute(
                """
                UPDATE suppression_list
                SET is_active = 0, removed_at = NOW()
                WHERE customer_mobile = %s
                  AND reason = 'opt_out_keyword'
                  AND is_active = 1
                """,
                (mobile,),
            )
            rows_affected = cursor.rowcount
            conn.commit()

            if rows_affected > 0:
                logger.info(
                    "Customer %s opted back in (removed from suppression). Keyword: %s",
                    mobile,
                    keyword,
                )
            else:
                logger.info(
                    "Customer %s sent opt-in keyword but was not suppressed. Keyword: %s",
                    mobile,
                    keyword,
                )

            # Record activity in customer_activity
            try:
                cursor.execute(
                    """
                    INSERT INTO customer_activity
                        (customer_mobile, activity_type, channel, details, created_at)
                    VALUES (%s, 'opt_in', 'whatsapp', %s, CURRENT_TIMESTAMP)
                    """,
                    (mobile, f'{{"keyword": "{keyword.strip().lower()}"}}'),
                )
                conn.commit()
            except Exception:
                logger.warning(
                    "Failed to record opt-in activity for %s", mobile
                )

            # Send confirmation message
            self._send_opt_in_confirmation(mobile)

        except Exception:
            if conn:
                try:
                    conn.rollback()
                except Exception:
                    pass
            logger.exception("Failed to process opt-in for mobile=%s", mobile)
            raise
        finally:
            if cursor:
                cursor.close()
            if conn and conn.is_connected():
                conn.close()

    def is_suppressed(self, mobile: str) -> bool:
        """
        Check if a customer is on the active suppression list.

        Returns True if there is ANY active (is_active=1) suppression record
        for the given mobile number, regardless of reason.

        Args:
            mobile: Customer mobile number to check.

        Returns:
            True if the customer is suppressed, False otherwise.
        """
        conn = None
        cursor = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor(dictionary=True)

            cursor.execute(
                """
                SELECT 1 FROM suppression_list
                WHERE customer_mobile = %s AND is_active = 1
                LIMIT 1
                """,
                (mobile,),
            )
            result = cursor.fetchone()
            return result is not None

        except Exception:
            logger.exception(
                "Failed to check suppression status for mobile=%s", mobile
            )
            # Fail-safe: if we can't check, treat as not suppressed
            # to avoid blocking legitimate sends on DB errors.
            # Operators can manually review.
            return False
        finally:
            if cursor:
                cursor.close()
            if conn and conn.is_connected():
                conn.close()

    def add_to_dnd(self, mobile: str, reason: str, operator: str) -> None:
        """
        Manually add a customer to the DND (Do Not Disturb) list.

        This is used by operators to manually suppress a customer with a
        documented reason.

        Args:
            mobile: Customer mobile number.
            reason: Reason for DND addition (free text explanation).
            operator: Name/ID of the operator adding the DND entry.
        """
        conn = None
        cursor = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor(dictionary=True)
            conn.start_transaction()

            # Check if already has an active manual_dnd entry
            cursor.execute(
                """
                SELECT id, is_active FROM suppression_list
                WHERE customer_mobile = %s AND reason = 'manual_dnd'
                LIMIT 1
                """,
                (mobile,),
            )
            existing = cursor.fetchone()

            if existing and existing.get("is_active"):
                # Already on DND — no action needed
                conn.commit()
                logger.info(
                    "Customer %s already on DND list, skipping.", mobile
                )
                return
            elif existing and not existing.get("is_active"):
                # Re-activate existing DND record
                cursor.execute(
                    """
                    UPDATE suppression_list
                    SET is_active = 1, added_by = %s, source_keyword = %s,
                        removed_at = NULL, created_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                    """,
                    (operator, reason, existing["id"]),
                )
            else:
                # Insert new DND record
                cursor.execute(
                    """
                    INSERT INTO suppression_list
                        (customer_mobile, reason, source_keyword, added_by, is_active, created_at)
                    VALUES (%s, 'manual_dnd', %s, %s, 1, CURRENT_TIMESTAMP)
                    """,
                    (mobile, reason, operator),
                )

            conn.commit()
            logger.info(
                "Customer %s added to DND by operator=%s reason=%s",
                mobile,
                operator,
                reason,
            )

            # Record activity
            try:
                cursor.execute(
                    """
                    INSERT INTO customer_activity
                        (customer_mobile, activity_type, channel, details, created_at)
                    VALUES (%s, 'opt_out', 'whatsapp', %s, CURRENT_TIMESTAMP)
                    """,
                    (
                        mobile,
                        f'{{"reason": "manual_dnd", "operator": "{operator}", "note": "{reason}"}}',
                    ),
                )
                conn.commit()
            except Exception:
                logger.warning(
                    "Failed to record DND activity for %s", mobile
                )

        except Exception:
            if conn:
                try:
                    conn.rollback()
                except Exception:
                    pass
            logger.exception(
                "Failed to add mobile=%s to DND list", mobile
            )
            raise
        finally:
            if cursor:
                cursor.close()
            if conn and conn.is_connected():
                conn.close()

    def _send_opt_out_confirmation(self, mobile: str) -> None:
        """Send a confirmation message acknowledging the opt-out."""
        if not self._send_message:
            logger.info(
                "No send_message_fn configured; skipping opt-out confirmation for %s",
                mobile,
            )
            return

        try:
            self._send_message(
                mobile,
                "text",
                text=(
                    "You have been unsubscribed from our messages. "
                    "You will no longer receive promotional messages from us. "
                    "Reply START to re-subscribe at any time."
                ),
            )
            logger.info("Opt-out confirmation sent to %s", mobile)
        except Exception:
            logger.warning(
                "Failed to send opt-out confirmation to %s", mobile
            )

    def _send_opt_in_confirmation(self, mobile: str) -> None:
        """Send a confirmation message acknowledging the re-subscription."""
        if not self._send_message:
            logger.info(
                "No send_message_fn configured; skipping opt-in confirmation for %s",
                mobile,
            )
            return

        try:
            self._send_message(
                mobile,
                "text",
                text=(
                    "Welcome back! You have been re-subscribed to our messages. "
                    "Reply STOP at any time to unsubscribe."
                ),
            )
            logger.info("Opt-in confirmation sent to %s", mobile)
        except Exception:
            logger.warning(
                "Failed to send opt-in confirmation to %s", mobile
            )
