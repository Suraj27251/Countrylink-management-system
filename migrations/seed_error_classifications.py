"""
Seed data for the error_classifications table.

Populates known WhatsApp Business API error codes mapped to failure categories
(transient, permanent, suppression) for the Retry_Categorizer service.

Uses INSERT IGNORE for idempotency — safe to run multiple times without
duplicating rows.

Requirement: 21.5 — THE Retry_Categorizer SHALL maintain an error classification
lookup table mapping WhatsApp Business API error codes to failure categories,
updatable by administrators without code changes.
"""

import logging

logger = logging.getLogger(__name__)

# Known WhatsApp Business API error codes with classification
# Format: (error_code, error_pattern, category, description, should_retry)
ERROR_CLASSIFICATIONS = [
    (131047, 'rate_limit', 'transient', 'Rate limit hit — too many messages sent too quickly', 1),
    (131026, 'invalid_number', 'permanent', 'Invalid phone number — recipient number is not valid', 0),
    (131056, 'user_blocked', 'suppression', 'Blocked by user — recipient has blocked this business', 0),
    (131053, 'media_download_failed', 'transient', 'Media download failed — could not fetch media for message', 1),
    (131031, 'account_locked', 'permanent', 'Business account locked — account has been restricted', 0),
    (131021, 'not_on_whatsapp', 'permanent', 'Recipient not on WhatsApp — number is not registered', 0),
    (131045, 'spam_rate_limit', 'transient', 'Spam rate limit — too many similar messages flagged', 1),
    (131049, 'message_expired', 'transient', 'Message too old or expired — delivery window exceeded', 0),
    (131051, 'unsupported_message_type', 'permanent', 'Unsupported message type — message format not supported', 0),
    (130472, 'experiment_paused', 'transient', 'Experiment paused — Meta has paused this campaign experiment', 1),
    (368, 'temporarily_blocked_policy', 'suppression', 'Temporarily blocked for policy violations', 0),
    (131000, 'generic_error', 'transient', 'Generic error — something went wrong, retry may succeed', 1),
]


def seed_error_classifications(connection):
    """
    Seed the error_classifications table with known WhatsApp Business API error codes.

    Uses INSERT IGNORE to make this idempotent — running multiple times will not
    create duplicate entries (relies on UNIQUE KEY uk_error_code on error_code column).

    Args:
        connection: An active mysql.connector connection to the database.
    """
    cursor = None
    try:
        cursor = connection.cursor()

        insert_sql = """
            INSERT IGNORE INTO error_classifications
                (error_code, error_pattern, category, description, should_retry)
            VALUES
                (%s, %s, %s, %s, %s)
        """

        rows_inserted = 0
        for error_code, error_pattern, category, description, should_retry in ERROR_CLASSIFICATIONS:
            cursor.execute(insert_sql, (error_code, error_pattern, category, description, should_retry))
            if cursor.rowcount > 0:
                rows_inserted += 1

        connection.commit()
        logger.info(
            "Seeded error_classifications: %d new rows inserted (%d total known codes)",
            rows_inserted,
            len(ERROR_CLASSIFICATIONS),
        )
        return rows_inserted

    except Exception as exc:
        if connection.is_connected():
            connection.rollback()
        logger.error("Failed to seed error_classifications: %s", exc, exc_info=True)
        raise
    finally:
        if cursor:
            cursor.close()
