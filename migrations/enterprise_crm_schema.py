"""
Enterprise WhatsApp CRM — Database Schema Migration

This module creates all required tables for the Enterprise CRM system and safely
alters existing tables to add new columns. All operations are idempotent:
- CREATE TABLE IF NOT EXISTS ensures tables are only created when missing
- A stored procedure `add_column_if_not_exists` handles safe ALTER TABLE additions
- ensure_multi_tenant_columns() adds organization_id, branch_id, tenant_id to all tables

Called at application startup via ensure_crm_tables().
"""

import logging
from contextlib import contextmanager
from mysql.connector import Error as MySQLError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CREATE TABLE statements (new tables)
# ---------------------------------------------------------------------------

CREATE_TABLE_STATEMENTS = [
    # Campaign management
    """
    CREATE TABLE IF NOT EXISTS campaigns (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        organization_id BIGINT NOT NULL DEFAULT 1,
        branch_id BIGINT NOT NULL DEFAULT 1,
        name VARCHAR(255) NOT NULL,
        description TEXT,
        campaign_type ENUM('promotional', 'transactional', 'reactivation', 'ab_test') NOT NULL DEFAULT 'promotional',
        status ENUM('draft', 'scheduled', 'pending_approval', 'approved', 'sending', 'paused', 'completed', 'cancelled', 'failed') NOT NULL DEFAULT 'draft',
        segment_id BIGINT,
        template_id BIGINT,
        channel VARCHAR(50) NOT NULL DEFAULT 'whatsapp',
        scheduled_at DATETIME,
        approved_at DATETIME,
        approved_by VARCHAR(255),
        started_at DATETIME,
        completed_at DATETIME,
        priority TINYINT NOT NULL DEFAULT 5,
        recurring_frequency ENUM('none', 'daily', 'weekly', 'monthly') NOT NULL DEFAULT 'none',
        recurring_end_date DATE,
        parent_campaign_id BIGINT COMMENT 'For A/B test winner rollout',
        ab_test_percentage DECIMAL(5,2) COMMENT '10.00 to 50.00',
        total_recipients INT DEFAULT 0,
        sent_count INT DEFAULT 0,
        delivered_count INT DEFAULT 0,
        read_count INT DEFAULT 0,
        failed_count INT DEFAULT 0,
        created_by VARCHAR(255),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        INDEX idx_campaigns_status (status),
        INDEX idx_campaigns_scheduled (scheduled_at),
        INDEX idx_campaigns_org (organization_id),
        INDEX idx_campaigns_branch (organization_id, branch_id),
        INDEX idx_campaigns_segment (segment_id),
        INDEX idx_campaigns_template (template_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,

    # A/B test variants
    """
    CREATE TABLE IF NOT EXISTS campaign_ab_variants (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        campaign_id BIGINT NOT NULL,
        template_id BIGINT NOT NULL,
        variant_label VARCHAR(10) NOT NULL COMMENT 'A, B, C, D',
        recipient_count INT DEFAULT 0,
        sent_count INT DEFAULT 0,
        delivered_count INT DEFAULT 0,
        read_count INT DEFAULT 0,
        response_count INT DEFAULT 0,
        is_winner TINYINT(1) DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_ab_campaign (campaign_id),
        FOREIGN KEY (campaign_id) REFERENCES campaigns(id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,

    # Audience segments
    """
    CREATE TABLE IF NOT EXISTS audience_segments (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        organization_id BIGINT NOT NULL DEFAULT 1,
        name VARCHAR(255) NOT NULL,
        description TEXT,
        filter_criteria JSON NOT NULL COMMENT 'Stored filter definition',
        estimated_count INT DEFAULT 0,
        created_by VARCHAR(255),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        INDEX idx_segments_org (organization_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,

    # Campaign messages (queue + delivery log)
    """
    CREATE TABLE IF NOT EXISTS campaign_messages (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        campaign_id BIGINT NOT NULL,
        ab_variant_id BIGINT,
        customer_mobile VARCHAR(20) NOT NULL,
        customer_name VARCHAR(255),
        template_id BIGINT NOT NULL,
        template_params JSON,
        channel VARCHAR(50) NOT NULL DEFAULT 'whatsapp',
        status ENUM('queued', 'sending', 'sent', 'delivered', 'read', 'failed', 'permanently_failed', 'skipped') NOT NULL DEFAULT 'queued',
        whatsapp_message_id VARCHAR(255),
        error_code INT,
        error_message TEXT,
        error_category ENUM('transient', 'permanent', 'suppression'),
        retry_count TINYINT DEFAULT 0,
        max_retries TINYINT DEFAULT 3,
        next_retry_at DATETIME,
        sent_at DATETIME,
        delivered_at DATETIME,
        read_at DATETIME,
        failed_at DATETIME,
        is_test_send TINYINT(1) NOT NULL DEFAULT 0 COMMENT 'Marks test send messages excluded from analytics',
        idempotency_key VARCHAR(255) NOT NULL COMMENT 'campaign_id + mobile + template_id',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        UNIQUE KEY uk_idempotency (idempotency_key),
        INDEX idx_cm_campaign_status (campaign_id, status),
        INDEX idx_cm_mobile (customer_mobile),
        INDEX idx_cm_whatsapp_msg (whatsapp_message_id),
        INDEX idx_cm_next_retry (status, next_retry_at),
        INDEX idx_cm_sent_at (sent_at),
        INDEX idx_cm_test_send (is_test_send),
        FOREIGN KEY (campaign_id) REFERENCES campaigns(id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,

    # Campaign templates (local registry, synced from Meta)
    """
    CREATE TABLE IF NOT EXISTS campaign_templates (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        organization_id BIGINT NOT NULL DEFAULT 1,
        template_name VARCHAR(255) NOT NULL,
        template_language VARCHAR(10) NOT NULL DEFAULT 'en',
        category ENUM('utility', 'marketing', 'authentication') NOT NULL DEFAULT 'marketing',
        status ENUM('pending', 'approved', 'rejected') NOT NULL DEFAULT 'pending',
        header_type ENUM('none', 'text', 'image', 'video', 'document') DEFAULT 'none',
        body_text TEXT,
        footer_text VARCHAR(60),
        placeholder_count INT DEFAULT 0,
        placeholder_mappings JSON COMMENT 'Maps {{1}} to customer fields',
        media_asset_id BIGINT,
        meta_template_id VARCHAR(255),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        INDEX idx_templates_org (organization_id),
        INDEX idx_templates_name (template_name),
        INDEX idx_templates_status (status)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,

    # Customer tags
    """
    CREATE TABLE IF NOT EXISTS customer_tags (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        organization_id BIGINT NOT NULL DEFAULT 1,
        branch_id BIGINT NOT NULL DEFAULT 1,
        customer_mobile VARCHAR(20) NOT NULL,
        tag_name VARCHAR(100) NOT NULL,
        added_by VARCHAR(255),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE KEY uk_customer_tag (customer_mobile, tag_name),
        INDEX idx_tags_tag (tag_name),
        INDEX idx_tags_org (organization_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,

    # Customer notes
    """
    CREATE TABLE IF NOT EXISTS customer_notes (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        organization_id BIGINT NOT NULL DEFAULT 1,
        branch_id BIGINT NOT NULL DEFAULT 1,
        customer_mobile VARCHAR(20) NOT NULL,
        note_text TEXT NOT NULL,
        added_by VARCHAR(255) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_notes_mobile (customer_mobile),
        INDEX idx_notes_org (organization_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,

    # Customer activity / interaction timeline
    """
    CREATE TABLE IF NOT EXISTS customer_activity (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        customer_mobile VARCHAR(20) NOT NULL,
        activity_type ENUM('message_sent', 'message_received', 'campaign_sent', 'note_added', 'tag_added', 'tag_removed', 'status_change', 'opt_out', 'opt_in') NOT NULL,
        channel VARCHAR(50) DEFAULT 'whatsapp',
        reference_id BIGINT COMMENT 'FK to related table record',
        details JSON,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_activity_mobile (customer_mobile),
        INDEX idx_activity_type (activity_type),
        INDEX idx_activity_created (created_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,

    # Campaign analytics (pre-computed summaries)
    """
    CREATE TABLE IF NOT EXISTS campaign_analytics (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        organization_id BIGINT NOT NULL DEFAULT 1,
        campaign_id BIGINT,
        metric_type VARCHAR(50) NOT NULL COMMENT 'delivery_rate, read_rate, quality_snapshot, etc.',
        metric_value DECIMAL(10,4),
        dimensions JSON COMMENT '{"zone": "...", "template": "...", "period": "..."}',
        period_start DATETIME,
        period_end DATETIME,
        computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_analytics_campaign (campaign_id),
        INDEX idx_analytics_type (metric_type),
        INDEX idx_analytics_period (period_start, period_end),
        INDEX idx_analytics_org (organization_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,

    # Automation rules
    """
    CREATE TABLE IF NOT EXISTS automation_rules (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        organization_id BIGINT NOT NULL DEFAULT 1,
        tenant_id BIGINT NOT NULL DEFAULT 1,
        name VARCHAR(255) NOT NULL,
        trigger_type ENUM('schedule', 'event', 'threshold') NOT NULL,
        trigger_config JSON NOT NULL,
        condition_config JSON,
        action_type ENUM('create_campaign_draft', 'notify_operator') NOT NULL,
        action_config JSON NOT NULL,
        is_active TINYINT(1) DEFAULT 1,
        last_triggered_at DATETIME,
        created_by VARCHAR(255),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        INDEX idx_rules_org (organization_id),
        INDEX idx_rules_active (is_active)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,

    # Media assets
    """
    CREATE TABLE IF NOT EXISTS media_assets (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        organization_id BIGINT NOT NULL DEFAULT 1,
        filename VARCHAR(255) NOT NULL,
        original_filename VARCHAR(255) NOT NULL,
        mime_type VARCHAR(100) NOT NULL,
        file_size_bytes BIGINT NOT NULL,
        media_type ENUM('image', 'video', 'document') NOT NULL,
        storage_path VARCHAR(500) NOT NULL,
        thumbnail_path VARCHAR(500),
        usage_count INT DEFAULT 0,
        uploaded_by VARCHAR(255),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_media_org (organization_id),
        INDEX idx_media_type (media_type)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,

    # Suppression list (opt-outs + blocks)
    """
    CREATE TABLE IF NOT EXISTS suppression_list (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        organization_id BIGINT NOT NULL DEFAULT 1,
        customer_mobile VARCHAR(20) NOT NULL,
        reason ENUM('opt_out_keyword', 'manual_dnd', 'user_blocked', 'spam_reported', 'invalid_number') NOT NULL,
        source_keyword VARCHAR(50),
        added_by VARCHAR(255),
        is_active TINYINT(1) DEFAULT 1,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        removed_at DATETIME,
        UNIQUE KEY uk_suppression_mobile (customer_mobile, reason),
        INDEX idx_suppression_active (is_active, customer_mobile),
        INDEX idx_suppression_org (organization_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,

    # Customer engagement scores
    """
    CREATE TABLE IF NOT EXISTS customer_engagement (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        customer_mobile VARCHAR(20) NOT NULL,
        messages_received_count INT DEFAULT 0,
        messages_read_count INT DEFAULT 0,
        response_count INT DEFAULT 0,
        avg_time_to_read_seconds INT,
        last_interaction_at DATETIME,
        interaction_score TINYINT DEFAULT 0 COMMENT '0-100',
        engagement_trend ENUM('increasing', 'stable', 'declining') DEFAULT 'stable',
        preferred_time_window ENUM('morning', 'afternoon', 'evening') DEFAULT 'afternoon',
        avg_response_time_seconds INT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        UNIQUE KEY uk_engagement_mobile (customer_mobile),
        INDEX idx_engagement_score (interaction_score)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,

    # Notifications
    """
    CREATE TABLE IF NOT EXISTS system_notifications (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        organization_id BIGINT NOT NULL DEFAULT 1,
        alert_type VARCHAR(50) NOT NULL COMMENT 'campaign_degraded, queue_overloaded, webhook_connectivity, template_rejected, quality_drop',
        severity ENUM('info', 'warning', 'critical') NOT NULL,
        title VARCHAR(255) NOT NULL,
        details JSON,
        target_operators JSON COMMENT 'List of operator names to notify',
        acknowledged_by VARCHAR(255),
        acknowledged_at DATETIME,
        delivered_push TINYINT(1) DEFAULT 0,
        delivered_whatsapp TINYINT(1) DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_notif_type (alert_type),
        INDEX idx_notif_severity (severity),
        INDEX idx_notif_ack (acknowledged_at),
        INDEX idx_notif_org (organization_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,

    # Error classification lookup
    """
    CREATE TABLE IF NOT EXISTS error_classifications (
        id INT AUTO_INCREMENT PRIMARY KEY,
        error_code INT NOT NULL,
        error_pattern VARCHAR(255),
        category ENUM('transient', 'permanent', 'suppression') NOT NULL,
        description VARCHAR(255),
        should_retry TINYINT(1) DEFAULT 0,
        UNIQUE KEY uk_error_code (error_code)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,

    # Cooldown tracking
    """
    CREATE TABLE IF NOT EXISTS message_cooldowns (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        customer_mobile VARCHAR(20) NOT NULL,
        campaign_id BIGINT NOT NULL,
        campaign_type VARCHAR(50) NOT NULL,
        sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_cooldown_mobile_type (customer_mobile, campaign_type, sent_at),
        INDEX idx_cooldown_sent (sent_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,

    # Quality metrics history
    """
    CREATE TABLE IF NOT EXISTS quality_metrics (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        period_start DATETIME NOT NULL,
        period_end DATETIME NOT NULL,
        blocked_count INT DEFAULT 0,
        failure_rate DECIMAL(5,4) DEFAULT 0,
        opt_out_rate DECIMAL(5,4) DEFAULT 0,
        read_rate DECIMAL(5,4) DEFAULT 0,
        quality_tier ENUM('green', 'yellow', 'red') DEFAULT 'green',
        computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_quality_period (period_start)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
]

# ---------------------------------------------------------------------------
# Stored procedure for safe ADD COLUMN IF NOT EXISTS
# MySQL does not natively support ADD COLUMN IF NOT EXISTS, so we use a
# procedure that checks INFORMATION_SCHEMA before altering.
# ---------------------------------------------------------------------------

CREATE_ADD_COLUMN_PROCEDURE = """
DROP PROCEDURE IF EXISTS add_column_if_not_exists;
"""

CREATE_ADD_COLUMN_PROCEDURE_BODY = """
CREATE PROCEDURE add_column_if_not_exists(
    IN p_table_name VARCHAR(64),
    IN p_column_name VARCHAR(64),
    IN p_column_definition TEXT,
    IN p_after_column VARCHAR(64)
)
BEGIN
    DECLARE v_column_exists INT DEFAULT 0;

    SELECT COUNT(*)
    INTO v_column_exists
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = DATABASE()
      AND TABLE_NAME = p_table_name
      AND COLUMN_NAME = p_column_name;

    IF v_column_exists = 0 THEN
        IF p_after_column IS NOT NULL AND p_after_column != '' THEN
            SET @ddl = CONCAT('ALTER TABLE `', p_table_name, '` ADD COLUMN `', p_column_name, '` ', p_column_definition, ' AFTER `', p_after_column, '`');
        ELSE
            SET @ddl = CONCAT('ALTER TABLE `', p_table_name, '` ADD COLUMN `', p_column_name, '` ', p_column_definition);
        END IF;
        PREPARE stmt FROM @ddl;
        EXECUTE stmt;
        DEALLOCATE PREPARE stmt;
    END IF;
END;
"""

# ---------------------------------------------------------------------------
# Stored procedure for safe ADD INDEX IF NOT EXISTS
# ---------------------------------------------------------------------------

CREATE_ADD_INDEX_PROCEDURE = """
DROP PROCEDURE IF EXISTS add_index_if_not_exists;
"""

CREATE_ADD_INDEX_PROCEDURE_BODY = """
CREATE PROCEDURE add_index_if_not_exists(
    IN p_table_name VARCHAR(64),
    IN p_index_name VARCHAR(64),
    IN p_index_columns TEXT
)
BEGIN
    DECLARE v_index_exists INT DEFAULT 0;

    SELECT COUNT(*)
    INTO v_index_exists
    FROM INFORMATION_SCHEMA.STATISTICS
    WHERE TABLE_SCHEMA = DATABASE()
      AND TABLE_NAME = p_table_name
      AND INDEX_NAME = p_index_name;

    IF v_index_exists = 0 THEN
        SET @ddl = CONCAT('ALTER TABLE `', p_table_name, '` ADD INDEX `', p_index_name, '` (', p_index_columns, ')');
        PREPARE stmt FROM @ddl;
        EXECUTE stmt;
        DEALLOCATE PREPARE stmt;
    END IF;
END;
"""

# ---------------------------------------------------------------------------
# ALTER TABLE statements for existing tables
# Uses the add_column_if_not_exists procedure for safe idempotent execution.
# ---------------------------------------------------------------------------

ALTER_EXISTING_TABLES = [
    # whatsapp_campaign_logs — add campaign_id and channel columns
    "CALL add_column_if_not_exists('whatsapp_campaign_logs', 'campaign_id', 'BIGINT', 'id')",
    "CALL add_column_if_not_exists('whatsapp_campaign_logs', 'channel', \"VARCHAR(50) DEFAULT 'whatsapp'\", 'campaign_id')",
    "CALL add_index_if_not_exists('whatsapp_campaign_logs', 'idx_wcl_campaign', 'campaign_id')",

    # operator_actions — add campaign_id column
    "CALL add_column_if_not_exists('operator_actions', 'campaign_id', 'BIGINT', 'target_id')",
    "CALL add_index_if_not_exists('operator_actions', 'idx_oa_campaign', 'campaign_id')",

    # renewal_records — add segmentation-related columns
    "CALL add_column_if_not_exists('renewal_records', 'area', 'VARCHAR(255)', 'zone_name')",
    "CALL add_column_if_not_exists('renewal_records', 'building', 'VARCHAR(255)', 'zone_name')",
    "CALL add_column_if_not_exists('renewal_records', 'network_type', 'VARCHAR(50)', 'zone_name')",
    "CALL add_column_if_not_exists('renewal_records', 'connectivity_mode', 'VARCHAR(50)', 'network_type')",
    "CALL add_column_if_not_exists('renewal_records', 'plan_category', 'VARCHAR(100)', 'plan_name')",
    "CALL add_column_if_not_exists('renewal_records', 'status', \"VARCHAR(50) DEFAULT 'active'\", 'category')",
    "CALL add_column_if_not_exists('renewal_records', 'activation_date', 'DATE', 'expiry_date')",
    "CALL add_column_if_not_exists('renewal_records', 'kyc_approved', 'TINYINT(1) DEFAULT 0', 'status')",
    "CALL add_column_if_not_exists('renewal_records', 'owner_tenant', 'VARCHAR(50)', 'kyc_approved')",

    # campaign_messages — add is_test_send column for test send distinction
    "CALL add_column_if_not_exists('campaign_messages', 'is_test_send', 'TINYINT(1) NOT NULL DEFAULT 0', 'failed_at')",
    "CALL add_index_if_not_exists('campaign_messages', 'idx_cm_test_send', 'is_test_send')",
]


# ---------------------------------------------------------------------------
# Multi-tenant column helpers
# ---------------------------------------------------------------------------

@contextmanager
def _get_cursor(get_mysql_connection):
    """Context manager for MySQL connection and cursor with auto-commit."""
    conn = None
    cursor = None
    try:
        conn = get_mysql_connection()
        cursor = conn.cursor()
        yield cursor
        conn.commit()
    except Exception:
        if conn and conn.is_connected():
            conn.rollback()
        raise
    finally:
        if cursor:
            try:
                cursor.close()
            except Exception:
                pass
        if conn and conn.is_connected():
            try:
                conn.close()
            except Exception:
                pass


def _column_exists(cursor, table_name, column_name):
    """Check if a column exists on a table using INFORMATION_SCHEMA."""
    cursor.execute(
        "SELECT COUNT(*) FROM information_schema.COLUMNS "
        "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s AND COLUMN_NAME = %s",
        (table_name, column_name)
    )
    result = cursor.fetchone()
    return result[0] > 0 if result else False


def _index_exists(cursor, table_name, index_name):
    """Check if an index exists on a table using INFORMATION_SCHEMA."""
    cursor.execute(
        "SELECT COUNT(*) FROM information_schema.STATISTICS "
        "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s AND INDEX_NAME = %s",
        (table_name, index_name)
    )
    result = cursor.fetchone()
    return result[0] > 0 if result else False


def _table_exists(cursor, table_name):
    """Check if a table exists in the current database."""
    cursor.execute(
        "SELECT COUNT(*) FROM information_schema.TABLES "
        "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s",
        (table_name,)
    )
    result = cursor.fetchone()
    return result[0] > 0 if result else False


def _add_column_if_not_exists(cursor, table_name, column_name, column_def):
    """
    Add a column to a table if it doesn't already exist.
    Uses try/except pattern since MySQL doesn't support ADD COLUMN IF NOT EXISTS.
    """
    if not _column_exists(cursor, table_name, column_name):
        try:
            sql = f"ALTER TABLE `{table_name}` ADD COLUMN `{column_name}` {column_def}"
            cursor.execute(sql)
            logger.info("Added column %s.%s", table_name, column_name)
            return True
        except MySQLError as e:
            # Handle race conditions where column was added between check and alter
            error_msg = str(e).lower()
            if 'duplicate column' in error_msg or 'already exists' in error_msg:
                logger.debug("Column %s.%s already exists (race condition)", table_name, column_name)
                return False
            raise
    else:
        logger.debug("Column %s.%s already exists, skipping", table_name, column_name)
        return False


def _add_index_if_not_exists(cursor, table_name, index_name, columns):
    """
    Add a composite index if it doesn't already exist.
    Uses try/except pattern for idempotent index creation.
    """
    if not _index_exists(cursor, table_name, index_name):
        try:
            cols = ", ".join(f"`{c}`" for c in columns)
            sql = f"CREATE INDEX `{index_name}` ON `{table_name}` ({cols})"
            cursor.execute(sql)
            logger.info("Created index %s on %s", index_name, table_name)
            return True
        except MySQLError as e:
            error_msg = str(e).lower()
            if 'duplicate' in error_msg or 'already exists' in error_msg:
                logger.debug("Index %s on %s already exists (race condition)", index_name, table_name)
                return False
            raise
    else:
        logger.debug("Index %s on %s already exists, skipping", index_name, table_name)
        return False


def ensure_multi_tenant_columns(get_connection_func):
    """
    Ensure all CRM tables have organization_id, branch_id, and tenant_id columns
    with DEFAULT 1 for backward compatibility with single-tenant operation.

    Also creates composite indexes: idx_{table}_tenant (organization_id, branch_id, tenant_id)
    on each table for efficient multi-tenant query scoping.

    This function is idempotent and safe to call multiple times.

    Validates: Requirements 24.1, 24.2, 24.3, 24.6
    """
    # Tables that already have organization_id AND branch_id in their CREATE TABLE
    # These only need tenant_id added if missing
    tables_with_org_and_branch = [
        'campaigns',
        'customer_tags',
        'customer_notes',
    ]

    # Tables that have organization_id but NOT branch_id in CREATE TABLE
    # These need branch_id and tenant_id added
    tables_with_org_only = [
        'audience_segments',
        'campaign_templates',
        'campaign_analytics',
        'media_assets',
        'suppression_list',
        'system_notifications',
    ]

    # automation_rules has organization_id AND tenant_id but NOT branch_id
    tables_with_org_and_tenant = [
        'automation_rules',
    ]

    # Tables that DON'T have any of the three multi-tenant columns in CREATE TABLE
    # These need all three added
    tables_without_multi_tenant = [
        'campaign_ab_variants',
        'campaign_messages',
        'customer_activity',
        'customer_engagement',
        'message_cooldowns',
        'quality_metrics',
        'error_classifications',
    ]

    # All tables that should have the composite tenant index
    all_crm_tables = (
        tables_with_org_and_branch
        + tables_with_org_only
        + tables_with_org_and_tenant
        + tables_without_multi_tenant
    )

    with _get_cursor(get_connection_func) as cursor:
        added_columns = 0
        added_indexes = 0

        # --- Tables with org_id + branch_id: add tenant_id only ---
        for table in tables_with_org_and_branch:
            if not _table_exists(cursor, table):
                logger.debug("Table %s does not exist yet, skipping", table)
                continue
            if _add_column_if_not_exists(
                cursor, table, 'tenant_id',
                'BIGINT NOT NULL DEFAULT 1'
            ):
                added_columns += 1

        # --- Tables with org_id only: add branch_id and tenant_id ---
        for table in tables_with_org_only:
            if not _table_exists(cursor, table):
                logger.debug("Table %s does not exist yet, skipping", table)
                continue
            if _add_column_if_not_exists(
                cursor, table, 'branch_id',
                'BIGINT NOT NULL DEFAULT 1'
            ):
                added_columns += 1
            if _add_column_if_not_exists(
                cursor, table, 'tenant_id',
                'BIGINT NOT NULL DEFAULT 1'
            ):
                added_columns += 1

        # --- Tables with org_id + tenant_id: add branch_id only ---
        for table in tables_with_org_and_tenant:
            if not _table_exists(cursor, table):
                logger.debug("Table %s does not exist yet, skipping", table)
                continue
            if _add_column_if_not_exists(
                cursor, table, 'branch_id',
                'BIGINT NOT NULL DEFAULT 1'
            ):
                added_columns += 1

        # --- Tables without any multi-tenant columns: add all three ---
        for table in tables_without_multi_tenant:
            if not _table_exists(cursor, table):
                logger.debug("Table %s does not exist yet, skipping", table)
                continue
            if _add_column_if_not_exists(
                cursor, table, 'organization_id',
                'BIGINT NOT NULL DEFAULT 1'
            ):
                added_columns += 1
            if _add_column_if_not_exists(
                cursor, table, 'branch_id',
                'BIGINT NOT NULL DEFAULT 1'
            ):
                added_columns += 1
            if _add_column_if_not_exists(
                cursor, table, 'tenant_id',
                'BIGINT NOT NULL DEFAULT 1'
            ):
                added_columns += 1

        # --- Create composite tenant indexes on all tables ---
        for table in all_crm_tables:
            if not _table_exists(cursor, table):
                logger.debug("Table %s does not exist yet, skipping index creation", table)
                continue
            index_name = f"idx_{table}_tenant"
            if _add_index_if_not_exists(
                cursor, table, index_name,
                ['organization_id', 'branch_id', 'tenant_id']
            ):
                added_indexes += 1

    logger.info(
        "ensure_multi_tenant_columns complete: %d columns added, %d indexes created",
        added_columns, added_indexes
    )
    return {'columns_added': added_columns, 'indexes_created': added_indexes}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def ensure_crm_tables(get_connection_func):
    """
    Create all Enterprise CRM tables and alter existing tables as needed.

    This function is idempotent and safe to call on every application startup.
    It uses CREATE TABLE IF NOT EXISTS for new tables and a stored procedure
    to safely add columns to existing tables only when they don't already exist.

    Args:
        get_connection_func: A callable that returns a mysql.connector connection.
    """
    conn = None
    cursor = None
    try:
        conn = get_connection_func()
        cursor = conn.cursor()

        # 1. Create helper stored procedures
        logger.info("CRM Migration: Creating helper stored procedures...")
        cursor.execute(CREATE_ADD_COLUMN_PROCEDURE)
        cursor.execute(CREATE_ADD_COLUMN_PROCEDURE_BODY)
        cursor.execute(CREATE_ADD_INDEX_PROCEDURE)
        cursor.execute(CREATE_ADD_INDEX_PROCEDURE_BODY)
        conn.commit()

        # 2. Create new tables
        logger.info("CRM Migration: Creating CRM tables (IF NOT EXISTS)...")
        for i, statement in enumerate(CREATE_TABLE_STATEMENTS):
            try:
                cursor.execute(statement)
                conn.commit()
            except MySQLError as e:
                logger.warning(
                    "CRM Migration: Table creation statement %d raised: %s", i + 1, e
                )
                conn.rollback()

        # 3. Alter existing tables (add columns + indexes)
        logger.info("CRM Migration: Altering existing tables...")
        for alter_stmt in ALTER_EXISTING_TABLES:
            try:
                cursor.execute(alter_stmt)
                conn.commit()
            except MySQLError as e:
                # Log but continue — table might not exist yet in dev environments
                logger.warning(
                    "CRM Migration: ALTER statement raised: %s — Statement: %s",
                    e,
                    alter_stmt[:100],
                )
                conn.rollback()

        logger.info("CRM Migration: Schema migration completed successfully.")

    except MySQLError as e:
        logger.error("CRM Migration: Fatal error during migration: %s", e, exc_info=True)
        if conn:
            try:
                conn.rollback()
            except Exception:
                pass
        raise
    finally:
        if cursor:
            try:
                cursor.close()
            except Exception:
                pass
        if conn:
            try:
                conn.close()
            except Exception:
                pass

    # 4. Ensure multi-tenant columns exist on all CRM tables
    logger.info("CRM Migration: Ensuring multi-tenant columns...")
    ensure_multi_tenant_columns(get_connection_func)
