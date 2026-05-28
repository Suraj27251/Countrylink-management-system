-- Renewal Campaign Tables for WhatsApp Template Sending
-- Database: countrylinks_user_database
-- Run this migration to create the required tables

-- 1. renewal_records - Source of customer renewal data
CREATE TABLE IF NOT EXISTS renewal_records (
    id INT AUTO_INCREMENT PRIMARY KEY,
    customer_name VARCHAR(255),
    mobile VARCHAR(20),
    account_id VARCHAR(100),
    plan_name VARCHAR(255),
    expiry_date DATE,
    days_remaining INT,
    category VARCHAR(50) COMMENT 'expired | today | upcoming',
    zone_name VARCHAR(100),
    amount VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY unique_account (account_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 2. whatsapp_campaign_logs - Logs every template send attempt
CREATE TABLE IF NOT EXISTS whatsapp_campaign_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    renewal_id INT COMMENT 'FK to renewal_records.id',
    mobile VARCHAR(20) COMMENT 'recipient phone number (10 digits)',
    template_name VARCHAR(100) COMMENT 'e.g. pack_expiry_alert',
    template_params JSON COMMENT 'e.g. ["Avinash", "ACC123", "2026-05-26"]',
    status VARCHAR(50) COMMENT 'sent | failed | pending',
    whatsapp_message_id VARCHAR(255) COMMENT 'Meta API message ID (null if failed)',
    operator_name VARCHAR(255) COMMENT 'who triggered it (e.g. whatsapp_inbox)',
    error_message TEXT COMMENT 'error details if failed',
    sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_mobile_template (mobile, template_name),
    INDEX idx_renewal_id (renewal_id),
    INDEX idx_sent_at (sent_at),
    INDEX idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 3. operator_actions - Audit trail for accountability
CREATE TABLE IF NOT EXISTS operator_actions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    operator_name VARCHAR(255) COMMENT 'e.g. whatsapp_inbox',
    action_type VARCHAR(100) COMMENT 'send_message | bulk_send',
    target_id INT COMMENT 'renewal_records.id',
    details JSON COMMENT '{"template": "...", "mobile": "..."}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_operator (operator_name),
    INDEX idx_action_type (action_type),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
