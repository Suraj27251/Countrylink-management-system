<?php

declare(strict_types=1);
date_default_timezone_set('Asia/Kolkata');

$host = getenv('DB_HOST') ?: 'localhost';
$dbName = getenv('DB_NAME') ?: 'countrylink_db';
$dbUser = getenv('DB_USER') ?: 'root';
$dbPass = getenv('DB_PASS') ?: '';
$charset = 'utf8mb4';

$dsn = "mysql:host={$host};dbname={$dbName};charset={$charset}";

$options = [
    PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION,
    PDO::ATTR_DEFAULT_FETCH_MODE => PDO::FETCH_ASSOC,
    PDO::ATTR_EMULATE_PREPARES => false,
];

try {
    $pdo = new PDO($dsn, $dbUser, $dbPass, $options);

    $pdo->exec(
        "CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(150) NOT NULL,
            email VARCHAR(150) NOT NULL UNIQUE,
            phone VARCHAR(20) NOT NULL,
            password VARCHAR(255) NOT NULL,
            status ENUM('pending','active') DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"
    );

    $pdo->exec(
        "CREATE TABLE IF NOT EXISTS otp_verifications (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NOT NULL,
            otp_code VARCHAR(10) NOT NULL,
            expires_at DATETIME NOT NULL,
            verified BOOLEAN DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"
    );

    $pdo->exec(
        "CREATE TABLE IF NOT EXISTS whatsapp_logs (
            id INT AUTO_INCREMENT PRIMARY KEY,
            invoice_id VARCHAR(100) NOT NULL,
            customer_name VARCHAR(255) NOT NULL,
            phone VARCHAR(30) NOT NULL,
            template_name VARCHAR(100) NOT NULL,
            status ENUM('sent','delivered','read','failed') NOT NULL DEFAULT 'sent',
            error_message VARCHAR(255) DEFAULT NULL,
            message_id VARCHAR(255) DEFAULT NULL,
            attempts INT NOT NULL DEFAULT 1,
            sent_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            sent_date DATE GENERATED ALWAYS AS (DATE(sent_at)) STORED,
            updated_at TIMESTAMP NULL DEFAULT NULL ON UPDATE CURRENT_TIMESTAMP,
            UNIQUE KEY unique_invoice_day (invoice_id, sent_date),
            KEY idx_whatsapp_logs_invoice_id (invoice_id),
            KEY idx_whatsapp_logs_status (status),
            KEY idx_whatsapp_logs_sent_at (sent_at),
            KEY idx_whatsapp_logs_message_id (message_id),
            KEY idx_invoice_status_date (invoice_id, status, sent_at)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"
    );

    // Backward-compatible hardening for existing deployments.
    // IMPORTANT: do not fail app bootstrap if schema migration syntax differs across MySQL/MariaDB versions.
    try {
        $columnStmt = $pdo->query("SHOW COLUMNS FROM whatsapp_logs");
        $existingColumns = [];
        foreach ($columnStmt->fetchAll() as $column) {
            $existingColumns[$column['Field']] = true;
        }

        if (!isset($existingColumns['error_message'])) {
            $pdo->exec("ALTER TABLE whatsapp_logs ADD COLUMN error_message VARCHAR(255) DEFAULT NULL");
        }
        if (!isset($existingColumns['sent_date'])) {
            $pdo->exec("ALTER TABLE whatsapp_logs ADD COLUMN sent_date DATE GENERATED ALWAYS AS (DATE(sent_at)) STORED");
        }

        $indexStmt = $pdo->query("SHOW INDEX FROM whatsapp_logs");
        $existingIndexes = [];
        foreach ($indexStmt->fetchAll() as $index) {
            $existingIndexes[$index['Key_name']] = true;
        }

        if (!isset($existingIndexes['unique_invoice_day'])) {
            $pdo->exec("ALTER TABLE whatsapp_logs ADD UNIQUE KEY unique_invoice_day (invoice_id, sent_date)");
        }
        if (!isset($existingIndexes['idx_invoice_status_date'])) {
            $pdo->exec("CREATE INDEX idx_invoice_status_date ON whatsapp_logs (invoice_id, status, sent_at)");
        }
    } catch (PDOException $migrationException) {
        error_log('whatsapp_logs migration warning: ' . $migrationException->getMessage());
    }
} catch (PDOException $e) {
    http_response_code(500);
    exit('Database connection failed.');
}
