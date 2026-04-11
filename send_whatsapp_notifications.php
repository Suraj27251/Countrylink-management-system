<?php

declare(strict_types=1);

date_default_timezone_set('Asia/Kolkata');

require_once __DIR__ . '/db.php';

const OVERDUE_TEMPLATE_NAME = 'payment_overdue_2';
const MAX_ATTEMPTS = 3;
const ERROR_LOG_FILE = '/home/countrylinks/whatsapp_error.log';

/**
 * Reuses existing WhatsApp sender if present.
 *
 * Expected signature:
 * sendWhatsAppTemplate(string $phone, string $templateName, array $params): array
 */
function sendOverdueTemplate(string $phone, string $templateName, array $params): array
{
    try {
        if (function_exists('sendWhatsAppTemplate')) {
            /** @var array<string,mixed> $result */
            $result = sendWhatsAppTemplate($phone, $templateName, $params);
            return $result;
        }

        // Fallback keeps the same credentials/config pattern used elsewhere in this codebase.
        $apiToken = getenv('META_ACCESS_TOKEN') ?: '';
        $phoneNumberId = getenv('META_PHONE_NUMBER_ID') ?: '';

        if ($apiToken === '' || $phoneNumberId === '') {
            return ['success' => false, 'message' => 'WhatsApp credentials are not configured.'];
        }

        $payload = [
            'messaging_product' => 'whatsapp',
            'to' => $phone,
            'type' => 'template',
            'template' => [
                'name' => $templateName,
                'language' => ['code' => 'en'],
                'components' => [[
                    'type' => 'body',
                    'parameters' => array_map(
                        static fn(string $value): array => ['type' => 'text', 'text' => $value],
                        $params
                    ),
                ]],
            ],
        ];

        $url = "https://graph.facebook.com/v20.0/{$phoneNumberId}/messages";
        $ch = curl_init($url);
        curl_setopt_array($ch, [
            CURLOPT_POST => true,
            CURLOPT_HTTPHEADER => [
                'Authorization: Bearer ' . $apiToken,
                'Content-Type: application/json',
            ],
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_POSTFIELDS => json_encode($payload, JSON_THROW_ON_ERROR),
            CURLOPT_TIMEOUT => 30,
        ]);

        $response = curl_exec($ch);
        $curlError = curl_error($ch);
        $statusCode = (int) curl_getinfo($ch, CURLINFO_HTTP_CODE);
        curl_close($ch);

        if ($curlError !== '') {
            logCriticalError('cURL error when sending WhatsApp template', ['error' => $curlError, 'phone' => $phone]);
            return ['success' => false, 'message' => $curlError];
        }

        $decoded = [];
        if (is_string($response) && $response !== '') {
            $decoded = json_decode($response, true) ?: [];
        }

        if ($statusCode >= 200 && $statusCode < 300) {
            return [
                'success' => true,
                'message_id' => $decoded['messages'][0]['id'] ?? null,
                'raw' => $decoded,
            ];
        }

        logCriticalError('WhatsApp API returned failure response', [
            'status_code' => $statusCode,
            'response' => $decoded,
            'phone' => $phone,
        ]);

        return [
            'success' => false,
            'message' => 'WhatsApp API error',
            'status_code' => $statusCode,
            'response' => $decoded,
        ];
    } catch (Throwable $e) {
        logCriticalError('Unexpected sender error', ['error' => $e->getMessage()]);
        return ['success' => false, 'message' => $e->getMessage()];
    }
}

function logCriticalError(string $message, array $context = []): void
{
    $line = sprintf(
        "[%s] %s %s\n",
        (new DateTimeImmutable())->format('Y-m-d H:i:s'),
        $message,
        $context ? json_encode($context, JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES) : ''
    );
    error_log($line, 3, ERROR_LOG_FILE);
}

function tableExists(PDO $pdo, string $tableName): bool
{
    $stmt = $pdo->prepare('SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = DATABASE() AND table_name = :table_name');
    $stmt->execute([':table_name' => $tableName]);
    return (int) $stmt->fetchColumn() > 0;
}

function columnExists(PDO $pdo, string $tableName, string $columnName): bool
{
    $stmt = $pdo->prepare('SELECT COUNT(*) FROM information_schema.columns WHERE table_schema = DATABASE() AND table_name = :table_name AND column_name = :column_name');
    $stmt->execute([':table_name' => $tableName, ':column_name' => $columnName]);
    return (int) $stmt->fetchColumn() > 0;
}

function fetchOverdueInvoices(PDO $pdo): array
{
    $hasZohoCustomers = tableExists($pdo, 'zoho_customers');
    $hasInvoiceCustomerId = columnExists($pdo, 'invoices', 'customer_id');

    if ($hasZohoCustomers && $hasInvoiceCustomerId) {
        $sql = "
            SELECT
                i.invoice_id,
                i.invoice_number,
                i.customer_name,
                i.total,
                i.due_date,
                i.phone AS invoice_phone,
                COALESCE(NULLIF(zc.phone, ''), NULLIF(i.phone, '')) AS resolved_phone
            FROM invoices i
            LEFT JOIN zoho_customers zc
                ON zc.zoho_contact_id = i.customer_id
            WHERE i.status IN ('overdue', 'unpaid')
        ";
    } else {
        $sql = "
            SELECT
                i.invoice_id,
                i.invoice_number,
                i.customer_name,
                i.total,
                i.due_date,
                i.phone AS invoice_phone,
                i.phone AS resolved_phone
            FROM invoices i
            WHERE i.status IN ('overdue', 'unpaid')
        ";
    }

    $stmt = $pdo->prepare($sql);
    $stmt->execute();
    return $stmt->fetchAll();
}

function fetchLatestStatusAndAttempts(PDO $pdo, string $invoiceId): ?array
{
    $stmt = $pdo->prepare('SELECT status, attempts FROM whatsapp_logs WHERE invoice_id = :invoice_id ORDER BY sent_at DESC, id DESC LIMIT 1');
    $stmt->execute([':invoice_id' => $invoiceId]);
    $row = $stmt->fetch();
    return $row ?: null;
}

function alreadySentToday(PDO $pdo, string $invoiceId): bool
{
    $stmt = $pdo->prepare('SELECT 1 FROM whatsapp_logs WHERE invoice_id = :invoice_id AND DATE(sent_at) = CURDATE() LIMIT 1');
    $stmt->execute([':invoice_id' => $invoiceId]);
    return (bool) $stmt->fetchColumn();
}

function insertWhatsAppLog(PDO $pdo, array $data): void
{
    $stmt = $pdo->prepare(
        'INSERT INTO whatsapp_logs (invoice_id, customer_name, phone, template_name, status, error_message, message_id, attempts, sent_at, updated_at)
         VALUES (:invoice_id, :customer_name, :phone, :template_name, :status, :error_message, :message_id, :attempts, NOW(), NOW())'
    );

    $stmt->execute([
        ':invoice_id' => (string) $data['invoice_id'],
        ':customer_name' => (string) $data['customer_name'],
        ':phone' => (string) $data['phone'],
        ':template_name' => (string) $data['template_name'],
        ':status' => (string) $data['status'],
        ':error_message' => $data['error_message'] !== null ? (string) $data['error_message'] : null,
        ':message_id' => $data['message_id'] !== null ? (string) $data['message_id'] : null,
        ':attempts' => (int) $data['attempts'],
    ]);
}

function buildTemplateParams(array $invoice): array
{
    $customerName = trim((string) ($invoice['customer_name'] ?? ''));
    $invoiceNumber = trim((string) ($invoice['invoice_number'] ?? ''));
    $totalRaw = $invoice['total'] ?? null;
    $dueDateRaw = trim((string) ($invoice['due_date'] ?? ''));

    if ($customerName === '' || $invoiceNumber === '' || $totalRaw === null || $dueDateRaw === '') {
        return ['ok' => false, 'error' => 'Missing template data'];
    }

    if (!is_numeric((string) $totalRaw)) {
        return ['ok' => false, 'error' => 'Invalid total amount'];
    }

    $dueTimestamp = strtotime($dueDateRaw);
    if ($dueTimestamp === false) {
        return ['ok' => false, 'error' => 'Invalid due_date'];
    }

    return [
        'ok' => true,
        'params' => [
            $customerName,
            $invoiceNumber,
            number_format((float) $totalRaw, 2, '.', ''),
            date('d-m-Y', $dueTimestamp),
        ],
    ];
}

try {
    $invoices = fetchOverdueInvoices($pdo);

    foreach ($invoices as $invoice) {
        $invoiceId = trim((string) ($invoice['invoice_id'] ?? ''));
        $customerName = trim((string) ($invoice['customer_name'] ?? 'Unknown'));
        $rawPhone = (string) ($invoice['resolved_phone'] ?? '');
        $phone = preg_replace('/\D+/', '', $rawPhone);

        if ($invoiceId === '') {
            logCriticalError('Skipping row with missing invoice_id', ['invoice' => $invoice]);
            continue;
        }

        // (1) Mandatory duplicate prevention for same invoice/day.
        if (alreadySentToday($pdo, $invoiceId)) {
            continue;
        }

        $latest = fetchLatestStatusAndAttempts($pdo, $invoiceId);

        // (2) Skip if already delivered/read.
        if ($latest !== null && in_array((string) $latest['status'], ['delivered', 'read'], true)) {
            continue;
        }

        $attempts = ($latest !== null ? (int) $latest['attempts'] : 0) + 1;
        if ($attempts > MAX_ATTEMPTS) {
            continue;
        }

        // (3) Strict phone validation.
        if ($phone === '' || strlen($phone) < 10) {
            insertWhatsAppLog($pdo, [
                'invoice_id' => $invoiceId,
                'customer_name' => $customerName,
                'phone' => $phone,
                'template_name' => OVERDUE_TEMPLATE_NAME,
                'status' => 'failed',
                'error_message' => 'Invalid or missing phone',
                'message_id' => null,
                'attempts' => $attempts,
            ]);
            continue;
        }

        // (4) Template parameter safety.
        $paramsResult = buildTemplateParams($invoice);
        if (($paramsResult['ok'] ?? false) !== true) {
            $errorMessage = (string) ($paramsResult['error'] ?? 'Invalid template data');
            insertWhatsAppLog($pdo, [
                'invoice_id' => $invoiceId,
                'customer_name' => $customerName,
                'phone' => $phone,
                'template_name' => OVERDUE_TEMPLATE_NAME,
                'status' => 'failed',
                'error_message' => $errorMessage,
                'message_id' => null,
                'attempts' => $attempts,
            ]);
            continue;
        }

        $response = sendOverdueTemplate($phone, OVERDUE_TEMPLATE_NAME, $paramsResult['params']);

        if (($response['success'] ?? false) !== true) {
            $errorMessage = (string) ($response['message'] ?? 'API failure response');
            insertWhatsAppLog($pdo, [
                'invoice_id' => $invoiceId,
                'customer_name' => $customerName,
                'phone' => $phone,
                'template_name' => OVERDUE_TEMPLATE_NAME,
                'status' => 'failed',
                'error_message' => $errorMessage,
                'message_id' => $response['message_id'] ?? null,
                'attempts' => $attempts,
            ]);
            continue;
        }

        // (5) message_id is mandatory for sent status.
        $messageId = trim((string) ($response['message_id'] ?? ''));
        if ($messageId === '') {
            insertWhatsAppLog($pdo, [
                'invoice_id' => $invoiceId,
                'customer_name' => $customerName,
                'phone' => $phone,
                'template_name' => OVERDUE_TEMPLATE_NAME,
                'status' => 'failed',
                'error_message' => 'Missing message_id in API response',
                'message_id' => null,
                'attempts' => $attempts,
            ]);
            continue;
        }

        insertWhatsAppLog($pdo, [
            'invoice_id' => $invoiceId,
            'customer_name' => $customerName,
            'phone' => $phone,
            'template_name' => OVERDUE_TEMPLATE_NAME,
            'status' => 'sent',
            'error_message' => null,
            'message_id' => $messageId,
            'attempts' => $attempts,
        ]);
    }
} catch (PDOException $e) {
    logCriticalError('Database error in send_whatsapp_notifications.php', ['error' => $e->getMessage()]);
    exit(1);
} catch (Throwable $e) {
    logCriticalError('Fatal error in send_whatsapp_notifications.php', ['error' => $e->getMessage()]);
    exit(1);
}
