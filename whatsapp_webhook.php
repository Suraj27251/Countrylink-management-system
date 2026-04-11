<?php

declare(strict_types=1);
date_default_timezone_set('Asia/Kolkata');

require_once __DIR__ . '/db.php';

header('Content-Type: application/json');
const ERROR_LOG_FILE = '/home/countrylinks/whatsapp_error.log';

function webhookResponse(int $statusCode, array $payload): void
{
    http_response_code($statusCode);
    echo json_encode($payload, JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES);
    exit;
}

function parseWebhookPayload(): ?array
{
    $raw = file_get_contents('php://input');
    if ($raw === false || trim($raw) === '') {
        return null;
    }

    $data = json_decode($raw, true);
    return is_array($data) ? $data : null;
}

function updateLogStatus(PDO $pdo, string $messageId, string $status): void
{
    $stmt = $pdo->prepare(
        'UPDATE whatsapp_logs SET status = :status, updated_at = NOW() WHERE message_id = :message_id'
    );
    $stmt->execute([
        ':status' => $status,
        ':message_id' => $messageId,
    ]);
}

function logWebhookError(string $message, array $context = []): void
{
    $line = sprintf(
        "[%s] %s %s\n",
        (new DateTimeImmutable())->format('Y-m-d H:i:s'),
        $message,
        $context ? json_encode($context, JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES) : ''
    );
    error_log($line, 3, ERROR_LOG_FILE);
}

if ($_SERVER['REQUEST_METHOD'] === 'GET') {
    // Optional verification support to stay compatible with existing Meta webhook verify flow.
    $mode = $_GET['hub_mode'] ?? $_GET['hub.mode'] ?? null;
    $token = $_GET['hub_verify_token'] ?? $_GET['hub.verify_token'] ?? null;
    $challenge = $_GET['hub_challenge'] ?? $_GET['hub.challenge'] ?? null;
    $verifyToken = getenv('WHATSAPP_WEBHOOK_VERIFY_TOKEN') ?: '';

    if ($mode === 'subscribe' && $verifyToken !== '' && hash_equals($verifyToken, (string) $token)) {
        echo (string) $challenge;
        exit;
    }

    webhookResponse(200, ['ok' => true, 'message' => 'Webhook endpoint ready']);
}

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    webhookResponse(405, ['ok' => false, 'error' => 'Method not allowed']);
}

$payload = parseWebhookPayload();
if ($payload === null) {
    webhookResponse(400, ['ok' => false, 'error' => 'Invalid JSON payload']);
}

$updated = 0;

try {
    $entries = $payload['entry'] ?? [];

    foreach ($entries as $entry) {
        $changes = $entry['changes'] ?? [];
        foreach ($changes as $change) {
            $statuses = $change['value']['statuses'] ?? [];
            foreach ($statuses as $statusEvent) {
                $messageId = (string) ($statusEvent['id'] ?? '');
                $status = (string) ($statusEvent['status'] ?? '');

                if ($messageId === '' || !in_array($status, ['delivered', 'read', 'failed'], true)) {
                    if ($messageId === '' && $status !== '') {
                        logWebhookError('Skipping status event due to missing message ID', ['event' => $statusEvent]);
                    }
                    continue;
                }

                updateLogStatus($pdo, $messageId, $status);
                $updated++;
            }
        }
    }

    webhookResponse(200, ['ok' => true, 'updated' => $updated]);
} catch (Throwable $e) {
    logWebhookError('whatsapp_webhook error', ['error' => $e->getMessage()]);
    webhookResponse(500, ['ok' => false, 'error' => 'Failed to process webhook']);
}
