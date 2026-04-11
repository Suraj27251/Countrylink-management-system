<?php

declare(strict_types=1);

date_default_timezone_set('Asia/Kolkata');

require_once __DIR__ . '/db.php';

const ZOHO_INVOICE_SYNC_LOG = '/home/countrylinks/whatsapp_error.log';

function logInvoiceSync(string $message, array $context = []): void
{
    $line = sprintf(
        "[%s] [invoice-sync] %s %s\n",
        (new DateTimeImmutable())->format('Y-m-d H:i:s'),
        $message,
        $context ? json_encode($context, JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES) : ''
    );
    error_log($line, 3, ZOHO_INVOICE_SYNC_LOG);
}

function getZohoAccessToken(): string
{
    $clientId = getenv('ZOHO_CLIENT_ID') ?: '';
    $clientSecret = getenv('ZOHO_CLIENT_SECRET') ?: '';
    $refreshToken = getenv('ZOHO_REFRESH_TOKEN') ?: '';

    if ($clientId === '' || $clientSecret === '' || $refreshToken === '') {
        throw new RuntimeException('Missing Zoho OAuth credentials.');
    }

    $url = 'https://accounts.zoho.in/oauth/v2/token';
    $payload = http_build_query([
        'refresh_token' => $refreshToken,
        'client_id' => $clientId,
        'client_secret' => $clientSecret,
        'grant_type' => 'refresh_token',
    ]);

    $ch = curl_init($url);
    curl_setopt_array($ch, [
        CURLOPT_POST => true,
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_POSTFIELDS => $payload,
        CURLOPT_TIMEOUT => 30,
    ]);

    $response = curl_exec($ch);
    $curlError = curl_error($ch);
    $statusCode = (int) curl_getinfo($ch, CURLINFO_HTTP_CODE);
    curl_close($ch);

    if ($curlError !== '') {
        throw new RuntimeException('Zoho token curl error: ' . $curlError);
    }

    $data = json_decode((string) $response, true) ?: [];
    if ($statusCode < 200 || $statusCode >= 300 || empty($data['access_token'])) {
        throw new RuntimeException('Zoho token request failed.');
    }

    return (string) $data['access_token'];
}

function syncZohoInvoices(PDO $pdo): void
{
    $orgId = getenv('ZOHO_ORG_ID') ?: '';
    $apiDomain = rtrim(getenv('ZOHO_API_DOMAIN') ?: 'https://www.zohoapis.in', '/');

    if ($orgId === '') {
        throw new RuntimeException('Missing ZOHO_ORG_ID.');
    }

    $accessToken = getZohoAccessToken();

    try {
        $pdo->exec("ALTER TABLE invoices ADD COLUMN zoho_contact_id VARCHAR(100) NULL");
    } catch (Throwable $e) {
        // ignore if already exists
    }

    $page = 1;
    $perPage = 200;

    while (true) {
        $url = sprintf('%s/books/v3/invoices?organization_id=%s&page=%d&per_page=%d', $apiDomain, urlencode($orgId), $page, $perPage);

        $ch = curl_init($url);
        curl_setopt_array($ch, [
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_HTTPHEADER => [
                'Authorization: Zoho-oauthtoken ' . $accessToken,
                'Content-Type: application/json',
            ],
            CURLOPT_TIMEOUT => 30,
        ]);

        $response = curl_exec($ch);
        $curlError = curl_error($ch);
        $statusCode = (int) curl_getinfo($ch, CURLINFO_HTTP_CODE);
        curl_close($ch);

        if ($curlError !== '') {
            throw new RuntimeException('Zoho invoice curl error: ' . $curlError);
        }

        $data = json_decode((string) $response, true) ?: [];
        if ($statusCode < 200 || $statusCode >= 300) {
            throw new RuntimeException('Zoho invoice API failed.');
        }

        $invoices = $data['invoices'] ?? [];
        if (!$invoices) {
            break;
        }

        $stmt = $pdo->prepare(
            'INSERT INTO invoices (invoice_id, invoice_number, customer_name, zoho_contact_id, status, total, due_date)
             VALUES (:invoice_id, :invoice_number, :customer_name, :zoho_contact_id, :status, :total, :due_date)
             ON DUPLICATE KEY UPDATE
                invoice_number = VALUES(invoice_number),
                customer_name = VALUES(customer_name),
                zoho_contact_id = VALUES(zoho_contact_id),
                status = VALUES(status),
                total = VALUES(total),
                due_date = VALUES(due_date)'
        );

        foreach ($invoices as $inv) {
            $zohoContactId = (string) ($inv['customer_id'] ?? ''); // REQUIRED LINK FIELD
            $stmt->execute([
                ':invoice_id' => (string) ($inv['invoice_id'] ?? ''),
                ':invoice_number' => (string) ($inv['invoice_number'] ?? ''),
                ':customer_name' => (string) ($inv['customer_name'] ?? ''),
                ':zoho_contact_id' => $zohoContactId,
                ':status' => (string) ($inv['status'] ?? ''),
                ':total' => (string) ($inv['total'] ?? 0),
                ':due_date' => (string) ($inv['due_date'] ?? ''),
            ]);
        }

        echo "Synced page {$page}, records: " . count($invoices) . "\n";
        $page++;
    }
}

try {
    syncZohoInvoices($pdo);
    echo "Invoice sync completed successfully.\n";
} catch (Throwable $e) {
    logInvoiceSync('Invoice sync failed', ['error' => $e->getMessage()]);
    echo 'Invoice sync failed: ' . $e->getMessage() . "\n";
    exit(1);
}
