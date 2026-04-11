<?php

declare(strict_types=1);
date_default_timezone_set('Asia/Kolkata');

require_once __DIR__ . '/db.php';

header('Content-Type: application/json');

$dateFilter = strtolower(trim((string) ($_GET['date'] ?? 'today')));
$statusFilter = strtolower(trim((string) ($_GET['status'] ?? 'all')));

$allowedDateFilters = ['today', 'last7days'];
$allowedStatuses = ['all', 'sent', 'delivered', 'read', 'failed'];

if (!in_array($dateFilter, $allowedDateFilters, true)) {
    http_response_code(422);
    echo json_encode(['error' => 'Invalid date filter']);
    exit;
}

if (!in_array($statusFilter, $allowedStatuses, true)) {
    http_response_code(422);
    echo json_encode(['error' => 'Invalid status filter']);
    exit;
}

$dateWhere = $dateFilter === 'today'
    ? 'DATE(wl.sent_at) = CURRENT_DATE()'
    : 'wl.sent_at >= (NOW() - INTERVAL 7 DAY)';

$statusWhere = '';
$params = [];
if ($statusFilter !== 'all') {
    $statusWhere = ' AND wl.status = :status';
    $params[':status'] = $statusFilter;
}

try {
    $summarySql = "
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN wl.status = 'delivered' THEN 1 ELSE 0 END) AS delivered,
            SUM(CASE WHEN wl.status = 'read' THEN 1 ELSE 0 END) AS `read_count`,
            SUM(CASE WHEN wl.status = 'failed' THEN 1 ELSE 0 END) AS failed,
            SUM(CASE WHEN wl.status = 'sent' THEN 1 ELSE 0 END) AS sent_only
        FROM whatsapp_logs wl
        WHERE {$dateWhere}{$statusWhere}
    ";

    $summaryStmt = $pdo->prepare($summarySql);
    $summaryStmt->execute($params);
    $summary = $summaryStmt->fetch() ?: [];

    $rowsSql = "
        SELECT
            wl.customer_name,
            wl.invoice_id,
            COALESCE(i.invoice_number, wl.invoice_id) AS invoice_number,
            i.total,
            wl.status,
            wl.sent_at
        FROM whatsapp_logs wl
        LEFT JOIN invoices i ON i.invoice_id = wl.invoice_id
        WHERE {$dateWhere}{$statusWhere}
        ORDER BY wl.sent_at DESC
        LIMIT 200
    ";

    $rowsStmt = $pdo->prepare($rowsSql);
    $rowsStmt->execute($params);
    $rows = $rowsStmt->fetchAll();

    echo json_encode([
        'summary' => [
            'total_sent_today' => (int) ($summary['total'] ?? 0),
            'delivered' => (int) ($summary['delivered'] ?? 0),
            'read' => (int) ($summary['read_count'] ?? 0),
            'failed' => (int) ($summary['failed'] ?? 0),
            'sent' => (int) ($summary['sent_only'] ?? 0),
        ],
        'rows' => $rows,
    ], JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES);
} catch (Throwable $e) {
    http_response_code(500);
    error_log('get_whatsapp_logs error: ' . $e->getMessage());
    echo json_encode(['error' => 'Failed to load WhatsApp logs']);
}
