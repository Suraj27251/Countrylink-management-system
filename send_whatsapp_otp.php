<?php

declare(strict_types=1);

function sendWhatsAppOtp(string $otpCode): array
{
    $adminNumber = '918149912379'; // country code + number
    $apiToken = getenv('META_ACCESS_TOKEN') ?: '';
    $phoneNumberId = getenv('META_PHONE_NUMBER_ID') ?: '';

    if ($apiToken === '' || $phoneNumberId === '') {
        return ['success' => false, 'message' => 'WhatsApp credentials are not configured.'];
    }

    $url = "https://graph.facebook.com/v20.0/{$phoneNumberId}/messages";

    $payload = [
        'messaging_product' => 'whatsapp',
        'to' => $adminNumber,
        'type' => 'template',
        'template' => [
            'name' => 'otp',
            'language' => ['code' => 'en'],
            'components' => [[
                'type' => 'body',
                'parameters' => [[
                    'type' => 'text',
                    'text' => $otpCode,
                ]],
            ]],
        ],
    ];

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
        return ['success' => false, 'message' => $curlError];
    }

    if ($statusCode >= 200 && $statusCode < 300) {
        return ['success' => true, 'message' => 'OTP sent to admin WhatsApp.'];
    }

    return [
        'success' => false,
        'message' => 'WhatsApp API error',
        'status_code' => $statusCode,
        'response' => $response,
    ];
}
