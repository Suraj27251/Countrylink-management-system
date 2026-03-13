<?php

declare(strict_types=1);

session_start();

require_once __DIR__ . '/db.php';
require_once __DIR__ . '/send_whatsapp_otp.php';

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    http_response_code(405);
    exit('Method not allowed.');
}

$name = trim(filter_input(INPUT_POST, 'name', FILTER_SANITIZE_FULL_SPECIAL_CHARS) ?? '');
$email = trim((string) filter_input(INPUT_POST, 'email', FILTER_SANITIZE_EMAIL));
$phone = preg_replace('/\D+/', '', (string) ($_POST['phone'] ?? ''));
$password = (string) ($_POST['password'] ?? '');
$confirmPassword = (string) ($_POST['confirm_password'] ?? '');

$errors = [];

if ($name === '' || $email === '' || $phone === '' || $password === '' || $confirmPassword === '') {
    $errors[] = 'All fields are required.';
}

if (!filter_var($email, FILTER_VALIDATE_EMAIL)) {
    $errors[] = 'Invalid email format.';
}

if (!preg_match('/^[0-9]{10,15}$/', $phone)) {
    $errors[] = 'Invalid phone number.';
}

if ($password !== $confirmPassword) {
    $errors[] = 'Passwords do not match.';
}

if (!empty($errors)) {
    $_SESSION['register_errors'] = $errors;
    header('Location: ' . ($_SERVER['HTTP_REFERER'] ?? 'signup.html'));
    exit;
}

try {
    $checkStmt = $pdo->prepare('SELECT id FROM users WHERE email = :email LIMIT 1');
    $checkStmt->execute([':email' => $email]);
    if ($checkStmt->fetch()) {
        $_SESSION['register_errors'] = ['Email is already registered.'];
        header('Location: ' . ($_SERVER['HTTP_REFERER'] ?? 'signup.html'));
        exit;
    }

    $passwordHash = password_hash($password, PASSWORD_DEFAULT);
    $otpCode = str_pad((string) random_int(0, 999999), 6, '0', STR_PAD_LEFT);
    $expiresAt = (new DateTime('+10 minutes'))->format('Y-m-d H:i:s');

    $pdo->beginTransaction();

    $insertUser = $pdo->prepare(
        'INSERT INTO users (name, email, phone, password, status) VALUES (:name, :email, :phone, :password, :status)'
    );

    $insertUser->execute([
        ':name' => $name,
        ':email' => $email,
        ':phone' => $phone,
        ':password' => $passwordHash,
        ':status' => 'pending',
    ]);

    $userId = (int) $pdo->lastInsertId();

    $insertOtp = $pdo->prepare(
        'INSERT INTO otp_verifications (user_id, otp_code, expires_at, verified) VALUES (:user_id, :otp_code, :expires_at, 0)'
    );

    $insertOtp->execute([
        ':user_id' => $userId,
        ':otp_code' => $otpCode,
        ':expires_at' => $expiresAt,
    ]);

    $whatsAppResult = sendWhatsAppOtp($otpCode);

    if (!$whatsAppResult['success']) {
        $pdo->rollBack();
        $_SESSION['register_errors'] = ['Registration failed while sending OTP. Please try again.'];
        header('Location: ' . ($_SERVER['HTTP_REFERER'] ?? 'signup.html'));
        exit;
    }

    $pdo->commit();

    $_SESSION['pending_user_id'] = $userId;
    $_SESSION['register_success'] = 'Registration successful. Enter the OTP sent to admin for approval.';
    header('Location: verify_otp.php');
    exit;
} catch (Throwable $e) {
    if ($pdo->inTransaction()) {
        $pdo->rollBack();
    }
    $_SESSION['register_errors'] = ['Something went wrong. Please try again.'];
    header('Location: ' . ($_SERVER['HTTP_REFERER'] ?? 'signup.html'));
    exit;
}
