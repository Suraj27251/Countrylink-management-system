<?php

declare(strict_types=1);

session_start();
require_once __DIR__ . '/db.php';

$userId = (int) ($_SESSION['pending_user_id'] ?? 0);
$error = '';

if ($userId <= 0) {
    $_SESSION['register_errors'] = ['Please complete signup before OTP verification.'];
    header('Location: signup.html');
    exit;
}

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $otp = trim((string) ($_POST['otp_code'] ?? ''));

    if (!preg_match('/^[0-9]{6}$/', $otp)) {
        $error = 'Please provide a valid OTP.';
    } else {
        try {
            $stmt = $pdo->prepare(
                'SELECT ov.id, ov.otp_code, ov.expires_at, ov.verified, u.status
                 FROM otp_verifications ov
                 INNER JOIN users u ON u.id = ov.user_id
                 WHERE ov.user_id = :user_id
                 ORDER BY ov.id DESC
                 LIMIT 1'
            );
            $stmt->execute([':user_id' => $userId]);
            $otpRow = $stmt->fetch();

            if (!$otpRow) {
                $error = 'No OTP request found for this user.';
            } elseif ($otpRow['status'] === 'active') {
                $error = 'This account is already approved. Please login.';
            } elseif ((int) $otpRow['verified'] === 1) {
                $error = 'This OTP has already been used.';
            } elseif (new DateTime($otpRow['expires_at']) < new DateTime()) {
                $error = 'OTP has expired. Please register again.';
            } elseif (!hash_equals((string) $otpRow['otp_code'], $otp)) {
                $error = 'Invalid OTP.';
            } else {
                $pdo->beginTransaction();

                $verifyStmt = $pdo->prepare('UPDATE otp_verifications SET verified = 1 WHERE id = :id AND verified = 0');
                $verifyStmt->execute([':id' => $otpRow['id']]);

                if ($verifyStmt->rowCount() !== 1) {
                    throw new RuntimeException('OTP verification conflict.');
                }

                $activateStmt = $pdo->prepare("UPDATE users SET status = 'active' WHERE id = :id AND status = 'pending'");
                $activateStmt->execute([':id' => $userId]);

                if ($activateStmt->rowCount() !== 1) {
                    throw new RuntimeException('User activation conflict.');
                }

                $pdo->commit();

                unset($_SESSION['pending_user_id']);
                $_SESSION['login_success'] = 'Account approved successfully. Please login.';
                header('Location: login.php');
                exit;
            }
        } catch (Throwable $e) {
            if ($pdo->inTransaction()) {
                $pdo->rollBack();
            }
            $error = 'Verification failed. Please try again.';
        }
    }
}
?>
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Verify OTP</title>
</head>
<body>
  <h2>Enter OTP</h2>
  <?php if ($error !== ''): ?>
    <p style="color: red;"><?php echo htmlspecialchars($error, ENT_QUOTES, 'UTF-8'); ?></p>
  <?php endif; ?>

  <form method="POST" action="verify_otp.php">
    <label for="otp_code">OTP</label>
    <input type="text" id="otp_code" name="otp_code" maxlength="6" pattern="[0-9]{6}" required>
    <button type="submit">Verify OTP</button>
  </form>
</body>
</html>
