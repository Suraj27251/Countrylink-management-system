<?php

declare(strict_types=1);

session_start();
require_once __DIR__ . '/db.php';

$error = '';
$success = $_SESSION['login_success'] ?? '';
unset($_SESSION['login_success']);

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $email = trim((string) filter_input(INPUT_POST, 'email', FILTER_SANITIZE_EMAIL));
    $password = (string) ($_POST['password'] ?? '');

    if ($email === '' || $password === '') {
        $error = 'Email and password are required.';
    } else {
        $stmt = $pdo->prepare('SELECT id, name, email, password, status FROM users WHERE email = :email LIMIT 1');
        $stmt->execute([':email' => $email]);
        $user = $stmt->fetch();

        if (!$user || !password_verify($password, $user['password'])) {
            $error = 'Invalid login credentials.';
        } elseif ($user['status'] !== 'active') {
            $error = 'Account not yet approved by admin.';
        } else {
            $_SESSION['user_id'] = (int) $user['id'];
            $_SESSION['user_name'] = $user['name'];
            session_regenerate_id(true);
            header('Location: dashboard.php');
            exit;
        }
    }
}
?>
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Login</title>
</head>
<body>
  <h2>Login</h2>
  <?php if ($success !== ''): ?>
    <p style="color: green;"><?php echo htmlspecialchars($success, ENT_QUOTES, 'UTF-8'); ?></p>
  <?php endif; ?>
  <?php if ($error !== ''): ?>
    <p style="color: red;"><?php echo htmlspecialchars($error, ENT_QUOTES, 'UTF-8'); ?></p>
  <?php endif; ?>

  <form method="POST" action="login.php">
    <label for="email">Email</label>
    <input type="email" id="email" name="email" required>

    <label for="password">Password</label>
    <input type="password" id="password" name="password" required>

    <button type="submit">Login</button>
  </form>
</body>
</html>
