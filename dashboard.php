<?php

declare(strict_types=1);

session_start();

if (!isset($_SESSION['user_id'])) {
    header('Location: login.php');
    exit;
}
?>
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Dashboard</title>
</head>
<body>
  <h1>Welcome, <?php echo htmlspecialchars((string) $_SESSION['user_name'], ENT_QUOTES, 'UTF-8'); ?></h1>
  <p>You are logged in.</p>
  <a href="logout.php">Logout</a>
</body>
</html>
