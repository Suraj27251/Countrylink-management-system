# Cron job setup

Run the overdue invoice WhatsApp notification worker every day at 9:00 AM server time:

```cron
0 9 * * * /usr/bin/php /workspace/Countrylink-management-system/send_whatsapp_notifications.php >> /workspace/Countrylink-management-system/logs/cron_whatsapp_notifications.log 2>&1
```

If your project is deployed elsewhere, update the absolute paths accordingly.
