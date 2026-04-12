# Cron Job Setup

## Required Cron Jobs

### 1. Sync Zoho Invoices
Run every day at **8:45 AM** to sync invoices and fetch plan names:

```cron
45 8 * * * /usr/bin/php /home/countrylinks/public_html/Countrylink-management-system-main/sync_zoho_invoices.php >> /home/countrylinks/logs/sync_zoho.log 2>&1
```

### 2. Send WhatsApp Notifications  
Run every day at **9:00 AM** to send overdue invoice notifications:

```cron
0 9 * * * /usr/bin/php /home/countrylinks/public_html/Countrylink-management-system-main/send_whatsapp_notifications.php >> /home/countrylinks/logs/whatsapp_notifications.log 2>&1
```

## Environment Variables Required

### WhatsApp Credentials (set in cPanel)
- `META_ACCESS_TOKEN` - WhatsApp Business API token
- `PHONE_NUMBER_ID` - WhatsApp Business Phone Number ID

### Database Credentials (set in cPanel)
- `MYSQL_DB_HOST` - MySQL host (localhost)
- `MYSQL_DB_NAME` - MySQL database name
- `MYSQL_DB_USER` - MySQL user
- `MYSQL_DB_PASSWORD` - MySQL password

## Notes

- Both jobs use Asia/Kolkata timezone
- Logs are written to `/home/countrylinks/logs/` for debugging
- The sync job must run BEFORE the WhatsApp job (8:45 AM → 9:00 AM)
- If deployed elsewhere, update the absolute paths in the cron commands accordingly
