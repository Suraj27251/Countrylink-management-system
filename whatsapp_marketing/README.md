# WhatsApp Marketing Module (Isolated)

This module is fully isolated and does not overwrite existing routes, templates, or tables.

## Files
- `routes.py`: Blueprint and APIs under `/marketing/*`
- `services.py`: WhatsApp send service with variable validation
- `template_sync.py`: Sync approved templates from Meta API
- `campaign_engine.py`: Queue processor + retry logic (max 3)
- `models.py`: DB schema + repository layer
- `schema.sql`: SQL schema for module tables

## Integration in Flask app (one-time include)
> This repo keeps existing files untouched per request. Use the snippet below when you are ready to wire it:

```python
from whatsapp_marketing import init_marketing

init_marketing(app, get_db_connection)
```

This registers the blueprint with `/marketing` prefix and initializes schema.

## API Endpoints
- `GET /marketing/dashboard`
- `GET /marketing/campaigns`
- `POST /marketing/campaigns`
- `POST /marketing/campaigns/run`
- `GET /marketing/templates`
- `GET /marketing/api/templates`
- `POST /marketing/templates/sync`
- `GET /marketing/webhook`
- `POST /marketing/webhook`

## Sample .env
```env
# Core
SECRET_KEY=change-this
DATABASE_PATH=/home/username/app/database/complaints.db

# WhatsApp Cloud API
WHATSAPP_API_VERSION=v20.0
META_ACCESS_TOKEN=EAAG...
PHONE_NUMBER_ID=123456789012345
WABA_ID=123456789012345
META_APP_SECRET=your_app_secret
WEBHOOK_VERIFY_TOKEN=your_verify_token
```

## Scheduling (no Celery/Redis)
Use cron in cPanel:

```bash
*/2 * * * * /usr/bin/curl -s -X POST https://yourdomain.com/marketing/campaigns/run >/dev/null 2>&1
```

## cPanel deployment steps
1. Upload module and templates files to the same project root.
2. Add environment variables in cPanel > Setup Python App.
3. Restart Python application.
4. Run schema SQL once (or first load calls `init_schema`).
5. Configure Meta webhook URL:
   - Verify: `https://yourdomain.com/marketing/webhook`
   - Events: messages + statuses
6. Add cron entry to run queued campaigns.

## Security and safety
- Env vars only (no hardcoded tokens)
- Signature verification for webhook (`X-Hub-Signature-256`)
- Variable-count validation before send
- Duplicate recipient prevention (`UNIQUE(campaign_id, contact_id)`)
- Retry cap of 3 for failed sends
