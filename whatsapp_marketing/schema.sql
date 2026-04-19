CREATE TABLE IF NOT EXISTS templates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    language TEXT NOT NULL,
    category TEXT,
    status TEXT NOT NULL,
    body_text TEXT,
    variables_count INTEGER DEFAULT 0,
    raw_json TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(name, language)
);

CREATE TABLE IF NOT EXISTS marketing_campaigns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    template_id INTEGER NOT NULL,
    scheduled_time TEXT,
    status TEXT NOT NULL DEFAULT 'draft',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(template_id) REFERENCES templates(id)
);

CREATE TABLE IF NOT EXISTS campaign_variable_mapping (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    campaign_id INTEGER NOT NULL,
    variable_position INTEGER NOT NULL,
    field_name TEXT NOT NULL,
    UNIQUE(campaign_id, variable_position),
    FOREIGN KEY(campaign_id) REFERENCES marketing_campaigns(id)
);

CREATE TABLE IF NOT EXISTS campaign_recipients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    campaign_id INTEGER NOT NULL,
    contact_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    retry_count INTEGER DEFAULT 0,
    external_message_id TEXT,
    error_message TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(campaign_id, contact_id),
    FOREIGN KEY(campaign_id) REFERENCES marketing_campaigns(id)
);

CREATE TABLE IF NOT EXISTS marketing_webhook_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
