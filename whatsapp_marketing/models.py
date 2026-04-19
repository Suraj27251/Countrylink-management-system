import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple


DDL_STATEMENTS = [
    """
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
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS marketing_campaigns (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        template_id INTEGER NOT NULL,
        scheduled_time TEXT,
        status TEXT NOT NULL DEFAULT 'draft',
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(template_id) REFERENCES templates(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS campaign_variable_mapping (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        campaign_id INTEGER NOT NULL,
        variable_position INTEGER NOT NULL,
        field_name TEXT NOT NULL,
        UNIQUE(campaign_id, variable_position),
        FOREIGN KEY(campaign_id) REFERENCES marketing_campaigns(id)
    )
    """,
    """
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
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS marketing_webhook_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_type TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """,
]


class MarketingRepository:
    def __init__(self, get_db_connection):
        self._get_db_connection = get_db_connection

    def init_schema(self) -> None:
        conn = self._get_db_connection()
        try:
            cur = conn.cursor()
            for ddl in DDL_STATEMENTS:
                cur.execute(ddl)
            conn.commit()
        finally:
            conn.close()

    def upsert_template(self, template: Dict[str, Any], variables_count: int, body_text: str) -> None:
        conn = self._get_db_connection()
        try:
            conn.execute(
                """
                INSERT INTO templates (name, language, category, status, body_text, variables_count, raw_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(name, language) DO UPDATE SET
                    category = excluded.category,
                    status = excluded.status,
                    body_text = excluded.body_text,
                    variables_count = excluded.variables_count,
                    raw_json = excluded.raw_json
                """,
                (
                    template.get("name", "").strip(),
                    template.get("language", "en"),
                    template.get("category", "MARKETING"),
                    template.get("status", "UNKNOWN"),
                    body_text,
                    variables_count,
                    json.dumps(template),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def list_templates(self) -> List[sqlite3.Row]:
        conn = self._get_db_connection()
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT id, name, language, category, status, body_text, variables_count, created_at
                FROM templates
                ORDER BY created_at DESC
                """
            ).fetchall()
            return rows
        finally:
            conn.close()

    def get_template(self, template_id: int) -> Optional[sqlite3.Row]:
        conn = self._get_db_connection()
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                """
                SELECT id, name, language, category, status, body_text, variables_count
                FROM templates WHERE id = ?
                """,
                (template_id,),
            ).fetchone()
            return row
        finally:
            conn.close()

    def create_campaign(self, name: str, template_id: int, scheduled_time: Optional[str], mappings: List[Tuple[int, str]]) -> int:
        conn = self._get_db_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO marketing_campaigns (name, template_id, scheduled_time, status)
                VALUES (?, ?, ?, 'scheduled')
                """,
                (name, template_id, scheduled_time),
            )
            campaign_id = cur.lastrowid
            for variable_position, field_name in mappings:
                cur.execute(
                    """
                    INSERT INTO campaign_variable_mapping (campaign_id, variable_position, field_name)
                    VALUES (?, ?, ?)
                    """,
                    (campaign_id, variable_position, field_name),
                )
            conn.commit()
            return campaign_id
        finally:
            conn.close()

    def list_campaigns(self) -> List[sqlite3.Row]:
        conn = self._get_db_connection()
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT c.id, c.name, c.scheduled_time, c.status, c.created_at,
                       t.name AS template_name,
                       SUM(CASE WHEN r.status='sent' THEN 1 ELSE 0 END) AS sent_count,
                       COUNT(r.id) AS recipient_count
                FROM marketing_campaigns c
                JOIN templates t ON t.id = c.template_id
                LEFT JOIN campaign_recipients r ON r.campaign_id = c.id
                GROUP BY c.id
                ORDER BY c.created_at DESC
                """
            ).fetchall()
            return rows
        finally:
            conn.close()

    def get_campaign(self, campaign_id: int) -> Optional[sqlite3.Row]:
        conn = self._get_db_connection()
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                """
                SELECT c.*, t.name AS template_name, t.language, t.variables_count
                FROM marketing_campaigns c
                JOIN templates t ON t.id = c.template_id
                WHERE c.id = ?
                """,
                (campaign_id,),
            ).fetchone()
            return row
        finally:
            conn.close()

    def get_campaign_mappings(self, campaign_id: int) -> Dict[int, str]:
        conn = self._get_db_connection()
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT variable_position, field_name
                FROM campaign_variable_mapping
                WHERE campaign_id = ?
                ORDER BY variable_position ASC
                """,
                (campaign_id,),
            ).fetchall()
            return {int(r["variable_position"]): r["field_name"] for r in rows}
        finally:
            conn.close()

    def list_contacts(self) -> List[sqlite3.Row]:
        conn = self._get_db_connection()
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT mobile AS contact_id,
                       COALESCE(MAX(NULLIF(name,'')), mobile) AS contact_name,
                       mobile
                FROM whatsapp_messages
                WHERE mobile IS NOT NULL AND TRIM(mobile) != ''
                GROUP BY mobile
                ORDER BY MAX(created_at) DESC
                """
            ).fetchall()
            return rows
        finally:
            conn.close()

    def enqueue_campaign_contacts(self, campaign_id: int, contact_ids: Iterable[str]) -> int:
        conn = self._get_db_connection()
        try:
            cur = conn.cursor()
            created = 0
            for contact_id in contact_ids:
                cur.execute(
                    """
                    INSERT OR IGNORE INTO campaign_recipients (campaign_id, contact_id, status, retry_count)
                    VALUES (?, ?, 'pending', 0)
                    """,
                    (campaign_id, contact_id),
                )
                if cur.rowcount:
                    created += 1
            conn.commit()
            return created
        finally:
            conn.close()

    def get_pending_recipients(self, limit: int = 50) -> List[sqlite3.Row]:
        conn = self._get_db_connection()
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT r.id AS recipient_id,
                       r.campaign_id,
                       r.contact_id,
                       r.retry_count,
                       c.name AS campaign_name,
                       c.status AS campaign_status,
                       t.name AS template_name,
                       t.language,
                       t.variables_count
                FROM campaign_recipients r
                JOIN marketing_campaigns c ON c.id = r.campaign_id
                JOIN templates t ON t.id = c.template_id
                WHERE r.status = 'pending'
                  AND r.retry_count < 3
                  AND c.status IN ('scheduled', 'running')
                  AND (c.scheduled_time IS NULL OR c.scheduled_time <= ?)
                ORDER BY c.scheduled_time ASC, r.id ASC
                LIMIT ?
                """,
                (datetime.utcnow().isoformat(), limit),
            ).fetchall()
            return rows
        finally:
            conn.close()

    def mark_campaign_running(self, campaign_id: int) -> None:
        conn = self._get_db_connection()
        try:
            conn.execute(
                "UPDATE marketing_campaigns SET status='running', updated_at=CURRENT_TIMESTAMP WHERE id=?",
                (campaign_id,),
            )
            conn.commit()
        finally:
            conn.close()

    def mark_recipient_result(self, recipient_id: int, status: str, external_message_id: Optional[str], error_message: Optional[str]) -> None:
        conn = self._get_db_connection()
        try:
            if status == 'failed':
                conn.execute(
                    """
                    UPDATE campaign_recipients
                    SET status=?, retry_count=retry_count+1, external_message_id=?, error_message=?, updated_at=CURRENT_TIMESTAMP
                    WHERE id=?
                    """,
                    (status, external_message_id, error_message, recipient_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE campaign_recipients
                    SET status=?, external_message_id=?, error_message=?, updated_at=CURRENT_TIMESTAMP
                    WHERE id=?
                    """,
                    (status, external_message_id, error_message, recipient_id),
                )
            conn.commit()
        finally:
            conn.close()

    def mark_completed_campaigns(self) -> None:
        conn = self._get_db_connection()
        try:
            conn.execute(
                """
                UPDATE marketing_campaigns
                SET status='completed', updated_at=CURRENT_TIMESTAMP
                WHERE id IN (
                    SELECT c.id
                    FROM marketing_campaigns c
                    LEFT JOIN campaign_recipients r ON r.campaign_id = c.id
                    GROUP BY c.id
                    HAVING SUM(CASE WHEN r.status IN ('pending', 'failed') AND r.retry_count < 3 THEN 1 ELSE 0 END) = 0
                )
                AND status IN ('scheduled', 'running')
                """
            )
            conn.commit()
        finally:
            conn.close()

    def save_webhook_log(self, event_type: str, payload: Dict[str, Any]) -> None:
        conn = self._get_db_connection()
        try:
            conn.execute(
                "INSERT INTO marketing_webhook_logs(event_type, payload_json) VALUES (?, ?)",
                (event_type, json.dumps(payload)),
            )
            conn.commit()
        finally:
            conn.close()

    def analytics_snapshot(self) -> Dict[str, Any]:
        conn = self._get_db_connection()
        conn.row_factory = sqlite3.Row
        try:
            total_contacts = conn.execute(
                "SELECT COUNT(DISTINCT mobile) AS count FROM whatsapp_messages WHERE mobile IS NOT NULL AND TRIM(mobile) != ''"
            ).fetchone()["count"]
            campaign_count = conn.execute("SELECT COUNT(*) AS count FROM marketing_campaigns").fetchone()["count"]
            sent_today = conn.execute(
                """
                SELECT COUNT(*) AS count FROM campaign_recipients
                WHERE status='sent' AND DATE(updated_at)=DATE('now')
                """
            ).fetchone()["count"]
            delivery = conn.execute(
                """
                SELECT
                    SUM(CASE WHEN status='sent' THEN 1 ELSE 0 END) AS sent_count,
                    COUNT(*) AS total_count
                FROM campaign_recipients
                """
            ).fetchone()
            total = delivery["total_count"] or 0
            sent = delivery["sent_count"] or 0
            rate = round((sent / total) * 100, 2) if total else 0

            by_day = conn.execute(
                """
                SELECT DATE(updated_at) AS day,
                       SUM(CASE WHEN status='sent' THEN 1 ELSE 0 END) AS sent_count
                FROM campaign_recipients
                GROUP BY DATE(updated_at)
                ORDER BY day DESC
                LIMIT 14
                """
            ).fetchall()
            campaign_perf = conn.execute(
                """
                SELECT c.name,
                       SUM(CASE WHEN r.status='sent' THEN 1 ELSE 0 END) AS sent_count,
                       COUNT(r.id) AS total_recipients
                FROM marketing_campaigns c
                LEFT JOIN campaign_recipients r ON r.campaign_id = c.id
                GROUP BY c.id
                ORDER BY c.created_at DESC
                LIMIT 8
                """
            ).fetchall()

            return {
                "total_contacts": total_contacts,
                "campaign_count": campaign_count,
                "sent_today": sent_today,
                "delivery_rate": rate,
                "messages_over_time": [dict(row) for row in reversed(by_day)],
                "campaign_stats": [dict(row) for row in campaign_perf],
            }
        finally:
            conn.close()
