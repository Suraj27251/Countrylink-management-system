import sys
from pathlib import Path

import sqlite3

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app as app_module


def _configure_test_db(tmp_path):
    test_db = tmp_path / "test_whatsapp.db"
    app_module.DB_PATH = str(test_db)
    app_module.init_db()
    return test_db


def test_webhook_post_stores_inbound_text_message(tmp_path):
    db_path = _configure_test_db(tmp_path)
    client = app_module.app.test_client()

    payload = {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "contacts": [{"wa_id": "919999999999", "profile": {"name": "Alice"}}],
                            "messages": [
                                {
                                    "id": "wamid.test.1",
                                    "from": "919999999999",
                                    "timestamp": "1760000000",
                                    "type": "text",
                                    "text": {"body": "Need support"},
                                }
                            ],
                        }
                    }
                ]
            }
        ],
    }

    response = client.post('/webhook', json=payload)

    assert response.status_code == 200

    conn = sqlite3.connect(db_path)
    row = conn.execute(
        """
        SELECT mobile, direction, message_type, text
        FROM whatsapp_messages
        WHERE message_id = 'wamid.test.1'
        """
    ).fetchone()
    conn.close()

    assert row is not None
    assert row[0] == "919999999999"
    assert row[1] == "inbound"
    assert row[2] == "text"
    assert row[3] == "Need support"


def test_extract_text_body_handles_interactive_flow_response_json():
    message = {
        "type": "interactive",
        "interactive": {
            "nfm_reply": {
                "response_json": {
                    "customer_name": "Ravi",
                    "issue": "No internet",
                    "priority": "high",
                }
            }
        },
    }

    body = app_module.extract_text_body(message)

    assert "Flow response" in body
    assert "customer_name: Ravi" in body
    assert "issue: No internet" in body
