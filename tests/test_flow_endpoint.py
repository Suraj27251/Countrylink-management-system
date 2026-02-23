import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sqlite3

import app as app_module


def _configure_test_db(tmp_path):
    test_db = tmp_path / "test.db"
    app_module.DB_PATH = str(test_db)
    app_module.init_db()
    return test_db


def test_parse_flow_payload_rejects_non_dict():
    payload, error = app_module.parse_flow_payload(None)
    assert payload is None
    assert error == "Invalid JSON payload"


def test_parse_flow_payload_strips_and_validates_fields():
    payload, error = app_module.parse_flow_payload(
        {"name": "  Alice  ", "mobile": " 999 ", "complaint": " Slow internet "}
    )
    assert error is None
    assert payload == {"name": "Alice", "mobile": "999", "complaint": "Slow internet"}


def test_flow_endpoint_returns_json_error_for_invalid_json(tmp_path):
    _configure_test_db(tmp_path)
    client = app_module.app.test_client()

    response = client.post("/flow-endpoint", data="not json", content_type="application/json")

    assert response.status_code == 400
    assert response.is_json
    assert response.get_json() == {"error": "Invalid JSON payload"}


def test_flow_endpoint_inserts_complaint_for_valid_payload(tmp_path):
    db_path = _configure_test_db(tmp_path)
    client = app_module.app.test_client()

    response = client.post(
        "/flow-endpoint",
        json={"name": "Bob", "mobile": "1234567890", "complaint": "Internet is down"},
    )

    assert response.status_code == 200
    assert response.get_json() == {"status": "received"}

    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT name, mobile, complaint, category, source FROM complaints ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()

    assert row is not None
    assert row[0] == "Bob"
    assert row[1] == "1234567890"
    assert row[2] == "Internet is down"
    assert row[4] == "Web"
