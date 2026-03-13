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


def test_payment_route_renders_payment_template(tmp_path):
    _configure_test_db(tmp_path)
    client = app_module.app.test_client()

    response = client.get("/payment")

    assert response.status_code == 200
    assert b"Internet Plan Checkout" in response.data


def test_razorpay_order_endpoint_returns_503_when_not_configured(tmp_path, monkeypatch):
    _configure_test_db(tmp_path)
    client = app_module.app.test_client()
    monkeypatch.delenv("RAZORPAY_KEY_ID", raising=False)
    monkeypatch.delenv("RAZORPAY_KEY_SECRET", raising=False)

    response = client.post("/api/payments/razorpay/order", json={"amount": 1500})

    assert response.status_code == 503
    assert response.get_json() == {"error": "Razorpay is not configured on server"}


def test_razorpay_order_endpoint_creates_order(tmp_path, monkeypatch):
    _configure_test_db(tmp_path)
    client = app_module.app.test_client()
    monkeypatch.setenv("RAZORPAY_KEY_ID", "rzp_test_key")
    monkeypatch.setenv("RAZORPAY_KEY_SECRET", "test_secret")

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"id": "order_123", "amount": 2499, "currency": "INR"}

    def fake_post(url, auth=None, json=None, timeout=0):
        assert url.endswith("/orders")
        assert auth == ("rzp_test_key", "test_secret")
        assert json["amount"] == 2499
        assert json["currency"] == "INR"
        return FakeResponse()

    monkeypatch.setattr(app_module.requests, "post", fake_post)

    response = client.post(
        "/api/payments/razorpay/order",
        json={"amount": 2499, "plan_name": "100 Mbps Unlimited", "billing_cycle": "monthly"},
    )

    assert response.status_code == 200
    assert response.get_json() == {
        "id": "order_123",
        "amount": 2499,
        "currency": "INR",
        "key": "rzp_test_key",
    }
