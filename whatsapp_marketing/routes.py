import hashlib
import hmac
import json
import os
from typing import Callable

from flask import Blueprint, current_app, jsonify, render_template, request

from .campaign_engine import CampaignEngine
from .models import MarketingRepository
from .services import WhatsAppMessageService
from .template_sync import TemplateSyncService


marketing_bp = Blueprint("marketing", __name__, url_prefix="/marketing")


def _repo() -> MarketingRepository:
    getter: Callable = current_app.config["MARKETING_GET_DB_CONNECTION"]
    return MarketingRepository(getter)


def _engine() -> CampaignEngine:
    return CampaignEngine(_repo(), WhatsAppMessageService())


def _template_sync() -> TemplateSyncService:
    return TemplateSyncService(_repo())


@marketing_bp.route("/dashboard", methods=["GET"])
def dashboard() -> str:
    repository = _repo()
    analytics = repository.analytics_snapshot()
    recent_campaigns = [dict(row) for row in repository.list_campaigns()[:10]]
    return render_template(
        "marketing_dashboard.html",
        analytics=analytics,
        recent_campaigns=recent_campaigns,
    )


@marketing_bp.route("/campaigns", methods=["GET"])
def campaigns_page() -> str:
    repository = _repo()
    return render_template(
        "marketing_campaigns.html",
        campaigns=[dict(r) for r in repository.list_campaigns()],
        templates=[dict(t) for t in repository.list_templates()],
        contacts=[dict(c) for c in repository.list_contacts()],
    )


@marketing_bp.route("/templates", methods=["GET"])
def templates_page() -> str:
    return render_template("marketing_templates.html", templates=_template_sync().list_templates())


@marketing_bp.route("/api/templates", methods=["GET"])
def api_list_templates():
    return jsonify({"templates": _template_sync().list_templates()})


@marketing_bp.route("/templates/sync", methods=["POST"])
def sync_templates():
    result = _template_sync().sync()
    return jsonify(result)


@marketing_bp.route("/campaigns", methods=["POST"])
def create_campaign():
    payload = request.get_json(silent=True) or {}
    campaign_id = _engine().create_campaign(payload)
    return jsonify({"campaign_id": campaign_id, "status": "created"}), 201


@marketing_bp.route("/campaigns/run", methods=["POST"])
def run_campaigns():
    payload = request.get_json(silent=True) or {}
    batch_size = int(payload.get("batch_size", 50))
    result = _engine().run_pending(batch_size=batch_size)
    return jsonify(result)


@marketing_bp.route("/webhook", methods=["GET"])
def webhook_verify():
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge", "")
    expected = os.environ.get("WEBHOOK_VERIFY_TOKEN", "")
    if mode == "subscribe" and token and token == expected:
        return challenge, 200
    return "Verification failed", 403


@marketing_bp.route("/webhook", methods=["POST"])
def webhook_event():
    payload = request.get_json(silent=True) or {}
    signature = request.headers.get("X-Hub-Signature-256", "")
    app_secret = os.environ.get("META_APP_SECRET", "")

    if app_secret and signature:
        raw = request.get_data() or b""
        digest = "sha256=" + hmac.new(app_secret.encode("utf-8"), raw, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(digest, signature):
            return jsonify({"error": "Invalid signature"}), 403

    repo = _repo()
    repo.save_webhook_log("webhook_event", payload)

    entries = payload.get("entry", [])
    for entry in entries:
        for change in entry.get("changes", []):
            value = change.get("value", {})
            if value.get("statuses"):
                repo.save_webhook_log("message_status", {"statuses": value.get("statuses")})
            if value.get("messages"):
                repo.save_webhook_log("incoming_message", {"messages": value.get("messages")})

    return jsonify({"status": "ok"}), 200


def init_marketing(app, get_db_connection) -> None:
    app.config["MARKETING_GET_DB_CONNECTION"] = get_db_connection
    repo = MarketingRepository(get_db_connection)
    repo.init_schema()
    app.register_blueprint(marketing_bp)
