from datetime import datetime
from typing import Any, Dict, List, Optional

from .models import MarketingRepository
from .services import WhatsAppMessageService, WhatsAppServiceError


class CampaignEngine:
    def __init__(self, repository: MarketingRepository, message_service: WhatsAppMessageService) -> None:
        self.repository = repository
        self.message_service = message_service

    def _resolve_contact_field(self, contact: Dict[str, Any], field_name: str) -> Optional[str]:
        if "." in field_name:
            _, key = field_name.split(".", 1)
        else:
            key = field_name
        value = contact.get(key)
        return "" if value is None else str(value)

    def create_campaign(self, payload: Dict[str, Any]) -> int:
        name = (payload.get("name") or "").strip()
        template_id = int(payload.get("template_id", 0))
        scheduled_time = payload.get("scheduled_time")
        mappings = payload.get("variable_mapping", [])

        if not name:
            raise ValueError("Campaign name is required")

        template = self.repository.get_template(template_id)
        if not template:
            raise ValueError("Invalid template selected")

        clean_mappings = []
        for item in mappings:
            try:
                pos = int(item.get("variable_position"))
                field = str(item.get("field_name", "")).strip()
                if pos <= 0 or not field:
                    continue
                clean_mappings.append((pos, field))
            except (TypeError, ValueError):
                continue

        contact_ids = payload.get("contact_ids") or []
        campaign_id = self.repository.create_campaign(name, template_id, scheduled_time, clean_mappings)
        self.repository.enqueue_campaign_contacts(campaign_id, contact_ids)
        return campaign_id

    def run_pending(self, batch_size: int = 50) -> Dict[str, int]:
        pending = self.repository.get_pending_recipients(limit=batch_size)
        contacts = {row["contact_id"]: dict(row) for row in self.repository.list_contacts()}

        results = {"processed": 0, "sent": 0, "failed": 0, "skipped": 0}
        for recipient in pending:
            results["processed"] += 1
            campaign_id = int(recipient["campaign_id"])
            self.repository.mark_campaign_running(campaign_id)

            mapping = self.repository.get_campaign_mappings(campaign_id)
            contact = contacts.get(recipient["contact_id"]) or {
                "contact_id": recipient["contact_id"],
                "contact_name": recipient["contact_id"],
                "mobile": recipient["contact_id"],
            }

            variables: List[str] = []
            invalid_mapping = False
            for pos in range(1, int(recipient["variables_count"]) + 1):
                field_name = mapping.get(pos)
                if not field_name:
                    invalid_mapping = True
                    break
                variables.append(self._resolve_contact_field(contact, field_name))

            if invalid_mapping:
                self.repository.mark_recipient_result(recipient["recipient_id"], "failed", None, "Invalid variable mapping")
                results["skipped"] += 1
                continue

            try:
                response = self.message_service.send_template_message(
                    to=contact.get("mobile", recipient["contact_id"]),
                    template_name=recipient["template_name"],
                    language=recipient["language"],
                    variables=variables,
                    expected_count=int(recipient["variables_count"]),
                )
                message_id = None
                msgs = response.get("messages") or []
                if msgs:
                    message_id = msgs[0].get("id")
                self.repository.mark_recipient_result(recipient["recipient_id"], "sent", message_id, None)
                results["sent"] += 1
            except (WhatsAppServiceError, ValueError) as exc:
                self.repository.mark_recipient_result(recipient["recipient_id"], "failed", None, str(exc))
                results["failed"] += 1

        self.repository.mark_completed_campaigns()
        return results
