import os
import re
from typing import Any, Dict, List

import requests

from .models import MarketingRepository


class TemplateSyncService:
    def __init__(self, repository: MarketingRepository) -> None:
        self.repository = repository

    @staticmethod
    def _extract_body_text(template: Dict[str, Any]) -> str:
        for component in template.get("components", []):
            if component.get("type") == "BODY":
                return component.get("text", "")
        return ""

    @staticmethod
    def _variables_count(text: str) -> int:
        matches = [int(v) for v in re.findall(r"\{\{\s*(\d+)\s*\}\}", text or "")]
        return max(matches) if matches else 0

    def sync(self) -> Dict[str, int]:
        token = os.environ.get("META_ACCESS_TOKEN", "")
        waba_id = os.environ.get("WABA_ID", "")
        version = os.environ.get("WHATSAPP_API_VERSION", "v20.0")
        if not token or not waba_id:
            raise RuntimeError("META_ACCESS_TOKEN and WABA_ID are required for template sync")

        url = f"https://graph.facebook.com/{version}/{waba_id}/message_templates"
        headers = {"Authorization": f"Bearer {token}"}

        synced = 0
        skipped = 0
        while url:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            payload = response.json()
            for template in payload.get("data", []):
                if template.get("status") != "APPROVED":
                    skipped += 1
                    continue
                body = self._extract_body_text(template)
                var_count = self._variables_count(body)
                self.repository.upsert_template(template, var_count, body)
                synced += 1
            url = payload.get("paging", {}).get("next")
        return {"synced": synced, "skipped": skipped}

    def list_templates(self) -> List[Dict[str, Any]]:
        return [dict(row) for row in self.repository.list_templates()]
