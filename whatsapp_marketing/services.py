import json
import os
import re
from typing import Any, Dict, List

import requests


class WhatsAppServiceError(Exception):
    pass


class WhatsAppMessageService:
    def __init__(self) -> None:
        self.api_version = os.environ.get("WHATSAPP_API_VERSION", "v20.0")
        self.access_token = os.environ.get("META_ACCESS_TOKEN", "")
        self.phone_number_id = os.environ.get("PHONE_NUMBER_ID", "")

    def _validate_env(self) -> None:
        missing = [
            key
            for key, value in {
                "META_ACCESS_TOKEN": self.access_token,
                "PHONE_NUMBER_ID": self.phone_number_id,
            }.items()
            if not value
        ]
        if missing:
            raise WhatsAppServiceError(f"Missing environment variables: {', '.join(missing)}")

    @staticmethod
    def count_expected_variables(template_body_text: str) -> int:
        indices = [int(match) for match in re.findall(r"\{\{\s*(\d+)\s*\}\}", template_body_text or "")]
        return max(indices) if indices else 0

    def send_template_message(self, to: str, template_name: str, language: str, variables: List[str], expected_count: int) -> Dict[str, Any]:
        self._validate_env()

        cleaned = re.sub(r"\D", "", to or "")
        if not cleaned:
            raise WhatsAppServiceError("Invalid destination number.")

        if expected_count != len(variables):
            raise WhatsAppServiceError(f"Invalid variable count. Expected {expected_count}, received {len(variables)}")

        components = []
        if variables:
            components.append(
                {
                    "type": "body",
                    "parameters": [{"type": "text", "text": str(value)} for value in variables],
                }
            )

        payload = {
            "messaging_product": "whatsapp",
            "to": cleaned,
            "type": "template",
            "template": {
                "name": template_name,
                "language": {"code": language},
                "components": components,
            },
        }

        url = f"https://graph.facebook.com/{self.api_version}/{self.phone_number_id}/messages"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }

        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
        data = response.json() if response.content else {}
        if response.status_code >= 400:
            error_details = data.get("error", {})
            raise WhatsAppServiceError(error_details.get("message", f"Meta API call failed with {response.status_code}"))
        return data
