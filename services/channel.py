"""
Channel abstraction layer for omnichannel message dispatch.

Provides a generic MessageDispatcher interface with WhatsApp as the initial
implementation. Future channels (SMS, Email, Telegram) implement the same ABC.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import requests


@dataclass
class DispatchResult:
    """Result of a message dispatch attempt."""
    success: bool
    message_id: Optional[str] = None
    error_code: Optional[int] = None
    error_message: Optional[str] = None


class MessageDispatcher(ABC):
    """Abstract base class for channel-specific message dispatchers."""

    @abstractmethod
    def send_template(
        self,
        recipient: str,
        template_name: str,
        params: List[str],
        media_url: Optional[str] = None,
        language: Optional[str] = None,
    ) -> DispatchResult:
        """
        Send a template message to a recipient.

        Args:
            recipient: The recipient phone number (with country code, e.g. "919876543210").
            template_name: The approved template name registered with the channel provider.
            params: Positional parameters to fill template placeholders.
            media_url: Optional media URL for templates with a header media component.
            language: Optional language code override (e.g. "en", "en_US").
                      If not provided, the dispatcher default is used.

        Returns:
            DispatchResult with success status and message ID or error details.
        """
        pass

    @abstractmethod
    def get_channel_name(self) -> str:
        """Return the canonical channel identifier (e.g. 'whatsapp', 'sms', 'email')."""
        pass


class WhatsAppDispatcher(MessageDispatcher):
    """
    WhatsApp Business API dispatcher using Meta's Cloud API.

    Uses the existing helper functions from app.py for authentication
    and configuration, passed in as callables to avoid circular imports.
    """

    def __init__(
        self,
        get_headers_fn=None,
        api_version: Optional[str] = None,
        get_phone_number_id_fn=None,
        template_language: str = "en",
    ):
        """
        Initialize the WhatsApp dispatcher.

        Args:
            get_headers_fn: Callable returning authorization headers dict.
                            Defaults to app.get_whatsapp_headers.
            api_version: WhatsApp API version string (e.g. "v20.0").
                         Defaults to app.WHATSAPP_API_VERSION.
            get_phone_number_id_fn: Callable returning the phone number ID.
                                    Defaults to app.get_whatsapp_phone_number_id.
            template_language: Default language code for templates.
        """
        # Lazy import to avoid circular dependency at module load time
        if get_headers_fn is None:
            from app import get_whatsapp_headers
            get_headers_fn = get_whatsapp_headers

        if api_version is None:
            from app import WHATSAPP_API_VERSION
            api_version = WHATSAPP_API_VERSION

        if get_phone_number_id_fn is None:
            from app import get_whatsapp_phone_number_id
            get_phone_number_id_fn = get_whatsapp_phone_number_id

        self._get_headers = get_headers_fn
        self._api_version = api_version
        self._get_phone_number_id = get_phone_number_id_fn
        self._template_language = template_language

    def get_channel_name(self) -> str:
        return "whatsapp"

    def send_template(
        self,
        recipient: str,
        template_name: str,
        params: List[str],
        media_url: Optional[str] = None,
        language: Optional[str] = None,
    ) -> DispatchResult:
        """
        Send a WhatsApp template message via Meta's Cloud API.

        Constructs the template payload and POSTs to:
        https://graph.facebook.com/{api_version}/{phone_number_id}/messages

        Args:
            recipient: Phone number with country code (e.g. "919876543210").
            template_name: Approved Meta template name.
            params: List of parameter values for body placeholders.
            media_url: Optional URL for header image/video/document component.
            language: Optional language code override (e.g. "en_US").
                      Uses self._template_language as fallback.

        Returns:
            DispatchResult with success=True and message_id on success,
            or success=False with error_code and error_message on failure.
        """
        try:
            headers = self._get_headers()
            phone_number_id = self._get_phone_number_id()
        except RuntimeError as exc:
            return DispatchResult(
                success=False,
                error_code=None,
                error_message=f"Configuration error: {exc}",
            )

        url = (
            f"https://graph.facebook.com/{self._api_version}"
            f"/{phone_number_id}/messages"
        )

        # Build template components
        components = []

        # Header component with media (if provided)
        if media_url:
            components.append({
                "type": "header",
                "parameters": [
                    {"type": "image", "image": {"link": media_url}}
                ],
            })

        # Body parameters
        if params:
            body_parameters = [
                {"type": "text", "text": str(p)} for p in params
            ]
            components.append({
                "type": "body",
                "parameters": body_parameters,
            })

        # Construct the full payload — use passed language or fall back to default
        language_code = language or self._template_language
        payload = {
            "messaging_product": "whatsapp",
            "to": recipient,
            "type": "template",
            "template": {
                "name": template_name,
                "language": {"code": language_code},
            },
        }

        if components:
            payload["template"]["components"] = components

        # Add Content-Type to headers
        request_headers = {**headers, "Content-Type": "application/json"}

        try:
            response = requests.post(
                url, json=payload, headers=request_headers, timeout=30
            )
        except requests.RequestException as exc:
            return DispatchResult(
                success=False,
                error_code=None,
                error_message=f"Network error: {exc}",
            )

        # Parse response
        if response.status_code == 200 or response.status_code == 201:
            try:
                data = response.json()
                message_id = (
                    data.get("messages", [{}])[0].get("id")
                    if data.get("messages")
                    else None
                )
                return DispatchResult(success=True, message_id=message_id)
            except (ValueError, IndexError, KeyError):
                # Response was 2xx but couldn't parse message ID
                return DispatchResult(success=True, message_id=None)
        else:
            # Error response from Meta API
            error_code = response.status_code
            error_message = None
            try:
                error_data = response.json()
                error_obj = error_data.get("error", {})
                error_code = error_obj.get("code", response.status_code)
                error_message = error_obj.get("message", response.text)
            except (ValueError, KeyError):
                error_message = response.text

            return DispatchResult(
                success=False,
                error_code=error_code,
                error_message=error_message,
            )
