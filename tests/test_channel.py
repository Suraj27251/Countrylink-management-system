"""Tests for the channel abstraction layer (services/channel.py)."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.channel import DispatchResult, MessageDispatcher, WhatsAppDispatcher


class TestDispatchResult:
    """Tests for the DispatchResult dataclass."""

    def test_success_result(self):
        result = DispatchResult(success=True, message_id="wamid.abc123")
        assert result.success is True
        assert result.message_id == "wamid.abc123"
        assert result.error_code is None
        assert result.error_message is None

    def test_failure_result(self):
        result = DispatchResult(
            success=False, error_code=131026, error_message="Invalid number"
        )
        assert result.success is False
        assert result.message_id is None
        assert result.error_code == 131026
        assert result.error_message == "Invalid number"

    def test_default_optional_fields(self):
        result = DispatchResult(success=True)
        assert result.message_id is None
        assert result.error_code is None
        assert result.error_message is None


class TestMessageDispatcherABC:
    """Tests for the MessageDispatcher abstract base class."""

    def test_cannot_instantiate_abc(self):
        try:
            MessageDispatcher()
            assert False, "Should not be able to instantiate ABC"
        except TypeError:
            pass

    def test_concrete_subclass_must_implement_methods(self):
        class IncompleteDispatcher(MessageDispatcher):
            pass

        try:
            IncompleteDispatcher()
            assert False, "Should not instantiate without implementing abstract methods"
        except TypeError:
            pass

    def test_concrete_subclass_works(self):
        class MockDispatcher(MessageDispatcher):
            def send_template(self, recipient, template_name, params, media_url=None):
                return DispatchResult(success=True, message_id="mock_id")

            def get_channel_name(self):
                return "mock"

        dispatcher = MockDispatcher()
        assert dispatcher.get_channel_name() == "mock"
        result = dispatcher.send_template("123", "test", [])
        assert result.success is True


class TestWhatsAppDispatcher:
    """Tests for the WhatsAppDispatcher implementation."""

    def _make_dispatcher(self, headers=None, api_version="v20.0", phone_id="12345"):
        """Create a dispatcher with mocked dependencies."""
        return WhatsAppDispatcher(
            get_headers_fn=lambda: headers or {"Authorization": "Bearer test_token"},
            api_version=api_version,
            get_phone_number_id_fn=lambda: phone_id,
        )

    def test_get_channel_name(self):
        dispatcher = self._make_dispatcher()
        assert dispatcher.get_channel_name() == "whatsapp"

    @patch("services.channel.requests.post")
    def test_send_template_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "messages": [{"id": "wamid.HBgLMTIzNDU2Nzg5MA=="}]
        }
        mock_post.return_value = mock_response

        dispatcher = self._make_dispatcher()
        result = dispatcher.send_template(
            recipient="919876543210",
            template_name="hello_world",
            params=["John"],
        )

        assert result.success is True
        assert result.message_id == "wamid.HBgLMTIzNDU2Nzg5MA=="
        assert result.error_code is None

        # Verify the API was called correctly
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        url = call_args[0][0] if call_args[0] else call_args[1].get("url")
        assert url == "https://graph.facebook.com/v20.0/12345/messages"

    @patch("services.channel.requests.post")
    def test_send_template_with_media_url(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "messages": [{"id": "wamid.media123"}]
        }
        mock_post.return_value = mock_response

        dispatcher = self._make_dispatcher()
        result = dispatcher.send_template(
            recipient="919876543210",
            template_name="promo_image",
            params=["Special Offer"],
            media_url="https://example.com/image.jpg",
        )

        assert result.success is True
        assert result.message_id == "wamid.media123"

        # Verify payload includes header component with media
        call_kwargs = mock_post.call_args
        payload = call_kwargs[1]["json"] if "json" in call_kwargs[1] else call_kwargs[1].get("json")
        template = payload["template"]
        assert "components" in template
        header_component = next(
            c for c in template["components"] if c["type"] == "header"
        )
        assert header_component["parameters"][0]["type"] == "image"
        assert header_component["parameters"][0]["image"]["link"] == "https://example.com/image.jpg"

    @patch("services.channel.requests.post")
    def test_send_template_with_body_params(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"messages": [{"id": "wamid.params"}]}
        mock_post.return_value = mock_response

        dispatcher = self._make_dispatcher()
        result = dispatcher.send_template(
            recipient="919876543210",
            template_name="renewal_reminder",
            params=["John", "Gold Plan", "2024-12-31"],
        )

        assert result.success is True

        # Verify body parameters
        call_kwargs = mock_post.call_args
        payload = call_kwargs[1]["json"] if "json" in call_kwargs[1] else call_kwargs[1].get("json")
        template = payload["template"]
        body_component = next(
            c for c in template["components"] if c["type"] == "body"
        )
        assert len(body_component["parameters"]) == 3
        assert body_component["parameters"][0] == {"type": "text", "text": "John"}
        assert body_component["parameters"][1] == {"type": "text", "text": "Gold Plan"}
        assert body_component["parameters"][2] == {"type": "text", "text": "2024-12-31"}

    @patch("services.channel.requests.post")
    def test_send_template_no_params_no_media(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"messages": [{"id": "wamid.noparam"}]}
        mock_post.return_value = mock_response

        dispatcher = self._make_dispatcher()
        result = dispatcher.send_template(
            recipient="919876543210",
            template_name="simple_hello",
            params=[],
        )

        assert result.success is True

        # Verify no components added when params and media are empty
        call_kwargs = mock_post.call_args
        payload = call_kwargs[1]["json"] if "json" in call_kwargs[1] else call_kwargs[1].get("json")
        template = payload["template"]
        assert "components" not in template

    @patch("services.channel.requests.post")
    def test_send_template_api_error(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        mock_response.json.return_value = {
            "error": {
                "message": "Invalid parameter",
                "type": "OAuthException",
                "code": 100,
            }
        }
        mock_post.return_value = mock_response

        dispatcher = self._make_dispatcher()
        result = dispatcher.send_template(
            recipient="invalid",
            template_name="hello_world",
            params=["Test"],
        )

        assert result.success is False
        assert result.error_code == 100
        assert "Invalid parameter" in result.error_message

    @patch("services.channel.requests.post")
    def test_send_template_rate_limit_error(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "Rate limit exceeded"
        mock_response.json.return_value = {
            "error": {
                "message": "Rate limit exceeded",
                "code": 131047,
            }
        }
        mock_post.return_value = mock_response

        dispatcher = self._make_dispatcher()
        result = dispatcher.send_template(
            recipient="919876543210",
            template_name="hello_world",
            params=[],
        )

        assert result.success is False
        assert result.error_code == 131047
        assert "Rate limit" in result.error_message

    @patch("services.channel.requests.post")
    def test_send_template_network_error(self, mock_post):
        import requests as req
        mock_post.side_effect = req.ConnectionError("Connection refused")

        dispatcher = self._make_dispatcher()
        result = dispatcher.send_template(
            recipient="919876543210",
            template_name="hello_world",
            params=[],
        )

        assert result.success is False
        assert result.error_code is None
        assert "Network error" in result.error_message

    @patch("services.channel.requests.post")
    def test_send_template_timeout(self, mock_post):
        import requests as req
        mock_post.side_effect = req.Timeout("Request timed out")

        dispatcher = self._make_dispatcher()
        result = dispatcher.send_template(
            recipient="919876543210",
            template_name="hello_world",
            params=[],
        )

        assert result.success is False
        assert "Network error" in result.error_message

    def test_configuration_error_missing_token(self):
        """Test graceful handling when token is not configured."""
        dispatcher = WhatsAppDispatcher(
            get_headers_fn=lambda: (_ for _ in ()).throw(RuntimeError("Missing META_ACCESS_TOKEN")),
            api_version="v20.0",
            get_phone_number_id_fn=lambda: "12345",
        )
        result = dispatcher.send_template("919876543210", "hello", [])
        assert result.success is False
        assert "Configuration error" in result.error_message

    def test_configuration_error_missing_phone_id(self):
        """Test graceful handling when phone number ID is not configured."""
        dispatcher = WhatsAppDispatcher(
            get_headers_fn=lambda: {"Authorization": "Bearer token"},
            api_version="v20.0",
            get_phone_number_id_fn=lambda: (_ for _ in ()).throw(RuntimeError("Missing PHONE_NUMBER_ID")),
        )
        result = dispatcher.send_template("919876543210", "hello", [])
        assert result.success is False
        assert "Configuration error" in result.error_message

    @patch("services.channel.requests.post")
    def test_payload_structure(self, mock_post):
        """Verify the exact payload structure sent to Meta API."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"messages": [{"id": "wamid.test"}]}
        mock_post.return_value = mock_response

        dispatcher = self._make_dispatcher(
            headers={"Authorization": "Bearer my_token"},
            api_version="v20.0",
            phone_id="9988776655",
        )
        dispatcher.send_template(
            recipient="919876543210",
            template_name="renewal_notice",
            params=["Suraj", "100 Mbps"],
            media_url=None,
        )

        call_args = mock_post.call_args
        url = call_args[0][0] if call_args[0] else call_args[1].get("url")
        assert url == "https://graph.facebook.com/v20.0/9988776655/messages"

        payload = call_args[1]["json"]
        assert payload["messaging_product"] == "whatsapp"
        assert payload["to"] == "919876543210"
        assert payload["type"] == "template"
        assert payload["template"]["name"] == "renewal_notice"
        assert payload["template"]["language"] == {"code": "en"}

    @patch("services.channel.requests.post")
    def test_custom_template_language(self, mock_post):
        """Verify custom template language is used."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"messages": [{"id": "wamid.lang"}]}
        mock_post.return_value = mock_response

        dispatcher = WhatsAppDispatcher(
            get_headers_fn=lambda: {"Authorization": "Bearer token"},
            api_version="v20.0",
            get_phone_number_id_fn=lambda: "12345",
            template_language="hi",
        )
        dispatcher.send_template("919876543210", "hindi_template", ["param1"])

        payload = mock_post.call_args[1]["json"]
        assert payload["template"]["language"] == {"code": "hi"}

    @patch("services.channel.requests.post")
    def test_send_template_201_status(self, mock_post):
        """Some API versions return 201 for successful creation."""
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"messages": [{"id": "wamid.201"}]}
        mock_post.return_value = mock_response

        dispatcher = self._make_dispatcher()
        result = dispatcher.send_template("919876543210", "test", [])

        assert result.success is True
        assert result.message_id == "wamid.201"
