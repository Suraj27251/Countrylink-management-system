"""
Tests for the TemplateValidator service.

Validates template placeholder parsing, mapping validation, customer parameter
validation, parameter sanitization, and template preview rendering.

Requirements: 20.1, 20.2, 20.3, 20.4, 20.5, 20.6, 20.7, 11.6
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.template_validator import (
    MAX_PARAM_LENGTH,
    Placeholder,
    ParamResult,
    TemplateValidator,
    ValidationResult,
)


class TestParsePlaceholders:
    """Tests for parse_placeholders() — Requirement 20.1"""

    def setup_method(self):
        self.validator = TemplateValidator()

    def test_empty_template(self):
        result = self.validator.parse_placeholders("")
        assert result == []

    def test_no_placeholders(self):
        result = self.validator.parse_placeholders("Hello, welcome to our service!")
        assert result == []

    def test_single_numeric_placeholder(self):
        result = self.validator.parse_placeholders("Hello {{1}}, welcome!")
        assert len(result) == 1
        assert result[0].name == "1"
        assert result[0].is_numeric is True
        assert result[0].position == 6

    def test_multiple_numeric_placeholders(self):
        template = "Hi {{1}}, your plan {{2}} expires on {{3}}"
        result = self.validator.parse_placeholders(template)
        assert len(result) == 3
        assert [p.name for p in result] == ["1", "2", "3"]
        assert all(p.is_numeric for p in result)

    def test_named_placeholders(self):
        template = "Hello {{name}}, your {{plan_name}} expires soon"
        result = self.validator.parse_placeholders(template)
        assert len(result) == 2
        assert result[0].name == "name"
        assert result[0].is_numeric is False
        assert result[1].name == "plan_name"
        assert result[1].is_numeric is False

    def test_mixed_numeric_and_named_placeholders(self):
        template = "Hi {{1}}, your {{plan_name}} is active"
        result = self.validator.parse_placeholders(template)
        assert len(result) == 2
        assert result[0].name == "1"
        assert result[0].is_numeric is True
        assert result[1].name == "plan_name"
        assert result[1].is_numeric is False

    def test_duplicate_placeholders_are_returned(self):
        template = "{{name}} - hello {{name}}"
        result = self.validator.parse_placeholders(template)
        assert len(result) == 2
        assert result[0].name == "name"
        assert result[1].name == "name"

    def test_position_tracking(self):
        template = "{{1}} and {{2}}"
        result = self.validator.parse_placeholders(template)
        assert result[0].position == 0
        assert result[1].position == 10

    def test_index_property_for_numeric(self):
        template = "Hello {{3}}"
        result = self.validator.parse_placeholders(template)
        assert result[0].index == 3

    def test_index_property_for_named(self):
        template = "Hello {{name}}"
        result = self.validator.parse_placeholders(template)
        assert result[0].index is None

    def test_underscore_in_placeholder_name(self):
        template = "{{first_name}} {{last_name}}"
        result = self.validator.parse_placeholders(template)
        assert len(result) == 2
        assert result[0].name == "first_name"
        assert result[1].name == "last_name"


class TestValidateMappings:
    """Tests for validate_mappings() — Requirements 20.2, 20.3, 20.4"""

    def setup_method(self):
        self.validator = TemplateValidator()

    def test_all_placeholders_mapped_is_valid(self):
        template = "Hi {{1}}, your plan {{2}} is active"
        mappings = {"1": "customer_name", "2": "plan_name"}
        result = self.validator.validate_mappings(template, mappings)
        assert result.is_valid is True
        assert result.unmapped_placeholders == []
        assert result.errors == []

    def test_missing_mapping_is_invalid(self):
        template = "Hi {{1}}, plan {{2}}, zone {{3}}"
        mappings = {"1": "customer_name", "2": "plan_name"}
        result = self.validator.validate_mappings(template, mappings)
        assert result.is_valid is False
        assert "3" in result.unmapped_placeholders

    def test_extra_mappings_detected(self):
        template = "Hi {{1}}"
        mappings = {"1": "customer_name", "2": "plan_name"}
        result = self.validator.validate_mappings(template, mappings)
        # Valid because all placeholders are mapped
        assert result.is_valid is True
        assert "2" in result.extra_mappings

    def test_empty_template_with_mappings(self):
        template = "No placeholders here"
        mappings = {"1": "customer_name"}
        result = self.validator.validate_mappings(template, mappings)
        assert result.is_valid is True
        assert result.extra_mappings == ["1"]

    def test_empty_mappings_with_placeholders(self):
        template = "Hi {{name}}"
        mappings = {}
        result = self.validator.validate_mappings(template, mappings)
        assert result.is_valid is False
        assert "name" in result.unmapped_placeholders

    def test_named_placeholders_mapping(self):
        template = "Hello {{name}}, your {{plan_name}} expires on {{expiry_date}}"
        mappings = {
            "name": "customer_name",
            "plan_name": "plan_name",
            "expiry_date": "expiry_date"
        }
        result = self.validator.validate_mappings(template, mappings)
        assert result.is_valid is True

    def test_error_messages_for_unmapped(self):
        template = "{{1}} {{2}} {{3}}"
        mappings = {"1": "name"}
        result = self.validator.validate_mappings(template, mappings)
        assert any("Unmapped" in e for e in result.errors)

    def test_duplicate_placeholders_only_need_one_mapping(self):
        template = "{{name}} is great, {{name}} is awesome"
        mappings = {"name": "customer_name"}
        result = self.validator.validate_mappings(template, mappings)
        assert result.is_valid is True


class TestValidateCustomerParams:
    """Tests for validate_customer_params() — Requirements 20.5, 20.6"""

    def setup_method(self):
        self.validator = TemplateValidator()

    def test_valid_customer_params(self):
        customer = {"customer_name": "Rajesh Kumar", "plan_name": "Eclipse 100"}
        mappings = {"1": "customer_name", "2": "plan_name"}
        result = self.validator.validate_customer_params(customer, mappings)
        assert result.is_valid is True
        assert result.resolved_params == {"1": "Rajesh Kumar", "2": "Eclipse 100"}
        assert result.invalid_params == []

    def test_null_value_is_invalid(self):
        customer = {"customer_name": None, "plan_name": "Eclipse 100"}
        mappings = {"1": "customer_name", "2": "plan_name"}
        result = self.validator.validate_customer_params(customer, mappings)
        assert result.is_valid is False
        assert any(p[0] == "1" and "null" in p[1] for p in result.invalid_params)

    def test_missing_field_is_invalid(self):
        customer = {"plan_name": "Eclipse 100"}
        mappings = {"1": "customer_name", "2": "plan_name"}
        result = self.validator.validate_customer_params(customer, mappings)
        assert result.is_valid is False
        assert any(p[0] == "1" for p in result.invalid_params)

    def test_empty_string_is_invalid(self):
        customer = {"customer_name": "", "plan_name": "Eclipse 100"}
        mappings = {"1": "customer_name", "2": "plan_name"}
        result = self.validator.validate_customer_params(customer, mappings)
        assert result.is_valid is False
        assert any(p[0] == "1" and "empty" in p[1] for p in result.invalid_params)

    def test_whitespace_only_is_invalid(self):
        customer = {"customer_name": "   ", "plan_name": "Eclipse 100"}
        mappings = {"1": "customer_name", "2": "plan_name"}
        result = self.validator.validate_customer_params(customer, mappings)
        assert result.is_valid is False
        assert any(p[0] == "1" and "empty" in p[1] for p in result.invalid_params)

    def test_value_exceeding_1024_chars_is_invalid(self):
        customer = {"customer_name": "A" * 1025, "plan_name": "Eclipse"}
        mappings = {"1": "customer_name", "2": "plan_name"}
        result = self.validator.validate_customer_params(customer, mappings)
        assert result.is_valid is False
        assert any(p[0] == "1" and "1024" in p[1] for p in result.invalid_params)

    def test_value_at_exactly_1024_chars_is_valid(self):
        customer = {"customer_name": "A" * 1024, "plan_name": "Eclipse"}
        mappings = {"1": "customer_name", "2": "plan_name"}
        result = self.validator.validate_customer_params(customer, mappings)
        assert result.is_valid is True
        assert result.resolved_params["1"] == "A" * 1024

    def test_numeric_value_converted_to_string(self):
        customer = {"customer_name": "Test", "amount": 500}
        mappings = {"1": "customer_name", "2": "amount"}
        result = self.validator.validate_customer_params(customer, mappings)
        assert result.is_valid is True
        assert result.resolved_params["2"] == "500"

    def test_multiple_invalid_params(self):
        customer = {"a": None, "b": "", "c": "valid"}
        mappings = {"1": "a", "2": "b", "3": "c"}
        result = self.validator.validate_customer_params(customer, mappings)
        assert result.is_valid is False
        assert len(result.invalid_params) == 2
        assert "3" in result.resolved_params


class TestSanitizeParam:
    """Tests for sanitize_param() — Requirement 11.6"""

    def setup_method(self):
        self.validator = TemplateValidator()

    def test_normal_ascii_text_unchanged(self):
        assert self.validator.sanitize_param("Hello World") == "Hello World"

    def test_removes_null_byte(self):
        assert self.validator.sanitize_param("Hello\x00World") == "HelloWorld"

    def test_removes_tab_character(self):
        # \t is U+0009 which is in the control range
        assert self.validator.sanitize_param("Hello\tWorld") == "HelloWorld"

    def test_removes_newline(self):
        # \n is U+000A which is in the control range
        assert self.validator.sanitize_param("Hello\nWorld") == "HelloWorld"

    def test_removes_carriage_return(self):
        assert self.validator.sanitize_param("Hello\rWorld") == "HelloWorld"

    def test_removes_delete_character(self):
        # U+007F DEL
        assert self.validator.sanitize_param("Hello\x7fWorld") == "HelloWorld"

    def test_removes_c1_control_characters(self):
        # U+0080–U+009F range
        assert self.validator.sanitize_param("Hello\x80\x8f\x9fWorld") == "HelloWorld"

    def test_preserves_hindi_characters(self):
        hindi_text = "नमस्ते दुनिया"
        assert self.validator.sanitize_param(hindi_text) == hindi_text

    def test_preserves_marathi_characters(self):
        marathi_text = "नमस्कार जग"
        assert self.validator.sanitize_param(marathi_text) == marathi_text

    def test_preserves_unicode_emoji(self):
        emoji_text = "Hello 👋 World 🌍"
        assert self.validator.sanitize_param(emoji_text) == emoji_text

    def test_preserves_arabic_characters(self):
        arabic_text = "مرحبا بالعالم"
        assert self.validator.sanitize_param(arabic_text) == arabic_text

    def test_output_length_leq_input_length(self):
        text = "Test\x00\x01\x02Text"
        result = self.validator.sanitize_param(text)
        assert len(result) <= len(text)

    def test_empty_string(self):
        assert self.validator.sanitize_param("") == ""

    def test_all_control_chars_produces_empty(self):
        assert self.validator.sanitize_param("\x00\x01\x02\x1f") == ""

    def test_mixed_control_and_unicode(self):
        text = "\x00नमस्ते\x1f World\x7f"
        result = self.validator.sanitize_param(text)
        assert result == "नमस्ते World"

    def test_preserves_space_character(self):
        # Space (U+0020) is NOT in control range
        assert self.validator.sanitize_param("Hello World") == "Hello World"

    def test_preserves_tilde(self):
        # Tilde (U+007E) is NOT in control range
        assert self.validator.sanitize_param("~test~") == "~test~"

    def test_unicode_after_c1_range_preserved(self):
        # U+00A0 (non-breaking space) and above should be preserved
        text = "Hello\u00a0World"  # U+00A0 is just after the C1 control block
        assert self.validator.sanitize_param(text) == text


class TestRenderPreview:
    """Tests for render_preview() — Requirement 20.7"""

    def setup_method(self):
        self.validator = TemplateValidator()

    def test_basic_render(self):
        template = "Hello {{1}}, your plan is {{2}}"
        customer = {"customer_name": "Rajesh", "plan_name": "Eclipse 100"}
        mappings = {"1": "customer_name", "2": "plan_name"}
        result = self.validator.render_preview(template, customer, mappings)
        assert result == "Hello Rajesh, your plan is Eclipse 100"

    def test_named_placeholders_render(self):
        template = "Hi {{name}}, {{plan}} expires soon"
        customer = {"customer_name": "Priya", "plan_name": "Fiber 50"}
        mappings = {"name": "customer_name", "plan": "plan_name"}
        result = self.validator.render_preview(template, customer, mappings)
        assert result == "Hi Priya, Fiber 50 expires soon"

    def test_unmapped_placeholder_left_as_is(self):
        template = "Hello {{1}}, your code is {{2}}"
        customer = {"customer_name": "Test"}
        mappings = {"1": "customer_name"}
        result = self.validator.render_preview(template, customer, mappings)
        assert result == "Hello Test, your code is {{2}}"

    def test_missing_customer_field_leaves_placeholder(self):
        template = "Hello {{1}}, zone {{2}}"
        customer = {"customer_name": "Test"}
        mappings = {"1": "customer_name", "2": "zone_name"}
        result = self.validator.render_preview(template, customer, mappings)
        assert result == "Hello Test, zone {{2}}"

    def test_sanitizes_values_during_render(self):
        template = "Hi {{1}}"
        customer = {"customer_name": "Rajesh\x00Kumar"}
        mappings = {"1": "customer_name"}
        result = self.validator.render_preview(template, customer, mappings)
        assert result == "Hi RajeshKumar"
        assert "\x00" not in result

    def test_empty_string_value_leaves_placeholder(self):
        template = "Hi {{1}}"
        customer = {"customer_name": ""}
        mappings = {"1": "customer_name"}
        result = self.validator.render_preview(template, customer, mappings)
        assert result == "Hi {{1}}"

    def test_none_value_leaves_placeholder(self):
        template = "Hi {{1}}"
        customer = {"customer_name": None}
        mappings = {"1": "customer_name"}
        result = self.validator.render_preview(template, customer, mappings)
        assert result == "Hi {{1}}"

    def test_numeric_value_rendered_as_string(self):
        template = "Your balance is {{1}} INR"
        customer = {"balance": 1500}
        mappings = {"1": "balance"}
        result = self.validator.render_preview(template, customer, mappings)
        assert result == "Your balance is 1500 INR"

    def test_hindi_text_preserved_in_render(self):
        template = "{{1}} को {{2}} की शुभकामनाएं"
        customer = {"name": "राजेश", "festival": "दीवाली"}
        mappings = {"1": "name", "2": "festival"}
        result = self.validator.render_preview(template, customer, mappings)
        assert result == "राजेश को दीवाली की शुभकामनाएं"

    def test_template_with_no_placeholders(self):
        template = "Welcome to our service!"
        customer = {"name": "Test"}
        mappings = {}
        result = self.validator.render_preview(template, customer, mappings)
        assert result == "Welcome to our service!"

    def test_duplicate_placeholder_rendered_both(self):
        template = "{{name}} says hello to {{name}}"
        customer = {"customer_name": "Priya"}
        mappings = {"name": "customer_name"}
        result = self.validator.render_preview(template, customer, mappings)
        assert result == "Priya says hello to Priya"
