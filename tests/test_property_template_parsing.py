"""
Property-based tests for template placeholder parsing and mapping validation.

Property 24: Template placeholder parsing and mapping validation
- For any template body with N placeholders, exactly N are identified
- Block approval if fewer than N mappings provided
- Block if any resolved value is null/empty/>1024 chars

**Validates: Requirements 20.1, 20.2, 20.5, 20.6**

Testing framework: Hypothesis (Python)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import string
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from services.template_validator import (
    MAX_PARAM_LENGTH,
    TemplateValidator,
)


# --- Hypothesis Strategies ---

def placeholder_names_strategy(min_size=1, max_size=10):
    """Generate a list of unique placeholder names (numeric or named)."""
    numeric_names = st.integers(min_value=1, max_value=99).map(str)
    named_names = st.text(
        alphabet=string.ascii_lowercase + "_",
        min_size=2,
        max_size=15
    ).filter(lambda s: not s.startswith("_") and not s.endswith("_") and "__" not in s and not s.isdigit())

    placeholder_name = st.one_of(numeric_names, named_names)
    return st.lists(
        placeholder_name,
        min_size=min_size,
        max_size=max_size,
        unique=True
    )


def template_body_with_placeholders(placeholder_names):
    """Build a template body string containing the given placeholders.

    Inserts each placeholder in {{name}} format surrounded by static text.
    """
    parts = []
    for name in placeholder_names:
        parts.append(f"Hello {{{{{name}}}}}")
    return " ".join(parts)


@st.composite
def template_and_placeholders(draw):
    """Strategy that generates a template body and its expected placeholder names."""
    names = draw(placeholder_names_strategy(min_size=1, max_size=8))
    body = template_body_with_placeholders(names)
    return body, names


@st.composite
def template_with_partial_mappings(draw):
    """Strategy generating a template with N placeholders and fewer than N mappings."""
    names = draw(placeholder_names_strategy(min_size=2, max_size=8))
    body = template_body_with_placeholders(names)

    # Pick a subset strictly smaller than all names
    num_to_map = draw(st.integers(min_value=0, max_value=len(names) - 1))
    mapped_names = names[:num_to_map]

    # Create mappings for the subset
    mappings = {name: f"field_{name}" for name in mapped_names}
    return body, names, mappings


@st.composite
def customer_with_invalid_values(draw):
    """Strategy generating customer data with at least one invalid value (null/empty/>1024 chars)."""
    names = draw(placeholder_names_strategy(min_size=1, max_size=6))
    body = template_body_with_placeholders(names)

    # Full mappings
    mappings = {name: f"field_{name}" for name in names}

    # Generate customer dict - at least one field must be invalid
    customer = {}
    invalid_type = draw(st.sampled_from(["null", "empty", "whitespace", "oversized"]))

    # Pick which placeholder will be invalid
    invalid_index = draw(st.integers(min_value=0, max_value=len(names) - 1))

    for i, name in enumerate(names):
        field_name = f"field_{name}"
        if i == invalid_index:
            if invalid_type == "null":
                customer[field_name] = None
            elif invalid_type == "empty":
                customer[field_name] = ""
            elif invalid_type == "whitespace":
                customer[field_name] = "   "
            elif invalid_type == "oversized":
                oversized_length = draw(st.integers(
                    min_value=MAX_PARAM_LENGTH + 1,
                    max_value=MAX_PARAM_LENGTH + 100
                ))
                customer[field_name] = "x" * oversized_length
        else:
            # Valid value
            customer[field_name] = draw(st.text(
                alphabet=string.ascii_letters + string.digits + " ",
                min_size=1,
                max_size=50
            ).filter(lambda s: s.strip() != ""))

    return body, names, mappings, customer, invalid_type


# --- Property Tests ---

class TestProperty24PlaceholderParsing:
    """Property 24: Template placeholder parsing and mapping validation.

    **Validates: Requirements 20.1, 20.2, 20.5, 20.6**
    """

    def setup_method(self):
        self.validator = TemplateValidator()

    @given(data=template_and_placeholders())
    @settings(max_examples=200)
    def test_parse_identifies_exactly_n_unique_placeholders(self, data):
        """Property: For any template body with N unique placeholders,
        parse_placeholders() identifies exactly N unique placeholder names.

        **Validates: Requirements 20.1**
        """
        template_body, expected_names = data
        result = self.validator.parse_placeholders(template_body)

        # Exactly N unique placeholders should be identified
        found_names = set(p.name for p in result)
        expected_set = set(expected_names)

        assert found_names == expected_set, (
            f"Expected placeholders {expected_set}, but found {found_names}"
        )

    @given(data=template_and_placeholders())
    @settings(max_examples=200)
    def test_parse_returns_correct_count_per_occurrence(self, data):
        """Property: For any template with N unique placeholders (each appearing once),
        parse_placeholders() returns exactly N results.

        **Validates: Requirements 20.1**
        """
        template_body, expected_names = data
        result = self.validator.parse_placeholders(template_body)

        # Each placeholder appears exactly once in our generated templates
        assert len(result) == len(expected_names), (
            f"Expected {len(expected_names)} placeholders, got {len(result)}"
        )

    @given(data=template_with_partial_mappings())
    @settings(max_examples=200)
    def test_fewer_mappings_than_placeholders_blocks_approval(self, data):
        """Property: When fewer than N mappings are provided for N placeholders,
        validate_mappings() returns is_valid=False (blocking approval).

        **Validates: Requirements 20.2**
        """
        template_body, all_names, partial_mappings = data

        result = self.validator.validate_mappings(template_body, partial_mappings)

        assert result.is_valid is False, (
            f"Expected invalid when {len(partial_mappings)} mappings provided "
            f"for {len(all_names)} placeholders"
        )
        assert len(result.unmapped_placeholders) > 0, (
            "Expected unmapped_placeholders to be non-empty"
        )

    @given(data=customer_with_invalid_values())
    @settings(max_examples=200)
    def test_invalid_resolved_values_block_dispatch(self, data):
        """Property: When any resolved customer value is null/empty/>1024 chars,
        validate_customer_params() returns is_valid=False (blocking dispatch).

        **Validates: Requirements 20.5, 20.6**
        """
        template_body, names, mappings, customer, invalid_type = data

        result = self.validator.validate_customer_params(customer, mappings)

        assert result.is_valid is False, (
            f"Expected invalid when customer has {invalid_type} value, "
            f"but got is_valid=True. Customer: {customer}"
        )
        assert len(result.invalid_params) > 0, (
            "Expected invalid_params to be non-empty"
        )

    @given(data=template_and_placeholders())
    @settings(max_examples=100)
    def test_complete_mappings_with_valid_values_passes(self, data):
        """Property: When all N placeholders have valid mappings and all customer
        values are non-null, non-empty, and ≤1024 chars, both validations pass.

        **Validates: Requirements 20.1, 20.2, 20.5, 20.6**
        """
        template_body, names = data

        # Complete mappings
        mappings = {name: f"field_{name}" for name in names}

        # Valid customer data for all fields
        customer = {f"field_{name}": f"value_for_{name}" for name in names}

        # Mapping validation should pass
        mapping_result = self.validator.validate_mappings(template_body, mappings)
        assert mapping_result.is_valid is True, (
            f"Expected valid mappings but got errors: {mapping_result.errors}"
        )

        # Customer param validation should pass
        param_result = self.validator.validate_customer_params(customer, mappings)
        assert param_result.is_valid is True, (
            f"Expected valid params but got invalid: {param_result.invalid_params}"
        )
