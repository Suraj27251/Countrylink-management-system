"""Template Validator Service for Enterprise WhatsApp CRM.

Validates WhatsApp message template placeholders against customer data fields,
ensures parameter values meet API constraints, sanitizes content, and renders
template previews.

Requirements: 20.1, 20.2, 20.3, 20.4, 20.5, 20.6, 20.7, 11.6
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# Regex pattern to match both numeric ({{1}}, {{2}}) and named ({{name}}, {{plan_name}}) placeholders
PLACEHOLDER_PATTERN = re.compile(r'\{\{(\w+)\}\}')

# Control character ranges to strip: U+0000–U+001F and U+007F–U+009F
CONTROL_CHAR_PATTERN = re.compile(
    r'[\u0000-\u001f\u007f-\u009f]'
)

# Maximum allowed parameter value length per WhatsApp Business API
MAX_PARAM_LENGTH = 1024


@dataclass
class Placeholder:
    """Represents a single placeholder found in a template body."""
    name: str
    position: int  # Start index in the template body
    is_numeric: bool = False  # True if placeholder is numeric like {{1}}, {{2}}

    @property
    def index(self) -> Optional[int]:
        """Return the numeric index if this is a numeric placeholder."""
        if self.is_numeric:
            return int(self.name)
        return None


@dataclass
class ValidationResult:
    """Result of validating placeholder-to-field mappings."""
    is_valid: bool
    unmapped_placeholders: List[str] = field(default_factory=list)
    extra_mappings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class ParamResult:
    """Result of validating resolved customer parameter values."""
    is_valid: bool
    resolved_params: Dict[str, str] = field(default_factory=dict)
    invalid_params: List[Tuple[str, str]] = field(default_factory=list)
    # Each tuple is (placeholder_name, reason)


class TemplateValidator:
    """Validates WhatsApp message template placeholders and parameters.

    This class is designed to be testable without database dependencies.
    Template data and customer records are accepted as parameters rather
    than queried from the database.
    """

    def parse_placeholders(self, template_body: str) -> List[Placeholder]:
        """Extract all placeholders from a template body.

        Finds both numeric ({{1}}, {{2}}) and named ({{name}}, {{plan_name}})
        style placeholder variables.

        Args:
            template_body: The template text containing placeholders.

        Returns:
            List of Placeholder objects with name, position, and type info.
            Duplicates are included (each occurrence is returned).

        Validates: Requirement 20.1
        """
        placeholders = []
        for match in PLACEHOLDER_PATTERN.finditer(template_body):
            name = match.group(1)
            position = match.start()
            is_numeric = name.isdigit()
            placeholders.append(Placeholder(
                name=name,
                position=position,
                is_numeric=is_numeric
            ))
        return placeholders

    def validate_mappings(
        self,
        template_body: str,
        mappings: Dict[str, str]
    ) -> ValidationResult:
        """Verify all placeholders have corresponding customer field mappings.

        Checks that every unique placeholder identified in the template body
        has a corresponding entry in the mappings dict, and identifies any
        extra mappings that don't correspond to a placeholder.

        Args:
            template_body: The template text containing placeholders.
            mappings: Dict mapping placeholder names to customer field names.
                      e.g. {"1": "customer_name", "2": "plan_name"}

        Returns:
            ValidationResult indicating validity, unmapped placeholders,
            and extra mappings.

        Validates: Requirements 20.2, 20.3, 20.4
        """
        placeholders = self.parse_placeholders(template_body)
        # Get unique placeholder names
        placeholder_names = set(p.name for p in placeholders)
        mapping_keys = set(mappings.keys())

        unmapped = sorted(placeholder_names - mapping_keys)
        extra = sorted(mapping_keys - placeholder_names)

        errors = []
        if unmapped:
            errors.append(
                f"Unmapped placeholders: {', '.join('{{' + n + '}}' for n in unmapped)}"
            )
        if extra:
            errors.append(
                f"Extra mappings without corresponding placeholders: {', '.join(extra)}"
            )

        return ValidationResult(
            is_valid=len(unmapped) == 0,
            unmapped_placeholders=unmapped,
            extra_mappings=extra,
            errors=errors
        )

    def validate_customer_params(
        self,
        customer: dict,
        mappings: Dict[str, str]
    ) -> ParamResult:
        """Verify resolved parameter values are valid for dispatch.

        For a given customer record and placeholder-to-field mappings,
        resolves each placeholder's value from customer data and verifies:
        - Not None
        - Not empty string (after stripping whitespace)
        - Length ≤ 1024 characters

        Args:
            customer: Dict of customer data fields.
                      e.g. {"customer_name": "Rajesh", "plan_name": "Eclipse 100"}
            mappings: Dict mapping placeholder names to customer field names.
                      e.g. {"1": "customer_name", "2": "plan_name"}

        Returns:
            ParamResult with resolved values and any invalid params.

        Validates: Requirements 20.5, 20.6
        """
        resolved_params = {}
        invalid_params = []

        for placeholder_name, field_name in mappings.items():
            value = customer.get(field_name)

            if value is None:
                invalid_params.append((placeholder_name, "value is null"))
                continue

            # Convert to string for validation
            str_value = str(value)

            if str_value.strip() == "":
                invalid_params.append((placeholder_name, "value is empty"))
                continue

            if len(str_value) > MAX_PARAM_LENGTH:
                invalid_params.append(
                    (placeholder_name,
                     f"value exceeds {MAX_PARAM_LENGTH} characters "
                     f"(length: {len(str_value)})")
                )
                continue

            resolved_params[placeholder_name] = str_value

        return ParamResult(
            is_valid=len(invalid_params) == 0,
            resolved_params=resolved_params,
            invalid_params=invalid_params
        )

    def sanitize_param(self, value: str) -> str:
        """Remove control characters while preserving printable Unicode.

        Strips characters in the ranges U+0000–U+001F and U+007F–U+009F
        while preserving all printable Unicode characters including
        Hindi/Marathi scripts.

        The output length is guaranteed to be ≤ the input length.

        Args:
            value: The parameter string value to sanitize.

        Returns:
            Sanitized string with control characters removed.

        Validates: Requirement 11.6
        """
        return CONTROL_CHAR_PATTERN.sub('', value)

    def render_preview(
        self,
        template_body: str,
        customer: dict,
        mappings: Dict[str, str]
    ) -> str:
        """Substitute customer data into template placeholders.

        Replaces all placeholders in the template body with actual customer
        values (sanitized). If a value cannot be resolved, the placeholder
        is left as-is.

        Args:
            template_body: The template text containing placeholders.
            customer: Dict of customer data fields.
            mappings: Dict mapping placeholder names to customer field names.

        Returns:
            Rendered template string with placeholders replaced by
            sanitized customer values.

        Validates: Requirement 20.7
        """
        def replace_placeholder(match: re.Match) -> str:
            placeholder_name = match.group(1)
            field_name = mappings.get(placeholder_name)
            if field_name is None:
                # No mapping for this placeholder, leave it as-is
                return match.group(0)

            value = customer.get(field_name)
            if value is None:
                # Customer doesn't have this field, leave placeholder
                return match.group(0)

            str_value = str(value)
            if str_value.strip() == "":
                return match.group(0)

            return self.sanitize_param(str_value)

        return PLACEHOLDER_PATTERN.sub(replace_placeholder, template_body)
