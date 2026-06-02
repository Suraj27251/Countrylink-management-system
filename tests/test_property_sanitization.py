"""
Property-based tests for TemplateValidator.sanitize_param() using Hypothesis.

**Validates: Requirements 11.6**

Property 17: Template parameter sanitization
- Sanitization removes control characters while preserving printable Unicode
  (Hindi/Marathi), and output length ≤ input length.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hypothesis import given, settings
from hypothesis import strategies as st

from services.template_validator import TemplateValidator

# Devanagari Unicode block: U+0900–U+097F (Hindi/Marathi script)
DEVANAGARI_ALPHABET = st.characters(min_codepoint=0x0900, max_codepoint=0x097F)

# Control character ranges that should be stripped
CONTROL_CHARS = (
    st.characters(min_codepoint=0x0000, max_codepoint=0x001F)
    | st.characters(min_codepoint=0x007F, max_codepoint=0x009F)
)

# Mixed alphabet including ASCII, Devanagari, control chars, and general Unicode
MIXED_ALPHABET = st.text(
    alphabet=st.characters(min_codepoint=0x0000, max_codepoint=0xFFFF),
)

validator = TemplateValidator()


class TestSanitizationProperties:
    """Property-based tests for sanitize_param() — Property 17."""

    @given(text=st.text(alphabet=st.characters(min_codepoint=0x0000, max_codepoint=0xFFFF)))
    @settings(max_examples=500)
    def test_output_length_leq_input_length(self, text: str):
        """
        Property: For any string input, sanitize_param() output length ≤ input length.

        Sanitization only removes characters, never adds them.

        **Validates: Requirements 11.6**
        """
        result = validator.sanitize_param(text)
        assert len(result) <= len(text)

    @given(text=st.text(alphabet=st.characters(min_codepoint=0x0000, max_codepoint=0xFFFF)))
    @settings(max_examples=500)
    def test_output_contains_no_control_characters(self, text: str):
        """
        Property: Output contains no characters in U+0000–U+001F or U+007F–U+009F.

        All control characters from both C0 and C1 blocks must be removed.

        **Validates: Requirements 11.6**
        """
        result = validator.sanitize_param(text)
        for char in result:
            code = ord(char)
            assert not (0x0000 <= code <= 0x001F), (
                f"Control character U+{code:04X} found in output"
            )
            assert not (0x007F <= code <= 0x009F), (
                f"Control character U+{code:04X} found in output"
            )

    @given(text=st.text(alphabet=st.characters(min_codepoint=0x0000, max_codepoint=0xFFFF)))
    @settings(max_examples=500)
    def test_printable_characters_preserved(self, text: str):
        """
        Property: All characters NOT in control ranges that exist in input also
        exist in output (characters are preserved).

        Non-control characters must never be removed by sanitization.

        **Validates: Requirements 11.6**
        """
        result = validator.sanitize_param(text)
        # Collect non-control characters from input
        for char in text:
            code = ord(char)
            is_control = (0x0000 <= code <= 0x001F) or (0x007F <= code <= 0x009F)
            if not is_control:
                assert char in result, (
                    f"Printable character '{char}' (U+{code:04X}) was removed"
                )

    @given(text=st.text(alphabet=DEVANAGARI_ALPHABET, min_size=1))
    @settings(max_examples=500)
    def test_devanagari_characters_never_removed(self, text: str):
        """
        Property: Hindi/Marathi characters (Devanagari: U+0900–U+097F) are never
        removed by sanitization.

        The Devanagari block contains no control characters, so all characters
        in a pure Devanagari string must be preserved exactly.

        **Validates: Requirements 11.6**
        """
        result = validator.sanitize_param(text)
        assert result == text, (
            f"Devanagari text was modified: input={text!r}, output={result!r}"
        )

    @given(text=st.text(alphabet=st.characters(min_codepoint=0x0000, max_codepoint=0xFFFF)))
    @settings(max_examples=500)
    def test_idempotent(self, text: str):
        """
        Property: The function is idempotent: sanitize(sanitize(x)) == sanitize(x).

        Once control characters are removed, applying sanitization again should
        produce the same result since there are no more control characters to remove.

        **Validates: Requirements 11.6**
        """
        once = validator.sanitize_param(text)
        twice = validator.sanitize_param(once)
        assert once == twice, (
            f"Not idempotent: sanitize(x)={once!r}, sanitize(sanitize(x))={twice!r}"
        )
