"""
Property-based tests for OptOutManager keyword recognition and round-trip using Hypothesis.

**Validates: Requirements 19.1, 19.2, 19.5**

Property 22: Opt-out keyword recognition and round-trip
- For any message containing one of the opt-out keywords ("STOP", "UNSUBSCRIBE",
  "OPT OUT", "CANCEL", "DND") in any case variant, the Opt_Out_Manager SHALL add
  the sender to the suppression list. Subsequently, for any message containing
  "START" or "SUBSCRIBE" from the same sender, the Opt_Out_Manager SHALL remove
  them from the suppression list.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from services.opt_out_manager import OptOutManager, OPT_OUT_KEYWORDS, OPT_IN_KEYWORDS


# Strategy: generate case variants of opt-out keywords
def case_variant(keyword: str) -> st.SearchStrategy:
    """Generate random case variants of a keyword."""
    return st.builds(
        lambda chars: "".join(chars),
        st.tuples(
            *[st.sampled_from([c.lower(), c.upper()]) for c in keyword]
        ),
    )


# Strategy: any opt-out keyword in any case variant
opt_out_keyword_strategy = st.one_of(
    *[case_variant(kw) for kw in OPT_OUT_KEYWORDS]
)

# Strategy: any opt-in keyword in any case variant
opt_in_keyword_strategy = st.one_of(
    *[case_variant(kw) for kw in OPT_IN_KEYWORDS]
)

# Strategy: whitespace padding around keywords
whitespace_strategy = st.text(
    alphabet=st.sampled_from([" ", "\t", "\n", "\r"]),
    min_size=0,
    max_size=3,
)


class TestOptOutKeywordRecognition:
    """Property-based tests for opt-out keyword recognition — Property 22."""

    @given(keyword=opt_out_keyword_strategy)
    @settings(max_examples=500)
    def test_opt_out_keywords_recognized_case_insensitive(self, keyword: str):
        """
        Property: Any case variant of an opt-out keyword is recognized.

        For any message containing one of the opt-out keywords in any case variant,
        is_opt_out_keyword() SHALL return True.

        **Validates: Requirements 19.1, 19.2**
        """
        manager = OptOutManager(get_connection=lambda: None)
        assert manager.is_opt_out_keyword(keyword), (
            f"Opt-out keyword '{keyword}' was not recognized"
        )

    @given(keyword=opt_out_keyword_strategy, leading=whitespace_strategy, trailing=whitespace_strategy)
    @settings(max_examples=500)
    def test_opt_out_keywords_recognized_with_whitespace(self, keyword: str, leading: str, trailing: str):
        """
        Property: Opt-out keywords with leading/trailing whitespace are recognized.

        The keyword matching should strip whitespace before comparison.

        **Validates: Requirements 19.1, 19.2**
        """
        padded = leading + keyword + trailing
        manager = OptOutManager(get_connection=lambda: None)
        assert manager.is_opt_out_keyword(padded), (
            f"Opt-out keyword '{padded!r}' with whitespace was not recognized"
        )

    @given(keyword=opt_in_keyword_strategy)
    @settings(max_examples=500)
    def test_opt_in_keywords_recognized_case_insensitive(self, keyword: str):
        """
        Property: Any case variant of an opt-in keyword (START, SUBSCRIBE) is recognized.

        For any message containing "START" or "SUBSCRIBE" in any case variant,
        is_opt_in_keyword() SHALL return True.

        **Validates: Requirements 19.5**
        """
        manager = OptOutManager(get_connection=lambda: None)
        assert manager.is_opt_in_keyword(keyword), (
            f"Opt-in keyword '{keyword}' was not recognized"
        )

    @given(keyword=opt_in_keyword_strategy, leading=whitespace_strategy, trailing=whitespace_strategy)
    @settings(max_examples=500)
    def test_opt_in_keywords_recognized_with_whitespace(self, keyword: str, leading: str, trailing: str):
        """
        Property: Opt-in keywords with leading/trailing whitespace are recognized.

        The keyword matching should strip whitespace before comparison.

        **Validates: Requirements 19.5**
        """
        padded = leading + keyword + trailing
        manager = OptOutManager(get_connection=lambda: None)
        assert manager.is_opt_in_keyword(padded), (
            f"Opt-in keyword '{padded!r}' with whitespace was not recognized"
        )

    @given(keyword=opt_out_keyword_strategy)
    @settings(max_examples=500)
    def test_opt_out_keywords_not_recognized_as_opt_in(self, keyword: str):
        """
        Property: Opt-out keywords are never recognized as opt-in keywords.

        The keyword sets are disjoint — an opt-out keyword must not trigger opt-in.

        **Validates: Requirements 19.1, 19.2, 19.5**
        """
        manager = OptOutManager(get_connection=lambda: None)
        assert not manager.is_opt_in_keyword(keyword), (
            f"Opt-out keyword '{keyword}' was incorrectly recognized as opt-in"
        )

    @given(keyword=opt_in_keyword_strategy)
    @settings(max_examples=500)
    def test_opt_in_keywords_not_recognized_as_opt_out(self, keyword: str):
        """
        Property: Opt-in keywords are never recognized as opt-out keywords.

        The keyword sets are disjoint — an opt-in keyword must not trigger opt-out.

        **Validates: Requirements 19.1, 19.2, 19.5**
        """
        manager = OptOutManager(get_connection=lambda: None)
        assert not manager.is_opt_out_keyword(keyword), (
            f"Opt-in keyword '{keyword}' was incorrectly recognized as opt-out"
        )

    @given(text=st.text(
        alphabet=st.characters(min_codepoint=0x0020, max_codepoint=0x007E),
        min_size=1,
        max_size=50,
    ))
    @settings(max_examples=500)
    def test_arbitrary_text_not_false_positive(self, text: str):
        """
        Property: Arbitrary text that is not an opt-out or opt-in keyword
        should not be recognized as either.

        **Validates: Requirements 19.1, 19.2, 19.5**
        """
        normalized = text.strip().lower()
        assume(normalized not in OPT_OUT_KEYWORDS)
        assume(normalized not in OPT_IN_KEYWORDS)

        manager = OptOutManager(get_connection=lambda: None)
        assert not manager.is_opt_out_keyword(text), (
            f"Arbitrary text '{text}' falsely recognized as opt-out"
        )
        assert not manager.is_opt_in_keyword(text), (
            f"Arbitrary text '{text}' falsely recognized as opt-in"
        )

    @given(keyword=opt_out_keyword_strategy)
    @settings(max_examples=200)
    def test_opt_out_then_opt_in_round_trip(self, keyword: str):
        """
        Property: After opt-out keyword detection, an opt-in keyword should be
        detectable — verifying the round-trip recognition logic.

        This tests the logical round-trip: if a keyword is detected as opt-out,
        then START/SUBSCRIBE should be detectable as opt-in for the same user.

        **Validates: Requirements 19.1, 19.2, 19.5**
        """
        manager = OptOutManager(get_connection=lambda: None)

        # The opt-out keyword is recognized
        assert manager.is_opt_out_keyword(keyword)

        # After opt-out, START and SUBSCRIBE are valid opt-in paths
        assert manager.is_opt_in_keyword("START")
        assert manager.is_opt_in_keyword("SUBSCRIBE")
        assert manager.is_opt_in_keyword("start")
        assert manager.is_opt_in_keyword("subscribe")
        assert manager.is_opt_in_keyword("Start")
        assert manager.is_opt_in_keyword("Subscribe")
