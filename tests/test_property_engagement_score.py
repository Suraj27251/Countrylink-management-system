"""
Property-based tests for Engagement Score Bounded Computation using Hypothesis.

**Validates: Requirements 23.2**

Property 29: Engagement score bounded computation
- For any customer engagement metrics (read_rate, response_rate, recency_score,
  frequency_score all in [0, 100]), the interaction_score SHALL equal
  round(0.4 * read_rate + 0.3 * response_rate + 0.2 * recency_score
  + 0.1 * frequency_score), bounded to [0, 100].
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hypothesis import given, settings
from hypothesis import strategies as st

from services.engagement_scorer import compute_interaction_score


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Valid input rates in [0, 100] (integers and floats)
valid_rate = st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)

# Boundary values for focused edge-case testing
boundary_rate = st.sampled_from([0.0, 50.0, 100.0])


# ---------------------------------------------------------------------------
# Property Tests
# ---------------------------------------------------------------------------
class TestEngagementScoreBounds:
    """Property-based tests for engagement score bounds — Property 29."""

    @given(
        read_rate=valid_rate,
        response_rate=valid_rate,
        recency_score=valid_rate,
        frequency_score=valid_rate,
    )
    @settings(max_examples=1000)
    def test_score_bounded_zero_to_hundred(
        self, read_rate, response_rate, recency_score, frequency_score
    ):
        """
        Property: The interaction_score SHALL always be bounded to [0, 100]
        for all valid inputs in [0, 100].

        **Validates: Requirements 23.2**
        """
        score = compute_interaction_score(
            read_rate, response_rate, recency_score, frequency_score
        )

        assert 0 <= score <= 100, (
            f"interaction_score {score} out of [0, 100] bounds "
            f"(read_rate={read_rate}, response_rate={response_rate}, "
            f"recency_score={recency_score}, frequency_score={frequency_score})"
        )

    @given(
        read_rate=valid_rate,
        response_rate=valid_rate,
        recency_score=valid_rate,
        frequency_score=valid_rate,
    )
    @settings(max_examples=1000)
    def test_score_equals_weighted_formula(
        self, read_rate, response_rate, recency_score, frequency_score
    ):
        """
        Property: The interaction_score SHALL equal
        round(0.4 * read_rate + 0.3 * response_rate + 0.2 * recency_score
              + 0.1 * frequency_score), clamped to [0, 100].

        **Validates: Requirements 23.2**
        """
        score = compute_interaction_score(
            read_rate, response_rate, recency_score, frequency_score
        )

        raw = (
            0.4 * read_rate
            + 0.3 * response_rate
            + 0.2 * recency_score
            + 0.1 * frequency_score
        )
        expected = max(0, min(100, round(raw)))

        assert score == expected, (
            f"interaction_score {score} != expected {expected} "
            f"(raw={raw}, read_rate={read_rate}, response_rate={response_rate}, "
            f"recency_score={recency_score}, frequency_score={frequency_score})"
        )

    @given(
        read_rate=valid_rate,
        response_rate=valid_rate,
        recency_score=valid_rate,
        frequency_score=valid_rate,
    )
    @settings(max_examples=500)
    def test_score_is_integer(
        self, read_rate, response_rate, recency_score, frequency_score
    ):
        """
        Property: The interaction_score SHALL always be an integer value.

        **Validates: Requirements 23.2**
        """
        score = compute_interaction_score(
            read_rate, response_rate, recency_score, frequency_score
        )

        assert isinstance(score, int), (
            f"interaction_score should be int, got {type(score).__name__}: {score}"
        )

    @given(
        read_rate=boundary_rate,
        response_rate=boundary_rate,
        recency_score=boundary_rate,
        frequency_score=boundary_rate,
    )
    @settings(max_examples=200)
    def test_boundary_inputs_produce_valid_score(
        self, read_rate, response_rate, recency_score, frequency_score
    ):
        """
        Property: At boundary values (0, 50, 100), the score SHALL remain
        in [0, 100] and match the weighted formula.

        **Validates: Requirements 23.2**
        """
        score = compute_interaction_score(
            read_rate, response_rate, recency_score, frequency_score
        )

        assert 0 <= score <= 100, (
            f"Boundary score {score} out of bounds "
            f"(read_rate={read_rate}, response_rate={response_rate}, "
            f"recency_score={recency_score}, frequency_score={frequency_score})"
        )

        raw = (
            0.4 * read_rate
            + 0.3 * response_rate
            + 0.2 * recency_score
            + 0.1 * frequency_score
        )
        expected = max(0, min(100, round(raw)))
        assert score == expected

    def test_all_zeros_returns_zero(self):
        """Minimum inputs should produce minimum score."""
        score = compute_interaction_score(0.0, 0.0, 0.0, 0.0)
        assert score == 0

    def test_all_max_returns_hundred(self):
        """Maximum inputs should produce maximum score."""
        score = compute_interaction_score(100.0, 100.0, 100.0, 100.0)
        assert score == 100

    def test_weights_sum_to_one(self):
        """
        When all inputs are equal, the score should equal that value
        (since 0.4 + 0.3 + 0.2 + 0.1 = 1.0).
        """
        score = compute_interaction_score(60.0, 60.0, 60.0, 60.0)
        assert score == 60
