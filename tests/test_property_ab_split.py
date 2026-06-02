"""
Property-based tests for A/B test even audience split.

Property 18: A/B test even audience split
- For any audience of size N and variant count V (2 ≤ V ≤ 4), the A/B split
  SHALL assign each variant either floor(N/V) or ceil(N/V) recipients, ensuring
  the difference between any two variants is at most 1 recipient.

**Validates: Requirements 13.2**

Testing framework: Hypothesis (Python)
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from blueprints.campaign_bp import CampaignService


# ---------------------------------------------------------------------------
# Hypothesis Strategies
# ---------------------------------------------------------------------------

# Valid audience sizes (non-negative integers)
audience_size_strategy = st.integers(min_value=0, max_value=100_000)

# Valid variant counts (2-4 as per design constraints)
variant_count_strategy = st.integers(min_value=2, max_value=4)


# ---------------------------------------------------------------------------
# Property Tests
# ---------------------------------------------------------------------------

class TestProperty18ABSplit:
    """Property 18: A/B test even audience split.

    **Validates: Requirements 13.2**
    """

    @given(
        audience_size=audience_size_strategy,
        variant_count=variant_count_strategy,
    )
    @settings(max_examples=500, deadline=None)
    def test_each_variant_gets_floor_or_ceil(self, audience_size, variant_count):
        """Property: For audience N and V variants, each variant gets exactly
        floor(N/V) or ceil(N/V) recipients.

        **Validates: Requirements 13.2**
        """
        result = CampaignService.compute_ab_split(audience_size, variant_count)

        floor_val = audience_size // variant_count
        ceil_val = math.ceil(audience_size / variant_count)

        for i, count in enumerate(result):
            assert count == floor_val or count == ceil_val, (
                f"Variant {i} got {count} recipients, but expected "
                f"floor({audience_size}/{variant_count})={floor_val} or "
                f"ceil({audience_size}/{variant_count})={ceil_val}"
            )

    @given(
        audience_size=audience_size_strategy,
        variant_count=variant_count_strategy,
    )
    @settings(max_examples=500, deadline=None)
    def test_sum_equals_audience_size(self, audience_size, variant_count):
        """Property: The total recipients assigned across all variants equals
        the original audience size (no recipients lost or gained).

        **Validates: Requirements 13.2**
        """
        result = CampaignService.compute_ab_split(audience_size, variant_count)

        assert sum(result) == audience_size, (
            f"Sum of split {result} is {sum(result)}, "
            f"expected {audience_size}"
        )

    @given(
        audience_size=audience_size_strategy,
        variant_count=variant_count_strategy,
    )
    @settings(max_examples=500, deadline=None)
    def test_max_difference_between_variants_is_one(self, audience_size, variant_count):
        """Property: The maximum difference between any two variant recipient
        counts is at most 1.

        **Validates: Requirements 13.2**
        """
        result = CampaignService.compute_ab_split(audience_size, variant_count)

        max_count = max(result)
        min_count = min(result)

        assert max_count - min_count <= 1, (
            f"Max difference is {max_count - min_count} (max={max_count}, "
            f"min={min_count}), but must be at most 1. "
            f"Split: {result} for N={audience_size}, V={variant_count}"
        )

    @given(
        audience_size=audience_size_strategy,
        variant_count=variant_count_strategy,
    )
    @settings(max_examples=500, deadline=None)
    def test_variant_count_matches_output_length(self, audience_size, variant_count):
        """Property: The result list length equals the requested variant count.

        **Validates: Requirements 13.2**
        """
        result = CampaignService.compute_ab_split(audience_size, variant_count)

        assert len(result) == variant_count, (
            f"Expected {variant_count} variants in result, got {len(result)}"
        )
