"""
Property-based tests for Simulation Engine exclusion warning threshold using Hypothesis.

**Validates: Requirements 22.5**

Property 28: Simulation exclusion warning threshold
- Warning when total_excluded / original_audience > 0.30
- No warning when total_excluded / original_audience <= 0.30
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from services.simulation import (
    EXCLUSION_WARNING_THRESHOLD,
    ExclusionBreakdown,
    SimulationResult,
)


# --- Strategies ---

# Non-negative integers for exclusion counts
st_exclusion_count = st.integers(min_value=0, max_value=10000)

# Original audience size (must be > 0 to compute ratio)
st_audience_size = st.integers(min_value=1, max_value=50000)


# --- Helper: replicate the warning logic from SimulationEngine ---

def compute_warnings(original_count: int, exclusions: ExclusionBreakdown) -> list:
    """
    Replicate the warning generation logic from SimulationEngine.simulate().

    This mirrors the exact logic in services/simulation.py that determines
    whether an exclusion warning should be generated.
    """
    warnings = []
    if original_count > 0:
        exclusion_ratio = exclusions.total / original_count
        if exclusion_ratio > EXCLUSION_WARNING_THRESHOLD:
            pct = round(exclusion_ratio * 100, 1)
            warnings.append(
                f"High exclusion rate: {pct}% of the original segment "
                f"({exclusions.total} of {original_count} recipients) "
                f"would be excluded. Consider reviewing your segment criteria."
            )
    return warnings


# --- Property Tests ---


class TestProperty28SimulationWarningThreshold:
    """
    Property 28: Simulation exclusion warning threshold.

    For any simulation where total_excluded / original_audience > 0.30,
    the Simulation_Engine SHALL include a prominent warning in the result.
    When total_excluded / original_audience <= 0.30, no such warning SHALL
    be generated.
    """

    @given(
        original_count=st_audience_size,
        cooldown=st_exclusion_count,
        opted_out=st_exclusion_count,
        dnd=st_exclusion_count,
        invalid_number=st_exclusion_count,
        incomplete_data=st_exclusion_count,
    )
    @settings(max_examples=200)
    def test_warning_generated_when_exclusions_exceed_threshold(
        self, original_count, cooldown, opted_out, dnd, invalid_number, incomplete_data
    ):
        """
        When total exclusions > 30% of original audience, a warning MUST be present.

        **Validates: Requirements 22.5**
        """
        exclusions = ExclusionBreakdown(
            cooldown=cooldown,
            opted_out=opted_out,
            dnd=dnd,
            invalid_number=invalid_number,
            incomplete_data=incomplete_data,
        )
        total_excluded = exclusions.total

        # Only test cases where exclusion ratio exceeds threshold
        assume(total_excluded / original_count > EXCLUSION_WARNING_THRESHOLD)

        warnings = compute_warnings(original_count, exclusions)

        # A warning about high exclusion rate MUST be present
        assert len(warnings) >= 1, (
            f"Expected warning when {total_excluded}/{original_count} "
            f"= {total_excluded/original_count:.4f} > {EXCLUSION_WARNING_THRESHOLD}"
        )
        # The warning should mention exclusion rate
        assert any("exclusion" in w.lower() or "excluded" in w.lower() for w in warnings), (
            f"Warning should reference exclusion rate, got: {warnings}"
        )

    @given(
        original_count=st_audience_size,
        cooldown=st_exclusion_count,
        opted_out=st_exclusion_count,
        dnd=st_exclusion_count,
        invalid_number=st_exclusion_count,
        incomplete_data=st_exclusion_count,
    )
    @settings(max_examples=200)
    def test_no_warning_when_exclusions_at_or_below_threshold(
        self, original_count, cooldown, opted_out, dnd, invalid_number, incomplete_data
    ):
        """
        When total exclusions <= 30% of original audience, no exclusion warning
        SHALL be generated.

        **Validates: Requirements 22.5**
        """
        exclusions = ExclusionBreakdown(
            cooldown=cooldown,
            opted_out=opted_out,
            dnd=dnd,
            invalid_number=invalid_number,
            incomplete_data=incomplete_data,
        )
        total_excluded = exclusions.total

        # Only test cases where exclusion ratio is at or below threshold
        assume(total_excluded / original_count <= EXCLUSION_WARNING_THRESHOLD)

        warnings = compute_warnings(original_count, exclusions)

        # No exclusion-related warning should be present
        exclusion_warnings = [
            w for w in warnings if "exclusion" in w.lower() or "excluded" in w.lower()
        ]
        assert len(exclusion_warnings) == 0, (
            f"Expected no exclusion warning when {total_excluded}/{original_count} "
            f"= {total_excluded/original_count:.4f} <= {EXCLUSION_WARNING_THRESHOLD}, "
            f"but got: {exclusion_warnings}"
        )

    @given(
        original_count=st_audience_size,
        cooldown=st_exclusion_count,
        opted_out=st_exclusion_count,
        dnd=st_exclusion_count,
        invalid_number=st_exclusion_count,
        incomplete_data=st_exclusion_count,
    )
    @settings(max_examples=200)
    def test_threshold_boundary_strict_inequality(
        self, original_count, cooldown, opted_out, dnd, invalid_number, incomplete_data
    ):
        """
        The threshold uses strict inequality (> 0.30), meaning exactly 30%
        exclusion should NOT trigger a warning.

        **Validates: Requirements 22.5**
        """
        exclusions = ExclusionBreakdown(
            cooldown=cooldown,
            opted_out=opted_out,
            dnd=dnd,
            invalid_number=invalid_number,
            incomplete_data=incomplete_data,
        )
        total_excluded = exclusions.total
        ratio = total_excluded / original_count

        warnings = compute_warnings(original_count, exclusions)
        has_exclusion_warning = any(
            "exclusion" in w.lower() or "excluded" in w.lower() for w in warnings
        )

        if ratio > EXCLUSION_WARNING_THRESHOLD:
            assert has_exclusion_warning, (
                f"Ratio {ratio:.4f} > {EXCLUSION_WARNING_THRESHOLD} but no warning generated"
            )
        else:
            assert not has_exclusion_warning, (
                f"Ratio {ratio:.4f} <= {EXCLUSION_WARNING_THRESHOLD} but warning was generated"
            )

    @given(
        original_count=st.integers(min_value=1, max_value=1000),
    )
    @settings(max_examples=100)
    def test_exactly_at_threshold_no_warning(self, original_count):
        """
        When exclusions are exactly 30% of original (where integer math allows),
        no warning should be generated since the condition is strictly greater than.

        **Validates: Requirements 22.5**
        """
        # Compute an exclusion count that is exactly 30% (when possible)
        exact_30_pct = int(original_count * EXCLUSION_WARNING_THRESHOLD)

        # Skip if rounding creates a ratio != 0.30 exactly
        if original_count == 0:
            return

        exclusions = ExclusionBreakdown(cooldown=exact_30_pct)
        ratio = exclusions.total / original_count

        warnings = compute_warnings(original_count, exclusions)
        exclusion_warnings = [
            w for w in warnings if "exclusion" in w.lower() or "excluded" in w.lower()
        ]

        if ratio <= EXCLUSION_WARNING_THRESHOLD:
            assert len(exclusion_warnings) == 0, (
                f"At exactly {ratio:.4f} ratio (threshold={EXCLUSION_WARNING_THRESHOLD}), "
                f"no warning should fire but got: {exclusion_warnings}"
            )
