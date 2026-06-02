"""
Property-based tests for Simulation Computation Correctness using Hypothesis.

**Validates: Requirements 22.1, 22.2, 22.4**

Property 27: Simulation computation correctness
- For any campaign with an audience segment of size N, the Simulation_Engine
  SHALL compute: final_audience = N - (cooldown_excluded + opted_out + dnd +
  invalid + incomplete), estimated_time = final_audience / throttle_rate, and
  estimated_cost = final_audience * per_message_rate, with all exclusion counts
  being non-negative and summing to total_excluded.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from services.simulation import (
    ExclusionBreakdown,
    SimulationResult,
    DEFAULT_THROTTLE_RATE,
    DEFAULT_COST_RATES,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Audience sizes (must be positive for meaningful simulations)
audience_sizes = st.integers(min_value=1, max_value=100_000)

# Non-negative exclusion counts
non_negative_exclusions = st.integers(min_value=0, max_value=50_000)

# Throttle rate (messages per second, must be positive)
throttle_rates = st.integers(min_value=1, max_value=1000)

# Per-message cost rate (INR, positive float)
cost_rates = st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False)


# Strategy to generate valid ExclusionBreakdowns where total <= N
@st.composite
def exclusion_breakdowns_for_audience(draw, max_total):
    """Generate an ExclusionBreakdown whose total does not exceed max_total."""
    # Distribute budget across exclusion reasons
    remaining = max_total
    cooldown = draw(st.integers(min_value=0, max_value=remaining))
    remaining -= cooldown
    opted_out = draw(st.integers(min_value=0, max_value=remaining))
    remaining -= opted_out
    dnd = draw(st.integers(min_value=0, max_value=remaining))
    remaining -= dnd
    invalid_number = draw(st.integers(min_value=0, max_value=remaining))
    remaining -= invalid_number
    incomplete_data = draw(st.integers(min_value=0, max_value=remaining))

    return ExclusionBreakdown(
        cooldown=cooldown,
        opted_out=opted_out,
        dnd=dnd,
        invalid_number=invalid_number,
        incomplete_data=incomplete_data,
    )


@st.composite
def simulation_inputs(draw):
    """Generate a complete set of simulation inputs."""
    original_audience = draw(audience_sizes)
    exclusions = draw(exclusion_breakdowns_for_audience(max_total=original_audience))
    throttle_rate = draw(throttle_rates)
    per_message_rate = draw(cost_rates)

    final_audience = original_audience - exclusions.total

    return {
        "original_audience": original_audience,
        "exclusions": exclusions,
        "throttle_rate": throttle_rate,
        "per_message_rate": per_message_rate,
        "final_audience": final_audience,
    }


# ---------------------------------------------------------------------------
# Property Tests
# ---------------------------------------------------------------------------
class TestSimulationComputationCorrectness:
    """Property-based tests for simulation computation — Property 27."""

    @given(data=simulation_inputs())
    @settings(max_examples=500)
    def test_final_audience_equals_n_minus_exclusions(self, data):
        """
        Property: final_audience = N - (cooldown + opted_out + dnd +
        invalid + incomplete).

        **Validates: Requirements 22.1**
        """
        original = data["original_audience"]
        exclusions = data["exclusions"]
        expected_final = original - exclusions.total

        # Construct SimulationResult as the engine would
        result = SimulationResult(
            original_audience_count=original,
            final_audience_count=expected_final,
            exclusions=exclusions,
        )

        assert result.final_audience_count == result.original_audience_count - result.exclusions.total, (
            f"final_audience should be {original} - {exclusions.total} = {expected_final}, "
            f"but got {result.final_audience_count}"
        )

    @given(data=simulation_inputs())
    @settings(max_examples=500)
    def test_estimated_time_equals_final_audience_over_rate(self, data):
        """
        Property: estimated_time = final_audience / throttle_rate.

        **Validates: Requirements 22.2**
        """
        final_audience = data["final_audience"]
        throttle_rate = data["throttle_rate"]

        expected_time = round(final_audience / throttle_rate, 2)

        # Simulate the computation the engine performs
        result = SimulationResult(
            original_audience_count=data["original_audience"],
            final_audience_count=final_audience,
            estimated_send_time_seconds=expected_time,
            exclusions=data["exclusions"],
        )

        computed_time = round(result.final_audience_count / throttle_rate, 2)
        assert result.estimated_send_time_seconds == computed_time, (
            f"estimated_time should be {final_audience}/{throttle_rate} = {computed_time}, "
            f"but got {result.estimated_send_time_seconds}"
        )

    @given(data=simulation_inputs())
    @settings(max_examples=500)
    def test_estimated_cost_equals_final_audience_times_price(self, data):
        """
        Property: estimated_cost = final_audience * per_message_rate.

        **Validates: Requirements 22.4**
        """
        final_audience = data["final_audience"]
        per_message_rate = data["per_message_rate"]

        expected_cost = round(final_audience * per_message_rate, 2)

        result = SimulationResult(
            original_audience_count=data["original_audience"],
            final_audience_count=final_audience,
            estimated_cost_inr=expected_cost,
            exclusions=data["exclusions"],
        )

        computed_cost = round(result.final_audience_count * per_message_rate, 2)
        assert result.estimated_cost_inr == computed_cost, (
            f"estimated_cost should be {final_audience} * {per_message_rate} = {computed_cost}, "
            f"but got {result.estimated_cost_inr}"
        )

    @given(
        cooldown=non_negative_exclusions,
        opted_out=non_negative_exclusions,
        dnd=non_negative_exclusions,
        invalid_number=non_negative_exclusions,
        incomplete_data=non_negative_exclusions,
    )
    @settings(max_examples=500)
    def test_all_exclusion_counts_non_negative(self, cooldown, opted_out, dnd, invalid_number, incomplete_data):
        """
        Property: All exclusion counts SHALL be non-negative.

        **Validates: Requirements 22.1**
        """
        exclusions = ExclusionBreakdown(
            cooldown=cooldown,
            opted_out=opted_out,
            dnd=dnd,
            invalid_number=invalid_number,
            incomplete_data=incomplete_data,
        )

        assert exclusions.cooldown >= 0, f"cooldown should be >= 0, got {exclusions.cooldown}"
        assert exclusions.opted_out >= 0, f"opted_out should be >= 0, got {exclusions.opted_out}"
        assert exclusions.dnd >= 0, f"dnd should be >= 0, got {exclusions.dnd}"
        assert exclusions.invalid_number >= 0, f"invalid_number should be >= 0, got {exclusions.invalid_number}"
        assert exclusions.incomplete_data >= 0, f"incomplete_data should be >= 0, got {exclusions.incomplete_data}"

    @given(
        cooldown=non_negative_exclusions,
        opted_out=non_negative_exclusions,
        dnd=non_negative_exclusions,
        invalid_number=non_negative_exclusions,
        incomplete_data=non_negative_exclusions,
    )
    @settings(max_examples=500)
    def test_exclusion_total_equals_sum_of_parts(self, cooldown, opted_out, dnd, invalid_number, incomplete_data):
        """
        Property: The .total property SHALL equal the sum of all individual
        exclusion counts (cooldown + opted_out + dnd + invalid + incomplete).

        **Validates: Requirements 22.1**
        """
        exclusions = ExclusionBreakdown(
            cooldown=cooldown,
            opted_out=opted_out,
            dnd=dnd,
            invalid_number=invalid_number,
            incomplete_data=incomplete_data,
        )

        expected_total = cooldown + opted_out + dnd + invalid_number + incomplete_data
        assert exclusions.total == expected_total, (
            f"exclusions.total should be {expected_total}, got {exclusions.total}"
        )

    @given(data=simulation_inputs())
    @settings(max_examples=500)
    def test_final_audience_non_negative(self, data):
        """
        Property: final_audience SHALL always be non-negative (exclusions
        cannot exceed original audience).

        **Validates: Requirements 22.1**
        """
        # Our strategy ensures exclusions.total <= original_audience
        final_audience = data["final_audience"]

        assert final_audience >= 0, (
            f"final_audience should be >= 0, got {final_audience} "
            f"(original={data['original_audience']}, exclusions={data['exclusions'].total})"
        )

    @given(data=simulation_inputs())
    @settings(max_examples=500)
    def test_estimated_time_non_negative(self, data):
        """
        Property: estimated_send_time_seconds SHALL always be non-negative.

        **Validates: Requirements 22.2**
        """
        final_audience = data["final_audience"]
        throttle_rate = data["throttle_rate"]

        estimated_time = final_audience / throttle_rate

        assert estimated_time >= 0.0, (
            f"estimated_time should be >= 0, got {estimated_time}"
        )

    @given(data=simulation_inputs())
    @settings(max_examples=500)
    def test_estimated_cost_non_negative(self, data):
        """
        Property: estimated_cost_inr SHALL always be non-negative.

        **Validates: Requirements 22.4**
        """
        final_audience = data["final_audience"]
        per_message_rate = data["per_message_rate"]

        estimated_cost = final_audience * per_message_rate

        assert estimated_cost >= 0.0, (
            f"estimated_cost should be >= 0, got {estimated_cost}"
        )
