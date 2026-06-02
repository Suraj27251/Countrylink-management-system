"""
Property-based tests for derived filter computation using Hypothesis.

**Validates: Requirements 3.6, 3.7**

Property 8: Derived filter computation correctness
- "days since last recharge" = (current_date - activation_date).days
  → SQL uses DATEDIFF(%s, r.activation_date) with reference_date as parameter
- "days inactive" = (current_date - expiry_date).days for expired customers
  → SQL uses DATEDIFF(%s, r.expiry_date) AND includes "r.expiry_date < %s" constraint
- For any reference_date, the date is correctly passed as a parameter (not interpolated)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datetime import date

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from services.segmentation import SegmentationService


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------
# Generate reasonable date ranges for min/max days
days_value = st.integers(min_value=0, max_value=3650)

# Generate reference dates within a reasonable range
reference_dates = st.dates(min_value=date(2020, 1, 1), max_value=date(2030, 12, 31))

# Strategy for days_since_last_recharge filter with min and/or max
days_since_recharge_filter = st.fixed_dictionaries({
    "min": st.one_of(st.none(), days_value),
    "max": st.one_of(st.none(), days_value),
}).filter(lambda d: d["min"] is not None or d["max"] is not None)

# Strategy for days_inactive filter with min and/or max
days_inactive_filter = st.fixed_dictionaries({
    "min": st.one_of(st.none(), days_value),
    "max": st.one_of(st.none(), days_value),
}).filter(lambda d: d["min"] is not None or d["max"] is not None)


# ---------------------------------------------------------------------------
# Property Tests
# ---------------------------------------------------------------------------
class TestDerivedFilterComputation:
    """Property-based tests for derived filter SQL generation — Property 8."""

    @given(
        range_filter=days_since_recharge_filter,
        ref_date=reference_dates,
    )
    @settings(max_examples=500)
    def test_days_since_last_recharge_uses_datediff_activation_date(
        self, range_filter: dict, ref_date: date
    ):
        """
        Property: For "days_since_last_recharge" with any min/max range,
        the SQL uses DATEDIFF(%s, r.activation_date) with reference_date
        as first parameter.

        **Validates: Requirements 3.6**
        """
        service = SegmentationService(lambda: None)
        clause, params = service.build_query(
            {"days_since_last_recharge": range_filter},
            reference_date=ref_date,
        )

        # Every DATEDIFF clause must reference r.activation_date
        if range_filter["min"] is not None:
            assert "DATEDIFF(%s, r.activation_date) >= %s" in clause, (
                f"Expected DATEDIFF(%s, r.activation_date) >= %s in clause, got: {clause}"
            )
        if range_filter["max"] is not None:
            assert "DATEDIFF(%s, r.activation_date) <= %s" in clause, (
                f"Expected DATEDIFF(%s, r.activation_date) <= %s in clause, got: {clause}"
            )

        # reference_date must be passed as a parameter (not interpolated into SQL)
        assert ref_date in params, (
            f"reference_date {ref_date} not found in params: {params}"
        )

        # The clause should NOT contain the date string itself
        date_str = ref_date.isoformat()
        assert date_str not in clause, (
            f"reference_date appears interpolated in SQL clause: {clause}"
        )

    @given(
        range_filter=days_inactive_filter,
        ref_date=reference_dates,
    )
    @settings(max_examples=500)
    def test_days_inactive_uses_datediff_expiry_date_with_constraint(
        self, range_filter: dict, ref_date: date
    ):
        """
        Property: For "days_inactive" with any min/max range, the SQL uses
        DATEDIFF(%s, r.expiry_date) AND includes "r.expiry_date < %s" constraint
        to ensure only expired customers are matched.

        **Validates: Requirements 3.7**
        """
        service = SegmentationService(lambda: None)
        clause, params = service.build_query(
            {"days_inactive": range_filter},
            reference_date=ref_date,
        )

        # Every DATEDIFF clause must reference r.expiry_date
        if range_filter["min"] is not None:
            assert "DATEDIFF(%s, r.expiry_date) >= %s" in clause, (
                f"Expected DATEDIFF(%s, r.expiry_date) >= %s in clause, got: {clause}"
            )
        if range_filter["max"] is not None:
            assert "DATEDIFF(%s, r.expiry_date) <= %s" in clause, (
                f"Expected DATEDIFF(%s, r.expiry_date) <= %s in clause, got: {clause}"
            )

        # Must include the expired-records constraint
        assert "r.expiry_date < %s" in clause, (
            f"Expected 'r.expiry_date < %s' constraint in clause, got: {clause}"
        )

        # reference_date must be passed as a parameter (not interpolated)
        assert ref_date in params, (
            f"reference_date {ref_date} not found in params: {params}"
        )

        # The clause should NOT contain the date string itself
        date_str = ref_date.isoformat()
        assert date_str not in clause, (
            f"reference_date appears interpolated in SQL clause: {clause}"
        )

    @given(
        ref_date=reference_dates,
        range_filter=days_since_recharge_filter,
    )
    @settings(max_examples=500)
    def test_days_since_recharge_reference_date_is_parameterized(
        self, ref_date: date, range_filter: dict
    ):
        """
        Property: For any reference_date and days_since_last_recharge filter,
        the date is correctly passed as a parameter (not interpolated into SQL).
        Each DATEDIFF clause contributes one reference_date parameter.

        **Validates: Requirements 3.6**
        """
        service = SegmentationService(lambda: None)
        clause, params = service.build_query(
            {"days_since_last_recharge": range_filter},
            reference_date=ref_date,
        )

        # Count how many DATEDIFF clauses we expect
        expected_datediff_count = 0
        if range_filter["min"] is not None:
            expected_datediff_count += 1
        if range_filter["max"] is not None:
            expected_datediff_count += 1

        # Each DATEDIFF clause should contribute one ref_date param
        ref_date_count = params.count(ref_date)
        assert ref_date_count == expected_datediff_count, (
            f"Expected {expected_datediff_count} reference_date params, "
            f"got {ref_date_count}. Params: {params}"
        )

    @given(
        ref_date=reference_dates,
        range_filter=days_inactive_filter,
    )
    @settings(max_examples=500)
    def test_days_inactive_reference_date_is_parameterized(
        self, ref_date: date, range_filter: dict
    ):
        """
        Property: For any reference_date and days_inactive filter, the date
        is correctly passed as a parameter (not interpolated). Each DATEDIFF
        clause contributes one ref_date param, plus one for the expiry constraint.

        **Validates: Requirements 3.7**
        """
        service = SegmentationService(lambda: None)
        clause, params = service.build_query(
            {"days_inactive": range_filter},
            reference_date=ref_date,
        )

        # Count expected reference_date occurrences in params:
        # - 1 per DATEDIFF clause (min and/or max)
        # - 1 for the "r.expiry_date < %s" constraint
        expected_datediff_count = 0
        if range_filter["min"] is not None:
            expected_datediff_count += 1
        if range_filter["max"] is not None:
            expected_datediff_count += 1
        expected_total = expected_datediff_count + 1  # +1 for expiry constraint

        ref_date_count = params.count(ref_date)
        assert ref_date_count == expected_total, (
            f"Expected {expected_total} reference_date params "
            f"({expected_datediff_count} DATEDIFF + 1 expiry constraint), "
            f"got {ref_date_count}. Params: {params}"
        )
