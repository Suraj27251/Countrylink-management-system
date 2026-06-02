"""
Property-based tests for Segmentation AND-filter correctness using Hypothesis.

**Validates: Requirements 3.1, 3.2**

Property 6: Segmentation AND-filter returns only matching customers
- For any combination of exact-match filters, build_query produces a WHERE clause
  with all filter fields joined by AND.
- Each filter criterion produces its own condition — if 3 filters given, 3 conditions
  in the WHERE clause.
- All parameter values appear in the params list (no SQL injection via string
  interpolation).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from services.segmentation import SegmentationService, EXACT_MATCH_FIELDS


# ---------------------------------------------------------------------------
# Strategies for generating random filter combinations
# ---------------------------------------------------------------------------

# All supported exact-match fields as a sorted list for deterministic strategies
EXACT_FIELDS_LIST = sorted(EXACT_MATCH_FIELDS)

# Strategy for generating a non-empty filter value (avoiding None and empty string)
filter_value_st = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "S", "Z"),
        min_codepoint=32,
        max_codepoint=0xFFFF,
    ),
    min_size=1,
    max_size=50,
)

# Strategy for generating a random subset of exact-match fields (1 to all fields)
exact_field_subset_st = st.lists(
    st.sampled_from(EXACT_FIELDS_LIST),
    min_size=1,
    max_size=len(EXACT_FIELDS_LIST),
    unique=True,
)

# Strategy for generating a dict of exact-match filters with random values
exact_filters_st = exact_field_subset_st.flatmap(
    lambda fields: st.fixed_dictionaries(
        {field: filter_value_st for field in fields}
    )
)


# ---------------------------------------------------------------------------
# Helper: create a SegmentationService (no DB needed for build_query)
# ---------------------------------------------------------------------------
def make_service():
    """Create a SegmentationService with a dummy connection factory."""
    return SegmentationService(lambda: None)


# ---------------------------------------------------------------------------
# Property Tests
# ---------------------------------------------------------------------------
class TestAndFilterCorrectness:
    """Property-based tests for segmentation AND-filter — Property 6."""

    @given(filters=exact_filters_st)
    @settings(max_examples=500)
    def test_all_filter_fields_joined_by_and(self, filters: dict):
        """
        Property: For any combination of exact-match filters, build_query
        produces a WHERE clause with all filter fields joined by AND.

        **Validates: Requirements 3.1, 3.2**
        """
        service = make_service()
        where_clause, params = service.build_query(filters)

        # The WHERE clause must not be the empty fallback
        assert where_clause != "1=1", (
            f"Non-empty filters {list(filters.keys())} should produce a real WHERE clause"
        )

        # Each field should appear as r.<field> = %s in the clause
        for field in filters:
            assert f"r.{field} = %s" in where_clause, (
                f"Field '{field}' missing from WHERE clause: {where_clause}"
            )

        # If more than one filter, AND must join them
        if len(filters) > 1:
            assert " AND " in where_clause, (
                f"Multiple filters should be joined by AND: {where_clause}"
            )

    @given(filters=exact_filters_st)
    @settings(max_examples=500)
    def test_condition_count_matches_filter_count(self, filters: dict):
        """
        Property: Each filter criterion produces its own condition. If N
        exact-match filters are given, exactly N conditions appear in the
        WHERE clause (each separated by AND).

        **Validates: Requirements 3.1, 3.2**
        """
        service = make_service()
        where_clause, params = service.build_query(filters)

        # Count the number of conditions by splitting on " AND "
        conditions = where_clause.split(" AND ")
        num_filters = len(filters)

        assert len(conditions) == num_filters, (
            f"Expected {num_filters} conditions for {num_filters} filters, "
            f"got {len(conditions)}. Clause: {where_clause}"
        )

    @given(filters=exact_filters_st)
    @settings(max_examples=500)
    def test_all_values_in_params_not_interpolated(self, filters: dict):
        """
        Property: All parameter values appear in the params list and are
        NOT interpolated into the SQL string. This prevents SQL injection.

        The key invariant: the WHERE clause uses only %s placeholders for
        values, and the number of placeholders equals the number of params.

        **Validates: Requirements 3.1, 3.2**
        """
        service = make_service()
        where_clause, params = service.build_query(filters)

        # Every filter value must be in the params list
        for field, value in filters.items():
            assert value in params, (
                f"Value for field '{field}' ({value!r}) not found in params: {params}"
            )

        # The number of %s placeholders must equal the number of params
        # This proves values are parameterized, not string-interpolated
        placeholder_count = where_clause.count("%s")
        assert placeholder_count == len(params), (
            f"Placeholder count ({placeholder_count}) != params count ({len(params)}). "
            f"Clause: {where_clause}"
        )

        # The WHERE clause should only contain field references (r.<field>),
        # operators (=, AND), and %s placeholders — never raw user values
        # embedded via f-string or .format(). We verify this by checking
        # that the clause structure is deterministic given only the field names.
        # Reconstruct the expected clause from just the field names.
        expected_parts = [f"r.{field} = %s" for field in filters]
        expected_clause = " AND ".join(expected_parts)
        assert where_clause == expected_clause, (
            f"WHERE clause structure mismatch.\n"
            f"Expected: {expected_clause}\n"
            f"Got:      {where_clause}"
        )
