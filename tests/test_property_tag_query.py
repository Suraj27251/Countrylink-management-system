"""
Property-based tests for tag queryability in segmentation.

Property 14: Tags assigned to customers are queryable in segmentation
- For any customer with assigned tags, the Segmentation_Engine SHALL include
  that customer in results when filtering by any of their assigned tags.
- Customers who do not have the filtered tag SHALL be excluded from results.
- The tags filter uses an EXISTS subquery on the customer_tags table.
- The tags filter supports both ANY mode (match any of the specified tags)
  and ALL mode (must have all specified tags).

**Validates: Requirements 7.5**

Testing framework: Hypothesis (Python)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from services.segmentation import SegmentationService, TAG_FIELD


# ---------------------------------------------------------------------------
# Strategies for generating tag filter inputs
# ---------------------------------------------------------------------------

# Strategy for a valid tag name (non-empty alphanumeric + underscores/hyphens)
tag_name_st = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-",
    min_size=1,
    max_size=30,
)

# Strategy for a list of unique tag names (1 to 5 tags)
tag_list_st = st.lists(
    tag_name_st,
    min_size=1,
    max_size=5,
    unique=True,
)

# Strategy for tag filter mode
tag_mode_st = st.sampled_from(["ANY", "ALL"])


# ---------------------------------------------------------------------------
# Helper: create a SegmentationService (no DB needed for build_query)
# ---------------------------------------------------------------------------
def make_service():
    """Create a SegmentationService with a dummy connection factory."""
    return SegmentationService(lambda: None)


# ---------------------------------------------------------------------------
# Property Tests
# ---------------------------------------------------------------------------
class TestProperty14TagQueryability:
    """Property-based tests for tag queryability in segmentation — Property 14.

    **Validates: Requirements 7.5**
    """

    @given(tags=tag_list_st)
    @settings(max_examples=200)
    def test_tag_filter_list_produces_exists_subquery(self, tags: list):
        """
        Property: For any list of tags used as a filter, build_query produces
        an EXISTS subquery that checks the customer_tags table for matching
        tag_name values. This ensures tagged customers appear in segment
        results when filtering by that tag.

        **Validates: Requirements 7.5**
        """
        service = make_service()
        filters = {TAG_FIELD: tags}

        where_clause, params = service.build_query(filters)

        # The WHERE clause must not be the empty fallback
        assert where_clause != "1=1", (
            f"Tag filter {tags} should produce a real WHERE clause"
        )

        # Must contain an EXISTS subquery referencing customer_tags
        assert "EXISTS" in where_clause, (
            f"Tag filter should produce EXISTS subquery, got: {where_clause}"
        )
        assert "customer_tags" in where_clause, (
            f"Tag filter should reference customer_tags table, got: {where_clause}"
        )

        # Must join on customer_mobile = r.mobile
        assert "ct.customer_mobile = r.mobile" in where_clause, (
            f"Tag subquery must correlate on customer_mobile = r.mobile, got: {where_clause}"
        )

        # Must filter by tag_name IN (...)
        assert "ct.tag_name IN" in where_clause, (
            f"Tag subquery must filter by tag_name IN (...), got: {where_clause}"
        )

        # All tag names must appear in the params (parameterized, not interpolated)
        for tag in tags:
            assert tag in params, (
                f"Tag '{tag}' should be in params list, got: {params}"
            )

    @given(tags=tag_list_st)
    @settings(max_examples=200)
    def test_tag_filter_any_mode_uses_exists(self, tags: list):
        """
        Property: When tags are provided with ANY mode (default), the query
        uses EXISTS with IN clause. A customer with ANY of the specified tags
        will match the subquery and appear in results.

        **Validates: Requirements 7.5**
        """
        service = make_service()
        filters = {TAG_FIELD: {"values": tags, "mode": "ANY"}}

        where_clause, params = service.build_query(filters)

        # Must use EXISTS for ANY mode
        assert "EXISTS (SELECT 1 FROM customer_tags ct" in where_clause, (
            f"ANY mode should use EXISTS subquery, got: {where_clause}"
        )

        # Must have correct number of placeholders for tags
        placeholders_in_subquery = ", ".join(["%s"] * len(tags))
        expected_in_clause = f"ct.tag_name IN ({placeholders_in_subquery})"
        assert expected_in_clause in where_clause, (
            f"Expected IN clause '{expected_in_clause}' in WHERE, got: {where_clause}"
        )

        # Params should contain all tag names
        for tag in tags:
            assert tag in params, (
                f"Tag '{tag}' missing from params: {params}"
            )

        # Params count should match number of tags (one %s per tag in IN clause)
        assert len(params) == len(tags), (
            f"Expected {len(tags)} params for {len(tags)} tags, got {len(params)}"
        )

    @given(tags=tag_list_st)
    @settings(max_examples=200)
    def test_tag_filter_all_mode_uses_count(self, tags: list):
        """
        Property: When tags are provided with ALL mode, the query uses
        a COUNT(DISTINCT ct.tag_name) subquery that equals the number of
        required tags. A customer must have ALL specified tags to match.

        **Validates: Requirements 7.5**
        """
        service = make_service()
        filters = {TAG_FIELD: {"values": tags, "mode": "ALL"}}

        where_clause, params = service.build_query(filters)

        # Must use COUNT subquery for ALL mode
        assert "COUNT(DISTINCT ct.tag_name)" in where_clause, (
            f"ALL mode should use COUNT(DISTINCT) subquery, got: {where_clause}"
        )
        assert "customer_tags" in where_clause, (
            f"ALL mode subquery should reference customer_tags, got: {where_clause}"
        )

        # Must correlate on customer_mobile = r.mobile
        assert "ct.customer_mobile = r.mobile" in where_clause, (
            f"ALL mode subquery must correlate on customer_mobile, got: {where_clause}"
        )

        # Must compare count to the number of required tags
        # Params should include all tag names + the tag count
        for tag in tags:
            assert tag in params, (
                f"Tag '{tag}' missing from params: {params}"
            )
        assert len(tags) in params, (
            f"Tag count {len(tags)} should be in params for ALL mode comparison, got: {params}"
        )

        # Total params = len(tags) for IN clause + 1 for the count comparison
        assert len(params) == len(tags) + 1, (
            f"Expected {len(tags) + 1} params for ALL mode, got {len(params)}"
        )

    @given(tags=tag_list_st)
    @settings(max_examples=200)
    def test_tag_filter_params_are_parameterized_not_interpolated(self, tags: list):
        """
        Property: Tag names are never interpolated directly into the SQL
        string — they are always passed as parameterized values (%s).
        This prevents SQL injection through tag names.

        **Validates: Requirements 7.5**
        """
        service = make_service()
        filters = {TAG_FIELD: tags}

        where_clause, params = service.build_query(filters)

        # Count %s placeholders — must equal number of params
        placeholder_count = where_clause.count("%s")
        assert placeholder_count == len(params), (
            f"Placeholder count ({placeholder_count}) != params count ({len(params)}). "
            f"Tags might be interpolated into SQL. Clause: {where_clause}"
        )

        # All tag values must be in the params list (proving they are parameterized)
        for tag in tags:
            assert tag in params, (
                f"Tag value '{tag}' not found in params — may not be parameterized. "
                f"Params: {params}"
            )

        # The number of params must equal the number of tags
        # (one placeholder per tag in the IN clause)
        assert len(params) == len(tags), (
            f"Expected {len(tags)} params for {len(tags)} tags, got {len(params)}. "
            f"Extra interpolation may be happening."
        )

    @given(
        customer_tags=tag_list_st,
        extra_tags=tag_list_st,
    )
    @settings(max_examples=200)
    def test_tag_filter_excludes_non_matching_tags(self, customer_tags, extra_tags):
        """
        Property: When filtering by tags that a customer does NOT have,
        the EXISTS subquery will not match. The filter uses tag_name IN (...)
        which only returns rows when at least one tag matches.

        We verify this structurally: a query filtering by tags X will only
        match customers who have a row in customer_tags with tag_name in X.
        If a customer has tags Y (disjoint from X), the subquery returns
        no rows and EXISTS evaluates to false.

        **Validates: Requirements 7.5**
        """
        # Ensure the filter tags are different from customer tags
        filter_tags = [t for t in extra_tags if t not in customer_tags]
        assume(len(filter_tags) > 0)

        service = make_service()
        filters = {TAG_FIELD: filter_tags}

        where_clause, params = service.build_query(filters)

        # The subquery structure ensures only matching tags are found:
        # EXISTS (SELECT 1 FROM customer_tags ct WHERE ct.customer_mobile = r.mobile
        #         AND ct.tag_name IN (%s, %s, ...))
        # For a customer whose tags are disjoint from filter_tags,
        # the IN clause will match no rows, EXISTS returns false.

        # Verify the structural guarantee: IN clause contains only the filter tags
        assert all(tag in params for tag in filter_tags), (
            f"All filter tags must be in params. Filter tags: {filter_tags}, params: {params}"
        )

        # Verify no customer_tags values leak into the query
        for tag in customer_tags:
            if tag not in filter_tags:
                assert tag not in params, (
                    f"Customer tag '{tag}' should not be in params when not in filter"
                )

    @given(tag=tag_name_st)
    @settings(max_examples=200)
    def test_single_tag_as_scalar_produces_valid_query(self, tag: str):
        """
        Property: When a single tag is passed as a scalar string (not a list),
        build_query wraps it into a list and still produces a valid EXISTS
        subquery. This ensures tag queryability works regardless of input format.

        **Validates: Requirements 7.5**
        """
        service = make_service()
        filters = {TAG_FIELD: tag}

        where_clause, params = service.build_query(filters)

        # Must produce a valid query (not fallback)
        assert where_clause != "1=1", (
            f"Single tag '{tag}' should produce a real WHERE clause"
        )

        # Must still reference customer_tags with EXISTS
        assert "EXISTS" in where_clause or "COUNT" in where_clause, (
            f"Single tag should produce a subquery, got: {where_clause}"
        )

        # The tag must be in params
        assert tag in params, (
            f"Single tag '{tag}' should be in params: {params}"
        )

    @given(tags=tag_list_st, mode=tag_mode_st)
    @settings(max_examples=200)
    def test_empty_tag_list_returns_fallback(self, tags: list, mode: str):
        """
        Property: When an empty tag list is provided (edge case), build_query
        should return the fallback clause "1=1" since there's nothing to filter.

        **Validates: Requirements 7.5**
        """
        service = make_service()

        # Test with empty list in dict format
        filters = {TAG_FIELD: {"values": [], "mode": mode}}
        where_clause, params = service.build_query(filters)

        assert where_clause == "1=1", (
            f"Empty tag list should produce fallback '1=1', got: {where_clause}"
        )
        assert params == [], (
            f"Empty tag list should produce empty params, got: {params}"
        )
