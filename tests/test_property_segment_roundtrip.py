"""
Property-based tests for segment save/load round-trip.

Property 7: Segment save/load round-trip
- Saving and loading a segment preserves semantically equivalent filter criteria.
- For any valid filter criteria dict, save followed by load returns the same
  filter criteria.
- JSON serialization/deserialization preserves all filter types: exact match
  strings, range dicts, boolean values, and tag lists.

**Validates: Requirements 3.3**

Testing framework: Hypothesis (Python)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from services.segmentation import (
    SegmentationService,
    EXACT_MATCH_FIELDS,
    BOOLEAN_FIELDS,
    RANGE_FIELDS,
    DERIVED_RANGE_FIELDS,
    TAG_FIELD,
)


# ---------------------------------------------------------------------------
# Mock infrastructure that captures stored JSON and returns it on load
# ---------------------------------------------------------------------------

class SegmentStoreCursor:
    """Mock cursor that captures the JSON stored by save_segment and
    returns it when load_segment reads back the row."""

    def __init__(self):
        self.stored_json = None
        self.lastrowid = 1
        self._last_sql = None

    def execute(self, sql, params=None):
        self._last_sql = sql
        if "INSERT INTO audience_segments" in sql and params is not None:
            # params order: (organization_id, name, description, filter_criteria, estimated_count, created_by)
            self.stored_json = params[3]
            self.lastrowid = 1

    def fetchone(self):
        # For COUNT(*) queries (estimate_count), return zero
        if self._last_sql and "COUNT(*)" in self._last_sql:
            return {"cnt": 0}
        # For SELECT * FROM audience_segments (load_segment), return a row
        # with filter_criteria as the stored JSON string
        if self._last_sql and "SELECT * FROM audience_segments" in self._last_sql:
            return {
                "id": 1,
                "organization_id": 1,
                "name": "test_segment",
                "description": "",
                "filter_criteria": self.stored_json,
                "estimated_count": 0,
                "created_by": "system",
            }
        return {"cnt": 0}

    def fetchall(self):
        return []

    def close(self):
        pass


class SegmentStoreConnection:
    """Mock connection used for both save and load operations."""

    def __init__(self, cursor: SegmentStoreCursor):
        self._cursor = cursor

    def cursor(self, dictionary=False):
        return self._cursor

    def start_transaction(self):
        pass

    def commit(self):
        pass

    def rollback(self):
        pass

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Hypothesis Strategies for valid filter criteria
# ---------------------------------------------------------------------------

# Exact match field values (non-empty strings)
exact_match_values = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_- ",
    min_size=1,
    max_size=50,
).filter(lambda s: s.strip() != "")

# Range filter values: dict with optional min/max integer keys
range_values = st.fixed_dictionaries(
    {},
    optional={
        "min": st.integers(min_value=0, max_value=365),
        "max": st.integers(min_value=0, max_value=365),
    },
).filter(lambda d: len(d) > 0)  # At least one of min/max must be present

# Boolean field values
boolean_values = st.booleans()

# Tag filter values: either a list of tags or a dict with values + mode
tag_list = st.lists(
    st.text(
        alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-",
        min_size=1,
        max_size=30,
    ),
    min_size=1,
    max_size=5,
)

tag_dict_values = st.fixed_dictionaries({
    "values": tag_list,
    "mode": st.sampled_from(["ANY", "ALL"]),
})

tag_values = st.one_of(tag_list, tag_dict_values)


@st.composite
def filter_criteria_strategy(draw):
    """Generate random filter criteria dicts combining various field types."""
    filters = {}

    # Optionally add some exact match fields
    exact_fields_to_include = draw(st.lists(
        st.sampled_from(sorted(EXACT_MATCH_FIELDS)),
        min_size=0,
        max_size=4,
        unique=True,
    ))
    for field in exact_fields_to_include:
        filters[field] = draw(exact_match_values)

    # Optionally add boolean fields
    if draw(st.booleans()):
        for field in BOOLEAN_FIELDS:
            filters[field] = draw(boolean_values)

    # Optionally add range fields
    range_fields_to_include = draw(st.lists(
        st.sampled_from(sorted(RANGE_FIELDS)),
        min_size=0,
        max_size=len(RANGE_FIELDS),
        unique=True,
    ))
    for field in range_fields_to_include:
        filters[field] = draw(range_values)

    # Optionally add derived range fields
    derived_fields_to_include = draw(st.lists(
        st.sampled_from(sorted(DERIVED_RANGE_FIELDS)),
        min_size=0,
        max_size=len(DERIVED_RANGE_FIELDS),
        unique=True,
    ))
    for field in derived_fields_to_include:
        filters[field] = draw(range_values)

    # Optionally add tags
    if draw(st.booleans()):
        filters[TAG_FIELD] = draw(tag_values)

    # Ensure at least one filter is present
    assume(len(filters) > 0)

    return filters


# ---------------------------------------------------------------------------
# Property Tests
# ---------------------------------------------------------------------------

class TestProperty7SegmentRoundTrip:
    """Property 7: Segment save/load round-trip.

    **Validates: Requirements 3.3**
    """

    @given(filters=filter_criteria_strategy())
    @settings(max_examples=200)
    def test_save_load_preserves_filter_criteria(self, filters):
        """Property: For any valid filter criteria dict, saving a segment
        and then loading it returns semantically equivalent filter criteria.

        **Validates: Requirements 3.3**
        """
        # Shared cursor captures the stored JSON and returns it on read
        shared_cursor = SegmentStoreCursor()

        def get_conn():
            return SegmentStoreConnection(shared_cursor)

        service = SegmentationService(get_conn)

        # Save the segment
        saved = service.save_segment(
            name="test_segment",
            filters=filters,
            description="",
            created_by="system",
            organization_id=1,
        )

        # The loaded filter_criteria should equal the original filters
        loaded_criteria = saved["filter_criteria"]
        assert loaded_criteria == filters, (
            f"Round-trip failed.\n"
            f"  Original: {filters}\n"
            f"  Loaded:   {loaded_criteria}"
        )

    @given(filters=filter_criteria_strategy())
    @settings(max_examples=200)
    def test_json_serialization_preserves_all_types(self, filters):
        """Property: JSON serialization/deserialization preserves all filter
        value types — exact match strings remain strings, range dicts remain
        dicts with numeric values, booleans remain booleans, and tag lists
        remain lists of strings.

        **Validates: Requirements 3.3**
        """
        # Directly test the JSON round-trip (the mechanism used by save/load)
        serialized = json.dumps(filters)
        deserialized = json.loads(serialized)

        # Verify structural equality
        assert deserialized == filters, (
            f"JSON round-trip failed.\n"
            f"  Original:     {filters}\n"
            f"  Deserialized: {deserialized}"
        )

        # Verify types are preserved for each field
        for field, value in filters.items():
            loaded_value = deserialized[field]

            if field in EXACT_MATCH_FIELDS:
                assert isinstance(loaded_value, str), (
                    f"Field '{field}' should be str after round-trip, got {type(loaded_value)}"
                )

            elif field in BOOLEAN_FIELDS:
                assert isinstance(loaded_value, bool), (
                    f"Field '{field}' should be bool after round-trip, got {type(loaded_value)}"
                )

            elif field in RANGE_FIELDS or field in DERIVED_RANGE_FIELDS:
                assert isinstance(loaded_value, dict), (
                    f"Field '{field}' should be dict after round-trip, got {type(loaded_value)}"
                )
                for key in loaded_value:
                    assert isinstance(loaded_value[key], int), (
                        f"Field '{field}[{key}]' should be int after round-trip, "
                        f"got {type(loaded_value[key])}"
                    )

            elif field == TAG_FIELD:
                if isinstance(value, list):
                    assert isinstance(loaded_value, list), (
                        f"Tags should be list after round-trip, got {type(loaded_value)}"
                    )
                    for item in loaded_value:
                        assert isinstance(item, str), (
                            f"Tag items should be str after round-trip, got {type(item)}"
                        )
                elif isinstance(value, dict):
                    assert isinstance(loaded_value, dict), (
                        f"Tags dict should be dict after round-trip, got {type(loaded_value)}"
                    )
                    assert isinstance(loaded_value.get("values"), list), (
                        f"Tags 'values' should be list after round-trip"
                    )
                    assert isinstance(loaded_value.get("mode"), str), (
                        f"Tags 'mode' should be str after round-trip"
                    )

    @given(filters=filter_criteria_strategy())
    @settings(max_examples=200)
    def test_save_load_produces_same_query_as_original(self, filters):
        """Property: The loaded filter criteria produce the same SQL query
        and parameters as the original criteria when passed to build_query(),
        confirming semantic equivalence.

        **Validates: Requirements 3.3**
        """
        from datetime import date

        shared_cursor = SegmentStoreCursor()

        def get_conn():
            return SegmentStoreConnection(shared_cursor)

        service = SegmentationService(get_conn)

        # Build query from original filters
        ref_date = date(2025, 1, 15)
        original_where, original_params = service.build_query(
            filters, reference_date=ref_date
        )

        # Save and load
        saved = service.save_segment(
            name="roundtrip_test",
            filters=filters,
            description="",
            created_by="system",
            organization_id=1,
        )
        loaded_criteria = saved["filter_criteria"]

        # Build query from loaded filters
        loaded_where, loaded_params = service.build_query(
            loaded_criteria, reference_date=ref_date
        )

        # The queries must be identical (same WHERE clause and params)
        assert original_where == loaded_where, (
            f"WHERE clause differs after round-trip.\n"
            f"  Original: {original_where}\n"
            f"  Loaded:   {loaded_where}"
        )
        assert original_params == loaded_params, (
            f"Query params differ after round-trip.\n"
            f"  Original: {original_params}\n"
            f"  Loaded:   {loaded_params}"
        )
