"""
Property-based tests for Timeline ordering consistency using Hypothesis.

**Validates: Requirements 7.2, 7.6**

Property 13: Timeline ordering consistency
- For any customer with interaction records of multiple types (messages, campaigns,
  notes, tags, status changes), the interaction timeline SHALL return all records
  merged in strictly reverse chronological order by created_at timestamp, with no
  records omitted.
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from services.crm import CRMService


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------
activity_types = st.sampled_from([
    "message_sent", "message_received", "campaign_sent", "note_added",
    "tag_added", "tag_removed", "status_change", "opt_out", "opt_in",
])

channels = st.sampled_from(["whatsapp", "system", "sms"])

# Generate a list of activity records with varying timestamps and types
def activity_record_strategy(mobile: str):
    """Generate a single activity record with a random timestamp and type."""
    return st.fixed_dictionaries({
        "id": st.integers(min_value=1, max_value=1000000),
        "customer_mobile": st.just(mobile),
        "activity_type": activity_types,
        "channel": channels,
        "reference_id": st.one_of(st.none(), st.integers(min_value=1, max_value=100000)),
        "details": st.one_of(
            st.none(),
            st.just(json.dumps({"note_text": "test note", "added_by": "operator1"})),
            st.just(json.dumps({"tag_name": "vip", "added_by": "operator2"})),
        ),
        "created_at": st.datetimes(
            min_value=datetime(2023, 1, 1),
            max_value=datetime(2025, 12, 31),
        ),
    })


mobile_numbers = st.from_regex(r"919[0-9]{9}", fullmatch=True)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------
class MockCursor:
    """Mock MySQL cursor with dictionary=True support."""

    def __init__(self):
        self.executed = []
        self.fetchone_result = None
        self.fetchall_result = []
        self.lastrowid = 1
        self._closed = False
        self.rowcount = 0
        self._fetch_queue = []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchone(self):
        return self.fetchone_result

    def fetchall(self):
        return self.fetchall_result

    def close(self):
        self._closed = True


class MockConnection:
    """Mock MySQL connection."""

    def __init__(self, cursor_instance=None):
        self._cursor = cursor_instance or MockCursor()
        self._committed = False
        self._rolled_back = False
        self._closed = False

    def cursor(self, dictionary=False):
        return self._cursor

    def start_transaction(self):
        pass

    def commit(self):
        self._committed = True

    def rollback(self):
        self._rolled_back = True

    def close(self):
        self._closed = True


# ---------------------------------------------------------------------------
# Property Tests
# ---------------------------------------------------------------------------
class TestTimelineOrderingConsistency:
    """Property-based tests for timeline ordering — Property 13."""

    @given(
        mobile=mobile_numbers,
        activities=st.lists(
            activity_types.flatmap(
                lambda at: st.fixed_dictionaries({
                    "id": st.integers(min_value=1, max_value=1000000),
                    "activity_type": st.just(at),
                    "channel": channels,
                    "reference_id": st.one_of(
                        st.none(), st.integers(min_value=1, max_value=100000)
                    ),
                    "details": st.one_of(
                        st.none(),
                        st.just(json.dumps({"note_text": "test", "added_by": "op"})),
                        st.just(json.dumps({"tag_name": "vip"})),
                    ),
                    "created_at": st.datetimes(
                        min_value=datetime(2023, 1, 1),
                        max_value=datetime(2025, 12, 31),
                    ),
                })
            ),
            min_size=2,
            max_size=50,
        ),
    )
    @settings(max_examples=200)
    def test_timeline_returns_records_in_reverse_chronological_order(
        self, mobile: str, activities: list
    ):
        """
        Property: For any set of interaction records of multiple types,
        get_interaction_timeline() returns them in strictly reverse chronological
        order (newest first, oldest last) by created_at timestamp.

        **Validates: Requirements 7.2, 7.6**
        """
        # Sort activities by created_at DESC to simulate what the DB would return
        sorted_activities = sorted(activities, key=lambda x: x["created_at"], reverse=True)

        # Assign customer_mobile to each activity
        for act in sorted_activities:
            act["customer_mobile"] = mobile

        total_count = len(sorted_activities)

        call_count = [0]

        def get_conn():
            call_count[0] += 1
            cursor = MockCursor()
            conn = MockConnection(cursor)

            if call_count[0] == 1:
                # COUNT query
                cursor.fetchone_result = {"total": total_count}
                # Activities query (ORDER BY created_at DESC) — simulate DB ordering
                cursor.fetchall_result = sorted_activities
            return conn

        service = CRMService(get_connection=get_conn)
        result = service.get_interaction_timeline(mobile=mobile, page=1, per_page=100)

        timeline = result["timeline"]

        # Verify all records returned (no omissions)
        assert len(timeline) == total_count, (
            f"Expected {total_count} timeline records, got {len(timeline)}"
        )

        # Verify reverse chronological order
        for i in range(len(timeline) - 1):
            current_ts = timeline[i]["timestamp"]
            next_ts = timeline[i + 1]["timestamp"]
            assert current_ts >= next_ts, (
                f"Timeline not in reverse chronological order at index {i}: "
                f"{current_ts} should be >= {next_ts}"
            )

    @given(
        mobile=mobile_numbers,
        activities=st.lists(
            activity_types.flatmap(
                lambda at: st.fixed_dictionaries({
                    "id": st.integers(min_value=1, max_value=1000000),
                    "activity_type": st.just(at),
                    "channel": channels,
                    "reference_id": st.one_of(
                        st.none(), st.integers(min_value=1, max_value=100000)
                    ),
                    "details": st.one_of(
                        st.none(),
                        st.just(json.dumps({"note_text": "test", "added_by": "op"})),
                    ),
                    "created_at": st.datetimes(
                        min_value=datetime(2023, 1, 1),
                        max_value=datetime(2025, 12, 31),
                    ),
                })
            ),
            min_size=1,
            max_size=50,
        ),
    )
    @settings(max_examples=200)
    def test_timeline_no_records_omitted(self, mobile: str, activities: list):
        """
        Property: For any set of customer_activity records, the timeline
        returned by get_interaction_timeline() contains ALL records — no
        records are omitted.

        **Validates: Requirements 7.2, 7.6**
        """
        # Sort as the DB would
        sorted_activities = sorted(activities, key=lambda x: x["created_at"], reverse=True)
        for act in sorted_activities:
            act["customer_mobile"] = mobile

        total_count = len(sorted_activities)

        # Track IDs to ensure all are present in output
        expected_ids = {act["id"] for act in sorted_activities}

        call_count = [0]

        def get_conn():
            call_count[0] += 1
            cursor = MockCursor()
            conn = MockConnection(cursor)

            if call_count[0] == 1:
                cursor.fetchone_result = {"total": total_count}
                cursor.fetchall_result = sorted_activities
            return conn

        service = CRMService(get_connection=get_conn)
        result = service.get_interaction_timeline(mobile=mobile, page=1, per_page=100)

        timeline = result["timeline"]

        # Verify count matches
        assert len(timeline) == total_count, (
            f"Expected {total_count} records, got {len(timeline)}. "
            f"Records were omitted from timeline."
        )

        # Verify all IDs are present
        returned_ids = {entry["id"] for entry in timeline}
        assert returned_ids == expected_ids, (
            f"Some records were omitted. "
            f"Missing IDs: {expected_ids - returned_ids}"
        )

    @given(
        mobile=mobile_numbers,
        num_activities=st.integers(min_value=2, max_value=30),
    )
    @settings(max_examples=200)
    def test_timeline_mixed_activity_types_maintain_order(
        self, mobile: str, num_activities: int
    ):
        """
        Property: When multiple activity types share the same timeline,
        records are still ordered by created_at DESC regardless of type.
        Different activity types do not form separate sequences — they are
        interleaved by timestamp.

        **Validates: Requirements 7.2, 7.6**
        """
        # Create a fixed set of activities with known timestamps
        # Each activity has a distinct timestamp spread across a range
        base_time = datetime(2024, 6, 1, 12, 0, 0)
        all_types = [
            "message_sent", "message_received", "campaign_sent",
            "note_added", "tag_added", "tag_removed", "status_change",
            "opt_out", "opt_in",
        ]

        activities = []
        for i in range(num_activities):
            activity_type = all_types[i % len(all_types)]
            # Each record is 1 hour apart
            timestamp = base_time + timedelta(hours=i)
            activities.append({
                "id": i + 1,
                "customer_mobile": mobile,
                "activity_type": activity_type,
                "channel": "whatsapp" if "message" in activity_type else "system",
                "reference_id": i + 100,
                "details": json.dumps({"type": activity_type, "index": i}),
                "created_at": timestamp,
            })

        # Sort by created_at DESC (as the DB ORDER BY would)
        sorted_activities = sorted(activities, key=lambda x: x["created_at"], reverse=True)
        total_count = len(sorted_activities)

        call_count = [0]

        def get_conn():
            call_count[0] += 1
            cursor = MockCursor()
            conn = MockConnection(cursor)

            if call_count[0] == 1:
                cursor.fetchone_result = {"total": total_count}
                cursor.fetchall_result = sorted_activities
            return conn

        service = CRMService(get_connection=get_conn)
        result = service.get_interaction_timeline(mobile=mobile, page=1, per_page=100)

        timeline = result["timeline"]

        # All records present
        assert len(timeline) == total_count

        # Verify strict reverse chronological order
        for i in range(len(timeline) - 1):
            current_ts = timeline[i]["timestamp"]
            next_ts = timeline[i + 1]["timestamp"]
            assert current_ts >= next_ts, (
                f"Timeline order violated at index {i}: "
                f"type={timeline[i]['type']} at {current_ts} followed by "
                f"type={timeline[i+1]['type']} at {next_ts}"
            )

        # Verify multiple activity types are represented
        types_in_timeline = {entry["type"] for entry in timeline}
        assert len(types_in_timeline) > 1, (
            "Expected multiple activity types in timeline to verify "
            "cross-type ordering"
        )

    @given(
        mobile=mobile_numbers,
        activities=st.lists(
            activity_types.flatmap(
                lambda at: st.fixed_dictionaries({
                    "id": st.integers(min_value=1, max_value=1000000),
                    "activity_type": st.just(at),
                    "channel": channels,
                    "reference_id": st.one_of(
                        st.none(), st.integers(min_value=1, max_value=100000)
                    ),
                    "details": st.one_of(
                        st.none(),
                        st.just(json.dumps({"note_text": "test", "added_by": "op"})),
                    ),
                    "created_at": st.datetimes(
                        min_value=datetime(2023, 1, 1),
                        max_value=datetime(2025, 12, 31),
                    ),
                })
            ),
            min_size=5,
            max_size=50,
        ),
    )
    @settings(max_examples=200)
    def test_timeline_preserves_all_activity_types(self, mobile: str, activities: list):
        """
        Property: The timeline does not filter out any activity_type.
        Every type of interaction record stored in customer_activity is
        returned in the timeline response.

        **Validates: Requirements 7.2, 7.6**
        """
        sorted_activities = sorted(activities, key=lambda x: x["created_at"], reverse=True)
        for act in sorted_activities:
            act["customer_mobile"] = mobile

        total_count = len(sorted_activities)
        expected_types = {act["activity_type"] for act in sorted_activities}

        call_count = [0]

        def get_conn():
            call_count[0] += 1
            cursor = MockCursor()
            conn = MockConnection(cursor)

            if call_count[0] == 1:
                cursor.fetchone_result = {"total": total_count}
                cursor.fetchall_result = sorted_activities
            return conn

        service = CRMService(get_connection=get_conn)
        result = service.get_interaction_timeline(mobile=mobile, page=1, per_page=100)

        timeline = result["timeline"]

        # Verify all activity types present in input are present in output
        returned_types = {entry["type"] for entry in timeline}
        assert expected_types == returned_types, (
            f"Some activity types were omitted. "
            f"Expected: {expected_types}, Got: {returned_types}, "
            f"Missing: {expected_types - returned_types}"
        )
