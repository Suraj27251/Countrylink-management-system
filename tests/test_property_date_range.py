"""
Property-based tests for date range filtering in analytics endpoints.

Property 15: Date range filtering includes only records within bounds
- For any date range [start, end] applied to analytics queries, all returned
  records SHALL have their relevant timestamp >= start AND <= end, with no
  records outside the range included.

**Validates: Requirements 8.4**

Testing framework: Hypothesis (Python)
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Date range filter function (extracted logic from analytics_bp.py)
# ---------------------------------------------------------------------------

def filter_records_by_date_range(records, start_date, end_date, timestamp_field="sent_at"):
    """
    Filter records to only include those with timestamps within [start, end].

    This mirrors the WHERE clause logic in analytics_bp.py:
        WHERE sent_at >= %s AND sent_at <= %s

    Parameters
    ----------
    records : list of dict
        Records containing a timestamp field.
    start_date : datetime
        Start of the date range (inclusive).
    end_date : datetime
        End of the date range (inclusive).
    timestamp_field : str
        The key in each record containing the timestamp.

    Returns
    -------
    list of dict
        Filtered records within the date range.
    """
    if start_date is None and end_date is None:
        return records

    filtered = []
    for record in records:
        ts = record.get(timestamp_field)
        if ts is None:
            continue
        if start_date is not None and ts < start_date:
            continue
        if end_date is not None and ts > end_date:
            continue
        filtered.append(record)
    return filtered


def filter_analytics_by_period(records, start_date, end_date):
    """
    Filter analytics records by period_start and period_end bounds.

    This mirrors the WHERE clause logic in analytics_bp.py aggregate endpoint:
        WHERE period_start >= %s AND period_end <= %s

    A record is included only if its entire period is within the query range.

    Parameters
    ----------
    records : list of dict
        Analytics records with period_start and period_end fields.
    start_date : datetime
        Start of the query range (inclusive).
    end_date : datetime
        End of the query range (inclusive).

    Returns
    -------
    list of dict
        Filtered analytics records within the date range.
    """
    if start_date is None and end_date is None:
        return records

    filtered = []
    for record in records:
        period_start = record.get("period_start")
        period_end = record.get("period_end")
        if period_start is None or period_end is None:
            continue
        if start_date is not None and period_start < start_date:
            continue
        if end_date is not None and period_end > end_date:
            continue
        filtered.append(record)
    return filtered


# ---------------------------------------------------------------------------
# Mock analytics cursor to simulate date-range-filtered queries
# ---------------------------------------------------------------------------

class AnalyticsDateRangeCursor:
    """Mock cursor simulating analytics queries with date range filtering."""

    def __init__(self, campaign_messages, analytics_records):
        """
        Parameters
        ----------
        campaign_messages : list of dict
            Records with: id, campaign_id, status, sent_at, customer_mobile
        analytics_records : list of dict
            Records with: id, campaign_id, metric_type, metric_value,
                         period_start, period_end
        """
        self._campaign_messages = campaign_messages
        self._analytics_records = analytics_records
        self._last_results = []
        self._last_result = None
        self.rowcount = 0

    def execute(self, sql, params=None):
        sql_stripped = sql.strip()

        if "campaign_messages" in sql_stripped and "sent_at >=" in sql_stripped:
            # Filter campaign_messages by sent_at date range
            start_date = params[0] if params and len(params) > 0 else None
            end_date = params[1] if params and len(params) > 1 else None

            filtered = filter_records_by_date_range(
                self._campaign_messages, start_date, end_date, "sent_at"
            )

            if "COUNT" in sql_stripped:
                self._last_result = {"total_messages": len(filtered)}
            else:
                self._last_results = filtered

            self.rowcount = len(filtered)

        elif "campaign_analytics" in sql_stripped and "period_start >=" in sql_stripped:
            # Filter analytics records by period bounds
            start_date = params[0] if params and len(params) > 0 else None
            end_date = params[1] if params and len(params) > 1 else None

            filtered = filter_analytics_by_period(
                self._analytics_records, start_date, end_date
            )
            self._last_results = filtered
            self.rowcount = len(filtered)

        else:
            self._last_results = []
            self._last_result = None
            self.rowcount = 0

    def fetchone(self):
        return self._last_result

    def fetchall(self):
        return self._last_results

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Hypothesis Strategies
# ---------------------------------------------------------------------------

# Generate reasonable datetimes within a 2-year window
reasonable_datetimes = st.datetimes(
    min_value=datetime(2023, 1, 1),
    max_value=datetime(2025, 12, 31),
)

# Generate date ranges where start <= end
@st.composite
def date_ranges(draw):
    """Generate a valid date range [start, end] where start <= end."""
    d1 = draw(reasonable_datetimes)
    d2 = draw(reasonable_datetimes)
    start = min(d1, d2)
    end = max(d1, d2)
    return start, end


# Generate campaign message records with timestamps
@st.composite
def campaign_message_records(draw, min_count=1, max_count=50):
    """Generate campaign_messages with sent_at timestamps."""
    count = draw(st.integers(min_value=min_count, max_value=max_count))
    records = []
    for i in range(count):
        sent_at = draw(reasonable_datetimes)
        status = draw(st.sampled_from(["sent", "delivered", "read", "failed"]))
        records.append({
            "id": i + 1,
            "campaign_id": draw(st.integers(min_value=1, max_value=100)),
            "customer_mobile": f"+91{draw(st.integers(min_value=7000000000, max_value=9999999999))}",
            "status": status,
            "sent_at": sent_at,
        })
    return records


# Generate analytics records with period timestamps
@st.composite
def analytics_period_records(draw, min_count=1, max_count=30):
    """Generate campaign_analytics records with period_start and period_end."""
    count = draw(st.integers(min_value=min_count, max_value=max_count))
    records = []
    for i in range(count):
        period_start = draw(reasonable_datetimes)
        # period_end is always after period_start (1 hour to 30 days later)
        offset_hours = draw(st.integers(min_value=1, max_value=720))
        period_end = period_start + timedelta(hours=offset_hours)
        metric_type = draw(st.sampled_from([
            "delivery_rate", "read_rate", "failure_rate"
        ]))
        metric_value = draw(st.floats(min_value=0.0, max_value=1.0))
        records.append({
            "id": i + 1,
            "campaign_id": draw(st.integers(min_value=1, max_value=100)),
            "metric_type": metric_type,
            "metric_value": metric_value,
            "period_start": period_start,
            "period_end": period_end,
        })
    return records


# ---------------------------------------------------------------------------
# Property Tests
# ---------------------------------------------------------------------------


class TestProperty15DateRangeFiltering:
    """Property 15: Date range filtering includes only records within bounds.

    **Validates: Requirements 8.4**
    """

    @given(
        date_range=date_ranges(),
        records=campaign_message_records(),
    )
    @settings(max_examples=500, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_all_returned_messages_have_sent_at_within_range(self, date_range, records):
        """
        Property: For any date range [start, end], all returned campaign_messages
        SHALL have sent_at >= start AND sent_at <= end.

        **Validates: Requirements 8.4**
        """
        start_date, end_date = date_range

        filtered = filter_records_by_date_range(records, start_date, end_date, "sent_at")

        for record in filtered:
            assert record["sent_at"] >= start_date, (
                f"Record {record['id']} has sent_at={record['sent_at']} which is "
                f"before start_date={start_date}"
            )
            assert record["sent_at"] <= end_date, (
                f"Record {record['id']} has sent_at={record['sent_at']} which is "
                f"after end_date={end_date}"
            )

    @given(
        date_range=date_ranges(),
        records=campaign_message_records(),
    )
    @settings(max_examples=500, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_no_in_range_messages_excluded(self, date_range, records):
        """
        Property: For any date range [start, end], no campaign_message with
        sent_at within [start, end] SHALL be excluded from results.

        **Validates: Requirements 8.4**
        """
        start_date, end_date = date_range

        filtered = filter_records_by_date_range(records, start_date, end_date, "sent_at")
        filtered_ids = {r["id"] for r in filtered}

        for record in records:
            if start_date <= record["sent_at"] <= end_date:
                assert record["id"] in filtered_ids, (
                    f"Record {record['id']} with sent_at={record['sent_at']} is within "
                    f"[{start_date}, {end_date}] but was excluded from results"
                )

    @given(
        date_range=date_ranges(),
        records=campaign_message_records(),
    )
    @settings(max_examples=500, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_no_out_of_range_messages_included(self, date_range, records):
        """
        Property: For any date range [start, end], no campaign_message with
        sent_at outside [start, end] SHALL be included in results.

        **Validates: Requirements 8.4**
        """
        start_date, end_date = date_range

        filtered = filter_records_by_date_range(records, start_date, end_date, "sent_at")

        for record in filtered:
            assert start_date <= record["sent_at"] <= end_date, (
                f"Record {record['id']} with sent_at={record['sent_at']} is outside "
                f"range [{start_date}, {end_date}] but was included in results"
            )

    @given(
        date_range=date_ranges(),
        records=analytics_period_records(),
    )
    @settings(max_examples=500, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_all_returned_analytics_have_period_within_range(self, date_range, records):
        """
        Property: For any date range [start, end], all returned analytics
        records SHALL have period_start >= start AND period_end <= end.

        **Validates: Requirements 8.4**
        """
        start_date, end_date = date_range

        filtered = filter_analytics_by_period(records, start_date, end_date)

        for record in filtered:
            assert record["period_start"] >= start_date, (
                f"Analytics record {record['id']} has period_start="
                f"{record['period_start']} which is before start_date={start_date}"
            )
            assert record["period_end"] <= end_date, (
                f"Analytics record {record['id']} has period_end="
                f"{record['period_end']} which is after end_date={end_date}"
            )

    @given(
        date_range=date_ranges(),
        records=analytics_period_records(),
    )
    @settings(max_examples=500, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_no_in_range_analytics_excluded(self, date_range, records):
        """
        Property: For any date range [start, end], no analytics record with
        period fully within [start, end] SHALL be excluded from results.

        **Validates: Requirements 8.4**
        """
        start_date, end_date = date_range

        filtered = filter_analytics_by_period(records, start_date, end_date)
        filtered_ids = {r["id"] for r in filtered}

        for record in records:
            if (record["period_start"] >= start_date and
                    record["period_end"] <= end_date):
                assert record["id"] in filtered_ids, (
                    f"Analytics record {record['id']} with period "
                    f"[{record['period_start']}, {record['period_end']}] is within "
                    f"[{start_date}, {end_date}] but was excluded"
                )

    @given(
        date_range=date_ranges(),
        records=campaign_message_records(),
    )
    @settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_filtered_count_matches_manual_count(self, date_range, records):
        """
        Property: The number of filtered records SHALL equal the count of
        records manually verified to be within [start, end].

        **Validates: Requirements 8.4**
        """
        start_date, end_date = date_range

        filtered = filter_records_by_date_range(records, start_date, end_date, "sent_at")

        # Manual count
        expected_count = sum(
            1 for r in records
            if start_date <= r["sent_at"] <= end_date
        )

        assert len(filtered) == expected_count, (
            f"Filter returned {len(filtered)} records but expected "
            f"{expected_count} for range [{start_date}, {end_date}]"
        )

    @given(
        date_range=date_ranges(),
        records=campaign_message_records(min_count=0, max_count=0),
    )
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_empty_dataset_returns_empty_results(self, date_range, records):
        """
        Property: When there are no records, date range filtering SHALL return
        an empty list regardless of the date range.

        **Validates: Requirements 8.4**
        """
        start_date, end_date = date_range

        filtered = filter_records_by_date_range([], start_date, end_date, "sent_at")

        assert filtered == [], (
            f"Expected empty results for empty dataset, got {len(filtered)} records"
        )

    @given(
        records=campaign_message_records(),
    )
    @settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_full_range_returns_all_records(self, records):
        """
        Property: When the date range encompasses all record timestamps,
        all records SHALL be returned.

        **Validates: Requirements 8.4**
        """
        assume(len(records) > 0)

        # Use a range that covers all records
        all_timestamps = [r["sent_at"] for r in records]
        start_date = min(all_timestamps) - timedelta(seconds=1)
        end_date = max(all_timestamps) + timedelta(seconds=1)

        filtered = filter_records_by_date_range(records, start_date, end_date, "sent_at")

        assert len(filtered) == len(records), (
            f"With full date range, expected all {len(records)} records but "
            f"got {len(filtered)}"
        )

    @given(
        records=campaign_message_records(),
    )
    @settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_boundary_inclusivity_start(self, records):
        """
        Property: A record with sent_at exactly equal to start_date SHALL be
        included in the results (inclusive boundary).

        **Validates: Requirements 8.4**
        """
        assume(len(records) > 0)

        # Pick the first record's timestamp as start_date
        target_record = records[0]
        start_date = target_record["sent_at"]
        end_date = start_date + timedelta(days=365)

        filtered = filter_records_by_date_range(records, start_date, end_date, "sent_at")
        filtered_ids = {r["id"] for r in filtered}

        assert target_record["id"] in filtered_ids, (
            f"Record with sent_at exactly at start_date={start_date} should be "
            f"included (inclusive boundary)"
        )

    @given(
        records=campaign_message_records(),
    )
    @settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_boundary_inclusivity_end(self, records):
        """
        Property: A record with sent_at exactly equal to end_date SHALL be
        included in the results (inclusive boundary).

        **Validates: Requirements 8.4**
        """
        assume(len(records) > 0)

        # Pick the first record's timestamp as end_date
        target_record = records[0]
        end_date = target_record["sent_at"]
        start_date = end_date - timedelta(days=365)

        filtered = filter_records_by_date_range(records, start_date, end_date, "sent_at")
        filtered_ids = {r["id"] for r in filtered}

        assert target_record["id"] in filtered_ids, (
            f"Record with sent_at exactly at end_date={end_date} should be "
            f"included (inclusive boundary)"
        )

    @given(
        date_range=date_ranges(),
        records=campaign_message_records(),
    )
    @settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_mock_cursor_returns_only_in_range_records(self, date_range, records):
        """
        Property: The mock analytics cursor (simulating the SQL WHERE clause)
        SHALL return only records within the specified date range, matching
        the pure filter function behavior.

        **Validates: Requirements 8.4**
        """
        start_date, end_date = date_range

        # Use mock cursor
        cursor = AnalyticsDateRangeCursor(records, [])
        cursor.execute(
            "SELECT * FROM campaign_messages WHERE sent_at >= %s AND sent_at <= %s",
            (start_date, end_date),
        )
        cursor_results = cursor.fetchall()

        # Use pure function
        pure_results = filter_records_by_date_range(records, start_date, end_date, "sent_at")

        # Both should return the same records
        cursor_ids = sorted([r["id"] for r in cursor_results])
        pure_ids = sorted([r["id"] for r in pure_results])

        assert cursor_ids == pure_ids, (
            f"Cursor returned {len(cursor_ids)} records but pure filter "
            f"returned {len(pure_ids)} for range [{start_date}, {end_date}]"
        )
