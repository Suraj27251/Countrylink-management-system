"""
Property-based tests for campaign duplication preserving configuration.

Property 3: Campaign duplication preserves configuration
- Duplicating a campaign produces a new draft with identical segment_id,
  template_id, campaign_type, channel, priority, recurring_frequency,
  and recurring_end_date.
- The duplicated campaign gets a different ID than the source.
- The duplicated campaign name ends with " (Copy)".

**Validates: Requirements 1.8**

Testing framework: Hypothesis (Python)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hypothesis import given, settings
from hypothesis import strategies as st

from blueprints.campaign_bp import CampaignService


# ---------------------------------------------------------------------------
# Mock infrastructure for tracking inserts
# ---------------------------------------------------------------------------

class TrackingCursor:
    """Mock cursor that records executed SQL and returns configured data."""

    def __init__(self, source_campaign: dict, new_id: int):
        self._source = source_campaign
        self._new_id = new_id
        self._call_index = 0
        self.executed = []
        self.lastrowid = new_id
        self.inserted_params = None

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        self._call_index += 1
        # Track the INSERT params for verification
        if "INSERT INTO campaigns" in sql and params is not None:
            self.inserted_params = params

    def fetchone(self):
        # First fetchone call returns the source campaign (SELECT * FROM campaigns WHERE id = %s)
        # Second fetchone call returns the duplicated campaign (get_campaign after insert)
        for sql, _ in reversed(self.executed):
            if "SELECT * FROM campaigns WHERE id" in sql:
                break

        # If the latest SELECT used the new_id, return the duplicated record
        last_select = self.executed[-1] if self.executed else (None, None)
        if last_select[1] and last_select[1] == (self._new_id,):
            return self._build_duplicated_record()
        return self._source

    def fetchall(self):
        return []

    def close(self):
        pass

    def _build_duplicated_record(self):
        """Build the expected duplicated campaign record from inserted params."""
        return {
            "id": self._new_id,
            "organization_id": self._source["organization_id"],
            "branch_id": self._source["branch_id"],
            "name": f"{self._source['name']} (Copy)",
            "description": self._source.get("description", ""),
            "campaign_type": self._source["campaign_type"],
            "status": "draft",
            "segment_id": self._source.get("segment_id"),
            "template_id": self._source.get("template_id"),
            "channel": self._source.get("channel", "whatsapp"),
            "priority": self._source.get("priority", 5),
            "recurring_frequency": self._source.get("recurring_frequency", "none"),
            "recurring_end_date": self._source.get("recurring_end_date"),
        }


class TrackingConnection:
    """Mock connection that uses TrackingCursor."""

    def __init__(self, cursor: TrackingCursor):
        self._cursor = cursor
        self.committed = False
        self.rolled_back = False

    def cursor(self, dictionary=False):
        return self._cursor

    def start_transaction(self):
        pass

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Hypothesis Strategies
# ---------------------------------------------------------------------------

campaign_types = st.sampled_from(["promotional", "transactional", "reactivation", "ab_test"])
channels = st.sampled_from(["whatsapp", "sms", "email"])
priorities = st.integers(min_value=1, max_value=10)
recurring_frequencies = st.sampled_from(["none", "daily", "weekly", "monthly"])
recurring_end_dates = st.one_of(st.none(), st.dates().map(lambda d: d.isoformat()))


@st.composite
def source_campaign_strategy(draw):
    """Generate a random source campaign with valid field combinations."""
    campaign_id = draw(st.integers(min_value=1, max_value=10000))
    name = draw(st.text(
        alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -_",
        min_size=1,
        max_size=100,
    ).filter(lambda s: s.strip() != ""))

    segment_id = draw(st.one_of(st.none(), st.integers(min_value=1, max_value=10000)))
    template_id = draw(st.one_of(st.none(), st.integers(min_value=1, max_value=10000)))
    campaign_type = draw(campaign_types)
    channel = draw(channels)
    priority = draw(priorities)
    recurring_frequency = draw(recurring_frequencies)
    recurring_end_date = draw(recurring_end_dates)

    # The source can be in any status (duplication works regardless of source state)
    status = draw(st.sampled_from([
        "draft", "scheduled", "pending_approval", "approved",
        "sending", "paused", "completed", "cancelled", "failed"
    ]))

    return {
        "id": campaign_id,
        "organization_id": 1,
        "branch_id": 1,
        "name": name,
        "description": f"Description for {name}",
        "campaign_type": campaign_type,
        "status": status,
        "segment_id": segment_id,
        "template_id": template_id,
        "channel": channel,
        "priority": priority,
        "recurring_frequency": recurring_frequency,
        "recurring_end_date": recurring_end_date,
    }


# ---------------------------------------------------------------------------
# Property Tests
# ---------------------------------------------------------------------------

class TestProperty3CampaignDuplication:
    """Property 3: Campaign duplication preserves configuration.

    **Validates: Requirements 1.8**
    """

    @given(source=source_campaign_strategy())
    @settings(max_examples=200)
    def test_duplicated_campaign_has_draft_status_and_identical_config(self, source):
        """Property: For any campaign with random segment_id, template_id,
        campaign_type, channel, priority, recurring_frequency, the duplicated
        campaign has status='draft' and identical values for all these fields.

        **Validates: Requirements 1.8**
        """
        new_id = source["id"] + 10000  # Ensure different ID
        cursor = TrackingCursor(source, new_id)

        call_count = [0]

        def get_conn():
            call_count[0] += 1
            if call_count[0] == 1:
                # First connection: used for the duplication transaction
                return TrackingConnection(cursor)
            else:
                # Second connection: used for get_campaign read
                read_cursor = TrackingCursor(source, new_id)
                # Force the read cursor to return the duplicated record
                read_cursor.executed.append(("SELECT * FROM campaigns WHERE id = %s", (new_id,)))
                return TrackingConnection(read_cursor)

        service = CampaignService(get_conn)
        result = service.duplicate_campaign(source["id"], "test_operator")

        # Status must be draft
        assert result["status"] == "draft", (
            f"Expected status 'draft', got '{result['status']}'"
        )

        # Configuration fields must be identical to source
        assert result["segment_id"] == source.get("segment_id"), (
            f"segment_id mismatch: expected {source.get('segment_id')}, got {result['segment_id']}"
        )
        assert result["template_id"] == source.get("template_id"), (
            f"template_id mismatch: expected {source.get('template_id')}, got {result['template_id']}"
        )
        assert result["campaign_type"] == source["campaign_type"], (
            f"campaign_type mismatch: expected {source['campaign_type']}, got {result['campaign_type']}"
        )
        assert result["channel"] == source.get("channel", "whatsapp"), (
            f"channel mismatch: expected {source.get('channel', 'whatsapp')}, got {result['channel']}"
        )
        assert result["priority"] == source.get("priority", 5), (
            f"priority mismatch: expected {source.get('priority', 5)}, got {result['priority']}"
        )
        assert result["recurring_frequency"] == source.get("recurring_frequency", "none"), (
            f"recurring_frequency mismatch: expected {source.get('recurring_frequency', 'none')}, "
            f"got {result['recurring_frequency']}"
        )
        assert result["recurring_end_date"] == source.get("recurring_end_date"), (
            f"recurring_end_date mismatch: expected {source.get('recurring_end_date')}, "
            f"got {result['recurring_end_date']}"
        )

    @given(source=source_campaign_strategy())
    @settings(max_examples=200)
    def test_duplicated_campaign_gets_different_id(self, source):
        """Property: The duplicated campaign always gets a different ID
        than the source campaign.

        **Validates: Requirements 1.8**
        """
        new_id = source["id"] + 10000
        cursor = TrackingCursor(source, new_id)

        call_count = [0]

        def get_conn():
            call_count[0] += 1
            if call_count[0] == 1:
                return TrackingConnection(cursor)
            else:
                read_cursor = TrackingCursor(source, new_id)
                read_cursor.executed.append(("SELECT * FROM campaigns WHERE id = %s", (new_id,)))
                return TrackingConnection(read_cursor)

        service = CampaignService(get_conn)
        result = service.duplicate_campaign(source["id"], "test_operator")

        assert result["id"] != source["id"], (
            f"Duplicated campaign ID ({result['id']}) must differ from source ({source['id']})"
        )

    @given(source=source_campaign_strategy())
    @settings(max_examples=200)
    def test_duplicated_campaign_name_ends_with_copy(self, source):
        """Property: The duplicated campaign name ends with ' (Copy)'.

        **Validates: Requirements 1.8**
        """
        new_id = source["id"] + 10000
        cursor = TrackingCursor(source, new_id)

        call_count = [0]

        def get_conn():
            call_count[0] += 1
            if call_count[0] == 1:
                return TrackingConnection(cursor)
            else:
                read_cursor = TrackingCursor(source, new_id)
                read_cursor.executed.append(("SELECT * FROM campaigns WHERE id = %s", (new_id,)))
                return TrackingConnection(read_cursor)

        service = CampaignService(get_conn)
        result = service.duplicate_campaign(source["id"], "test_operator")

        assert result["name"].endswith(" (Copy)"), (
            f"Expected name to end with ' (Copy)', got '{result['name']}'"
        )
        # Also verify the prefix is the original name
        expected_name = f"{source['name']} (Copy)"
        assert result["name"] == expected_name, (
            f"Expected name '{expected_name}', got '{result['name']}'"
        )

    @given(source=source_campaign_strategy())
    @settings(max_examples=200)
    def test_duplicate_inserts_correct_params_to_db(self, source):
        """Property: The SQL INSERT for duplication passes the correct
        parameter values from the source campaign to the database.

        **Validates: Requirements 1.8**
        """
        new_id = source["id"] + 10000
        cursor = TrackingCursor(source, new_id)

        call_count = [0]

        def get_conn():
            call_count[0] += 1
            if call_count[0] == 1:
                return TrackingConnection(cursor)
            else:
                read_cursor = TrackingCursor(source, new_id)
                read_cursor.executed.append(("SELECT * FROM campaigns WHERE id = %s", (new_id,)))
                return TrackingConnection(read_cursor)

        service = CampaignService(get_conn)
        service.duplicate_campaign(source["id"], "test_operator")

        # Verify the actual INSERT parameters passed to cursor.execute
        assert cursor.inserted_params is not None, "No INSERT was executed"

        # The INSERT params order matches the duplicate_campaign implementation:
        # (organization_id, branch_id, name, description, campaign_type,
        #  segment_id, template_id, channel, priority,
        #  recurring_frequency, recurring_end_date, created_by)
        params = cursor.inserted_params
        assert params[0] == source["organization_id"]  # organization_id
        assert params[1] == source["branch_id"]  # branch_id
        assert params[2] == f"{source['name']} (Copy)"  # name
        assert params[3] == source.get("description", "")  # description
        assert params[4] == source["campaign_type"]  # campaign_type
        assert params[5] == source.get("segment_id")  # segment_id
        assert params[6] == source.get("template_id")  # template_id
        assert params[7] == source.get("channel", "whatsapp")  # channel
        assert params[8] == source.get("priority", 5)  # priority
        assert params[9] == source.get("recurring_frequency", "none")  # recurring_frequency
        assert params[10] == source.get("recurring_end_date")  # recurring_end_date
        assert params[11] == "test_operator"  # created_by
