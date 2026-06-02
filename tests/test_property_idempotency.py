"""
Property-based tests for idempotency preventing duplicate message delivery.

Property 10: Idempotency prevents duplicate message delivery
- For any campaign with a recipient list containing duplicate mobile numbers,
  or during recovery re-processing, the system SHALL produce exactly one
  campaign_messages record per unique (campaign_id, customer_mobile, template_id)
  combination, enforced by the idempotency_key unique constraint.

**Validates: Requirements 4.5, 26.3**

Testing framework: Hypothesis (Python)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from services.sending_queue import SendingQueue


# ---------------------------------------------------------------------------
# Mock infrastructure simulating MySQL INSERT IGNORE on unique idempotency_key
# ---------------------------------------------------------------------------

class IdempotencyTrackingCursor:
    """Mock cursor that simulates INSERT IGNORE with unique idempotency_key constraint."""

    def __init__(self, campaign: dict, template: dict):
        self._campaign = campaign
        self._template = template
        self._seen_keys = set()  # Tracks idempotency_keys that have been "inserted"
        self.inserted_records = []  # All actually inserted records
        self.attempted_inserts = []  # All attempted inserts (including duplicates)
        self.rowcount = 0
        self.lastrowid = 0

    def execute(self, sql, params=None):
        if "SELECT template_id, segment_id FROM campaigns WHERE id" in sql:
            # Return campaign lookup
            self._last_result = self._campaign
        elif "SELECT template_name, body_text, placeholder_mappings" in sql:
            # Return template lookup
            self._last_result = self._template
        elif "INSERT IGNORE INTO campaign_messages" in sql and params is not None:
            # Simulate INSERT IGNORE with unique constraint on idempotency_key
            # idempotency_key is the last parameter in the INSERT
            idempotency_key = params[-1]
            self.attempted_inserts.append({
                "campaign_id": params[0],
                "customer_mobile": params[1],
                "customer_name": params[2],
                "template_id": params[3],
                "template_params": params[4],
                "channel": params[5],
                "idempotency_key": idempotency_key,
            })

            if idempotency_key not in self._seen_keys:
                # New record — insert succeeds
                self._seen_keys.add(idempotency_key)
                self.inserted_records.append({
                    "campaign_id": params[0],
                    "customer_mobile": params[1],
                    "template_id": params[3],
                    "idempotency_key": idempotency_key,
                })
                self.rowcount = 1
                self.lastrowid += 1
            else:
                # Duplicate — INSERT IGNORE silently skips
                self.rowcount = 0
        elif "UPDATE campaigns SET total_recipients" in sql:
            # Update total_recipients — no-op for test
            self.rowcount = 1
        else:
            self._last_result = None
            self.rowcount = 0

    def fetchone(self):
        return self._last_result

    def fetchall(self):
        return []

    def close(self):
        pass


class IdempotencyTrackingConnection:
    """Mock connection using IdempotencyTrackingCursor."""

    def __init__(self, cursor: IdempotencyTrackingCursor):
        self._cursor = cursor
        self.committed = False

    def cursor(self, dictionary=False):
        return self._cursor

    def start_transaction(self):
        pass

    def commit(self):
        self.committed = True

    def rollback(self):
        pass

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Hypothesis Strategies
# ---------------------------------------------------------------------------

# Generate valid mobile numbers (digits, 7-15 chars as per international standards)
mobile_number_strategy = st.text(
    alphabet="0123456789",
    min_size=7,
    max_size=15,
).filter(lambda s: s.strip() != "")

# Generate campaign IDs
campaign_id_strategy = st.integers(min_value=1, max_value=10000)

# Generate template IDs
template_id_strategy = st.integers(min_value=1, max_value=1000)


@st.composite
def recipients_with_duplicates(draw):
    """Generate a recipient list that contains duplicate mobile numbers."""
    # Generate unique mobiles first
    unique_count = draw(st.integers(min_value=1, max_value=10))
    unique_mobiles = draw(
        st.lists(
            mobile_number_strategy,
            min_size=unique_count,
            max_size=unique_count,
            unique=True,
        )
    )

    # Create base recipients from unique mobiles
    recipients = [
        {"mobile": mobile, "customer_name": f"Customer {i}"}
        for i, mobile in enumerate(unique_mobiles)
    ]

    # Now add duplicates by repeating some or all mobiles
    duplicate_count = draw(st.integers(min_value=1, max_value=max(1, len(unique_mobiles) * 2)))
    for _ in range(duplicate_count):
        mobile_to_dup = draw(st.sampled_from(unique_mobiles))
        recipients.append({"mobile": mobile_to_dup, "customer_name": "Duplicate"})

    # Shuffle the list to randomize order
    shuffled = draw(st.permutations(recipients))
    return list(shuffled), unique_mobiles


@st.composite
def recipients_all_unique(draw):
    """Generate a recipient list with all unique mobile numbers."""
    count = draw(st.integers(min_value=1, max_value=20))
    mobiles = draw(
        st.lists(
            mobile_number_strategy,
            min_size=count,
            max_size=count,
            unique=True,
        )
    )
    recipients = [
        {"mobile": mobile, "customer_name": f"Customer {i}"}
        for i, mobile in enumerate(mobiles)
    ]
    return recipients, mobiles


# ---------------------------------------------------------------------------
# Property Tests
# ---------------------------------------------------------------------------

class TestProperty10Idempotency:
    """Property 10: Idempotency prevents duplicate message delivery.

    **Validates: Requirements 4.5, 26.3**
    """

    @given(
        data=recipients_with_duplicates(),
        campaign_id=campaign_id_strategy,
        template_id=template_id_strategy,
    )
    @settings(max_examples=200, deadline=None)
    def test_duplicate_mobiles_produce_exactly_one_record_each(
        self, data, campaign_id, template_id
    ):
        """Property: For any campaign with duplicate mobile numbers in the
        recipient list, exactly one campaign_message record is created per
        unique (campaign_id, mobile, template_id) combination.

        **Validates: Requirements 4.5, 26.3**
        """
        recipients, unique_mobiles = data

        campaign = {"template_id": template_id, "segment_id": 1}
        template = {
            "template_name": "test_template",
            "body_text": "Hello {{1}}",
            "placeholder_mappings": '{"1": "customer_name"}',
        }

        cursor = IdempotencyTrackingCursor(campaign, template)
        conn = IdempotencyTrackingConnection(cursor)

        def get_conn():
            return conn

        queue = SendingQueue(get_connection=get_conn)
        enqueued = queue.enqueue_campaign(campaign_id, recipients)

        # The number of inserted records must equal the number of unique mobiles
        assert len(cursor.inserted_records) == len(unique_mobiles), (
            f"Expected {len(unique_mobiles)} records but got "
            f"{len(cursor.inserted_records)}. Recipients had {len(recipients)} "
            f"entries with {len(unique_mobiles)} unique mobiles."
        )

        # Enqueued count returned must match inserted records
        assert enqueued == len(unique_mobiles), (
            f"enqueue_campaign returned {enqueued}, expected {len(unique_mobiles)}"
        )

        # Verify each unique mobile has exactly one record
        inserted_mobiles = [r["customer_mobile"] for r in cursor.inserted_records]
        for mobile in unique_mobiles:
            count = inserted_mobiles.count(mobile)
            assert count == 1, (
                f"Mobile {mobile} has {count} records, expected exactly 1"
            )

    @given(
        data=recipients_with_duplicates(),
        campaign_id=campaign_id_strategy,
        template_id=template_id_strategy,
    )
    @settings(max_examples=200, deadline=None)
    def test_idempotency_key_format_is_deterministic(
        self, data, campaign_id, template_id
    ):
        """Property: The idempotency_key for each record follows the deterministic
        format '{campaign_id}_{mobile}_{template_id}', ensuring the same combination
        always produces the same key.

        **Validates: Requirements 4.5, 26.3**
        """
        recipients, unique_mobiles = data

        campaign = {"template_id": template_id, "segment_id": 1}
        template = {
            "template_name": "test_template",
            "body_text": "Hello {{1}}",
            "placeholder_mappings": '{"1": "customer_name"}',
        }

        cursor = IdempotencyTrackingCursor(campaign, template)
        conn = IdempotencyTrackingConnection(cursor)

        def get_conn():
            return conn

        queue = SendingQueue(get_connection=get_conn)
        queue.enqueue_campaign(campaign_id, recipients)

        # Verify each inserted record has the correct idempotency_key format
        for record in cursor.inserted_records:
            expected_key = f"{campaign_id}_{record['customer_mobile']}_{template_id}"
            assert record["idempotency_key"] == expected_key, (
                f"Expected idempotency_key '{expected_key}', "
                f"got '{record['idempotency_key']}'"
            )

    @given(
        data=recipients_all_unique(),
        campaign_id=campaign_id_strategy,
        template_id=template_id_strategy,
    )
    @settings(max_examples=200, deadline=None)
    def test_unique_recipients_all_enqueued(
        self, data, campaign_id, template_id
    ):
        """Property: When all recipients have unique mobiles, the enqueue count
        equals the total recipient count (no duplicates rejected).

        **Validates: Requirements 4.5, 26.3**
        """
        recipients, unique_mobiles = data

        campaign = {"template_id": template_id, "segment_id": 1}
        template = {
            "template_name": "test_template",
            "body_text": "Hello {{1}}",
            "placeholder_mappings": '{"1": "customer_name"}',
        }

        cursor = IdempotencyTrackingCursor(campaign, template)
        conn = IdempotencyTrackingConnection(cursor)

        def get_conn():
            return conn

        queue = SendingQueue(get_connection=get_conn)
        enqueued = queue.enqueue_campaign(campaign_id, recipients)

        # All unique recipients should be enqueued
        assert enqueued == len(unique_mobiles), (
            f"Expected all {len(unique_mobiles)} unique recipients to be "
            f"enqueued, but got {enqueued}"
        )

        # No duplicates means attempted inserts == inserted records
        assert len(cursor.attempted_inserts) == len(cursor.inserted_records), (
            f"With unique recipients, attempted ({len(cursor.attempted_inserts)}) "
            f"should equal inserted ({len(cursor.inserted_records)})"
        )

    @given(
        data=recipients_with_duplicates(),
        campaign_id=campaign_id_strategy,
        template_id=template_id_strategy,
    )
    @settings(max_examples=200, deadline=None)
    def test_repeated_enqueue_same_campaign_remains_idempotent(
        self, data, campaign_id, template_id
    ):
        """Property: Calling enqueue_campaign multiple times with the same
        recipients for the same campaign produces no additional records
        (simulating recovery re-processing).

        **Validates: Requirements 4.5, 26.3**
        """
        recipients, unique_mobiles = data

        campaign = {"template_id": template_id, "segment_id": 1}
        template = {
            "template_name": "test_template",
            "body_text": "Hello {{1}}",
            "placeholder_mappings": '{"1": "customer_name"}',
        }

        cursor = IdempotencyTrackingCursor(campaign, template)
        conn = IdempotencyTrackingConnection(cursor)

        def get_conn():
            return conn

        queue = SendingQueue(get_connection=get_conn)

        # First enqueue
        first_enqueued = queue.enqueue_campaign(campaign_id, recipients)
        first_record_count = len(cursor.inserted_records)

        # Second enqueue with same data (simulating recovery re-processing)
        second_enqueued = queue.enqueue_campaign(campaign_id, recipients)
        second_record_count = len(cursor.inserted_records)

        # No new records should have been inserted on second call
        assert second_record_count == first_record_count, (
            f"Second enqueue produced {second_record_count - first_record_count} "
            f"new records; expected 0 (idempotency violation)"
        )

        # Second call should return 0 enqueued
        assert second_enqueued == 0, (
            f"Second enqueue returned {second_enqueued}, expected 0 "
            f"(all records already exist)"
        )

        # Total unique records must equal unique mobiles
        assert first_record_count == len(unique_mobiles), (
            f"Total records {first_record_count} should equal "
            f"unique mobiles {len(unique_mobiles)}"
        )
