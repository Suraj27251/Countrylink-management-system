"""
Property-based tests for webhook status updates applying to correct records.

Property 11: Webhook status updates apply to correct records
- For any valid webhook status callback containing a whatsapp_message_id and a
  status (delivered, read, failed), the system SHALL update exactly the
  campaign_message record matching that whatsapp_message_id, and the new status
  timestamp SHALL be recorded.

**Validates: Requirements 5.2**

Testing framework: Hypothesis (Python)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from services.delivery_tracker import DeliveryTracker


# ---------------------------------------------------------------------------
# Mock infrastructure simulating MySQL for delivery tracker
# ---------------------------------------------------------------------------

class WebhookStatusCursor:
    """Mock cursor that simulates campaign_messages table lookups and updates."""

    def __init__(self, messages: list):
        """
        Args:
            messages: list of dicts representing campaign_messages rows, each with:
                id, campaign_id, whatsapp_message_id, status (current), customer_mobile
        """
        self._messages = {m["whatsapp_message_id"]: dict(m) for m in messages}
        self._last_result = None
        self._updates = []  # Track all UPDATE operations
        self.rowcount = 0

    def execute(self, sql, params=None):
        sql_lower = sql.strip().lower()

        if "select id, campaign_id, status as current_status, customer_mobile" in sql_lower:
            # Lookup campaign message by whatsapp_message_id
            whatsapp_msg_id = params[0] if params else None
            if whatsapp_msg_id and whatsapp_msg_id in self._messages:
                msg = self._messages[whatsapp_msg_id]
                self._last_result = {
                    "id": msg["id"],
                    "campaign_id": msg["campaign_id"],
                    "current_status": msg["status"],
                    "customer_mobile": msg["customer_mobile"],
                }
            else:
                self._last_result = None
            self.rowcount = 1 if self._last_result else 0

        elif "select id from campaign_messages" in sql_lower:
            # is_campaign_message check
            whatsapp_msg_id = params[0] if params else None
            if whatsapp_msg_id and whatsapp_msg_id in self._messages:
                self._last_result = {"id": self._messages[whatsapp_msg_id]["id"]}
            else:
                self._last_result = None

        elif "update campaign_messages" in sql_lower:
            # Track the UPDATE and apply it to our in-memory store
            self._updates.append({"sql": sql, "params": params})
            # Find which message is being updated (last param is the message id)
            message_id = params[-1] if params else None
            if message_id:
                for msg in self._messages.values():
                    if msg["id"] == message_id:
                        # Determine new status from SQL
                        if "status = 'delivered'" in sql_lower:
                            msg["status"] = "delivered"
                            msg["delivered_at"] = params[0]
                        elif "status = 'read'" in sql_lower:
                            msg["status"] = "read"
                            msg["read_at"] = params[0]
                        elif "status = 'sent'" in sql_lower:
                            msg["status"] = "sent"
                            msg["sent_at"] = params[0]
                        elif "status = 'failed'" in sql_lower:
                            msg["status"] = "failed"
                            msg["failed_at"] = params[0]
                            msg["error_code"] = params[1]
                            msg["error_message"] = params[2]
                        break
            self.rowcount = 1

        elif "select" in sql_lower and "sum(" in sql_lower:
            # _update_campaign_counts query
            self._last_result = {
                "sent_count": 0,
                "delivered_count": 0,
                "read_count": 0,
                "failed_count": 0,
            }

        elif "update campaigns" in sql_lower:
            # _update_campaign_counts update
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

    @property
    def updates(self):
        return self._updates

    def get_message_state(self, whatsapp_message_id):
        """Get the current state of a message after updates."""
        return self._messages.get(whatsapp_message_id)


class WebhookStatusConnection:
    """Mock connection for webhook status tracking tests."""

    def __init__(self, cursor: WebhookStatusCursor):
        self._cursor = cursor
        self.committed = False
        self.rolled_back = False
        self._connected = True

    def cursor(self, dictionary=False):
        return self._cursor

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True

    def close(self):
        self._connected = False

    def is_connected(self):
        return self._connected


# ---------------------------------------------------------------------------
# Hypothesis Strategies
# ---------------------------------------------------------------------------

# WhatsApp message IDs are typically alphanumeric strings
whatsapp_message_id_strategy = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789_",
    min_size=10,
    max_size=40,
).filter(lambda s: s.strip() != "")

# Valid statuses for webhook callbacks
webhook_status_strategy = st.sampled_from(["delivered", "read", "failed"])

# Unix timestamps (recent, within reasonable range)
timestamp_strategy = st.integers(min_value=1600000000, max_value=1800000000).map(str)

# Error codes for failed messages
error_code_strategy = st.sampled_from([131047, 131026, 131056, 470, 500, 131051])

# Error messages
error_message_strategy = st.text(min_size=5, max_size=100, alphabet=st.characters(
    whitelist_categories=("L", "N", "P", "Z"),
))

# Campaign IDs
campaign_id_strategy = st.integers(min_value=1, max_value=10000)

# Message record IDs
message_id_strategy = st.integers(min_value=1, max_value=100000)

# Mobile numbers
mobile_strategy = st.text(alphabet="0123456789", min_size=10, max_size=13)

# Initial status that allows transition to delivered/read/failed
valid_prior_status_strategy = st.sampled_from(["sent", "delivered", "sending"])


@st.composite
def campaign_message_record(draw, whatsapp_msg_id=None, status=None):
    """Generate a single campaign_message record."""
    return {
        "id": draw(message_id_strategy),
        "campaign_id": draw(campaign_id_strategy),
        "whatsapp_message_id": whatsapp_msg_id or draw(whatsapp_message_id_strategy),
        "status": status or draw(valid_prior_status_strategy),
        "customer_mobile": draw(mobile_strategy),
    }


@st.composite
def multiple_campaign_messages(draw):
    """Generate multiple campaign_message records with unique whatsapp_message_ids."""
    count = draw(st.integers(min_value=2, max_value=10))
    msg_ids = draw(
        st.lists(
            whatsapp_message_id_strategy,
            min_size=count,
            max_size=count,
            unique=True,
        )
    )
    messages = []
    for i, msg_id in enumerate(msg_ids):
        messages.append({
            "id": i + 1,
            "campaign_id": draw(campaign_id_strategy),
            "whatsapp_message_id": msg_id,
            "status": draw(valid_prior_status_strategy),
            "customer_mobile": draw(mobile_strategy),
        })
    return messages


# ---------------------------------------------------------------------------
# Property Tests
# ---------------------------------------------------------------------------

class TestProperty11WebhookStatusUpdates:
    """Property 11: Webhook status updates apply to correct records.

    **Validates: Requirements 5.2**
    """

    @given(
        messages=multiple_campaign_messages(),
        target_index=st.integers(min_value=0, max_value=9),
        new_status=st.sampled_from(["delivered", "read"]),
        timestamp=timestamp_strategy,
    )
    @settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_webhook_updates_exactly_matching_record(
        self, messages, target_index, new_status, timestamp
    ):
        """Property: A webhook status update with a whatsapp_message_id updates
        exactly the campaign_message record matching that ID, and no other records
        are modified.

        **Validates: Requirements 5.2**
        """
        # Ensure target_index is within bounds
        target_index = target_index % len(messages)
        target_message = messages[target_index]

        # Ensure the target message has a status that allows transition
        target_message["status"] = "sent"

        # Take snapshot of other messages' statuses before update
        other_statuses_before = {
            m["whatsapp_message_id"]: m["status"]
            for m in messages
            if m["whatsapp_message_id"] != target_message["whatsapp_message_id"]
        }

        cursor = WebhookStatusCursor(messages)
        conn = WebhookStatusConnection(cursor)

        def get_conn():
            return WebhookStatusConnection(cursor)

        tracker = DeliveryTracker(get_connection=get_conn)
        result = tracker.process_status_update(
            whatsapp_message_id=target_message["whatsapp_message_id"],
            status=new_status,
            timestamp=timestamp,
        )

        # The update should succeed
        assert result is True, (
            f"process_status_update returned False for valid update: "
            f"msg_id={target_message['whatsapp_message_id']}, status={new_status}"
        )

        # Verify the target message was updated
        updated_msg = cursor.get_message_state(target_message["whatsapp_message_id"])
        assert updated_msg["status"] == new_status, (
            f"Expected status '{new_status}' but got '{updated_msg['status']}' "
            f"for whatsapp_message_id={target_message['whatsapp_message_id']}"
        )

        # Verify no other messages were modified
        for msg_id, original_status in other_statuses_before.items():
            current_msg = cursor.get_message_state(msg_id)
            assert current_msg["status"] == original_status, (
                f"Non-target message {msg_id} was modified: "
                f"expected status '{original_status}', got '{current_msg['status']}'"
            )

    @given(
        messages=multiple_campaign_messages(),
        target_index=st.integers(min_value=0, max_value=9),
        timestamp=timestamp_strategy,
    )
    @settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_webhook_status_timestamp_is_recorded(
        self, messages, target_index, timestamp
    ):
        """Property: When a webhook status update is processed, the corresponding
        timestamp field (delivered_at, read_at, or failed_at) is recorded on the
        matching record.

        **Validates: Requirements 5.2**
        """
        target_index = target_index % len(messages)
        target_message = messages[target_index]

        # Ensure the target allows delivered transition
        target_message["status"] = "sent"

        cursor = WebhookStatusCursor(messages)

        def get_conn():
            return WebhookStatusConnection(cursor)

        tracker = DeliveryTracker(get_connection=get_conn)
        result = tracker.process_status_update(
            whatsapp_message_id=target_message["whatsapp_message_id"],
            status="delivered",
            timestamp=timestamp,
        )

        assert result is True

        # Verify the timestamp was recorded
        updated_msg = cursor.get_message_state(target_message["whatsapp_message_id"])
        assert "delivered_at" in updated_msg, (
            f"delivered_at not set on message after 'delivered' status update"
        )
        assert updated_msg["delivered_at"] is not None, (
            f"delivered_at is None after 'delivered' status update"
        )

    @given(
        messages=multiple_campaign_messages(),
        timestamp=timestamp_strategy,
        error_code=error_code_strategy,
        error_message=error_message_strategy,
    )
    @settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_failed_status_records_error_details(
        self, messages, timestamp, error_code, error_message
    ):
        """Property: When a webhook delivers a 'failed' status, the error_code and
        error_message are recorded alongside the failed_at timestamp on exactly the
        matching record.

        **Validates: Requirements 5.2**
        """
        # Pick the first message as target and set status to allow failure
        target_message = messages[0]
        target_message["status"] = "sent"

        cursor = WebhookStatusCursor(messages)

        def get_conn():
            return WebhookStatusConnection(cursor)

        tracker = DeliveryTracker(get_connection=get_conn)
        result = tracker.process_status_update(
            whatsapp_message_id=target_message["whatsapp_message_id"],
            status="failed",
            timestamp=timestamp,
            error_code=error_code,
            error_message=error_message,
        )

        assert result is True

        # Verify error details were recorded on the correct message
        updated_msg = cursor.get_message_state(target_message["whatsapp_message_id"])
        assert updated_msg["status"] == "failed", (
            f"Expected status 'failed' but got '{updated_msg['status']}'"
        )
        assert updated_msg.get("failed_at") is not None, (
            "failed_at timestamp was not recorded"
        )
        assert updated_msg.get("error_code") == error_code, (
            f"Expected error_code {error_code}, got {updated_msg.get('error_code')}"
        )
        assert updated_msg.get("error_message") == error_message, (
            f"Expected error_message '{error_message}', "
            f"got '{updated_msg.get('error_message')}'"
        )

        # Verify other messages were NOT modified
        for msg in messages[1:]:
            other_msg = cursor.get_message_state(msg["whatsapp_message_id"])
            assert other_msg.get("error_code") is None or other_msg["whatsapp_message_id"] == target_message["whatsapp_message_id"], (
                f"Non-target message {msg['whatsapp_message_id']} had error_code set"
            )

    @given(
        nonexistent_msg_id=whatsapp_message_id_strategy,
        messages=multiple_campaign_messages(),
        new_status=webhook_status_strategy,
        timestamp=timestamp_strategy,
    )
    @settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_nonexistent_message_id_does_not_modify_any_record(
        self, nonexistent_msg_id, messages, new_status, timestamp
    ):
        """Property: When a webhook arrives with a whatsapp_message_id that does not
        match any campaign_message record, no records are modified and the function
        returns False.

        **Validates: Requirements 5.2**
        """
        # Ensure the nonexistent ID is truly not in our messages
        existing_ids = {m["whatsapp_message_id"] for m in messages}
        assume(nonexistent_msg_id not in existing_ids)

        # Take snapshot of all statuses
        statuses_before = {
            m["whatsapp_message_id"]: m["status"] for m in messages
        }

        cursor = WebhookStatusCursor(messages)

        def get_conn():
            return WebhookStatusConnection(cursor)

        tracker = DeliveryTracker(get_connection=get_conn)
        result = tracker.process_status_update(
            whatsapp_message_id=nonexistent_msg_id,
            status=new_status,
            timestamp=timestamp,
        )

        # Should return False — no matching record
        assert result is False, (
            f"process_status_update returned True for non-existent "
            f"whatsapp_message_id={nonexistent_msg_id}"
        )

        # Verify no records were modified
        for msg_id, original_status in statuses_before.items():
            current_msg = cursor.get_message_state(msg_id)
            assert current_msg["status"] == original_status, (
                f"Message {msg_id} was modified despite non-matching webhook: "
                f"expected '{original_status}', got '{current_msg['status']}'"
            )

        # Verify no UPDATE queries were executed
        assert len(cursor.updates) == 0, (
            f"UPDATE queries were executed for non-existent message: "
            f"{len(cursor.updates)} updates"
        )
