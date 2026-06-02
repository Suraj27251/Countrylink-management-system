"""
Property-based tests for stale message recovery reset.

Property 30: Stale message recovery reset
- For any campaign_message record with status "sending" and updated_at older
  than 5 minutes from current time, the Recovery_Manager SHALL reset its status
  to "queued" for re-dispatch, ensuring no message remains permanently stuck
  in "sending" state.

**Validates: Requirements 26.6**

Testing framework: Hypothesis (Python)
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from services.recovery_manager import RecoveryManager


# ---------------------------------------------------------------------------
# Mock infrastructure simulating MySQL queries for stale message detection
# ---------------------------------------------------------------------------


class StaleRecoveryCursor:
    """Mock cursor that simulates campaign_messages table with stale detection."""

    def __init__(self, messages: list, stale_minutes: int = 5):
        """
        Parameters
        ----------
        messages : list of dict
            Each dict has: id, status, updated_at, idempotency_key, campaign_id
        stale_minutes : int
            The threshold used in the WHERE clause
        """
        self._messages = {m["id"]: dict(m) for m in messages}
        self._stale_minutes = stale_minutes
        self._last_result = None
        self._last_results = []
        self.rowcount = 0

    def execute(self, sql, params=None):
        sql_stripped = sql.strip()

        if "SELECT id" in sql_stripped and "status = 'sending'" in sql_stripped and "DATE_SUB" in sql_stripped:
            # identify_stale_messages query
            stale_minutes = params[0] if params else self._stale_minutes
            now = datetime.now()
            threshold = now - timedelta(minutes=stale_minutes)
            stale = [
                {"id": m["id"]}
                for m in self._messages.values()
                if m["status"] == "sending" and m["updated_at"] < threshold
            ]
            self._last_results = stale
            self.rowcount = len(stale)

        elif "SELECT id, idempotency_key, campaign_id" in sql_stripped and "WHERE id IN" in sql_stripped:
            # deduplicate_and_requeue: fetch messages by IDs
            ids = list(params) if params else []
            results = []
            for msg_id in ids:
                if msg_id in self._messages:
                    m = self._messages[msg_id]
                    results.append({
                        "id": m["id"],
                        "idempotency_key": m["idempotency_key"],
                        "campaign_id": m["campaign_id"],
                    })
            self._last_results = results
            self.rowcount = len(results)

        elif "SELECT id FROM campaign_messages" in sql_stripped and "status IN ('sent', 'delivered', 'read')" in sql_stripped:
            # deduplicate_and_requeue: check if another record with same
            # idempotency_key already succeeded
            # For stale recovery tests, we assume no duplicates exist
            self._last_result = None
            self.rowcount = 0

        elif "UPDATE campaign_messages SET status = 'queued'" in sql_stripped:
            # Reset to queued
            if params:
                msg_id = params[0]
                if msg_id in self._messages:
                    self._messages[msg_id]["status"] = "queued"
            self.rowcount = 1

        elif "UPDATE campaign_messages SET status = 'skipped'" in sql_stripped:
            # Skip duplicates
            if params:
                msg_id = params[0]
                if msg_id in self._messages:
                    self._messages[msg_id]["status"] = "skipped"
            self.rowcount = 1

        else:
            self._last_result = None
            self._last_results = []
            self.rowcount = 0

    def fetchone(self):
        return self._last_result

    def fetchall(self):
        return self._last_results

    def close(self):
        pass


class StaleRecoveryConnection:
    """Mock connection using StaleRecoveryCursor."""

    def __init__(self, cursor: StaleRecoveryCursor):
        self._cursor = cursor
        self.committed = False

    def cursor(self, dictionary=False):
        return self._cursor

    def commit(self):
        self.committed = True

    def rollback(self):
        pass

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Hypothesis Strategies
# ---------------------------------------------------------------------------

# Generate message IDs
message_id_strategy = st.integers(min_value=1, max_value=100000)

# Generate campaign IDs
campaign_id_strategy = st.integers(min_value=1, max_value=10000)

# Generate stale durations (minutes older than the 5-minute threshold)
stale_minutes_extra = st.integers(min_value=1, max_value=1440)  # 1 min to 24 hours extra

# Generate fresh durations (minutes within the threshold - NOT stale)
fresh_minutes = st.integers(min_value=0, max_value=4)  # 0 to 4 minutes (within 5-min threshold)


@st.composite
def stale_sending_messages(draw):
    """Generate a list of campaign_messages that are in 'sending' state and
    have updated_at older than 5 minutes ago (stale)."""
    count = draw(st.integers(min_value=1, max_value=20))
    now = datetime.now()

    messages = []
    used_ids = set()
    for i in range(count):
        msg_id = draw(message_id_strategy.filter(lambda x: x not in used_ids))
        used_ids.add(msg_id)
        campaign_id = draw(campaign_id_strategy)
        extra_minutes = draw(stale_minutes_extra)
        updated_at = now - timedelta(minutes=5 + extra_minutes)

        messages.append({
            "id": msg_id,
            "status": "sending",
            "updated_at": updated_at,
            "idempotency_key": f"{campaign_id}_{msg_id}_tmpl",
            "campaign_id": campaign_id,
        })

    return messages


@st.composite
def fresh_sending_messages(draw):
    """Generate messages in 'sending' state with updated_at within the last
    5 minutes (NOT stale)."""
    count = draw(st.integers(min_value=1, max_value=20))
    now = datetime.now()

    messages = []
    used_ids = set()
    for i in range(count):
        msg_id = draw(message_id_strategy.filter(lambda x: x not in used_ids))
        used_ids.add(msg_id)
        campaign_id = draw(campaign_id_strategy)
        minutes_ago = draw(fresh_minutes)
        updated_at = now - timedelta(minutes=minutes_ago)

        messages.append({
            "id": msg_id,
            "status": "sending",
            "updated_at": updated_at,
            "idempotency_key": f"{campaign_id}_{msg_id}_tmpl",
            "campaign_id": campaign_id,
        })

    return messages


@st.composite
def mixed_messages(draw):
    """Generate a mix of stale sending, fresh sending, and other statuses."""
    now = datetime.now()
    used_ids = set()

    # Stale sending messages
    stale_count = draw(st.integers(min_value=1, max_value=10))
    stale_msgs = []
    for _ in range(stale_count):
        msg_id = draw(message_id_strategy.filter(lambda x: x not in used_ids))
        used_ids.add(msg_id)
        campaign_id = draw(campaign_id_strategy)
        extra_minutes = draw(stale_minutes_extra)
        updated_at = now - timedelta(minutes=5 + extra_minutes)
        stale_msgs.append({
            "id": msg_id,
            "status": "sending",
            "updated_at": updated_at,
            "idempotency_key": f"{campaign_id}_{msg_id}_tmpl",
            "campaign_id": campaign_id,
        })

    # Fresh sending messages (should NOT be reset)
    fresh_count = draw(st.integers(min_value=0, max_value=10))
    fresh_msgs = []
    for _ in range(fresh_count):
        msg_id = draw(message_id_strategy.filter(lambda x: x not in used_ids))
        used_ids.add(msg_id)
        campaign_id = draw(campaign_id_strategy)
        minutes_ago = draw(fresh_minutes)
        updated_at = now - timedelta(minutes=minutes_ago)
        fresh_msgs.append({
            "id": msg_id,
            "status": "sending",
            "updated_at": updated_at,
            "idempotency_key": f"{campaign_id}_{msg_id}_tmpl",
            "campaign_id": campaign_id,
        })

    # Messages in other statuses (should NOT be touched)
    other_count = draw(st.integers(min_value=0, max_value=10))
    other_msgs = []
    other_statuses = ["queued", "sent", "delivered", "read", "failed", "skipped"]
    for _ in range(other_count):
        msg_id = draw(message_id_strategy.filter(lambda x: x not in used_ids))
        used_ids.add(msg_id)
        campaign_id = draw(campaign_id_strategy)
        status = draw(st.sampled_from(other_statuses))
        # Could be any age
        minutes_ago = draw(st.integers(min_value=0, max_value=1440))
        updated_at = now - timedelta(minutes=minutes_ago)
        other_msgs.append({
            "id": msg_id,
            "status": status,
            "updated_at": updated_at,
            "idempotency_key": f"{campaign_id}_{msg_id}_tmpl",
            "campaign_id": campaign_id,
        })

    all_msgs = stale_msgs + fresh_msgs + other_msgs
    return all_msgs, stale_msgs, fresh_msgs, other_msgs


# ---------------------------------------------------------------------------
# Property Tests
# ---------------------------------------------------------------------------


class TestProperty30StaleMessageRecovery:
    """Property 30: Stale message recovery reset.

    **Validates: Requirements 26.6**
    """

    @given(messages=stale_sending_messages())
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_all_stale_sending_messages_are_identified(self, messages):
        """Property: For any set of campaign_messages in 'sending' state with
        updated_at older than 5 minutes, identify_stale_messages SHALL return
        all of their IDs.

        **Validates: Requirements 26.6**
        """
        cursor = StaleRecoveryCursor(messages, stale_minutes=5)
        conn = StaleRecoveryConnection(cursor)

        def get_conn():
            return conn

        manager = RecoveryManager(get_connection=get_conn, stale_minutes=5)
        stale_ids = manager.identify_stale_messages(stale_minutes=5)

        # All messages are stale sending — all should be identified
        expected_ids = sorted([m["id"] for m in messages])
        actual_ids = sorted(stale_ids)

        assert actual_ids == expected_ids, (
            f"Expected all {len(expected_ids)} stale messages to be identified, "
            f"but got {len(actual_ids)}. Missing: {set(expected_ids) - set(actual_ids)}"
        )

    @given(messages=fresh_sending_messages())
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_fresh_sending_messages_not_identified_as_stale(self, messages):
        """Property: Messages in 'sending' state with updated_at within the last
        5 minutes SHALL NOT be identified as stale.

        **Validates: Requirements 26.6**
        """
        cursor = StaleRecoveryCursor(messages, stale_minutes=5)
        conn = StaleRecoveryConnection(cursor)

        def get_conn():
            return conn

        manager = RecoveryManager(get_connection=get_conn, stale_minutes=5)
        stale_ids = manager.identify_stale_messages(stale_minutes=5)

        # None of the fresh messages should be identified as stale
        assert stale_ids == [], (
            f"Expected 0 stale messages (all are fresh), but got {len(stale_ids)}: "
            f"{stale_ids}"
        )

    @given(data=mixed_messages())
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_only_stale_sending_messages_identified_in_mixed_set(self, data):
        """Property: In a mixed set of messages (stale sending, fresh sending,
        and other statuses), only the stale sending messages SHALL be identified.

        **Validates: Requirements 26.6**
        """
        all_msgs, stale_msgs, fresh_msgs, other_msgs = data

        cursor = StaleRecoveryCursor(all_msgs, stale_minutes=5)
        conn = StaleRecoveryConnection(cursor)

        def get_conn():
            return conn

        manager = RecoveryManager(get_connection=get_conn, stale_minutes=5)
        stale_ids = manager.identify_stale_messages(stale_minutes=5)

        expected_stale_ids = sorted([m["id"] for m in stale_msgs])
        actual_stale_ids = sorted(stale_ids)

        assert actual_stale_ids == expected_stale_ids, (
            f"Expected {len(expected_stale_ids)} stale messages, got "
            f"{len(actual_stale_ids)}. "
            f"Extra: {set(actual_stale_ids) - set(expected_stale_ids)}, "
            f"Missing: {set(expected_stale_ids) - set(actual_stale_ids)}"
        )

    @given(messages=stale_sending_messages())
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_stale_messages_reset_to_queued(self, messages):
        """Property: For any stale messages identified, deduplicate_and_requeue
        SHALL reset their status to 'queued' for re-dispatch.

        **Validates: Requirements 26.6**
        """
        cursor = StaleRecoveryCursor(messages, stale_minutes=5)
        conn = StaleRecoveryConnection(cursor)

        def get_conn():
            return conn

        manager = RecoveryManager(get_connection=get_conn, stale_minutes=5)

        # Identify stale messages
        stale_ids = manager.identify_stale_messages(stale_minutes=5)
        assume(len(stale_ids) > 0)

        # Create a fresh cursor/connection for the requeue operation
        cursor2 = StaleRecoveryCursor(messages, stale_minutes=5)
        conn2 = StaleRecoveryConnection(cursor2)

        def get_conn2():
            return conn2

        manager2 = RecoveryManager(get_connection=get_conn2, stale_minutes=5)
        requeued_count = manager2.deduplicate_and_requeue(stale_ids)

        # All stale messages should be requeued (no duplicates in our test setup)
        assert requeued_count == len(stale_ids), (
            f"Expected {len(stale_ids)} messages to be requeued, "
            f"got {requeued_count}"
        )

        # Verify all messages are now in 'queued' status
        for msg_id in stale_ids:
            assert cursor2._messages[msg_id]["status"] == "queued", (
                f"Message {msg_id} should have status 'queued' after recovery, "
                f"but has '{cursor2._messages[msg_id]['status']}'"
            )

    @given(data=mixed_messages())
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_no_message_stuck_in_sending_after_recovery(self, data):
        """Property: After recovery, no message that was stale (in 'sending'
        for >5 minutes) SHALL remain in 'sending' state — it must be reset
        to 'queued'.

        **Validates: Requirements 26.6**
        """
        all_msgs, stale_msgs, fresh_msgs, other_msgs = data
        assume(len(stale_msgs) > 0)

        # Step 1: Identify stale messages
        cursor1 = StaleRecoveryCursor(all_msgs, stale_minutes=5)
        conn1 = StaleRecoveryConnection(cursor1)

        def get_conn1():
            return conn1

        manager1 = RecoveryManager(get_connection=get_conn1, stale_minutes=5)
        stale_ids = manager1.identify_stale_messages(stale_minutes=5)

        # Step 2: Reset stale messages
        cursor2 = StaleRecoveryCursor(all_msgs, stale_minutes=5)
        conn2 = StaleRecoveryConnection(cursor2)

        def get_conn2():
            return conn2

        manager2 = RecoveryManager(get_connection=get_conn2, stale_minutes=5)
        manager2.deduplicate_and_requeue(stale_ids)

        # Verify: no stale message remains in 'sending' state
        for msg in stale_msgs:
            msg_state = cursor2._messages[msg["id"]]["status"]
            assert msg_state == "queued", (
                f"Stale message {msg['id']} still has status '{msg_state}' "
                f"after recovery — should be 'queued'"
            )

        # Verify: fresh sending messages are NOT affected
        for msg in fresh_msgs:
            msg_state = cursor2._messages[msg["id"]]["status"]
            assert msg_state == "sending", (
                f"Fresh message {msg['id']} had its status changed to "
                f"'{msg_state}' — should remain 'sending'"
            )

        # Verify: other-status messages are NOT affected
        for msg in other_msgs:
            msg_state = cursor2._messages[msg["id"]]["status"]
            assert msg_state == msg["status"], (
                f"Message {msg['id']} (status='{msg['status']}') was changed "
                f"to '{msg_state}' — should be untouched"
            )
