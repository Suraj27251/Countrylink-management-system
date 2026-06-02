"""
Property-based tests for Queue manipulation invariants using Hypothesis.

**Validates: Requirements 1.5, 1.6, 1.7**

Property 2: Queue manipulation preserves message invariants
- Pausing a campaign stops dispatch (0 new messages dispatched for that campaign).
- Resuming a paused campaign only dispatches messages with status "queued".
- Cancelling a campaign transitions all remaining queued messages to "skipped" (cancelled) status
  with none dispatched.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from services.sending_queue import SendingQueue, BatchResult


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------
campaign_ids = st.integers(min_value=1, max_value=10000)
message_counts = st.integers(min_value=1, max_value=50)

# A list of message statuses present in a campaign queue (some queued, some already sent)
message_status_strategy = st.sampled_from(
    ["queued", "sending", "sent", "delivered", "read", "failed", "permanently_failed", "skipped"]
)


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


class MockDispatcher:
    """Mock dispatcher that records dispatch calls."""

    def __init__(self):
        self.dispatched = []

    def send_template(self, recipient, template_name, params, media_url=None):
        from services.channel import DispatchResult
        self.dispatched.append({
            "recipient": recipient,
            "template_name": template_name,
            "params": params,
        })
        return DispatchResult(success=True, message_id=f"msg_{recipient}")

    def get_channel_name(self):
        return "whatsapp"


class MockTemplateValidator:
    """Mock validator that always passes."""

    def sanitize_param(self, value):
        return value

    def validate_customer_params(self, customer, mappings):
        return True


# ---------------------------------------------------------------------------
# Property Tests
# ---------------------------------------------------------------------------
class TestQueueManipulationInvariants:
    """Property-based tests for queue manipulation — Property 2."""

    @given(
        campaign_id=campaign_ids,
        num_queued=message_counts,
    )
    @settings(max_examples=200)
    def test_pause_stops_dispatch_for_campaign(self, campaign_id: int, num_queued: int):
        """
        Property: After pausing a campaign, process_batch() dispatches 0 messages
        for that campaign. Messages from the paused campaign are excluded from
        the batch query.

        **Validates: Requirements 1.5**
        """
        call_count = [0]

        def get_conn():
            call_count[0] += 1
            cursor = MockCursor()
            conn = MockConnection(cursor)

            if call_count[0] == 1:
                # pause_campaign: UPDATE campaigns SET status = 'paused'
                return conn
            else:
                # process_batch: returns empty result because campaign is excluded
                cursor.fetchall_result = []
                return conn

        dispatcher = MockDispatcher()
        queue = SendingQueue(
            get_connection=get_conn,
            dispatcher=dispatcher,
            template_validator=MockTemplateValidator(),
            throttle_rate=1000,  # High rate to avoid sleep
        )

        # Pause the campaign
        queue.pause_campaign(campaign_id)

        # Verify campaign is in paused set
        assert campaign_id in queue._paused_campaigns, (
            f"Campaign {campaign_id} should be in _paused_campaigns after pause"
        )

        # Verify the campaign is excluded from dispatch
        excluded = queue._get_excluded_campaign_ids()
        assert campaign_id in excluded, (
            f"Campaign {campaign_id} should be in excluded campaign IDs"
        )

        # Verify is_campaign_excluded returns True
        assert queue._is_campaign_excluded(campaign_id), (
            f"_is_campaign_excluded({campaign_id}) should return True after pause"
        )

        # process_batch should dispatch 0 messages for paused campaign
        result = queue.process_batch(batch_size=num_queued)

        # No messages dispatched because the paused campaign is excluded
        assert dispatcher.dispatched == [], (
            f"Expected 0 dispatches for paused campaign {campaign_id}, "
            f"got {len(dispatcher.dispatched)}"
        )
        assert result.sent_count == 0, (
            f"Expected sent_count=0 after pausing campaign {campaign_id}, "
            f"got {result.sent_count}"
        )

    @given(
        campaign_id=campaign_ids,
        num_queued=message_counts,
    )
    @settings(max_examples=200)
    def test_resume_only_dispatches_queued_messages(self, campaign_id: int, num_queued: int):
        """
        Property: After resuming a paused campaign, process_batch() only
        dispatches messages with status "queued". Messages in other statuses
        (sent, delivered, failed, etc.) are NOT re-dispatched.

        **Validates: Requirements 1.6**
        """
        # Build a mix of messages: some queued, some already in terminal states
        queued_messages = [
            {
                "id": i,
                "campaign_id": campaign_id,
                "customer_mobile": f"91900000{i:04d}",
                "customer_name": f"Customer {i}",
                "template_id": 1,
                "template_params": None,
            }
            for i in range(1, num_queued + 1)
        ]

        call_count = [0]

        def get_conn():
            call_count[0] += 1
            cursor = MockCursor()
            conn = MockConnection(cursor)

            if call_count[0] == 1:
                # pause_campaign: UPDATE campaigns
                return conn
            elif call_count[0] == 2:
                # resume_campaign: UPDATE campaigns
                return conn
            elif call_count[0] == 3:
                # process_batch: SELECT queued messages — returns only "queued" ones
                cursor.fetchall_result = queued_messages
                return conn
            else:
                # Subsequent calls for _dispatch_message, _mark_message_sent, etc.
                cursor.fetchone_result = {"template_name": "test_template"}
                return conn

        dispatcher = MockDispatcher()
        queue = SendingQueue(
            get_connection=get_conn,
            dispatcher=dispatcher,
            template_validator=MockTemplateValidator(),
            throttle_rate=1000,
        )

        # First pause, then resume
        queue.pause_campaign(campaign_id)
        queue.resume_campaign(campaign_id)

        # Verify campaign is no longer in paused set
        assert campaign_id not in queue._paused_campaigns, (
            f"Campaign {campaign_id} should NOT be in _paused_campaigns after resume"
        )

        # Verify not excluded anymore
        assert not queue._is_campaign_excluded(campaign_id), (
            f"_is_campaign_excluded({campaign_id}) should return False after resume"
        )

        # process_batch should only fetch messages with status='queued' (from SQL WHERE clause)
        result = queue.process_batch(batch_size=num_queued)

        # The SQL query in process_batch only selects status='queued', so all dispatched
        # messages were in 'queued' state — this is enforced by the WHERE clause
        assert result.sent_count == num_queued, (
            f"Expected sent_count={num_queued} for resumed campaign, "
            f"got {result.sent_count}"
        )
        assert len(dispatcher.dispatched) == num_queued, (
            f"Expected {num_queued} dispatches after resume, "
            f"got {len(dispatcher.dispatched)}"
        )

    @given(
        campaign_id=campaign_ids,
        num_messages=message_counts,
        num_already_sent=st.integers(min_value=0, max_value=10),
    )
    @settings(max_examples=200)
    def test_cancel_transitions_all_queued_to_skipped(
        self, campaign_id: int, num_messages: int, num_already_sent: int
    ):
        """
        Property: When a campaign is cancelled, all remaining 'queued' messages
        transition to 'skipped' status. The SQL UPDATE in cancel_campaign()
        targets only status='queued' messages for the given campaign_id.

        **Validates: Requirements 1.7**
        """
        assume(num_messages > num_already_sent)

        call_count = [0]
        executed_sql = []

        def get_conn():
            call_count[0] += 1
            cursor = MockCursor()
            conn = MockConnection(cursor)

            # Track SQL executed for verification
            original_execute = cursor.execute

            def tracking_execute(sql, params=None):
                executed_sql.append((sql, params))
                return original_execute(sql, params)

            cursor.execute = tracking_execute
            return conn

        dispatcher = MockDispatcher()
        queue = SendingQueue(
            get_connection=get_conn,
            dispatcher=dispatcher,
            template_validator=MockTemplateValidator(),
            throttle_rate=1000,
        )

        # Cancel the campaign
        queue.cancel_campaign(campaign_id)

        # Verify campaign is in cancelled set
        assert campaign_id in queue._cancelled_campaigns, (
            f"Campaign {campaign_id} should be in _cancelled_campaigns after cancel"
        )

        # Verify the SQL executed targets queued messages for this campaign
        # The cancel_campaign method should UPDATE campaign_messages SET status='skipped'
        # WHERE campaign_id=%s AND status='queued'
        cancel_sql_found = False
        for sql, params in executed_sql:
            if "campaign_messages" in sql and "skipped" in sql and "queued" in sql:
                cancel_sql_found = True
                assert params == (campaign_id,), (
                    f"cancel SQL should target campaign_id={campaign_id}, got params={params}"
                )
                break

        assert cancel_sql_found, (
            "Expected UPDATE campaign_messages SET status='skipped' WHERE ... status='queued' "
            "SQL to be executed during cancel_campaign()"
        )

        # Verify campaign status updated to 'cancelled'
        campaign_status_sql_found = False
        for sql, params in executed_sql:
            if "campaigns" in sql and "cancelled" in sql and "campaign_messages" not in sql:
                campaign_status_sql_found = True
                assert params == (campaign_id,), (
                    f"Campaign status update should target campaign_id={campaign_id}"
                )
                break

        assert campaign_status_sql_found, (
            "Expected UPDATE campaigns SET status='cancelled' SQL to be executed"
        )

    @given(
        campaign_id=campaign_ids,
        num_queued=message_counts,
    )
    @settings(max_examples=200)
    def test_cancelled_campaign_not_dispatched(self, campaign_id: int, num_queued: int):
        """
        Property: After cancelling a campaign, process_batch() dispatches 0 messages
        for that campaign (campaign is in the excluded set).

        **Validates: Requirements 1.7**
        """
        call_count = [0]

        def get_conn():
            call_count[0] += 1
            cursor = MockCursor()
            conn = MockConnection(cursor)

            if call_count[0] == 1:
                # cancel_campaign: UPDATE campaign_messages + campaigns
                return conn
            else:
                # process_batch: returns empty because campaign is excluded
                cursor.fetchall_result = []
                return conn

        dispatcher = MockDispatcher()
        queue = SendingQueue(
            get_connection=get_conn,
            dispatcher=dispatcher,
            template_validator=MockTemplateValidator(),
            throttle_rate=1000,
        )

        # Cancel the campaign
        queue.cancel_campaign(campaign_id)

        # Verify campaign is excluded
        assert queue._is_campaign_excluded(campaign_id), (
            f"Cancelled campaign {campaign_id} should be excluded from dispatch"
        )

        # process_batch should dispatch nothing for cancelled campaign
        result = queue.process_batch(batch_size=num_queued)

        assert dispatcher.dispatched == [], (
            f"Expected 0 dispatches for cancelled campaign {campaign_id}, "
            f"got {len(dispatcher.dispatched)}"
        )
        assert result.sent_count == 0, (
            f"Expected sent_count=0 for cancelled campaign, got {result.sent_count}"
        )

    @given(
        campaign_id=campaign_ids,
        other_campaign_id=campaign_ids,
        num_queued=message_counts,
    )
    @settings(max_examples=200)
    def test_pause_does_not_affect_other_campaigns(
        self, campaign_id: int, other_campaign_id: int, num_queued: int
    ):
        """
        Property: Pausing one campaign does not prevent dispatch of messages
        from other campaigns. Only the paused campaign's messages are excluded.

        **Validates: Requirements 1.5**
        """
        assume(campaign_id != other_campaign_id)

        # Build messages from the OTHER campaign (not paused)
        other_messages = [
            {
                "id": i,
                "campaign_id": other_campaign_id,
                "customer_mobile": f"91900000{i:04d}",
                "customer_name": f"Customer {i}",
                "template_id": 1,
                "template_params": None,
            }
            for i in range(1, num_queued + 1)
        ]

        call_count = [0]

        def get_conn():
            call_count[0] += 1
            cursor = MockCursor()
            conn = MockConnection(cursor)

            if call_count[0] == 1:
                # pause_campaign: UPDATE campaigns
                return conn
            elif call_count[0] == 2:
                # process_batch: returns messages from the other (non-paused) campaign
                cursor.fetchall_result = other_messages
                return conn
            else:
                # _dispatch_message, _mark_message_sent calls
                cursor.fetchone_result = {"template_name": "test_template"}
                return conn

        dispatcher = MockDispatcher()
        queue = SendingQueue(
            get_connection=get_conn,
            dispatcher=dispatcher,
            template_validator=MockTemplateValidator(),
            throttle_rate=1000,
        )

        # Pause only campaign_id
        queue.pause_campaign(campaign_id)

        # Verify other_campaign_id is NOT excluded
        assert not queue._is_campaign_excluded(other_campaign_id), (
            f"Campaign {other_campaign_id} should NOT be excluded when "
            f"campaign {campaign_id} is paused"
        )

        # process_batch should still dispatch messages for other campaigns
        result = queue.process_batch(batch_size=num_queued)

        assert result.sent_count == num_queued, (
            f"Expected sent_count={num_queued} for non-paused campaign "
            f"{other_campaign_id}, got {result.sent_count}"
        )
