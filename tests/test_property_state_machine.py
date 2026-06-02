"""
Property-based tests for Campaign state machine transitions using Hypothesis.

**Validates: Requirements 1.2**

Property 1: Campaign state machine transition validity
- Only valid transitions are allowed and all invalid transitions are rejected with error.
- Terminal states (completed, failed, cancelled) have no valid outgoing transitions.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from blueprints.campaign_bp import CampaignService, VALID_TRANSITIONS


# ---------------------------------------------------------------------------
# All known campaign states (from design state machine)
# ---------------------------------------------------------------------------
ALL_STATES = [
    "draft", "scheduled", "pending_approval", "approved",
    "sending", "paused", "completed", "failed", "cancelled",
]

TERMINAL_STATES = ["completed", "failed", "cancelled"]

# Build list of all valid (from_state, to_state) pairs
VALID_PAIRS = [
    (from_state, to_state)
    for from_state, targets in VALID_TRANSITIONS.items()
    for to_state in targets
]

# Build list of all invalid (from_state, to_state) pairs
INVALID_PAIRS = [
    (from_state, to_state)
    for from_state in ALL_STATES
    for to_state in ALL_STATES
    if to_state not in VALID_TRANSITIONS.get(from_state, set())
]

# Hypothesis strategies
valid_states = st.sampled_from(ALL_STATES)
terminal_states = st.sampled_from(TERMINAL_STATES)
valid_transition_pairs = st.sampled_from(VALID_PAIRS)
invalid_transition_pairs = st.sampled_from(INVALID_PAIRS)


# ---------------------------------------------------------------------------
# Mock helpers (matching existing test_campaign_service.py patterns)
# ---------------------------------------------------------------------------
class MockCursor:
    """Mock MySQL cursor with dictionary=True support."""

    def __init__(self):
        self.executed = []
        self.fetchone_result = None
        self.fetchall_result = []
        self.lastrowid = 1
        self._closed = False

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
class TestStateMachineTransitionValidity:
    """Property-based tests for campaign state machine — Property 1."""

    @given(pair=valid_transition_pairs)
    @settings(max_examples=500)
    def test_valid_transitions_succeed(self, pair: tuple):
        """
        Property: For any valid state and any target state that IS in
        VALID_TRANSITIONS[from_state], transition_state() should succeed
        and return the campaign in the new state.

        **Validates: Requirements 1.2**
        """
        from_state, to_state = pair
        call_count = [0]

        def get_conn():
            call_count[0] += 1
            if call_count[0] == 1:
                # For transition_state (write connection)
                c = MockCursor()
                c.fetchone_result = {"id": 1, "status": from_state}
                return MockConnection(c)
            else:
                # For get_campaign (read after transition)
                c = MockCursor()
                c.fetchone_result = {"id": 1, "status": to_state, "name": "Test"}
                return MockConnection(c)

        service = CampaignService(get_conn)
        result = service.transition_state(1, to_state, "operator_test")
        assert result["status"] == to_state, (
            f"Expected status '{to_state}' after transition from '{from_state}', "
            f"got '{result['status']}'"
        )

    @given(pair=invalid_transition_pairs)
    @settings(max_examples=500)
    def test_invalid_transitions_rejected(self, pair: tuple):
        """
        Property: For any valid state and any target state NOT in
        VALID_TRANSITIONS[from_state], transition_state() should raise ValueError.

        **Validates: Requirements 1.2**
        """
        from_state, to_state = pair

        cursor = MockCursor()
        cursor.fetchone_result = {"id": 1, "status": from_state}
        conn = MockConnection(cursor)
        service = CampaignService(lambda: conn)

        try:
            service.transition_state(1, to_state, "operator_test")
            assert False, (
                f"Transition '{from_state}' → '{to_state}' should have raised ValueError "
                f"but succeeded"
            )
        except ValueError as e:
            assert "Invalid transition" in str(e), (
                f"ValueError message should contain 'Invalid transition', got: {e}"
            )

    @given(terminal_state=terminal_states, to_state=valid_states)
    @settings(max_examples=500)
    def test_terminal_states_have_no_outgoing_transitions(self, terminal_state: str, to_state: str):
        """
        Property: Terminal states (completed, failed, cancelled) have no valid
        outgoing transitions. Any attempt to transition from a terminal state
        should raise ValueError.

        **Validates: Requirements 1.2**
        """
        cursor = MockCursor()
        cursor.fetchone_result = {"id": 1, "status": terminal_state}
        conn = MockConnection(cursor)
        service = CampaignService(lambda: conn)

        try:
            service.transition_state(1, to_state, "operator_test")
            assert False, (
                f"Terminal state '{terminal_state}' should not allow transition to "
                f"'{to_state}', but it succeeded"
            )
        except ValueError:
            pass  # Expected — terminal states reject all transitions
