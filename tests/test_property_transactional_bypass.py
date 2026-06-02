"""
Property-based tests for transactional message cooldown bypass using Hypothesis.

**Validates: Requirements 17.6**

Property 21: Transactional messages bypass cooldown
- For any transactional message (payment confirmations, service alerts),
  the CooldownManager SHALL never block delivery regardless of the customer's
  promotional cooldown state or frequency count.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unittest.mock import MagicMock

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from services.cooldown_manager import CooldownManager, CooldownResult


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Mobile number strategy: realistic phone numbers
MOBILE_NUMBERS = st.from_regex(r"91[6-9][0-9]{9}", fullmatch=True)

# Quality tiers that affect cooldown windows
QUALITY_TIERS = st.sampled_from(["green", "yellow", "red"])

# Number of prior promotional messages in the cooldown window (simulates heavy cooldown state)
PRIOR_WINDOW_COUNT = st.integers(min_value=0, max_value=100)

# Number of prior promotional messages in the rolling 7-day period
PRIOR_WEEKLY_COUNT = st.integers(min_value=0, max_value=50)

# Transactional campaign type with various case permutations
TRANSACTIONAL_TYPES = st.sampled_from([
    "transactional",
    "Transactional",
    "TRANSACTIONAL",
    "TransActional",
])

# Non-transactional campaign types (for contrast/negative tests)
NON_TRANSACTIONAL_TYPES = st.sampled_from([
    "promotional",
    "reactivation",
    "ab_test",
])


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

class MockCursor:
    """Mock MySQL cursor that simulates heavy cooldown state."""

    def __init__(self, window_count: int = 5, weekly_count: int = 10):
        self._window_count = window_count
        self._weekly_count = weekly_count
        self._call_idx = 0
        self._closed = False
        self.executed = []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchone(self):
        if self._call_idx == 0:
            self._call_idx += 1
            return {"cnt": self._window_count}
        else:
            self._call_idx += 1
            return {"cnt": self._weekly_count}

    def close(self):
        self._closed = True


class MockConnection:
    """Mock MySQL connection that returns a configured cursor."""

    def __init__(self, cursor_instance):
        self._cursor = cursor_instance

    def cursor(self, dictionary=False):
        return self._cursor

    def commit(self):
        pass


# ---------------------------------------------------------------------------
# Property Tests
# ---------------------------------------------------------------------------

class TestTransactionalBypassProperty:
    """Property-based tests for transactional message cooldown bypass — Property 21."""

    @given(
        mobile=MOBILE_NUMBERS,
        campaign_type=TRANSACTIONAL_TYPES,
        quality_tier=QUALITY_TIERS,
        window_count=PRIOR_WINDOW_COUNT,
        weekly_count=PRIOR_WEEKLY_COUNT,
    )
    @settings(max_examples=500)
    def test_transactional_never_blocked_regardless_of_cooldown_state(
        self,
        mobile: str,
        campaign_type: str,
        quality_tier: str,
        window_count: int,
        weekly_count: int,
    ):
        """
        Property: For any transactional message, the CooldownManager SHALL never
        block delivery regardless of the customer's promotional cooldown state
        (any number of prior messages in window) or frequency count (any number
        of prior messages in 7-day period), under any quality tier.

        **Validates: Requirements 17.6**
        """
        # Set up a mock connection that would block promotional messages
        cursor = MockCursor(window_count=window_count, weekly_count=weekly_count)
        conn = MockConnection(cursor)
        mgr = CooldownManager(lambda: conn, quality_tier=quality_tier)

        result = mgr.check_cooldown(mobile, campaign_type)

        # Transactional messages must ALWAYS be allowed
        assert result.allowed is True, (
            f"Transactional message was blocked! "
            f"mobile={mobile}, type={campaign_type}, tier={quality_tier}, "
            f"window_count={window_count}, weekly_count={weekly_count}"
        )
        assert result.reason is None, (
            f"Transactional message had a block reason: {result.reason}"
        )
        assert result.excluded_count == 0, (
            f"Transactional message had excluded_count={result.excluded_count}"
        )

    @given(
        mobile=MOBILE_NUMBERS,
        campaign_type=TRANSACTIONAL_TYPES,
        quality_tier=QUALITY_TIERS,
    )
    @settings(max_examples=300)
    def test_transactional_never_queries_database(
        self,
        mobile: str,
        campaign_type: str,
        quality_tier: str,
    ):
        """
        Property: For any transactional message, the CooldownManager SHALL
        bypass all cooldown checks entirely and never query the database.
        This ensures transactional messages have zero latency overhead from
        cooldown enforcement.

        **Validates: Requirements 17.6**
        """
        cursor = MockCursor(window_count=99, weekly_count=99)
        conn = MockConnection(cursor)
        mgr = CooldownManager(lambda: conn, quality_tier=quality_tier)

        mgr.check_cooldown(mobile, campaign_type)

        # No database queries should have been executed for transactional
        assert len(cursor.executed) == 0, (
            f"Transactional message triggered {len(cursor.executed)} DB queries "
            f"(expected 0). type={campaign_type}, tier={quality_tier}"
        )

    @given(
        mobile=MOBILE_NUMBERS,
        non_transactional_type=NON_TRANSACTIONAL_TYPES,
        quality_tier=QUALITY_TIERS,
        window_count=st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=300)
    def test_non_transactional_blocked_when_cooldown_active(
        self,
        mobile: str,
        non_transactional_type: str,
        quality_tier: str,
        window_count: int,
    ):
        """
        Property (contrast): For any non-transactional message with an active
        cooldown state (window_count > 0), the CooldownManager SHALL block
        the message. This confirms transactional bypass is specific to
        transactional types only and not a general bypass.

        **Validates: Requirements 17.6**
        """
        cursor = MockCursor(window_count=window_count, weekly_count=0)
        conn = MockConnection(cursor)
        mgr = CooldownManager(lambda: conn, quality_tier=quality_tier)

        result = mgr.check_cooldown(mobile, non_transactional_type)

        # Non-transactional messages with active cooldown MUST be blocked
        assert result.allowed is False, (
            f"Non-transactional message was NOT blocked when cooldown active! "
            f"mobile={mobile}, type={non_transactional_type}, tier={quality_tier}, "
            f"window_count={window_count}"
        )
        assert result.reason == "cooldown_active", (
            f"Expected reason 'cooldown_active', got '{result.reason}'"
        )
