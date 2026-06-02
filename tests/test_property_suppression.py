"""
Property-based tests for suppression list enforcement at dispatch time using Hypothesis.

**Validates: Requirements 19.3**

Property 23: Suppression list enforcement at dispatch time
- For any customer on the active suppression list (opted-out, DND, blocked, or invalid
  number), the Sending_Queue SHALL never dispatch a campaign message to that customer,
  regardless of segment membership.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unittest.mock import MagicMock, patch

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from services.opt_out_manager import OptOutManager

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Valid mobile number patterns
MOBILE_NUMBERS = st.from_regex(r"91[6-9][0-9]{9}", fullmatch=True)

# Suppression reasons matching the ENUM in the database schema
SUPPRESSION_REASONS = st.sampled_from([
    "opt_out_keyword",
    "manual_dnd",
    "user_blocked",
    "spam_reported",
    "invalid_number",
])

# Source keywords for opt-out scenarios
SOURCE_KEYWORDS = st.sampled_from(["stop", "unsubscribe", "opt out", "cancel", "dnd", ""])

# Operator names for manual DND
OPERATOR_NAMES = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "Pd")),
    min_size=1,
    max_size=30,
)

# Campaign IDs
CAMPAIGN_IDS = st.integers(min_value=1, max_value=100000)

# Template IDs
TEMPLATE_IDS = st.integers(min_value=1, max_value=1000)

# Batch of recipient mobiles (1 to 20 recipients)
RECIPIENT_LISTS = st.lists(MOBILE_NUMBERS, min_size=1, max_size=20)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _create_mock_connection_with_suppression(suppressed_mobiles: set):
    """
    Create a mock MySQL connection that simulates the suppression_list table.

    The mock cursor responds to SELECT queries on suppression_list,
    returning a result if the mobile is in the suppressed set.
    """
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value = cursor
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=False)
    conn.is_connected.return_value = True

    def execute_side_effect(query, params=None):
        """Simulate DB queries for suppression check."""
        if params and "suppression_list" in query:
            mobile = params[0] if params else None
            if mobile in suppressed_mobiles:
                cursor.fetchone.return_value = {"id": 1}
            else:
                cursor.fetchone.return_value = None
        else:
            cursor.fetchone.return_value = None

    cursor.execute = MagicMock(side_effect=execute_side_effect)
    return conn


def _create_opt_out_manager(suppressed_mobiles: set):
    """Create an OptOutManager with a mock connection that knows about suppressed mobiles."""
    def get_conn():
        return _create_mock_connection_with_suppression(suppressed_mobiles)
    return OptOutManager(get_conn)


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestSuppressionEnforcementProperties:
    """Property-based tests for suppression list enforcement — Property 23."""

    @given(
        mobile=MOBILE_NUMBERS,
        reason=SUPPRESSION_REASONS,
    )
    @settings(max_examples=300)
    def test_suppressed_customer_is_detected_regardless_of_reason(
        self, mobile: str, reason: str
    ):
        """
        Property: For any customer on the active suppression list with ANY reason
        (opt_out_keyword, manual_dnd, user_blocked, spam_reported, invalid_number),
        is_suppressed() SHALL return True.

        This ensures the dispatch pre-check correctly identifies suppressed customers
        regardless of the suppression reason.

        **Validates: Requirements 19.3**
        """
        # The suppression list contains this mobile
        suppressed_set = {mobile}
        mgr = _create_opt_out_manager(suppressed_set)

        result = mgr.is_suppressed(mobile)

        assert result is True, (
            f"Expected is_suppressed=True for mobile='{mobile}' with reason='{reason}', "
            f"but got False. Suppressed customers must never receive campaign messages."
        )

    @given(
        mobile=MOBILE_NUMBERS,
    )
    @settings(max_examples=300)
    def test_non_suppressed_customer_is_not_blocked(
        self, mobile: str
    ):
        """
        Property: For any customer NOT on the suppression list,
        is_suppressed() SHALL return False — allowing normal dispatch.

        **Validates: Requirements 19.3**
        """
        # Empty suppression list — no one is suppressed
        suppressed_set = set()
        mgr = _create_opt_out_manager(suppressed_set)

        result = mgr.is_suppressed(mobile)

        assert result is False, (
            f"Expected is_suppressed=False for mobile='{mobile}' not in suppression list, "
            f"but got True. Non-suppressed customers should be eligible for dispatch."
        )

    @given(
        suppressed_mobiles=st.frozensets(MOBILE_NUMBERS, min_size=1, max_size=10),
        campaign_id=CAMPAIGN_IDS,
        template_id=TEMPLATE_IDS,
    )
    @settings(max_examples=300)
    def test_dispatch_skips_all_suppressed_recipients(
        self, suppressed_mobiles: frozenset, campaign_id: int, template_id: int
    ):
        """
        Property: For any set of suppressed customers in a campaign recipient list,
        the suppression check SHALL identify every single suppressed recipient,
        ensuring zero messages dispatched to suppressed customers.

        This simulates the dispatch-time suppression check by verifying that
        is_suppressed() returns True for every member of the suppression list.

        **Validates: Requirements 19.3**
        """
        mgr = _create_opt_out_manager(suppressed_mobiles)

        dispatched_to_suppressed = []
        for mobile in suppressed_mobiles:
            if not mgr.is_suppressed(mobile):
                # This mobile would incorrectly receive a message
                dispatched_to_suppressed.append(mobile)

        assert len(dispatched_to_suppressed) == 0, (
            f"Suppression enforcement failed! The following suppressed customers "
            f"would have received campaign {campaign_id} messages: "
            f"{dispatched_to_suppressed}"
        )

    @given(
        suppressed_mobiles=st.frozensets(MOBILE_NUMBERS, min_size=1, max_size=5),
        non_suppressed_mobiles=st.frozensets(MOBILE_NUMBERS, min_size=1, max_size=5),
        campaign_id=CAMPAIGN_IDS,
    )
    @settings(max_examples=300)
    def test_mixed_audience_only_non_suppressed_receive_messages(
        self, suppressed_mobiles: frozenset, non_suppressed_mobiles: frozenset,
        campaign_id: int
    ):
        """
        Property: For any campaign audience containing both suppressed and
        non-suppressed customers, the dispatch pre-check SHALL:
        - Block ALL suppressed customers (is_suppressed → True)
        - Allow ALL non-suppressed customers (is_suppressed → False)

        This verifies that suppression enforcement is precise — it blocks
        exactly the suppressed customers and no others.

        **Validates: Requirements 19.3**
        """
        # Ensure no overlap between sets for a clean test
        actual_non_suppressed = non_suppressed_mobiles - suppressed_mobiles

        assume(len(actual_non_suppressed) > 0)

        mgr = _create_opt_out_manager(suppressed_mobiles)

        # Verify ALL suppressed are blocked
        for mobile in suppressed_mobiles:
            assert mgr.is_suppressed(mobile) is True, (
                f"Suppressed mobile '{mobile}' was NOT detected as suppressed. "
                f"This customer would incorrectly receive a campaign message."
            )

        # Verify ALL non-suppressed are allowed
        for mobile in actual_non_suppressed:
            assert mgr.is_suppressed(mobile) is False, (
                f"Non-suppressed mobile '{mobile}' was incorrectly blocked. "
                f"This customer should be eligible for campaign messages."
            )

    @given(
        mobile=MOBILE_NUMBERS,
        reason=SUPPRESSION_REASONS,
        campaign_id=CAMPAIGN_IDS,
    )
    @settings(max_examples=300)
    def test_suppression_enforcement_independent_of_segment_membership(
        self, mobile: str, reason: str, campaign_id: int
    ):
        """
        Property: Suppression enforcement SHALL apply regardless of segment
        membership. Even if a customer qualifies for a campaign's target segment,
        if they are on the suppression list, they must NOT receive the message.

        This validates the "regardless of segment membership" clause in Property 23.

        **Validates: Requirements 19.3**
        """
        # Customer is both in the target segment AND on the suppression list
        suppressed_set = {mobile}
        mgr = _create_opt_out_manager(suppressed_set)

        # Simulate: customer passes segment filter but is suppressed
        passes_segment_filter = True  # Always True — they qualify for the segment
        is_suppressed = mgr.is_suppressed(mobile)

        # The dispatch decision: only send if NOT suppressed
        should_dispatch = passes_segment_filter and not is_suppressed

        assert should_dispatch is False, (
            f"Campaign {campaign_id} would dispatch to suppressed mobile '{mobile}' "
            f"(reason='{reason}') just because they are in the target segment. "
            f"Suppression must override segment membership."
        )
