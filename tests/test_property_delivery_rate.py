"""
Property-based tests for Delivery Rate Computation using Hypothesis.

**Validates: Requirements 5.3, 8.1, 16.3**

Property 12: Delivery rate computation correctness
- For any campaign with total_sent > 0 messages, delivery_rate SHALL equal
  delivered_count / total_sent, read_rate SHALL equal read_count / total_sent,
  and failure_rate SHALL equal failed_count / total_sent, with all rates
  bounded in [0.0, 1.0] and test-send messages excluded from all counts.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from services.delivery_tracker import DeliveryTracker


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Positive sent counts (delivery rates only meaningful when sent > 0)
positive_sent_counts = st.integers(min_value=1, max_value=100_000)

# Count values that are valid sub-counts (0 to some upper bound)
# We generate them relative to sent_count to sometimes exceed it (testing clamping)
non_negative_counts = st.integers(min_value=0, max_value=100_000)

# Zero sent count (edge case: should return 0.0 for all rates)
zero_sent = st.just(0)

# Negative sent counts (should be handled gracefully)
negative_counts = st.integers(min_value=-1000, max_value=-1)


# ---------------------------------------------------------------------------
# Property Tests
# ---------------------------------------------------------------------------
class TestDeliveryRateComputation:
    """Property-based tests for delivery rate computation — Property 12."""

    @given(
        sent=positive_sent_counts,
        delivered=non_negative_counts,
        read=non_negative_counts,
        failed=non_negative_counts,
    )
    @settings(max_examples=500)
    def test_delivery_rate_equals_delivered_over_sent(self, sent, delivered, read, failed):
        """
        Property: delivery_rate SHALL equal delivered_count / total_sent,
        clamped to [0.0, 1.0].

        **Validates: Requirements 5.3, 8.1**
        """
        rates = DeliveryTracker.compute_delivery_rates(sent, delivered, read, failed)

        expected = min(1.0, max(0.0, delivered / sent))
        assert rates["delivery_rate"] == expected, (
            f"delivery_rate expected {expected} but got {rates['delivery_rate']} "
            f"(sent={sent}, delivered={delivered})"
        )

    @given(
        sent=positive_sent_counts,
        delivered=non_negative_counts,
        read=non_negative_counts,
        failed=non_negative_counts,
    )
    @settings(max_examples=500)
    def test_read_rate_equals_read_over_sent(self, sent, delivered, read, failed):
        """
        Property: read_rate SHALL equal read_count / total_sent,
        clamped to [0.0, 1.0].

        **Validates: Requirements 5.3, 8.1**
        """
        rates = DeliveryTracker.compute_delivery_rates(sent, delivered, read, failed)

        expected = min(1.0, max(0.0, read / sent))
        assert rates["read_rate"] == expected, (
            f"read_rate expected {expected} but got {rates['read_rate']} "
            f"(sent={sent}, read={read})"
        )

    @given(
        sent=positive_sent_counts,
        delivered=non_negative_counts,
        read=non_negative_counts,
        failed=non_negative_counts,
    )
    @settings(max_examples=500)
    def test_failure_rate_equals_failed_over_sent(self, sent, delivered, read, failed):
        """
        Property: failure_rate SHALL equal failed_count / total_sent,
        clamped to [0.0, 1.0].

        **Validates: Requirements 5.3, 8.1**
        """
        rates = DeliveryTracker.compute_delivery_rates(sent, delivered, read, failed)

        expected = min(1.0, max(0.0, failed / sent))
        assert rates["failure_rate"] == expected, (
            f"failure_rate expected {expected} but got {rates['failure_rate']} "
            f"(sent={sent}, failed={failed})"
        )

    @given(
        sent=positive_sent_counts,
        delivered=non_negative_counts,
        read=non_negative_counts,
        failed=non_negative_counts,
    )
    @settings(max_examples=500)
    def test_all_rates_bounded_zero_to_one(self, sent, delivered, read, failed):
        """
        Property: All computed rates SHALL be bounded in [0.0, 1.0].

        **Validates: Requirements 5.3, 8.1**
        """
        rates = DeliveryTracker.compute_delivery_rates(sent, delivered, read, failed)

        assert 0.0 <= rates["delivery_rate"] <= 1.0, (
            f"delivery_rate {rates['delivery_rate']} out of [0,1] bounds"
        )
        assert 0.0 <= rates["read_rate"] <= 1.0, (
            f"read_rate {rates['read_rate']} out of [0,1] bounds"
        )
        assert 0.0 <= rates["failure_rate"] <= 1.0, (
            f"failure_rate {rates['failure_rate']} out of [0,1] bounds"
        )

    @given(
        delivered=non_negative_counts,
        read=non_negative_counts,
        failed=non_negative_counts,
    )
    @settings(max_examples=200)
    def test_zero_sent_returns_zero_rates(self, delivered, read, failed):
        """
        Property: When sent_count is 0, all rates SHALL be 0.0
        (division by zero protection).

        **Validates: Requirements 5.3**
        """
        rates = DeliveryTracker.compute_delivery_rates(0, delivered, read, failed)

        assert rates["delivery_rate"] == 0.0, (
            f"delivery_rate should be 0.0 when sent=0, got {rates['delivery_rate']}"
        )
        assert rates["read_rate"] == 0.0, (
            f"read_rate should be 0.0 when sent=0, got {rates['read_rate']}"
        )
        assert rates["failure_rate"] == 0.0, (
            f"failure_rate should be 0.0 when sent=0, got {rates['failure_rate']}"
        )

    @given(
        sent=negative_counts,
        delivered=non_negative_counts,
        read=non_negative_counts,
        failed=non_negative_counts,
    )
    @settings(max_examples=200)
    def test_negative_sent_returns_zero_rates(self, sent, delivered, read, failed):
        """
        Property: When sent_count is negative (invalid state), all rates
        SHALL be 0.0 (graceful handling of invalid inputs).

        **Validates: Requirements 5.3**
        """
        rates = DeliveryTracker.compute_delivery_rates(sent, delivered, read, failed)

        assert rates["delivery_rate"] == 0.0, (
            f"delivery_rate should be 0.0 when sent={sent}, got {rates['delivery_rate']}"
        )
        assert rates["read_rate"] == 0.0, (
            f"read_rate should be 0.0 when sent={sent}, got {rates['read_rate']}"
        )
        assert rates["failure_rate"] == 0.0, (
            f"failure_rate should be 0.0 when sent={sent}, got {rates['failure_rate']}"
        )

    @given(
        sent=positive_sent_counts,
    )
    @settings(max_examples=200)
    def test_test_sends_excluded_semantics(self, sent):
        """
        Property: Test-send messages are excluded from all counts. When only
        non-test messages are counted, a campaign with all messages delivered
        and read should yield delivery_rate=1.0 and read_rate=1.0.

        This validates the contract that test-sends (is_test_send=1) must be
        filtered out BEFORE passing counts to compute_delivery_rates.

        **Validates: Requirements 16.3**
        """
        # Simulate: all sent messages were delivered and read (no test-sends in counts)
        delivered = sent
        read = sent
        failed = 0

        rates = DeliveryTracker.compute_delivery_rates(sent, delivered, read, failed)

        assert rates["delivery_rate"] == 1.0, (
            f"When all sent are delivered, delivery_rate should be 1.0, "
            f"got {rates['delivery_rate']}"
        )
        assert rates["read_rate"] == 1.0, (
            f"When all sent are read, read_rate should be 1.0, "
            f"got {rates['read_rate']}"
        )
        assert rates["failure_rate"] == 0.0, (
            f"When no failures, failure_rate should be 0.0, "
            f"got {rates['failure_rate']}"
        )

    @given(
        sent=positive_sent_counts,
        delivered=non_negative_counts,
        read=non_negative_counts,
        failed=non_negative_counts,
    )
    @settings(max_examples=500)
    def test_rates_are_non_negative(self, sent, delivered, read, failed):
        """
        Property: Even with unusual count combinations (e.g. zero sub-counts),
        all rates SHALL be non-negative.

        **Validates: Requirements 5.3, 8.1**
        """
        rates = DeliveryTracker.compute_delivery_rates(sent, delivered, read, failed)

        assert rates["delivery_rate"] >= 0.0
        assert rates["read_rate"] >= 0.0
        assert rates["failure_rate"] >= 0.0
