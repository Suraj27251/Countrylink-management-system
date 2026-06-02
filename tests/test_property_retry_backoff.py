"""
Property-based tests for Retry Backoff Computation using Hypothesis.

**Validates: Requirements 4.3**

Property 9: Retry backoff computation
- For any retry attempt number N (1, 2, or 3), the computed backoff delay
  SHALL equal 5 * 3^(N-1) seconds (producing 5s, 15s, 45s).
- No message SHALL be retried more than 3 times.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from services.retry_categorizer import RetryCategorizerService


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------
# Valid retry attempt numbers (1-based)
valid_attempt_numbers = st.sampled_from([1, 2, 3])

# Retry counts that have NOT exceeded the limit (0, 1, 2 mean attempts 1, 2, 3 are next)
retry_counts_within_limit = st.integers(min_value=0, max_value=2)

# Retry counts that HAVE exceeded or met the limit (>= 3)
retry_counts_exceeded = st.integers(min_value=3, max_value=100)

# Arbitrary max_retries values (always 3 per design, but test robustness)
max_retries_strategy = st.just(3)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------
class MockCursor:
    """Mock MySQL cursor."""

    def __init__(self):
        self.executed = []
        self.fetchone_result = None
        self._closed = False

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchone(self):
        return self.fetchone_result

    def close(self):
        self._closed = True


class MockConnection:
    """Mock MySQL connection."""

    def __init__(self, cursor_instance=None):
        self._cursor = cursor_instance or MockCursor()
        self._committed = False

    def cursor(self, dictionary=False):
        return self._cursor

    def commit(self):
        self._committed = True

    def rollback(self):
        pass

    def is_connected(self):
        return True


# ---------------------------------------------------------------------------
# Property Tests
# ---------------------------------------------------------------------------
class TestRetryBackoffComputation:
    """Property-based tests for retry backoff — Property 9."""

    @given(attempt=valid_attempt_numbers)
    @settings(max_examples=200)
    def test_backoff_formula_correctness(self, attempt: int):
        """
        Property: For any retry attempt number N in {1, 2, 3}, the computed
        backoff delay SHALL equal exactly 5 * 3^(N-1) seconds.

        Expected values:
          - N=1: 5 * 3^0 = 5 seconds
          - N=2: 5 * 3^1 = 15 seconds
          - N=3: 5 * 3^2 = 45 seconds

        **Validates: Requirements 4.3**
        """
        expected_delay = 5 * (3 ** (attempt - 1))
        actual_delay = RetryCategorizerService.compute_backoff(attempt)

        assert actual_delay == expected_delay, (
            f"For attempt {attempt}, expected backoff {expected_delay}s "
            f"but got {actual_delay}s"
        )

    @given(retry_count=retry_counts_exceeded)
    @settings(max_examples=200)
    def test_no_retry_beyond_max_retries(self, retry_count: int):
        """
        Property: No message SHALL be retried more than 3 times. When
        retry_count >= max_retries (3), should_retry() must return False.

        **Validates: Requirements 4.3**
        """
        cursor = MockCursor()
        cursor.fetchone_result = {
            "id": 1,
            "campaign_id": 100,
            "retry_count": retry_count,
            "max_retries": 3,
            "error_code": 131047,
            "error_category": "transient",
            "status": "failed",
        }
        conn = MockConnection(cursor)
        service = RetryCategorizerService(lambda: conn)

        decision = service.should_retry(1)

        assert decision.should_retry is False, (
            f"Message with retry_count={retry_count} (>= max_retries=3) "
            f"should NOT be retried, but should_retry returned True"
        )
        assert "exceeded" in decision.reason.lower() or "max" in decision.reason.lower(), (
            f"Rejection reason should mention max retries exceeded, got: '{decision.reason}'"
        )

    @given(retry_count=retry_counts_within_limit)
    @settings(max_examples=200)
    def test_retry_allowed_within_limit(self, retry_count: int):
        """
        Property: When retry_count < max_retries (3) and error_category is
        'transient', should_retry() must return True with the correct
        backoff delay for the next attempt.

        **Validates: Requirements 4.3**
        """
        cursor = MockCursor()
        cursor.fetchone_result = {
            "id": 1,
            "campaign_id": 100,
            "retry_count": retry_count,
            "max_retries": 3,
            "error_code": 131047,
            "error_category": "transient",
            "status": "failed",
        }
        conn = MockConnection(cursor)
        service = RetryCategorizerService(lambda: conn)

        decision = service.should_retry(1)

        assert decision.should_retry is True, (
            f"Message with retry_count={retry_count} (< max_retries=3) and "
            f"transient error should be retried, but should_retry returned False"
        )

        # Verify the backoff delay matches the formula for the NEXT attempt
        next_attempt = retry_count + 1
        expected_backoff = 5 * (3 ** (next_attempt - 1))
        assert decision.backoff_seconds == expected_backoff, (
            f"For retry_count={retry_count} (next attempt={next_attempt}), "
            f"expected backoff {expected_backoff}s but got {decision.backoff_seconds}s"
        )

    @given(
        retry_count=retry_counts_within_limit,
        error_category=st.sampled_from(["permanent", "suppression"]),
    )
    @settings(max_examples=200)
    def test_non_transient_errors_never_retried(self, retry_count: int, error_category: str):
        """
        Property: Only transient errors are retried. For any message with
        error_category 'permanent' or 'suppression', should_retry() must
        return False regardless of retry_count.

        **Validates: Requirements 4.3**
        """
        cursor = MockCursor()
        cursor.fetchone_result = {
            "id": 1,
            "campaign_id": 100,
            "retry_count": retry_count,
            "max_retries": 3,
            "error_code": 131026,
            "error_category": error_category,
            "status": "failed",
        }
        conn = MockConnection(cursor)
        service = RetryCategorizerService(lambda: conn)

        decision = service.should_retry(1)

        assert decision.should_retry is False, (
            f"Message with error_category='{error_category}' should NOT be retried "
            f"regardless of retry_count={retry_count}, but should_retry returned True"
        )

    @given(attempt=st.integers(min_value=1, max_value=3))
    @settings(max_examples=200)
    def test_backoff_is_strictly_increasing(self, attempt: int):
        """
        Property: The backoff delay is strictly increasing with each attempt.
        For attempts 1 < 2 < 3, delays 5 < 15 < 45.

        **Validates: Requirements 4.3**
        """
        assume(attempt < 3)

        current_backoff = RetryCategorizerService.compute_backoff(attempt)
        next_backoff = RetryCategorizerService.compute_backoff(attempt + 1)

        assert next_backoff > current_backoff, (
            f"Backoff must be strictly increasing: attempt {attempt} = {current_backoff}s "
            f"should be less than attempt {attempt + 1} = {next_backoff}s"
        )
