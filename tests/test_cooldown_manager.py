"""
Unit tests for CooldownManager — cooldown enforcement, frequency limits,
transactional bypass, and quality tier integration.
"""

import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from services.cooldown_manager import CooldownManager, CooldownResult


class MockCursor:
    """Mock MySQL cursor with dictionary=True support."""

    def __init__(self):
        self.executed = []
        self.fetchone_results = []
        self._call_idx = 0
        self._closed = False

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchone(self):
        if self._call_idx < len(self.fetchone_results):
            result = self.fetchone_results[self._call_idx]
            self._call_idx += 1
            return result
        return None

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

    def close(self):
        pass


class TestCooldownManagerInit(unittest.TestCase):
    """Tests for CooldownManager initialization and configuration."""

    def test_default_quality_tier_is_green(self):
        """Default quality tier should be green."""
        mgr = CooldownManager(lambda: MockConnection())
        self.assertEqual(mgr.quality_tier, "green")

    def test_custom_quality_tier(self):
        """Quality tier should be settable via constructor."""
        mgr = CooldownManager(lambda: MockConnection(), quality_tier="yellow")
        self.assertEqual(mgr.quality_tier, "yellow")

    def test_quality_tier_case_insensitive(self):
        """Quality tier should be stored lowercase."""
        mgr = CooldownManager(lambda: MockConnection(), quality_tier="YELLOW")
        self.assertEqual(mgr.quality_tier, "yellow")

    def test_quality_tier_setter(self):
        """Quality tier property should be mutable."""
        mgr = CooldownManager(lambda: MockConnection())
        mgr.quality_tier = "red"
        self.assertEqual(mgr.quality_tier, "red")


class TestTransactionalBypass(unittest.TestCase):
    """Tests that transactional messages bypass all cooldown checks."""

    def test_transactional_always_allowed(self):
        """Transactional campaign type should always return allowed=True."""
        mgr = CooldownManager(lambda: MockConnection())
        result = mgr.check_cooldown("919876543210", "transactional")
        self.assertTrue(result.allowed)
        self.assertIsNone(result.reason)
        self.assertEqual(result.excluded_count, 0)

    def test_transactional_case_insensitive(self):
        """Transactional bypass should work regardless of case."""
        mgr = CooldownManager(lambda: MockConnection())
        result = mgr.check_cooldown("919876543210", "Transactional")
        self.assertTrue(result.allowed)

    def test_transactional_no_db_query(self):
        """Transactional messages should not hit the database."""
        cursor = MockCursor()
        conn = MockConnection(cursor)
        mgr = CooldownManager(lambda: conn)
        mgr.check_cooldown("919876543210", "transactional")
        # No queries should have been executed
        self.assertEqual(len(cursor.executed), 0)


class TestPromotionalCooldownWindow(unittest.TestCase):
    """Tests for the time-based promotional cooldown window."""

    def test_allowed_when_no_recent_messages(self):
        """Customer with no recent messages should be allowed."""
        cursor = MockCursor()
        # First query (window check): 0 messages in window
        # Second query (7-day check): 0 messages in 7 days
        cursor.fetchone_results = [{"cnt": 0}, {"cnt": 0}]
        conn = MockConnection(cursor)
        mgr = CooldownManager(lambda: conn)

        result = mgr.check_cooldown("919876543210", "promotional")
        self.assertTrue(result.allowed)
        self.assertIsNone(result.reason)

    def test_blocked_when_message_within_72h(self):
        """Customer with a promotional message within 72h should be blocked."""
        cursor = MockCursor()
        # First query (window check): 1 message found in window
        cursor.fetchone_results = [{"cnt": 1}]
        conn = MockConnection(cursor)
        mgr = CooldownManager(lambda: conn)

        result = mgr.check_cooldown("919876543210", "promotional")
        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, "cooldown_active")
        self.assertEqual(result.excluded_count, 1)

    def test_green_tier_uses_72h_window(self):
        """Green tier should use 72-hour promotional window."""
        cursor = MockCursor()
        cursor.fetchone_results = [{"cnt": 0}, {"cnt": 0}]
        conn = MockConnection(cursor)
        mgr = CooldownManager(lambda: conn, quality_tier="green")

        mgr.check_cooldown("919876543210", "promotional")

        # Verify the cutoff time used in the query
        _, params = cursor.executed[0]
        cutoff = params[1]
        expected_cutoff = datetime.now() - timedelta(hours=72)
        # Allow 2 second tolerance for test execution
        self.assertAlmostEqual(
            cutoff.timestamp(), expected_cutoff.timestamp(), delta=2
        )

    def test_yellow_tier_uses_120h_window(self):
        """Yellow tier should extend the promotional window to 120 hours."""
        cursor = MockCursor()
        cursor.fetchone_results = [{"cnt": 0}, {"cnt": 0}]
        conn = MockConnection(cursor)
        mgr = CooldownManager(lambda: conn, quality_tier="yellow")

        mgr.check_cooldown("919876543210", "promotional")

        # Verify the cutoff time uses 120h window
        _, params = cursor.executed[0]
        cutoff = params[1]
        expected_cutoff = datetime.now() - timedelta(hours=120)
        # Allow 2 second tolerance for test execution
        self.assertAlmostEqual(
            cutoff.timestamp(), expected_cutoff.timestamp(), delta=2
        )


class TestWeeklyFrequencyLimit(unittest.TestCase):
    """Tests for the 2-per-7-day rolling limit."""

    def test_allowed_when_under_weekly_limit(self):
        """Customer with fewer than 2 messages in 7 days should be allowed."""
        cursor = MockCursor()
        # Window check: no recent message
        # 7-day check: 1 message (under limit of 2)
        cursor.fetchone_results = [{"cnt": 0}, {"cnt": 1}]
        conn = MockConnection(cursor)
        mgr = CooldownManager(lambda: conn)

        result = mgr.check_cooldown("919876543210", "promotional")
        self.assertTrue(result.allowed)

    def test_blocked_when_at_weekly_limit(self):
        """Customer with 2 messages in 7 days should be blocked."""
        cursor = MockCursor()
        # Window check: no recent message (outside 72h but inside 7 days)
        # 7-day check: 2 messages (at limit)
        cursor.fetchone_results = [{"cnt": 0}, {"cnt": 2}]
        conn = MockConnection(cursor)
        mgr = CooldownManager(lambda: conn)

        result = mgr.check_cooldown("919876543210", "promotional")
        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, "weekly_limit_exceeded")
        self.assertEqual(result.excluded_count, 2)

    def test_blocked_when_over_weekly_limit(self):
        """Customer with more than 2 messages in 7 days should be blocked."""
        cursor = MockCursor()
        cursor.fetchone_results = [{"cnt": 0}, {"cnt": 3}]
        conn = MockConnection(cursor)
        mgr = CooldownManager(lambda: conn)

        result = mgr.check_cooldown("919876543210", "promotional")
        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, "weekly_limit_exceeded")


class TestRecordSend(unittest.TestCase):
    """Tests for recording sends in the cooldown table."""

    def test_record_send_inserts_to_table(self):
        """record_send should INSERT into message_cooldowns."""
        cursor = MockCursor()
        conn = MockConnection(cursor)
        mgr = CooldownManager(lambda: conn)

        mgr.record_send("919876543210", 42, "promotional")

        self.assertEqual(len(cursor.executed), 1)
        sql, params = cursor.executed[0]
        self.assertIn("INSERT INTO message_cooldowns", sql)
        self.assertEqual(params[0], "919876543210")
        self.assertEqual(params[1], 42)
        self.assertEqual(params[2], "promotional")
        # Fourth param should be a datetime (sent_at)
        self.assertIsInstance(params[3], datetime)

    def test_record_send_commits(self):
        """record_send should commit the transaction."""
        cursor = MockCursor()
        conn = MockConnection(cursor)
        mgr = CooldownManager(lambda: conn)

        mgr.record_send("919876543210", 42, "promotional")
        self.assertTrue(conn._committed)

    def test_record_send_closes_cursor(self):
        """record_send should close the cursor after operation."""
        cursor = MockCursor()
        conn = MockConnection(cursor)
        mgr = CooldownManager(lambda: conn)

        mgr.record_send("919876543210", 42, "promotional")
        self.assertTrue(cursor._closed)


class TestCooldownResult(unittest.TestCase):
    """Tests for CooldownResult dataclass."""

    def test_dataclass_fields(self):
        """CooldownResult should have expected fields with defaults."""
        result = CooldownResult(allowed=True)
        self.assertTrue(result.allowed)
        self.assertIsNone(result.reason)
        self.assertEqual(result.excluded_count, 0)

    def test_dataclass_with_all_fields(self):
        """CooldownResult should accept all fields."""
        result = CooldownResult(
            allowed=False, reason="cooldown_active", excluded_count=3
        )
        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, "cooldown_active")
        self.assertEqual(result.excluded_count, 3)


class TestCampaignTypeCoverage(unittest.TestCase):
    """Tests for different campaign types and their cooldown behavior."""

    def test_reactivation_subject_to_cooldown(self):
        """Reactivation campaigns should be subject to cooldown checks."""
        cursor = MockCursor()
        cursor.fetchone_results = [{"cnt": 1}]
        conn = MockConnection(cursor)
        mgr = CooldownManager(lambda: conn)

        result = mgr.check_cooldown("919876543210", "reactivation")
        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, "cooldown_active")

    def test_ab_test_subject_to_cooldown(self):
        """A/B test campaigns should be subject to cooldown checks."""
        cursor = MockCursor()
        cursor.fetchone_results = [{"cnt": 1}]
        conn = MockConnection(cursor)
        mgr = CooldownManager(lambda: conn)

        result = mgr.check_cooldown("919876543210", "ab_test")
        self.assertFalse(result.allowed)

    def test_promotional_subject_to_cooldown(self):
        """Promotional campaigns should be subject to cooldown checks."""
        cursor = MockCursor()
        cursor.fetchone_results = [{"cnt": 1}]
        conn = MockConnection(cursor)
        mgr = CooldownManager(lambda: conn)

        result = mgr.check_cooldown("919876543210", "promotional")
        self.assertFalse(result.allowed)


class TestQualityTierIntegration(unittest.TestCase):
    """Tests for quality tier affecting cooldown window."""

    def test_promo_window_hours_green(self):
        """Green tier should return 72 hours."""
        mgr = CooldownManager(lambda: MockConnection(), quality_tier="green")
        self.assertEqual(mgr._get_promo_window_hours(), 72)

    def test_promo_window_hours_yellow(self):
        """Yellow tier should return 120 hours."""
        mgr = CooldownManager(lambda: MockConnection(), quality_tier="yellow")
        self.assertEqual(mgr._get_promo_window_hours(), 120)

    def test_promo_window_hours_red(self):
        """Red tier should return 72 hours (default)."""
        mgr = CooldownManager(lambda: MockConnection(), quality_tier="red")
        self.assertEqual(mgr._get_promo_window_hours(), 72)

    def test_changing_tier_affects_window(self):
        """Changing tier dynamically should affect the window calculation."""
        mgr = CooldownManager(lambda: MockConnection(), quality_tier="green")
        self.assertEqual(mgr._get_promo_window_hours(), 72)

        mgr.quality_tier = "yellow"
        self.assertEqual(mgr._get_promo_window_hours(), 120)


if __name__ == "__main__":
    unittest.main()
