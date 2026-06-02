"""
Unit tests for SimulationEngine — pre-send campaign analysis.

Tests cover:
- Basic simulation with no exclusions
- Cooldown exclusions
- Opt-out/DND exclusions
- Invalid number exclusions
- Incomplete template parameter exclusions
- Duplicate recipient detection
- Warning generation when exclusions > 30%
- Estimated send time and cost calculations
- Edge cases: empty segment, no template, transactional bypass
"""

import json
import unittest
from unittest.mock import MagicMock, patch

from services.simulation import (
    DEFAULT_COST_RATES,
    DEFAULT_THROTTLE_RATE,
    EXCLUSION_WARNING_THRESHOLD,
    ExclusionBreakdown,
    SimulationEngine,
    SimulationResult,
)


class MockCursor:
    """Mock MySQL cursor with dictionary=True support."""

    def __init__(self):
        self.executed = []
        self.results_queue = []
        self._call_idx = 0
        self._closed = False

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchone(self):
        if self._call_idx < len(self.results_queue):
            result = self.results_queue[self._call_idx]
            self._call_idx += 1
            return result
        return None

    def fetchall(self):
        if self._call_idx < len(self.results_queue):
            result = self.results_queue[self._call_idx]
            self._call_idx += 1
            return result if isinstance(result, list) else []
        return []

    def close(self):
        self._closed = True


class MockConnection:
    """Mock MySQL connection."""

    def __init__(self, cursor_instance=None):
        self._cursor = cursor_instance or MockCursor()

    def cursor(self, dictionary=False):
        return self._cursor

    def close(self):
        pass


class TestSimulationEngineInit(unittest.TestCase):
    """Tests for SimulationEngine initialization."""

    def test_default_throttle_rate(self):
        engine = SimulationEngine(lambda: MockConnection())
        self.assertEqual(engine.throttle_rate, DEFAULT_THROTTLE_RATE)

    def test_custom_throttle_rate(self):
        engine = SimulationEngine(lambda: MockConnection(), throttle_rate=100)
        self.assertEqual(engine.throttle_rate, 100)

    def test_default_cost_rates(self):
        engine = SimulationEngine(lambda: MockConnection())
        self.assertEqual(engine.cost_rates, DEFAULT_COST_RATES)

    def test_custom_cost_rates(self):
        custom = {"marketing": 1.0, "utility": 0.5}
        engine = SimulationEngine(lambda: MockConnection(), cost_rates=custom)
        self.assertEqual(engine.cost_rates, custom)


class TestSimulationEngineSimulate(unittest.TestCase):
    """Tests for SimulationEngine.simulate() method."""

    def _make_engine_with_cursor(self, cursor):
        conn = MockConnection(cursor)
        return SimulationEngine(lambda: conn)

    def test_campaign_not_found_raises(self):
        """Should raise ValueError when campaign doesn't exist."""
        cursor = MockCursor()
        # Campaign query returns None
        cursor.results_queue = [None]
        engine = self._make_engine_with_cursor(cursor)

        with self.assertRaises(ValueError) as ctx:
            engine.simulate(999)
        self.assertIn("999", str(ctx.exception))

    def test_empty_segment_returns_zero_audience(self):
        """Campaign with no segment_id should return zero audience."""
        cursor = MockCursor()
        # Campaign: no segment_id
        cursor.results_queue = [
            {"id": 1, "name": "Test", "segment_id": None, "template_id": None,
             "campaign_type": "promotional", "status": "draft"},
        ]
        engine = self._make_engine_with_cursor(cursor)
        result = engine.simulate(1)

        self.assertEqual(result.original_audience_count, 0)
        self.assertEqual(result.final_audience_count, 0)
        self.assertEqual(result.estimated_send_time_seconds, 0.0)
        self.assertEqual(result.estimated_cost_inr, 0.0)

    def test_all_eligible_no_exclusions(self):
        """All recipients eligible — no exclusions, no duplicates."""
        cursor = MockCursor()
        cursor.results_queue = [
            # Campaign
            {"id": 1, "name": "Promo", "segment_id": 10, "template_id": 20,
             "campaign_type": "promotional", "status": "draft"},
            # Segment filter_criteria
            {"filter_criteria": json.dumps({"status": "active"})},
            # Recipients (fetchall)
            [
                {"mobile": "919876543210", "customer_name": "Alice", "plan_name": "Eclipse 100", "zone_name": "Z1", "area": "A1", "expiry_date": None},
                {"mobile": "919876543211", "customer_name": "Bob", "plan_name": "Eclipse 200", "zone_name": "Z2", "area": "A2", "expiry_date": None},
            ],
            # Template placeholder_mappings
            {"placeholder_mappings": json.dumps({"1": "customer_name"})},
            # Suppression check for Alice
            None,
            # Cooldown check for Alice
            {"cnt": 0},
            # Suppression check for Bob
            None,
            # Cooldown check for Bob
            {"cnt": 0},
            # Template category
            {"category": "marketing"},
        ]
        engine = self._make_engine_with_cursor(cursor)
        result = engine.simulate(1)

        self.assertEqual(result.original_audience_count, 2)
        self.assertEqual(result.final_audience_count, 2)
        self.assertEqual(result.exclusions.total, 0)
        self.assertEqual(result.duplicate_count, 0)
        # Estimated time: 2 / 80 = 0.025
        self.assertAlmostEqual(result.estimated_send_time_seconds, 0.03, places=2)
        # Estimated cost: 2 * 0.76 = 1.52
        self.assertAlmostEqual(result.estimated_cost_inr, 1.52, places=2)
        self.assertEqual(result.warnings, [])

    def test_duplicate_recipients_detected(self):
        """Duplicate mobiles in segment should be detected."""
        cursor = MockCursor()
        cursor.results_queue = [
            # Campaign
            {"id": 1, "name": "Promo", "segment_id": 10, "template_id": None,
             "campaign_type": "promotional", "status": "draft"},
            # Segment
            {"filter_criteria": json.dumps({"status": "active"})},
            # Recipients with duplicate
            [
                {"mobile": "919876543210", "customer_name": "Alice"},
                {"mobile": "919876543210", "customer_name": "Alice"},  # duplicate
                {"mobile": "919876543211", "customer_name": "Bob"},
            ],
            # Template mappings (no template)
            # Suppression for Alice
            None,
            # Cooldown for Alice
            {"cnt": 0},
            # Suppression for Bob
            None,
            # Cooldown for Bob
            {"cnt": 0},
            # Template category (no template_id so returns default)
        ]
        engine = self._make_engine_with_cursor(cursor)
        result = engine.simulate(1)

        self.assertEqual(result.original_audience_count, 3)
        self.assertEqual(result.duplicate_count, 1)
        self.assertEqual(result.final_audience_count, 2)

    def test_invalid_number_excluded(self):
        """Invalid mobile numbers should be excluded."""
        cursor = MockCursor()
        cursor.results_queue = [
            # Campaign
            {"id": 1, "name": "Promo", "segment_id": 10, "template_id": None,
             "campaign_type": "promotional", "status": "draft"},
            # Segment
            {"filter_criteria": json.dumps({"status": "active"})},
            # Recipients — one invalid
            [
                {"mobile": "919876543210", "customer_name": "Alice"},
                {"mobile": "abc", "customer_name": "Invalid"},
                {"mobile": "", "customer_name": "Empty"},
            ],
            # Suppression for Alice
            None,
            # Cooldown for Alice
            {"cnt": 0},
            # Template category
        ]
        engine = self._make_engine_with_cursor(cursor)
        result = engine.simulate(1)

        self.assertEqual(result.exclusions.invalid_number, 2)
        self.assertEqual(result.final_audience_count, 1)

    def test_opted_out_excluded(self):
        """Opted-out customers should be excluded."""
        cursor = MockCursor()
        cursor.results_queue = [
            # Campaign
            {"id": 1, "name": "Promo", "segment_id": 10, "template_id": None,
             "campaign_type": "promotional", "status": "draft"},
            # Segment
            {"filter_criteria": json.dumps({"status": "active"})},
            # Recipients
            [
                {"mobile": "919876543210", "customer_name": "Alice"},
                {"mobile": "919876543211", "customer_name": "Bob"},
            ],
            # Suppression for Alice — opted out
            {"reason": "opt_out_keyword"},
            # Suppression for Bob — not suppressed
            None,
            # Cooldown for Bob
            {"cnt": 0},
            # Template category
        ]
        engine = self._make_engine_with_cursor(cursor)
        result = engine.simulate(1)

        self.assertEqual(result.exclusions.opted_out, 1)
        self.assertEqual(result.final_audience_count, 1)

    def test_dnd_excluded(self):
        """DND-suppressed customers should be excluded."""
        cursor = MockCursor()
        cursor.results_queue = [
            # Campaign
            {"id": 1, "name": "Promo", "segment_id": 10, "template_id": None,
             "campaign_type": "promotional", "status": "draft"},
            # Segment
            {"filter_criteria": json.dumps({"status": "active"})},
            # Recipients
            [
                {"mobile": "919876543210", "customer_name": "Alice"},
            ],
            # Suppression for Alice — manual DND
            {"reason": "manual_dnd"},
            # Template category
        ]
        engine = self._make_engine_with_cursor(cursor)
        result = engine.simulate(1)

        self.assertEqual(result.exclusions.dnd, 1)
        self.assertEqual(result.final_audience_count, 0)

    def test_cooldown_excluded(self):
        """Customers in cooldown should be excluded."""
        cursor = MockCursor()
        cursor.results_queue = [
            # Campaign
            {"id": 1, "name": "Promo", "segment_id": 10, "template_id": None,
             "campaign_type": "promotional", "status": "draft"},
            # Segment
            {"filter_criteria": json.dumps({"status": "active"})},
            # Recipients
            [
                {"mobile": "919876543210", "customer_name": "Alice"},
            ],
            # Suppression for Alice — not suppressed
            None,
            # Cooldown for Alice — in cooldown
            {"cnt": 1},
            # Template category
        ]
        engine = self._make_engine_with_cursor(cursor)
        result = engine.simulate(1)

        self.assertEqual(result.exclusions.cooldown, 1)
        self.assertEqual(result.final_audience_count, 0)

    def test_incomplete_params_excluded(self):
        """Recipients with incomplete template params should be excluded."""
        cursor = MockCursor()
        cursor.results_queue = [
            # Campaign
            {"id": 1, "name": "Promo", "segment_id": 10, "template_id": 20,
             "campaign_type": "promotional", "status": "draft"},
            # Segment
            {"filter_criteria": json.dumps({"status": "active"})},
            # Recipients — Bob missing plan_name
            [
                {"mobile": "919876543210", "customer_name": "Alice", "plan_name": "Eclipse 100"},
                {"mobile": "919876543211", "customer_name": "Bob", "plan_name": None},
            ],
            # Template mappings require customer_name and plan_name
            {"placeholder_mappings": json.dumps({"1": "customer_name", "2": "plan_name"})},
            # Suppression for Alice
            None,
            # Cooldown for Alice
            {"cnt": 0},
            # Suppression for Bob
            None,
            # Cooldown for Bob
            {"cnt": 0},
            # Template category
            {"category": "marketing"},
        ]
        engine = self._make_engine_with_cursor(cursor)
        result = engine.simulate(1)

        self.assertEqual(result.exclusions.incomplete_data, 1)
        self.assertEqual(result.final_audience_count, 1)

    def test_warning_generated_above_threshold(self):
        """Warning should be generated when exclusions > 30% of original."""
        cursor = MockCursor()
        # 10 recipients, 4 excluded (40% > 30%)
        recipients = [
            {"mobile": f"91987654321{i}", "customer_name": f"User{i}"}
            for i in range(10)
        ]
        cursor.results_queue = [
            # Campaign
            {"id": 1, "name": "Promo", "segment_id": 10, "template_id": None,
             "campaign_type": "promotional", "status": "draft"},
            # Segment
            {"filter_criteria": json.dumps({"status": "active"})},
            # Recipients
            recipients,
        ]
        # First 4 will be suppressed (opted out), rest will pass
        for i in range(10):
            if i < 4:
                cursor.results_queue.append({"reason": "opt_out_keyword"})
            else:
                cursor.results_queue.append(None)  # not suppressed
                cursor.results_queue.append({"cnt": 0})  # not in cooldown

        engine = self._make_engine_with_cursor(cursor)
        result = engine.simulate(1)

        self.assertEqual(result.exclusions.opted_out, 4)
        self.assertEqual(result.final_audience_count, 6)
        self.assertTrue(len(result.warnings) >= 1)
        self.assertIn("40.0%", result.warnings[0])

    def test_no_warning_below_threshold(self):
        """No warning when exclusions <= 30% of original."""
        cursor = MockCursor()
        # 10 recipients, 2 excluded (20% <= 30%)
        recipients = [
            {"mobile": f"91987654321{i}", "customer_name": f"User{i}"}
            for i in range(10)
        ]
        cursor.results_queue = [
            # Campaign
            {"id": 1, "name": "Promo", "segment_id": 10, "template_id": None,
             "campaign_type": "promotional", "status": "draft"},
            # Segment
            {"filter_criteria": json.dumps({"status": "active"})},
            # Recipients
            recipients,
        ]
        # First 2 will be suppressed, rest pass
        for i in range(10):
            if i < 2:
                cursor.results_queue.append({"reason": "opt_out_keyword"})
            else:
                cursor.results_queue.append(None)
                cursor.results_queue.append({"cnt": 0})

        engine = self._make_engine_with_cursor(cursor)
        result = engine.simulate(1)

        self.assertEqual(result.exclusions.opted_out, 2)
        self.assertEqual(result.final_audience_count, 8)
        # No exclusion-rate warning (duplicates warning not expected either)
        exclusion_warnings = [w for w in result.warnings if "exclusion" in w.lower()]
        self.assertEqual(len(exclusion_warnings), 0)

    def test_transactional_bypasses_cooldown(self):
        """Transactional campaigns should bypass cooldown checks."""
        cursor = MockCursor()
        cursor.results_queue = [
            # Campaign — transactional type
            {"id": 1, "name": "Payment Confirm", "segment_id": 10, "template_id": None,
             "campaign_type": "transactional", "status": "draft"},
            # Segment
            {"filter_criteria": json.dumps({"status": "active"})},
            # Recipients
            [
                {"mobile": "919876543210", "customer_name": "Alice"},
            ],
            # Suppression for Alice — not suppressed
            None,
            # Note: No cooldown check for transactional
            # Template category
        ]
        engine = self._make_engine_with_cursor(cursor)
        result = engine.simulate(1)

        self.assertEqual(result.exclusions.cooldown, 0)
        self.assertEqual(result.final_audience_count, 1)

    def test_estimated_cost_uses_template_category(self):
        """Cost should use the template's category rate."""
        cursor = MockCursor()
        cursor.results_queue = [
            # Campaign
            {"id": 1, "name": "Auth", "segment_id": 10, "template_id": 20,
             "campaign_type": "promotional", "status": "draft"},
            # Segment
            {"filter_criteria": json.dumps({"status": "active"})},
            # Recipients (5 people)
            [{"mobile": f"91987654321{i}", "customer_name": f"U{i}", "plan_name": "Plan"} for i in range(5)],
            # Template mappings — empty (no params needed)
            {"placeholder_mappings": None},
            # All pass suppression and cooldown
        ]
        for _ in range(5):
            cursor.results_queue.append(None)  # not suppressed
            cursor.results_queue.append({"cnt": 0})  # not in cooldown
        # Template category: authentication
        cursor.results_queue.append({"category": "authentication"})

        engine = self._make_engine_with_cursor(cursor)
        result = engine.simulate(1)

        # Cost = 5 * 0.28 = 1.40
        self.assertAlmostEqual(result.estimated_cost_inr, 1.40, places=2)


class TestExclusionBreakdown(unittest.TestCase):
    """Tests for ExclusionBreakdown data class."""

    def test_total_property(self):
        eb = ExclusionBreakdown(cooldown=2, opted_out=3, dnd=1, invalid_number=4, incomplete_data=5)
        self.assertEqual(eb.total, 15)

    def test_default_zero(self):
        eb = ExclusionBreakdown()
        self.assertEqual(eb.total, 0)


class TestSimulationHelpers(unittest.TestCase):
    """Tests for SimulationEngine helper methods."""

    def test_is_valid_mobile_valid_numbers(self):
        engine = SimulationEngine(lambda: MockConnection())
        self.assertTrue(engine._is_valid_mobile("919876543210"))
        self.assertTrue(engine._is_valid_mobile("+919876543210"))
        self.assertTrue(engine._is_valid_mobile("1234567"))  # min 7 digits

    def test_is_valid_mobile_invalid_numbers(self):
        engine = SimulationEngine(lambda: MockConnection())
        self.assertFalse(engine._is_valid_mobile(""))
        self.assertFalse(engine._is_valid_mobile(None))
        self.assertFalse(engine._is_valid_mobile("abc"))
        self.assertFalse(engine._is_valid_mobile("123"))  # too short
        self.assertFalse(engine._is_valid_mobile("1234567890123456"))  # too long (16 digits)
        self.assertFalse(engine._is_valid_mobile("91-9876-543210"))  # contains non-digit

    def test_has_complete_params_all_present(self):
        engine = SimulationEngine(lambda: MockConnection())
        recipient = {"customer_name": "Alice", "plan_name": "Eclipse 100"}
        mappings = {"1": "customer_name", "2": "plan_name"}
        self.assertTrue(engine._has_complete_params(recipient, mappings))

    def test_has_complete_params_missing_field(self):
        engine = SimulationEngine(lambda: MockConnection())
        recipient = {"customer_name": "Alice"}
        mappings = {"1": "customer_name", "2": "plan_name"}
        self.assertFalse(engine._has_complete_params(recipient, mappings))

    def test_has_complete_params_empty_value(self):
        engine = SimulationEngine(lambda: MockConnection())
        recipient = {"customer_name": "Alice", "plan_name": "  "}
        mappings = {"1": "customer_name", "2": "plan_name"}
        self.assertFalse(engine._has_complete_params(recipient, mappings))

    def test_has_complete_params_none_value(self):
        engine = SimulationEngine(lambda: MockConnection())
        recipient = {"customer_name": "Alice", "plan_name": None}
        mappings = {"1": "customer_name", "2": "plan_name"}
        self.assertFalse(engine._has_complete_params(recipient, mappings))


if __name__ == "__main__":
    unittest.main()
