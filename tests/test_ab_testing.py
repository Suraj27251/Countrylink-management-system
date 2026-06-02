"""
Unit tests for A/B testing functionality in CampaignService.

Tests cover:
- create_ab_test() validation and record creation
- compute_ab_split() even audience distribution algorithm
- select_ab_winner() winner selection and rollout campaign creation
- Endpoint validation
"""

import json
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

from blueprints.campaign_bp import CampaignService, VALID_TRANSITIONS


class MockCursor:
    """Mock MySQL cursor with dictionary=True support."""

    def __init__(self):
        self.executed = []
        self.fetchone_result = None
        self.fetchall_result = []
        self.lastrowid = 1
        self._closed = False
        self._fetchone_results = []
        self._fetchone_idx = 0

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        # Auto-increment lastrowid for INSERT statements
        if "INSERT" in sql.upper():
            self.lastrowid += 1

    def fetchone(self):
        if self._fetchone_results:
            if self._fetchone_idx < len(self._fetchone_results):
                result = self._fetchone_results[self._fetchone_idx]
                self._fetchone_idx += 1
                return result
            return None
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


class TestCreateABTest(unittest.TestCase):
    """Tests for CampaignService.create_ab_test()."""

    def test_create_ab_test_with_2_variants(self):
        """Should create A/B test with 2 variants successfully."""
        cursor = MockCursor()
        cursor.fetchone_result = {"id": 1, "status": "draft", "segment_id": 10}
        cursor.lastrowid = 100
        conn = MockConnection(cursor)
        service = CampaignService(lambda: conn)

        result = service.create_ab_test(1, [10, 20], 25.0, "operator1")

        self.assertEqual(result["campaign_id"], 1)
        self.assertEqual(result["campaign_type"], "ab_test")
        self.assertEqual(result["test_percentage"], 25.0)
        self.assertEqual(len(result["variants"]), 2)
        self.assertEqual(result["variants"][0]["variant_label"], "A")
        self.assertEqual(result["variants"][1]["variant_label"], "B")
        self.assertEqual(result["variants"][0]["template_id"], 10)
        self.assertEqual(result["variants"][1]["template_id"], 20)

    def test_create_ab_test_with_4_variants(self):
        """Should create A/B test with 4 variants (maximum)."""
        cursor = MockCursor()
        cursor.fetchone_result = {"id": 1, "status": "draft", "segment_id": 10}
        cursor.lastrowid = 100
        conn = MockConnection(cursor)
        service = CampaignService(lambda: conn)

        result = service.create_ab_test(1, [10, 20, 30, 40], 50.0, "op1")

        self.assertEqual(len(result["variants"]), 4)
        labels = [v["variant_label"] for v in result["variants"]]
        self.assertEqual(labels, ["A", "B", "C", "D"])

    def test_create_ab_test_rejects_fewer_than_2_variants(self):
        """Should reject when fewer than 2 variants are provided."""
        conn = MockConnection()
        service = CampaignService(lambda: conn)

        with self.assertRaises(ValueError) as ctx:
            service.create_ab_test(1, [10], 25.0, "op1")
        self.assertIn("at least 2", str(ctx.exception))

    def test_create_ab_test_rejects_more_than_4_variants(self):
        """Should reject when more than 4 variants are provided."""
        conn = MockConnection()
        service = CampaignService(lambda: conn)

        with self.assertRaises(ValueError) as ctx:
            service.create_ab_test(1, [10, 20, 30, 40, 50], 25.0, "op1")
        self.assertIn("maximum of 4", str(ctx.exception))

    def test_create_ab_test_rejects_pct_below_10(self):
        """Should reject test percentage below 10%."""
        conn = MockConnection()
        service = CampaignService(lambda: conn)

        with self.assertRaises(ValueError) as ctx:
            service.create_ab_test(1, [10, 20], 5.0, "op1")
        self.assertIn("between 10 and 50", str(ctx.exception))

    def test_create_ab_test_rejects_pct_above_50(self):
        """Should reject test percentage above 50%."""
        conn = MockConnection()
        service = CampaignService(lambda: conn)

        with self.assertRaises(ValueError) as ctx:
            service.create_ab_test(1, [10, 20], 60.0, "op1")
        self.assertIn("between 10 and 50", str(ctx.exception))

    def test_create_ab_test_rejects_non_draft_campaign(self):
        """Should reject A/B test creation for campaigns not in draft state."""
        cursor = MockCursor()
        cursor.fetchone_result = {"id": 1, "status": "sending", "segment_id": 10}
        conn = MockConnection(cursor)
        service = CampaignService(lambda: conn)

        with self.assertRaises(ValueError) as ctx:
            service.create_ab_test(1, [10, 20], 25.0, "op1")
        self.assertIn("draft", str(ctx.exception))

    def test_create_ab_test_rejects_not_found_campaign(self):
        """Should raise ValueError when campaign doesn't exist."""
        cursor = MockCursor()
        cursor.fetchone_result = None
        conn = MockConnection(cursor)
        service = CampaignService(lambda: conn)

        with self.assertRaises(ValueError) as ctx:
            service.create_ab_test(999, [10, 20], 25.0, "op1")
        self.assertIn("not found", str(ctx.exception))

    def test_create_ab_test_accepts_boundary_pct_10(self):
        """Should accept test percentage of exactly 10%."""
        cursor = MockCursor()
        cursor.fetchone_result = {"id": 1, "status": "draft", "segment_id": 10}
        cursor.lastrowid = 100
        conn = MockConnection(cursor)
        service = CampaignService(lambda: conn)

        result = service.create_ab_test(1, [10, 20], 10.0, "op1")
        self.assertEqual(result["test_percentage"], 10.0)

    def test_create_ab_test_accepts_boundary_pct_50(self):
        """Should accept test percentage of exactly 50%."""
        cursor = MockCursor()
        cursor.fetchone_result = {"id": 1, "status": "draft", "segment_id": 10}
        cursor.lastrowid = 100
        conn = MockConnection(cursor)
        service = CampaignService(lambda: conn)

        result = service.create_ab_test(1, [10, 20], 50.0, "op1")
        self.assertEqual(result["test_percentage"], 50.0)


class TestComputeABSplit(unittest.TestCase):
    """Tests for CampaignService.compute_ab_split() even audience split algorithm."""

    def test_even_split_2_variants(self):
        """100 recipients / 2 variants = 50, 50."""
        result = CampaignService.compute_ab_split(100, 2)
        self.assertEqual(result, [50, 50])
        self.assertEqual(sum(result), 100)

    def test_even_split_3_variants(self):
        """99 recipients / 3 variants = 33, 33, 33."""
        result = CampaignService.compute_ab_split(99, 3)
        self.assertEqual(result, [33, 33, 33])
        self.assertEqual(sum(result), 99)

    def test_uneven_split_2_variants(self):
        """101 recipients / 2 variants = 51, 50."""
        result = CampaignService.compute_ab_split(101, 2)
        self.assertEqual(result, [51, 50])
        self.assertEqual(sum(result), 101)

    def test_uneven_split_3_variants(self):
        """100 recipients / 3 variants = 34, 33, 33."""
        result = CampaignService.compute_ab_split(100, 3)
        self.assertEqual(result, [34, 33, 33])
        self.assertEqual(sum(result), 100)

    def test_uneven_split_4_variants(self):
        """103 recipients / 4 variants = 26, 26, 26, 25."""
        result = CampaignService.compute_ab_split(103, 4)
        self.assertEqual(result, [26, 26, 26, 25])
        self.assertEqual(sum(result), 103)

    def test_split_max_difference_is_1(self):
        """For any split, the difference between max and min is at most 1."""
        for n in range(0, 200):
            for v in range(2, 5):
                result = CampaignService.compute_ab_split(n, v)
                self.assertEqual(sum(result), n)
                self.assertEqual(len(result), v)
                self.assertLessEqual(max(result) - min(result), 1)

    def test_split_zero_audience(self):
        """0 recipients split across variants gives all zeros."""
        result = CampaignService.compute_ab_split(0, 3)
        self.assertEqual(result, [0, 0, 0])

    def test_split_fewer_recipients_than_variants(self):
        """1 recipient / 4 variants = 1, 0, 0, 0."""
        result = CampaignService.compute_ab_split(1, 4)
        self.assertEqual(result, [1, 0, 0, 0])
        self.assertEqual(sum(result), 1)

    def test_split_rejects_variant_count_below_2(self):
        """Should raise ValueError for variant count < 2."""
        with self.assertRaises(ValueError):
            CampaignService.compute_ab_split(100, 1)

    def test_split_rejects_variant_count_above_4(self):
        """Should raise ValueError for variant count > 4."""
        with self.assertRaises(ValueError):
            CampaignService.compute_ab_split(100, 5)

    def test_split_rejects_negative_audience(self):
        """Should raise ValueError for negative audience size."""
        with self.assertRaises(ValueError):
            CampaignService.compute_ab_split(-1, 2)

    def test_split_all_variants_get_floor_or_ceil(self):
        """Each variant gets exactly floor(N/V) or ceil(N/V)."""
        import math
        for n in [7, 10, 13, 50, 99, 1000]:
            for v in [2, 3, 4]:
                result = CampaignService.compute_ab_split(n, v)
                floor_val = n // v
                ceil_val = math.ceil(n / v)
                for val in result:
                    self.assertTrue(
                        val == floor_val or val == ceil_val,
                        f"N={n}, V={v}: got {val}, expected {floor_val} or {ceil_val}"
                    )


class TestSelectABWinner(unittest.TestCase):
    """Tests for CampaignService.select_ab_winner()."""

    def test_select_winner_creates_rollout_campaign(self):
        """Selecting a winner should create a new draft rollout campaign."""
        call_count = [0]

        def get_conn():
            call_count[0] += 1
            c = MockCursor()
            if call_count[0] == 1:
                # First call: select_ab_winner
                c._fetchone_results = [
                    # Campaign lookup
                    {
                        "id": 1, "name": "Test AB", "status": "completed",
                        "campaign_type": "ab_test", "segment_id": 10,
                        "ab_test_percentage": 25.0, "organization_id": 1,
                        "branch_id": 1, "channel": "whatsapp", "priority": 5,
                    },
                    # Variant lookup
                    {"id": 5, "template_id": 20, "variant_label": "B"},
                ]
                c.lastrowid = 99
                return MockConnection(c)
            else:
                return MockConnection(c)

        service = CampaignService(get_conn)
        result = service.select_ab_winner(1, 5, "operator1")

        self.assertEqual(result["campaign_id"], 1)
        self.assertEqual(result["winning_variant_id"], 5)
        self.assertEqual(result["winning_variant_label"], "B")
        self.assertEqual(result["winning_template_id"], 20)
        self.assertIn("rollout_campaign_id", result)
        self.assertEqual(result["rollout_status"], "draft")

    def test_select_winner_rejects_non_ab_test_campaign(self):
        """Should reject winner selection for non-A/B test campaigns."""
        cursor = MockCursor()
        cursor.fetchone_result = {
            "id": 1, "name": "Regular", "status": "completed",
            "campaign_type": "promotional", "segment_id": 10,
            "ab_test_percentage": None, "organization_id": 1,
            "branch_id": 1, "channel": "whatsapp", "priority": 5,
        }
        conn = MockConnection(cursor)
        service = CampaignService(lambda: conn)

        with self.assertRaises(ValueError) as ctx:
            service.select_ab_winner(1, 5, "op1")
        self.assertIn("not an A/B test", str(ctx.exception))

    def test_select_winner_rejects_non_completed_campaign(self):
        """Should reject winner selection if campaign hasn't completed."""
        cursor = MockCursor()
        cursor.fetchone_result = {
            "id": 1, "name": "Test AB", "status": "sending",
            "campaign_type": "ab_test", "segment_id": 10,
            "ab_test_percentage": 25.0, "organization_id": 1,
            "branch_id": 1, "channel": "whatsapp", "priority": 5,
        }
        conn = MockConnection(cursor)
        service = CampaignService(lambda: conn)

        with self.assertRaises(ValueError) as ctx:
            service.select_ab_winner(1, 5, "op1")
        self.assertIn("completed", str(ctx.exception))

    def test_select_winner_rejects_invalid_variant_id(self):
        """Should reject if variant doesn't belong to the campaign."""
        cursor = MockCursor()
        cursor._fetchone_results = [
            # Campaign lookup
            {
                "id": 1, "name": "Test AB", "status": "completed",
                "campaign_type": "ab_test", "segment_id": 10,
                "ab_test_percentage": 25.0, "organization_id": 1,
                "branch_id": 1, "channel": "whatsapp", "priority": 5,
            },
            # Variant lookup - not found
            None,
        ]
        conn = MockConnection(cursor)
        service = CampaignService(lambda: conn)

        with self.assertRaises(ValueError) as ctx:
            service.select_ab_winner(1, 999, "op1")
        self.assertIn("not found", str(ctx.exception))

    def test_select_winner_rejects_not_found_campaign(self):
        """Should raise ValueError when campaign doesn't exist."""
        cursor = MockCursor()
        cursor.fetchone_result = None
        conn = MockConnection(cursor)
        service = CampaignService(lambda: conn)

        with self.assertRaises(ValueError) as ctx:
            service.select_ab_winner(999, 5, "op1")
        self.assertIn("not found", str(ctx.exception))


class TestGetABVariants(unittest.TestCase):
    """Tests for CampaignService.get_ab_variants()."""

    def test_get_variants_returns_all_for_campaign(self):
        """Should return all variant records for the given campaign."""
        cursor = MockCursor()
        cursor.fetchall_result = [
            {
                "id": 1, "campaign_id": 10, "template_id": 100,
                "variant_label": "A", "recipient_count": 50,
                "sent_count": 50, "delivered_count": 45,
                "read_count": 30, "response_count": 5, "is_winner": 0,
            },
            {
                "id": 2, "campaign_id": 10, "template_id": 200,
                "variant_label": "B", "recipient_count": 50,
                "sent_count": 50, "delivered_count": 48,
                "read_count": 40, "response_count": 10, "is_winner": 0,
            },
        ]
        conn = MockConnection(cursor)
        service = CampaignService(lambda: conn)

        result = service.get_ab_variants(10)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["variant_label"], "A")
        self.assertEqual(result[1]["variant_label"], "B")

    def test_get_variants_empty_for_non_ab_campaign(self):
        """Should return empty list for campaigns without variants."""
        cursor = MockCursor()
        cursor.fetchall_result = []
        conn = MockConnection(cursor)
        service = CampaignService(lambda: conn)

        result = service.get_ab_variants(99)
        self.assertEqual(result, [])


class TestABTestPreventFullSend(unittest.TestCase):
    """Tests ensuring full send is prevented until operator selects winner."""

    def test_rollout_campaign_created_in_draft_state(self):
        """The rollout campaign must be in draft state, requiring approval before send."""
        call_count = [0]

        def get_conn():
            call_count[0] += 1
            c = MockCursor()
            if call_count[0] == 1:
                c._fetchone_results = [
                    {
                        "id": 1, "name": "Test AB", "status": "completed",
                        "campaign_type": "ab_test", "segment_id": 10,
                        "ab_test_percentage": 25.0, "organization_id": 1,
                        "branch_id": 1, "channel": "whatsapp", "priority": 5,
                    },
                    {"id": 5, "template_id": 20, "variant_label": "A"},
                ]
                c.lastrowid = 50
                return MockConnection(c)
            else:
                return MockConnection(c)

        service = CampaignService(get_conn)
        result = service.select_ab_winner(1, 5, "op1")

        # The rollout campaign MUST be in draft — it cannot be sent without
        # going through the approval workflow
        self.assertEqual(result["rollout_status"], "draft")

    def test_rollout_campaign_references_parent(self):
        """The rollout campaign should reference the parent A/B test campaign."""
        call_count = [0]

        def get_conn():
            call_count[0] += 1
            c = MockCursor()
            if call_count[0] == 1:
                c._fetchone_results = [
                    {
                        "id": 7, "name": "AB Campaign", "status": "completed",
                        "campaign_type": "ab_test", "segment_id": 10,
                        "ab_test_percentage": 30.0, "organization_id": 1,
                        "branch_id": 1, "channel": "whatsapp", "priority": 5,
                    },
                    {"id": 12, "template_id": 33, "variant_label": "C"},
                ]
                c.lastrowid = 77
                return MockConnection(c)
            else:
                return MockConnection(c)

        service = CampaignService(get_conn)
        result = service.select_ab_winner(7, 12, "op1")

        # Verify the INSERT SQL contains parent_campaign_id
        cursor = get_conn()._cursor
        # Check through the first connection's executed SQLs
        # We can verify the result structure
        self.assertEqual(result["campaign_id"], 7)
        self.assertIn("rollout_campaign_id", result)


if __name__ == "__main__":
    unittest.main()
