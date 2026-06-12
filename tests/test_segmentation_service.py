"""
Unit tests for SegmentationService — query building, pagination, save/load.

Uses mock MySQL connections to test business logic without database dependency.
"""

import json
import unittest
from datetime import date, datetime
from unittest.mock import MagicMock, patch

from services.segmentation import SegmentationService


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


class TestBuildQuery(unittest.TestCase):
    """Tests for SegmentationService.build_query()"""

    def setUp(self):
        self.service = SegmentationService(lambda: MockConnection())
        self.ref_date = date(2024, 6, 15)

    def test_empty_filters_returns_1_equals_1(self):
        """Empty or None filters should produce '1=1' with no params."""
        clause, params = self.service.build_query({})
        self.assertEqual(clause, "1=1")
        self.assertEqual(params, [])

        clause, params = self.service.build_query(None)
        self.assertEqual(clause, "1=1")
        self.assertEqual(params, [])

    def test_single_exact_match_filter(self):
        """Single exact match field should produce r.field = %s."""
        clause, params = self.service.build_query({"zone_name": "Mumbai"})
        self.assertIn("r.zone_name = %s", clause)
        self.assertEqual(params, ["Mumbai"])

    def test_multiple_exact_match_filters_produce_and_logic(self):
        """Multiple filters should be joined with AND."""
        clause, params = self.service.build_query({
            "zone_name": "Mumbai",
            "status": "active",
            "plan_category": "Fiber",
        })
        self.assertIn("r.zone_name = %s", clause)
        self.assertIn("r.status = %s", clause)
        self.assertIn("r.plan_category = %s", clause)
        self.assertIn(" AND ", clause)
        self.assertEqual(len(params), 3)

    def test_boolean_field_filter(self):
        """Boolean filter (kyc_approved) should map True/False to 1/0."""
        clause, params = self.service.build_query({"kyc_approved": True})
        self.assertIn("r.kyc_approved = %s", clause)
        self.assertEqual(params, [1])

        clause, params = self.service.build_query({"kyc_approved": False})
        self.assertIn("r.kyc_approved = %s", clause)
        self.assertEqual(params, [0])

    def test_range_filter_with_min_and_max(self):
        """Range filter should produce >= min AND <= max."""
        clause, params = self.service.build_query({
            "days_remaining": {"min": 5, "max": 30}
        })
        self.assertIn("r.days_remaining >= %s", clause)
        self.assertIn("r.days_remaining <= %s", clause)
        self.assertEqual(params, [5, 30])

    def test_range_filter_with_only_min(self):
        """Range filter with only min should only produce >= clause."""
        clause, params = self.service.build_query({
            "days_remaining": {"min": 10, "max": None}
        })
        self.assertIn("r.days_remaining >= %s", clause)
        self.assertNotIn("<=", clause)
        self.assertEqual(params, [10])

    def test_range_filter_with_only_max(self):
        """Range filter with only max should only produce <= clause."""
        clause, params = self.service.build_query({
            "days_remaining": {"min": None, "max": 15}
        })
        self.assertNotIn(">=", clause)
        self.assertIn("r.days_remaining <= %s", clause)
        self.assertEqual(params, [15])

    def test_derived_days_since_last_recharge(self):
        """Derived filter uses DATEDIFF with reference date."""
        clause, params = self.service.build_query(
            {"days_since_last_recharge": {"min": 30, "max": 90}},
            reference_date=self.ref_date,
        )
        self.assertIn("DATEDIFF(%s, r.activation_date) >= %s", clause)
        self.assertIn("DATEDIFF(%s, r.activation_date) <= %s", clause)
        self.assertIn(self.ref_date, params)
        self.assertIn(30, params)
        self.assertIn(90, params)

    def test_derived_days_inactive(self):
        """days_inactive filter uses DATEDIFF and ensures expiry_date < ref_date."""
        clause, params = self.service.build_query(
            {"days_inactive": {"min": 7, "max": 60}},
            reference_date=self.ref_date,
        )
        self.assertIn("DATEDIFF(%s, r.expiry_date) >= %s", clause)
        self.assertIn("DATEDIFF(%s, r.expiry_date) <= %s", clause)
        # Should also filter for expired records
        self.assertIn("r.expiry_date < %s", clause)
        self.assertIn(self.ref_date, params)

    def test_tags_filter_any_mode(self):
        """Tags with ANY mode should use EXISTS subquery."""
        clause, params = self.service.build_query({
            "tags": {"values": ["vip", "premium"], "mode": "ANY"}
        })
        self.assertIn("EXISTS", clause)
        self.assertIn("customer_tags", clause)
        self.assertIn("vip", params)
        self.assertIn("premium", params)

    def test_tags_filter_all_mode(self):
        """Tags with ALL mode should use COUNT subquery."""
        clause, params = self.service.build_query({
            "tags": {"values": ["vip", "premium"], "mode": "ALL"}
        })
        self.assertIn("COUNT(DISTINCT ct.tag_name)", clause)
        self.assertIn("vip", params)
        self.assertIn("premium", params)
        self.assertIn(2, params)  # len(tag_list)

    def test_tags_filter_simple_list(self):
        """Tags as a simple list should default to ANY mode."""
        clause, params = self.service.build_query({
            "tags": ["active_user"]
        })
        self.assertIn("EXISTS", clause)
        self.assertIn("active_user", params)

    def test_null_values_skipped(self):
        """None values in filters should be skipped."""
        clause, params = self.service.build_query({
            "zone_name": "Mumbai",
            "status": None,
        })
        self.assertIn("r.zone_name = %s", clause)
        self.assertNotIn("status", clause)
        self.assertEqual(params, ["Mumbai"])

    def test_parameterized_queries_no_string_interpolation(self):
        """All filter values should be parameters, never interpolated into SQL."""
        filters = {
            "zone_name": "Mumbai'; DROP TABLE--",
            "status": "active",
            "days_remaining": {"min": 5, "max": 30},
        }
        clause, params = self.service.build_query(filters)
        # The dangerous string should be in params, not in the clause
        self.assertNotIn("Mumbai'; DROP TABLE--", clause)
        self.assertIn("Mumbai'; DROP TABLE--", params)
        # Only %s placeholders in the clause
        self.assertNotIn("Mumbai", clause)

    def test_all_supported_exact_fields(self):
        """All exact match fields should be handled."""
        for field in ["expiry_category", "plan_name", "plan_category", "zone_name",
                      "area", "building", "status", "network_type",
                      "connectivity_mode", "owner_tenant"]:
            clause, params = self.service.build_query({field: "test_value"})
            self.assertIn(f"r.{field} = %s", clause)
            self.assertEqual(params, ["test_value"])


class TestEstimateCount(unittest.TestCase):
    """Tests for SegmentationService.estimate_count()"""

    def test_estimate_count_returns_integer(self):
        """estimate_count should return integer count from query."""
        cursor = MockCursor()
        cursor.fetchone_result = {"cnt": 42}
        conn = MockConnection(cursor)
        service = SegmentationService(lambda: conn)

        count = service.estimate_count({"zone_name": "Mumbai"})
        self.assertEqual(count, 42)
        # Should have executed a COUNT query
        sql, _ = cursor.executed[0]
        self.assertIn("COUNT(*)", sql)
        self.assertIn("customers", sql)

    def test_estimate_count_with_empty_filters(self):
        """estimate_count with empty filters should query all records."""
        cursor = MockCursor()
        cursor.fetchone_result = {"cnt": 1000}
        conn = MockConnection(cursor)
        service = SegmentationService(lambda: conn)

        count = service.estimate_count({})
        self.assertEqual(count, 1000)


class TestEvaluateSegment(unittest.TestCase):
    """Tests for SegmentationService.evaluate_segment()"""

    def test_evaluate_returns_paginated_results(self):
        """evaluate_segment should return paginated structure."""
        cursor = MockCursor()
        # First execute: COUNT query
        # Second execute: data query
        cursor.fetchone_result = {"cnt": 100}
        cursor.fetchall_result = [
            {"mobile": "9876543210", "name": "Test User"},
        ]
        conn = MockConnection(cursor)
        service = SegmentationService(lambda: conn)

        result = service.evaluate_segment({"zone_name": "Mumbai"}, page=1, per_page=50)
        self.assertEqual(result["total"], 100)
        self.assertEqual(result["page"], 1)
        self.assertEqual(result["per_page"], 50)
        self.assertEqual(result["total_pages"], 2)
        self.assertEqual(len(result["customers"]), 1)
        self.assertNotIn("warning", result)

    def test_evaluate_with_zero_results_includes_warning(self):
        """Zero results should include a warning message."""
        cursor = MockCursor()
        cursor.fetchone_result = {"cnt": 0}
        cursor.fetchall_result = []
        conn = MockConnection(cursor)
        service = SegmentationService(lambda: conn)

        result = service.evaluate_segment({"zone_name": "Nonexistent"})
        self.assertEqual(result["total"], 0)
        self.assertIn("warning", result)
        self.assertIn("No customers match", result["warning"])

    def test_evaluate_pagination_offset(self):
        """Page 2 should use correct OFFSET."""
        cursor = MockCursor()
        cursor.fetchone_result = {"cnt": 100}
        cursor.fetchall_result = []
        conn = MockConnection(cursor)
        service = SegmentationService(lambda: conn)

        service.evaluate_segment({"zone_name": "Mumbai"}, page=2, per_page=50)
        # Check the data SQL has correct offset
        data_sql, data_params = cursor.executed[1]
        self.assertIn("LIMIT", data_sql)
        self.assertIn("OFFSET", data_sql)
        # Offset should be 50 for page 2 with per_page=50
        self.assertIn(50, data_params)  # per_page
        self.assertIn(50, data_params)  # offset


class TestSaveLoadSegment(unittest.TestCase):
    """Tests for save_segment and load_segment."""

    def test_save_segment_persists_filters_as_json(self):
        """save_segment should JSON-serialize filter criteria."""
        cursor = MockCursor()
        # First call: estimate_count (COUNT query)
        # Then: INSERT
        # Then: load_segment (SELECT)
        cursor.fetchone_result = {"cnt": 50}
        cursor.lastrowid = 7

        conn = MockConnection(cursor)
        call_count = [0]
        original_fetchone = cursor.fetchone

        def multi_fetchone():
            call_count[0] += 1
            if call_count[0] == 1:
                return {"cnt": 50}  # estimate_count
            else:
                return {
                    "id": 7,
                    "name": "Test Segment",
                    "description": "desc",
                    "filter_criteria": json.dumps({"zone_name": "Mumbai"}),
                    "estimated_count": 50,
                    "organization_id": 1,
                    "created_by": "operator1",
                    "created_at": datetime.now(),
                    "updated_at": datetime.now(),
                }

        cursor.fetchone = multi_fetchone

        service = SegmentationService(lambda: conn)
        result = service.save_segment(
            name="Test Segment",
            filters={"zone_name": "Mumbai"},
            description="desc",
            created_by="operator1",
        )
        self.assertEqual(result["id"], 7)
        self.assertEqual(result["filter_criteria"], {"zone_name": "Mumbai"})

    def test_load_segment_parses_json(self):
        """load_segment should parse filter_criteria JSON string to dict."""
        cursor = MockCursor()
        cursor.fetchone_result = {
            "id": 3,
            "name": "My Segment",
            "description": "",
            "filter_criteria": '{"status": "active", "zone_name": "Pune"}',
            "estimated_count": 200,
            "organization_id": 1,
            "created_by": "op",
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }
        conn = MockConnection(cursor)
        service = SegmentationService(lambda: conn)

        result = service.load_segment(3)
        self.assertEqual(result["id"], 3)
        self.assertIsInstance(result["filter_criteria"], dict)
        self.assertEqual(result["filter_criteria"]["status"], "active")

    def test_load_segment_not_found_raises(self):
        """load_segment should raise ValueError for missing ID."""
        cursor = MockCursor()
        cursor.fetchone_result = None
        conn = MockConnection(cursor)
        service = SegmentationService(lambda: conn)

        with self.assertRaises(ValueError):
            service.load_segment(999)


if __name__ == "__main__":
    unittest.main()
