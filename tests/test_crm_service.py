"""
Unit tests for CRMService — customer profile, notes, tags, interaction timeline,
campaign history, opt-out/DND status, and engagement score display.

Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 19.6, 23.7
"""

import json
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

from services.crm import CRMService


class MockCursor:
    """Mock MySQL cursor with dictionary=True support."""

    def __init__(self):
        self.executed = []
        self.fetchone_results = []
        self.fetchall_results = []
        self._fetchone_idx = 0
        self._fetchall_idx = 0
        self._closed = False
        self.rowcount = 0
        self.lastrowid = 1

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchone(self):
        if self._fetchone_idx < len(self.fetchone_results):
            result = self.fetchone_results[self._fetchone_idx]
            self._fetchone_idx += 1
            return result
        return None

    def fetchall(self):
        if self._fetchall_idx < len(self.fetchall_results):
            result = self.fetchall_results[self._fetchall_idx]
            self._fetchall_idx += 1
            return result
        return []

    def close(self):
        self._closed = True


class MockConnection:
    """Mock MySQL connection."""

    def __init__(self, cursor_instance=None):
        self._cursor = cursor_instance or MockCursor()
        self._committed = False
        self._rolled_back = False
        self._connected = True

    def cursor(self, dictionary=False):
        return self._cursor

    def commit(self):
        self._committed = True

    def rollback(self):
        self._rolled_back = True

    def start_transaction(self):
        pass

    def close(self):
        self._connected = False

    def is_connected(self):
        return self._connected


class TestGetCustomerProfile(unittest.TestCase):
    """Tests for CRMService.get_customer_profile()."""

    def _make_service(self, cursor):
        conn = MockConnection(cursor)
        return CRMService(lambda: conn)

    def test_profile_by_id_returns_full_profile(self):
        """Profile by ID returns customer data + opt-out + engagement + tags."""
        cursor = MockCursor()
        # renewal_records query
        cursor.fetchone_results.append({
            "id": 1,
            "mobile": "919876543210",
            "customer_name": "Rahul Sharma",
            "plan_name": "Unlimited 100",
            "plan_category": "broadband",
            "validity": 30,
            "days_remaining": 15,
            "status": "active",
            "zone_name": "Zone A",
            "area": "Sector 5",
            "building": "Tower B",
            "network_type": "fiber",
            "connectivity_mode": "FTTH",
            "activation_date": "2024-01-15",
            "expiry_date": "2024-12-15",
            "kyc_approved": 1,
            "owner_tenant": "owner",
        })
        # suppression_list query -> no suppression
        cursor.fetchall_results.append([])
        # customer_engagement query -> has engagement data
        cursor.fetchone_results.append({
            "interaction_score": 75,
            "engagement_trend": "increasing",
            "last_interaction_at": datetime(2024, 6, 1, 10, 0, 0),
            "messages_received_count": 10,
            "messages_read_count": 8,
            "response_count": 3,
            "avg_time_to_read_seconds": 120,
            "preferred_time_window": "morning",
            "avg_response_time_seconds": 300,
        })
        # customer_tags query
        cursor.fetchall_results.append([
            {"tag_name": "VIP", "added_by": "admin", "created_at": datetime(2024, 5, 1)},
        ])

        service = self._make_service(cursor)
        profile = service.get_customer_profile(customer_id=1)

        self.assertEqual(profile["mobile"], "919876543210")
        self.assertEqual(profile["customer_name"], "Rahul Sharma")
        self.assertEqual(profile["plan_name"], "Unlimited 100")
        self.assertEqual(profile["status"], "active")
        self.assertFalse(profile["opt_out_status"]["opted_out"])
        self.assertFalse(profile["dnd_status"]["dnd_active"])
        self.assertEqual(profile["engagement"]["score"], 75)
        self.assertEqual(profile["engagement"]["trend"], "increasing")
        self.assertEqual(profile["tags"], ["VIP"])

    def test_profile_with_opt_out_and_dnd(self):
        """Profile shows opt-out and DND status from suppression list."""
        cursor = MockCursor()
        cursor.fetchone_results.append({
            "id": 2,
            "mobile": "919876543211",
            "customer_name": "Priya Singh",
            "plan_name": "Basic 50",
            "plan_category": "broadband",
            "validity": 30,
            "days_remaining": 0,
            "status": "expired",
            "zone_name": "Zone B",
            "area": None,
            "building": None,
            "network_type": None,
            "connectivity_mode": None,
            "activation_date": None,
            "expiry_date": "2024-05-01",
            "kyc_approved": 0,
            "owner_tenant": None,
        })
        # suppression_list: opted-out + DND
        cursor.fetchall_results.append([
            {
                "reason": "opt_out_keyword",
                "source_keyword": "STOP",
                "created_at": datetime(2024, 6, 15),
            },
            {
                "reason": "manual_dnd",
                "source_keyword": None,
                "created_at": datetime(2024, 6, 10),
            },
        ])
        # No engagement data
        cursor.fetchone_results.append(None)
        # No tags
        cursor.fetchall_results.append([])

        service = self._make_service(cursor)
        profile = service.get_customer_profile(customer_id=2)

        self.assertTrue(profile["opt_out_status"]["opted_out"])
        self.assertEqual(profile["opt_out_status"]["source"], "keyword")
        self.assertEqual(profile["opt_out_status"]["keyword"], "STOP")
        self.assertTrue(profile["dnd_status"]["dnd_active"])
        self.assertEqual(profile["engagement"]["score"], 0)
        self.assertEqual(profile["engagement"]["trend"], "stable")

    def test_profile_not_found_raises_value_error(self):
        """Profile raises ValueError when customer not found."""
        cursor = MockCursor()
        cursor.fetchone_results.append(None)

        service = self._make_service(cursor)
        with self.assertRaises(ValueError) as ctx:
            service.get_customer_profile(customer_id=999)
        self.assertIn("not found", str(ctx.exception))

    def test_profile_requires_id_or_mobile(self):
        """Profile raises ValueError if neither customer_id nor mobile given."""
        cursor = MockCursor()
        service = self._make_service(cursor)
        with self.assertRaises(ValueError):
            service.get_customer_profile()


class TestInteractionTimeline(unittest.TestCase):
    """Tests for CRMService.get_interaction_timeline()."""

    def _make_service(self, cursor):
        conn = MockConnection(cursor)
        return CRMService(lambda: conn)

    def test_timeline_returns_reverse_chronological_events(self):
        """Timeline returns events sorted by timestamp descending."""
        cursor = MockCursor()
        # _resolve_mobile query
        cursor.fetchone_results.append({"mobile": "919876543210"})
        # COUNT query
        cursor.fetchone_results.append({"total": 3})
        # Activities
        cursor.fetchall_results.append([
            {
                "id": 3,
                "customer_mobile": "919876543210",
                "activity_type": "note_added",
                "channel": "system",
                "reference_id": 10,
                "details": json.dumps({"note_text": "Called customer"}),
                "created_at": datetime(2024, 6, 3, 12, 0, 0),
            },
            {
                "id": 2,
                "customer_mobile": "919876543210",
                "activity_type": "campaign_sent",
                "channel": "whatsapp",
                "reference_id": 5,
                "details": json.dumps({"campaign_name": "Summer Sale"}),
                "created_at": datetime(2024, 6, 2, 10, 0, 0),
            },
            {
                "id": 1,
                "customer_mobile": "919876543210",
                "activity_type": "tag_added",
                "channel": "system",
                "reference_id": 1,
                "details": json.dumps({"tag_name": "VIP"}),
                "created_at": datetime(2024, 6, 1, 8, 0, 0),
            },
        ])

        service = self._make_service(cursor)
        result = service.get_interaction_timeline(customer_id=1, page=1, per_page=50)

        self.assertEqual(result["total"], 3)
        self.assertEqual(len(result["timeline"]), 3)
        # Verify reverse chronological order
        timestamps = [e["timestamp"] for e in result["timeline"]]
        self.assertEqual(timestamps, sorted(timestamps, reverse=True))

    def test_timeline_pagination(self):
        """Timeline supports pagination."""
        cursor = MockCursor()
        cursor.fetchone_results.append({"mobile": "919876543210"})
        cursor.fetchone_results.append({"total": 100})
        cursor.fetchall_results.append([])

        service = self._make_service(cursor)
        result = service.get_interaction_timeline(customer_id=1, page=2, per_page=20)

        self.assertEqual(result["total"], 100)
        self.assertEqual(result["page"], 2)
        self.assertEqual(result["per_page"], 20)
        self.assertEqual(result["total_pages"], 5)


class TestAddNote(unittest.TestCase):
    """Tests for CRMService.add_note()."""

    def _make_service(self, cursor):
        conn = MockConnection(cursor)
        return CRMService(lambda: conn)

    def test_add_note_success(self):
        """Adding a note persists it and records activity."""
        cursor = MockCursor()
        # _resolve_mobile
        cursor.fetchone_results.append({"mobile": "919876543210"})
        cursor.lastrowid = 42
        # Final select for the created note
        cursor.fetchone_results.append({
            "id": 42,
            "customer_mobile": "919876543210",
            "note_text": "Customer called about upgrade",
            "added_by": "operator1",
            "created_at": datetime(2024, 6, 1),
        })

        service = self._make_service(cursor)
        note = service.add_note(
            customer_id=1, note="Customer called about upgrade", operator="operator1"
        )

        self.assertEqual(note["id"], 42)
        self.assertEqual(note["note_text"], "Customer called about upgrade")
        self.assertEqual(note["added_by"], "operator1")

    def test_add_note_empty_text_raises(self):
        """Empty note text raises ValueError."""
        cursor = MockCursor()
        service = self._make_service(cursor)
        with self.assertRaises(ValueError) as ctx:
            service.add_note(mobile="919876543210", note="", operator="admin")
        self.assertIn("empty", str(ctx.exception).lower())

    def test_add_note_no_operator_raises(self):
        """Missing operator name raises ValueError."""
        cursor = MockCursor()
        service = self._make_service(cursor)
        with self.assertRaises(ValueError) as ctx:
            service.add_note(mobile="919876543210", note="A note", operator="")
        self.assertIn("operator", str(ctx.exception).lower())


class TestTags(unittest.TestCase):
    """Tests for CRMService.add_tags() and remove_tag()."""

    def _make_service(self, cursor):
        conn = MockConnection(cursor)
        return CRMService(lambda: conn)

    def test_add_tags_returns_added_list(self):
        """add_tags returns the list of successfully added tags."""
        cursor = MockCursor()
        # _resolve_mobile
        cursor.fetchone_results.append({"mobile": "919876543210"})
        # Simulate INSERT IGNORE success (rowcount > 0)
        cursor.rowcount = 1
        cursor.lastrowid = 10

        service = self._make_service(cursor)
        added = service.add_tags(
            customer_id=1, tags=["VIP", "upgrade_interested"], operator="admin"
        )

        # Both tags get rowcount=1 (mock doesn't change per call, so both succeed)
        self.assertEqual(len(added), 2)
        self.assertIn("VIP", added)
        self.assertIn("upgrade_interested", added)

    def test_add_tags_empty_raises(self):
        """Empty tags list raises ValueError."""
        cursor = MockCursor()
        service = self._make_service(cursor)
        with self.assertRaises(ValueError):
            service.add_tags(mobile="919876543210", tags=[], operator="admin")

    def test_add_tags_no_operator_raises(self):
        """Missing operator raises ValueError."""
        cursor = MockCursor()
        service = self._make_service(cursor)
        with self.assertRaises(ValueError):
            service.add_tags(mobile="919876543210", tags=["VIP"], operator="")

    def test_remove_tag_success(self):
        """remove_tag returns True when tag existed."""
        cursor = MockCursor()
        # _resolve_mobile
        cursor.fetchone_results.append({"mobile": "919876543210"})
        cursor.rowcount = 1  # Simulates successful DELETE

        service = self._make_service(cursor)
        removed = service.remove_tag(
            customer_id=1, tag="VIP", operator="admin"
        )

        self.assertTrue(removed)

    def test_remove_tag_not_found(self):
        """remove_tag returns False when tag doesn't exist."""
        cursor = MockCursor()
        cursor.fetchone_results.append({"mobile": "919876543210"})
        cursor.rowcount = 0  # No rows deleted

        service = self._make_service(cursor)
        removed = service.remove_tag(
            customer_id=1, tag="nonexistent", operator="admin"
        )

        self.assertFalse(removed)

    def test_remove_tag_empty_raises(self):
        """Empty tag name raises ValueError."""
        cursor = MockCursor()
        service = self._make_service(cursor)
        with self.assertRaises(ValueError):
            service.remove_tag(mobile="919876543210", tag="", operator="admin")


class TestCampaignHistory(unittest.TestCase):
    """Tests for CRMService.get_campaign_history()."""

    def _make_service(self, cursor):
        conn = MockConnection(cursor)
        return CRMService(lambda: conn)

    def test_campaign_history_returns_deliveries(self):
        """Campaign history returns list of campaign deliveries."""
        cursor = MockCursor()
        # _resolve_mobile
        cursor.fetchone_results.append({"mobile": "919876543210"})
        # Campaign history query
        cursor.fetchall_results.append([
            {
                "message_id": 1,
                "campaign_id": 10,
                "campaign_name": "Summer Sale",
                "campaign_type": "promotional",
                "template_name": "summer_offer",
                "delivery_status": "delivered",
                "sent_at": datetime(2024, 6, 1, 10, 0, 0),
                "delivered_at": datetime(2024, 6, 1, 10, 0, 5),
                "read_at": datetime(2024, 6, 1, 10, 5, 0),
                "failed_at": None,
                "error_message": None,
                "channel": "whatsapp",
            },
            {
                "message_id": 2,
                "campaign_id": 8,
                "campaign_name": "Renewal Reminder",
                "campaign_type": "transactional",
                "template_name": "renewal_reminder",
                "delivery_status": "read",
                "sent_at": datetime(2024, 5, 15, 9, 0, 0),
                "delivered_at": datetime(2024, 5, 15, 9, 0, 3),
                "read_at": datetime(2024, 5, 15, 9, 10, 0),
                "failed_at": None,
                "error_message": None,
                "channel": "whatsapp",
            },
        ])

        service = self._make_service(cursor)
        history = service.get_campaign_history(customer_id=1)

        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["campaign_name"], "Summer Sale")
        self.assertEqual(history[0]["delivery_status"], "delivered")
        self.assertEqual(history[1]["campaign_name"], "Renewal Reminder")

    def test_campaign_history_empty(self):
        """Campaign history returns empty list if no campaigns targeted customer."""
        cursor = MockCursor()
        cursor.fetchone_results.append({"mobile": "919876543210"})
        cursor.fetchall_results.append([])

        service = self._make_service(cursor)
        history = service.get_campaign_history(customer_id=1)

        self.assertEqual(history, [])


if __name__ == "__main__":
    unittest.main()
