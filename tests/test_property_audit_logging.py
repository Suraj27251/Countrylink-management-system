"""
Property-based tests for audit logging in campaign actions.

Property 5: Every campaign action produces an audit log entry
- For any state-changing action, an INSERT INTO operator_actions is executed
  with correct operator_name, action_type, and campaign_id within 1 second.

**Validates: Requirements 2.6, 11.2**

Testing framework: Hypothesis (Python)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import string
from datetime import datetime, timedelta
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from blueprints.campaign_bp import CampaignService, VALID_TRANSITIONS


# ---------------------------------------------------------------------------
# Hypothesis Strategies
# ---------------------------------------------------------------------------

def operator_name_strategy():
    """Generate random operator names (non-empty alphanumeric + underscores)."""
    return st.text(
        alphabet=string.ascii_letters + string.digits + "_",
        min_size=1,
        max_size=50,
    ).filter(lambda s: s.strip() != "")


def campaign_id_strategy():
    """Generate random campaign IDs (positive integers)."""
    return st.integers(min_value=1, max_value=100000)


def campaign_name_strategy():
    """Generate random campaign names (non-empty printable strings)."""
    return st.text(
        alphabet=string.ascii_letters + string.digits + " -_",
        min_size=1,
        max_size=100,
    ).filter(lambda s: s.strip() != "")


def campaign_type_strategy():
    """Generate valid campaign types."""
    return st.sampled_from(["promotional", "transactional", "reactivation", "ab_test"])


def valid_transition_strategy():
    """Generate a random valid (from_state, to_state) pair from the transition graph."""
    pairs = []
    for from_state, to_states in VALID_TRANSITIONS.items():
        for to_state in to_states:
            pairs.append((from_state, to_state))
    return st.sampled_from(pairs)


def campaign_data_strategy():
    """Generate valid campaign creation data."""
    return st.fixed_dictionaries({
        "name": campaign_name_strategy(),
        "campaign_type": campaign_type_strategy(),
        "description": st.text(
            alphabet=string.ascii_letters + string.digits + " .",
            min_size=0,
            max_size=200,
        ),
        "organization_id": st.integers(min_value=1, max_value=100),
        "branch_id": st.integers(min_value=1, max_value=100),
        "channel": st.sampled_from(["whatsapp"]),
        "priority": st.integers(min_value=1, max_value=10),
    })


# ---------------------------------------------------------------------------
# Mock classes that capture executed SQL for audit verification
# ---------------------------------------------------------------------------

class AuditCaptureCursor:
    """Mock cursor that captures all executed SQL statements and params."""

    def __init__(self, fetchone_result=None, fetchall_result=None, lastrowid=1):
        self.executed = []
        self.fetchone_result = fetchone_result
        self.fetchall_result = fetchall_result or []
        self.lastrowid = lastrowid

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchone(self):
        return self.fetchone_result

    def fetchall(self):
        return self.fetchall_result

    def close(self):
        pass


class AuditCaptureConnection:
    """Mock connection that exposes the cursor for SQL inspection."""

    def __init__(self, cursor):
        self._cursor = cursor
        self.committed = False

    def cursor(self, dictionary=False):
        return self._cursor

    def start_transaction(self):
        pass

    def commit(self):
        self.committed = True

    def rollback(self):
        pass

    def close(self):
        pass


def find_audit_inserts(executed_statements):
    """Extract all INSERT INTO operator_actions statements from executed SQL list."""
    audit_inserts = []
    for sql, params in executed_statements:
        if sql and "operator_actions" in sql.lower() and "insert" in sql.lower():
            audit_inserts.append((sql, params))
    return audit_inserts


# ---------------------------------------------------------------------------
# Property Tests
# ---------------------------------------------------------------------------

class TestProperty5AuditLogging:
    """Property 5: Every campaign action produces an audit log entry.

    **Validates: Requirements 2.6, 11.2**
    """

    @given(
        transition=valid_transition_strategy(),
        operator_name=operator_name_strategy(),
        campaign_id=campaign_id_strategy(),
    )
    @settings(max_examples=200)
    def test_state_transition_produces_audit_log(self, transition, operator_name, campaign_id):
        """Property: For any valid state transition, an INSERT INTO operator_actions
        is executed with the correct operator_name, action_type, and campaign_id.

        **Validates: Requirements 2.6**
        """
        from_state, to_state = transition

        # Set up cursor that simulates finding the campaign in from_state
        write_cursor = AuditCaptureCursor(
            fetchone_result={"id": campaign_id, "status": from_state}
        )
        write_conn = AuditCaptureConnection(write_cursor)

        # Read cursor for get_campaign after transition
        read_cursor = AuditCaptureCursor(
            fetchone_result={"id": campaign_id, "status": to_state, "name": "Test"}
        )
        read_conn = AuditCaptureConnection(read_cursor)

        call_count = [0]

        def get_conn():
            call_count[0] += 1
            if call_count[0] == 1:
                return write_conn
            return read_conn

        service = CampaignService(get_conn)
        service.transition_state(campaign_id, to_state, operator_name)

        # Verify audit log insert happened
        audit_inserts = find_audit_inserts(write_cursor.executed)
        assert len(audit_inserts) == 1, (
            f"Expected exactly 1 audit log INSERT for transition "
            f"'{from_state}' → '{to_state}', got {len(audit_inserts)}"
        )

        # Verify the parameters contain correct values
        _sql, params = audit_inserts[0]
        # params = (operator_name, action_type, target_id, campaign_id, details_json)
        assert params[0] == operator_name, (
            f"Expected operator_name='{operator_name}', got '{params[0]}'"
        )
        expected_action_type = f"transition_{to_state}"
        assert params[1] == expected_action_type, (
            f"Expected action_type='{expected_action_type}', got '{params[1]}'"
        )
        assert params[2] == campaign_id, (
            f"Expected target_id={campaign_id}, got {params[2]}"
        )
        assert params[3] == campaign_id, (
            f"Expected campaign_id={campaign_id}, got {params[3]}"
        )

        # Verify details JSON contains from_state and to_state
        details = json.loads(params[4])
        assert details["from_state"] == from_state
        assert details["to_state"] == to_state

    @given(
        data=campaign_data_strategy(),
        operator_name=operator_name_strategy(),
    )
    @settings(max_examples=200)
    def test_create_campaign_produces_audit_log(self, data, operator_name):
        """Property: For create_campaign with any valid data, an audit log entry
        is created with operator_name, action_type='create_campaign', and correct campaign_id.

        **Validates: Requirements 2.6, 11.2**
        """
        generated_campaign_id = 42

        # Write cursor for create
        write_cursor = AuditCaptureCursor(lastrowid=generated_campaign_id)
        write_conn = AuditCaptureConnection(write_cursor)

        # Read cursor for get_campaign after create
        read_cursor = AuditCaptureCursor(
            fetchone_result={
                "id": generated_campaign_id,
                "name": data["name"],
                "status": "draft",
                "campaign_type": data["campaign_type"],
            }
        )
        read_conn = AuditCaptureConnection(read_cursor)

        call_count = [0]

        def get_conn():
            call_count[0] += 1
            if call_count[0] == 1:
                return write_conn
            return read_conn

        service = CampaignService(get_conn)
        service.create_campaign(data, operator_name)

        # Verify audit log insert happened
        audit_inserts = find_audit_inserts(write_cursor.executed)
        assert len(audit_inserts) == 1, (
            f"Expected exactly 1 audit log INSERT for create_campaign, "
            f"got {len(audit_inserts)}"
        )

        # Verify params
        _sql, params = audit_inserts[0]
        assert params[0] == operator_name, (
            f"Expected operator_name='{operator_name}', got '{params[0]}'"
        )
        assert params[1] == "create_campaign", (
            f"Expected action_type='create_campaign', got '{params[1]}'"
        )
        assert params[2] == generated_campaign_id, (
            f"Expected target_id={generated_campaign_id}, got {params[2]}"
        )
        assert params[3] == generated_campaign_id, (
            f"Expected campaign_id={generated_campaign_id}, got {params[3]}"
        )

        # Verify details contains the campaign name
        details = json.loads(params[4])
        assert details["name"] == data["name"]
        assert details["campaign_type"] == data["campaign_type"]

    @given(
        operator_name=operator_name_strategy(),
        source_campaign_id=campaign_id_strategy(),
    )
    @settings(max_examples=200)
    def test_duplicate_campaign_produces_audit_log_with_source_id(self, operator_name, source_campaign_id):
        """Property: For duplicate_campaign, an audit log is created with
        source_campaign_id in details JSON.

        **Validates: Requirements 2.6, 11.2**
        """
        new_campaign_id = source_campaign_id + 1000

        # Source campaign data
        source_campaign = {
            "id": source_campaign_id,
            "organization_id": 1,
            "branch_id": 1,
            "name": "Original Campaign",
            "description": "A campaign",
            "campaign_type": "promotional",
            "segment_id": 10,
            "template_id": 20,
            "channel": "whatsapp",
            "priority": 5,
            "recurring_frequency": "none",
            "recurring_end_date": None,
            "status": "completed",
        }

        # Write cursor: first fetchone returns source campaign, insert returns new ID
        write_cursor = AuditCaptureCursor(
            fetchone_result=source_campaign,
            lastrowid=new_campaign_id,
        )
        write_conn = AuditCaptureConnection(write_cursor)

        # Read cursor for get_campaign after duplicate
        read_cursor = AuditCaptureCursor(
            fetchone_result={
                "id": new_campaign_id,
                "name": "Original Campaign (Copy)",
                "status": "draft",
                "segment_id": 10,
                "template_id": 20,
                "campaign_type": "promotional",
            }
        )
        read_conn = AuditCaptureConnection(read_cursor)

        call_count = [0]

        def get_conn():
            call_count[0] += 1
            if call_count[0] == 1:
                return write_conn
            return read_conn

        service = CampaignService(get_conn)
        service.duplicate_campaign(source_campaign_id, operator_name)

        # Verify audit log insert happened
        audit_inserts = find_audit_inserts(write_cursor.executed)
        assert len(audit_inserts) == 1, (
            f"Expected exactly 1 audit log INSERT for duplicate_campaign, "
            f"got {len(audit_inserts)}"
        )

        # Verify params
        _sql, params = audit_inserts[0]
        assert params[0] == operator_name, (
            f"Expected operator_name='{operator_name}', got '{params[0]}'"
        )
        assert params[1] == "duplicate_campaign", (
            f"Expected action_type='duplicate_campaign', got '{params[1]}'"
        )
        # The audit log uses new_campaign_id as the target
        assert params[2] == new_campaign_id, (
            f"Expected target_id={new_campaign_id}, got {params[2]}"
        )
        assert params[3] == new_campaign_id, (
            f"Expected campaign_id={new_campaign_id}, got {params[3]}"
        )

        # Verify details contains source_campaign_id
        details = json.loads(params[4])
        assert "source_campaign_id" in details, (
            f"Expected 'source_campaign_id' in details, got: {details}"
        )
        assert details["source_campaign_id"] == source_campaign_id, (
            f"Expected source_campaign_id={source_campaign_id}, "
            f"got {details['source_campaign_id']}"
        )
