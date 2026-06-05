"""
Property-based tests for role-based permission enforcement using Hypothesis.

**Validates: Requirements 11.1**

Property 16: Role-based permission enforcement
- Operators without "campaign_send" permission get HTTP 403 on approve/send
- Operators with "campaign_send" permission OR admin role succeed (200 or appropriate response)
- Non-privileged transitions don't require campaign_send permission
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
from unittest.mock import patch, MagicMock

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from blueprints.campaign_bp import PRIVILEGED_ACTIONS, VALID_TRANSITIONS

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Roles that are NOT admin
NON_ADMIN_ROLES = st.sampled_from(["operator", "viewer", "editor", "manager", "support", ""])

# Admin role
ADMIN_ROLE = st.just("admin")

# All possible roles (including admin)
ALL_ROLES = st.sampled_from(["admin", "operator", "viewer", "editor", "manager", "support", ""])

# Permissions that do NOT include "campaign_send"
NON_PRIVILEGED_PERMISSIONS = st.lists(
    st.sampled_from(["view_campaigns", "edit_campaigns", "manage_users", "view_reports", "export_data", ""]),
    max_size=5,
).filter(lambda perms: "campaign_send" not in perms)

# Permissions that include "campaign_send"
PRIVILEGED_PERMISSIONS = st.lists(
    st.sampled_from(["campaign_send", "view_campaigns", "edit_campaigns", "manage_users", "view_reports"]),
    min_size=1,
    max_size=5,
).filter(lambda perms: "campaign_send" in perms)

# Privileged target states (require campaign_send)
PRIVILEGED_STATES = st.sampled_from(sorted(PRIVILEGED_ACTIONS))

# Non-privileged target states (don't require campaign_send)
# These are transitions that do NOT require campaign_send permission
NON_PRIVILEGED_STATES = st.sampled_from([
    s for s in ["scheduled", "pending_approval", "paused", "cancelled", "completed", "failed"]
    if s not in PRIVILEGED_ACTIONS
])

# User name strategy
USER_NAMES = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "Pd")),
    min_size=1,
    max_size=30,
)


# ---------------------------------------------------------------------------
# Flask test app fixture
# ---------------------------------------------------------------------------

def _create_test_app():
    """Create a minimal Flask app with campaign blueprint for testing."""
    from flask import Flask
    from blueprints.campaign_bp import campaign_bp

    test_app = Flask(__name__)
    test_app.secret_key = "test-secret-key"
    test_app.config["TESTING"] = True
    test_app.register_blueprint(campaign_bp)
    return test_app


# ---------------------------------------------------------------------------
# Mock the CampaignService to avoid needing a real database
# ---------------------------------------------------------------------------

def _mock_get_service(current_state="pending_approval"):
    """Create a mock CampaignService that returns controlled results."""
    mock_service = MagicMock()
    mock_service.transition_state.return_value = {
        "id": 1,
        "status": "approved",
        "name": "Test Campaign",
    }
    return mock_service


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestRoleBasedPermissionProperties:
    """Property-based tests for role-based permission enforcement — Property 16."""

    @given(
        role=NON_ADMIN_ROLES,
        permissions=NON_PRIVILEGED_PERMISSIONS,
        target_state=PRIVILEGED_STATES,
        user_name=USER_NAMES,
    )
    @settings(max_examples=300)
    def test_unprivileged_user_gets_403_on_privileged_transitions(
        self, role: str, permissions: list, target_state: str, user_name: str
    ):
        """
        Property: For any user WITHOUT "campaign_send" permission and NOT admin role,
        transitioning to a privileged state ("approved" or "sending") returns HTTP 403.

        **Validates: Requirements 11.1**
        """
        app = _create_test_app()

        with app.test_client() as client:
            # Set up session with user who lacks campaign_send permission
            with client.session_transaction() as sess:
                sess["user_id"] = 1
                sess["user_name"] = user_name
                sess["user_role"] = role
                sess["permissions"] = permissions

            # Mock the CampaignService to prevent actual DB calls
            with patch("blueprints.campaign_bp._get_service") as mock_get_svc:
                response = client.post(
                    "/api/campaigns/1/transition",
                    data=json.dumps({"new_state": target_state}),
                    content_type="application/json",
                )

            assert response.status_code == 403, (
                f"Expected 403 for role='{role}', permissions={permissions}, "
                f"target_state='{target_state}', got {response.status_code}"
            )
            data = response.get_json()
            assert "campaign_send" in data["error"].lower() or "forbidden" in data["error"].lower()

    @given(
        permissions=PRIVILEGED_PERMISSIONS,
        target_state=PRIVILEGED_STATES,
        user_name=USER_NAMES,
    )
    @settings(max_examples=300)
    def test_user_with_campaign_send_permission_succeeds(
        self, permissions: list, target_state: str, user_name: str
    ):
        """
        Property: For any user WITH "campaign_send" permission (non-admin role),
        transitioning to a privileged state is allowed (not blocked by permissions).

        The endpoint may return 200 (success) or 400 (invalid state transition),
        but NOT 403 (forbidden).

        **Validates: Requirements 11.1**
        """
        app = _create_test_app()

        with app.test_client() as client:
            with client.session_transaction() as sess:
                sess["user_id"] = 1
                sess["user_name"] = user_name
                sess["user_role"] = "operator"
                sess["permissions"] = permissions

            with patch("blueprints.campaign_bp._get_service") as mock_get_svc:
                mock_service = MagicMock()
                mock_service.transition_state.return_value = {
                    "id": 1, "status": target_state, "name": "Test"
                }
                mock_get_svc.return_value = mock_service

                response = client.post(
                    "/api/campaigns/1/transition",
                    data=json.dumps({"new_state": target_state}),
                    content_type="application/json",
                )

            # Permission check should NOT block — must not be 403
            assert response.status_code != 403, (
                f"Got 403 for user with permissions={permissions}, "
                f"target_state='{target_state}' — should be allowed"
            )

    @given(
        target_state=PRIVILEGED_STATES,
        user_name=USER_NAMES,
    )
    @settings(max_examples=300)
    def test_admin_role_always_succeeds(
        self, target_state: str, user_name: str
    ):
        """
        Property: For any user with admin role, transitioning to a privileged state
        is allowed regardless of explicit permissions list.

        **Validates: Requirements 11.1**
        """
        app = _create_test_app()

        with app.test_client() as client:
            with client.session_transaction() as sess:
                sess["user_id"] = 1
                sess["user_name"] = user_name
                sess["user_role"] = "admin"
                sess["permissions"] = []  # No explicit permissions, but admin bypasses

            with patch("blueprints.campaign_bp._get_service") as mock_get_svc:
                mock_service = MagicMock()
                mock_service.transition_state.return_value = {
                    "id": 1, "status": target_state, "name": "Test"
                }
                mock_get_svc.return_value = mock_service

                response = client.post(
                    "/api/campaigns/1/transition",
                    data=json.dumps({"new_state": target_state}),
                    content_type="application/json",
                )

            # Admin should never be blocked by permissions
            assert response.status_code != 403, (
                f"Got 403 for admin user on target_state='{target_state}' — "
                f"admin should always be allowed"
            )

    @given(
        role=NON_ADMIN_ROLES,
        permissions=NON_PRIVILEGED_PERMISSIONS,
        target_state=NON_PRIVILEGED_STATES,
        user_name=USER_NAMES,
    )
    @settings(max_examples=300)
    def test_non_privileged_transitions_dont_require_campaign_send(
        self, role: str, permissions: list, target_state: str, user_name: str
    ):
        """
        Property: Non-privileged transitions (scheduled, pending_approval, paused,
        cancelled, completed, failed) do NOT require "campaign_send" permission.
        Users without this permission should not get 403 for these transitions.

        They may get 400 (invalid transition from current state) or 200 (success),
        but never 403 due to missing campaign_send permission.

        **Validates: Requirements 11.1**
        """
        app = _create_test_app()

        with app.test_client() as client:
            with client.session_transaction() as sess:
                sess["user_id"] = 1
                sess["user_name"] = user_name
                sess["user_role"] = role
                sess["permissions"] = permissions

            with patch("blueprints.campaign_bp._get_service") as mock_get_svc:
                mock_service = MagicMock()
                mock_service.transition_state.return_value = {
                    "id": 1, "status": target_state, "name": "Test"
                }
                mock_get_svc.return_value = mock_service

                response = client.post(
                    "/api/campaigns/1/transition",
                    data=json.dumps({"new_state": target_state}),
                    content_type="application/json",
                )

            # Non-privileged transitions should NOT be blocked by permission check
            assert response.status_code != 403, (
                f"Got 403 for non-privileged transition to '{target_state}' — "
                f"permission check should not apply for non-privileged states"
            )

    def test_uppercase_admin_can_approve_campaign(self):
        """Regression: admin role comparisons are case-insensitive for approval."""
        app = _create_test_app()

        with app.test_client() as client:
            with client.session_transaction() as sess:
                sess["user_id"] = 1
                sess["user_name"] = "Admin User"
                sess["user_role"] = "Admin"
                sess["permissions"] = []

            with patch("blueprints.campaign_bp._get_service") as mock_get_svc:
                mock_service = MagicMock()
                mock_service.approve_campaign.return_value = {
                    "id": 1,
                    "status": "sending",
                    "name": "Test Campaign",
                }
                mock_get_svc.return_value = mock_service

                response = client.post("/api/campaigns/1/approve")

            assert response.status_code == 200
            mock_service.approve_campaign.assert_called_once_with(1, "Admin User")

    def test_session_refreshes_campaign_send_permission_from_auth_database(self, tmp_path, monkeypatch):
        """Regression: stale sessions can approve when DB has campaign_send permission."""
        import sqlite3

        db_path = tmp_path / "auth.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                CREATE TABLE users (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    email TEXT,
                    password_hash TEXT,
                    role TEXT,
                    permissions TEXT
                )
                """
            )
            conn.execute(
                "INSERT INTO users (id, role, permissions) VALUES (?, ?, ?)",
                (7, "operator", '["campaign_send"]'),
            )
            conn.commit()

        def get_test_db_connection():
            return sqlite3.connect(db_path)

        monkeypatch.setattr("auth.DB_PATH", str(db_path))
        monkeypatch.setattr("auth.get_db_connection", get_test_db_connection)
        app = _create_test_app()

        with app.test_client() as client:
            with client.session_transaction() as sess:
                sess["user_id"] = 7
                sess["user_name"] = "Operator"
                # Simulate an older session missing role/permissions.

            with patch("blueprints.campaign_bp._get_service") as mock_get_svc:
                mock_service = MagicMock()
                mock_service.approve_campaign.return_value = {
                    "id": 1,
                    "status": "sending",
                    "name": "Test Campaign",
                }
                mock_get_svc.return_value = mock_service

                response = client.post("/api/campaigns/1/approve")

            assert response.status_code == 200
            mock_service.approve_campaign.assert_called_once_with(1, "Operator")
