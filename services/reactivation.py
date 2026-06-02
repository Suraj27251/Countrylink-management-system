"""
Reactivation Workflows and Automation Rules Service.

Provides:
- Pre-configured reactivation workflow templates (expired recovery, inactive comeback,
  disconnected re-engagement, speed upgrade, festive promotions)
- Automation rules CRUD with trigger types (schedule, event, threshold),
  condition config, and actions (create_campaign_draft, notify_operator)
- Reactivation success tracking (status change within 30 days of campaign)
- All automation-generated campaigns require operator approval

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reactivation Workflow Templates
# ---------------------------------------------------------------------------

@dataclass
class ReactivationWorkflow:
    """Pre-configured reactivation workflow definition."""
    workflow_id: str
    name: str
    description: str
    campaign_type: str
    suggested_segment_filters: Dict
    suggested_template_name: str
    cooldown_days: int = 7
    success_window_days: int = 30


# The 5 built-in reactivation workflow templates per Requirement 6.1
REACTIVATION_WORKFLOWS: Dict[str, ReactivationWorkflow] = {
    "expired_recovery": ReactivationWorkflow(
        workflow_id="expired_recovery",
        name="Expired Plan Recovery",
        description="Target customers whose plans have expired, encouraging them to renew.",
        campaign_type="reactivation",
        suggested_segment_filters={
            "status": "expired",
            "days_inactive": {"min": 1, "max": 30},
        },
        suggested_template_name="plan_renewal_reminder",
        cooldown_days=7,
        success_window_days=30,
    ),
    "inactive_comeback": ReactivationWorkflow(
        workflow_id="inactive_comeback",
        name="Inactive Customer Comeback",
        description="Re-engage customers who have been inactive for an extended period.",
        campaign_type="reactivation",
        suggested_segment_filters={
            "status": "inactive",
            "days_inactive": {"min": 7, "max": 60},
        },
        suggested_template_name="comeback_offer",
        cooldown_days=7,
        success_window_days=30,
    ),
    "disconnected_reengagement": ReactivationWorkflow(
        workflow_id="disconnected_reengagement",
        name="Disconnected Customer Re-engagement",
        description="Reach out to disconnected customers with special reconnection offers.",
        campaign_type="reactivation",
        suggested_segment_filters={
            "status": "disconnected",
            "days_inactive": {"min": 1},
        },
        suggested_template_name="reconnection_offer",
        cooldown_days=7,
        success_window_days=30,
    ),
    "speed_upgrade": ReactivationWorkflow(
        workflow_id="speed_upgrade",
        name="Speed Upgrade Offers",
        description="Offer speed upgrades to active customers on older/lower-tier plans.",
        campaign_type="reactivation",
        suggested_segment_filters={
            "status": "active",
            "plan_category": "basic",
        },
        suggested_template_name="speed_upgrade_offer",
        cooldown_days=7,
        success_window_days=30,
    ),
    "festive_promotions": ReactivationWorkflow(
        workflow_id="festive_promotions",
        name="Festive Promotional Campaigns",
        description="Seasonal and festive promotional campaigns with special discount offers.",
        campaign_type="reactivation",
        suggested_segment_filters={
            "status": "active",
        },
        suggested_template_name="festive_offer",
        cooldown_days=7,
        success_window_days=30,
    ),
}


# ---------------------------------------------------------------------------
# Reactivation & Automation Service
# ---------------------------------------------------------------------------

class ReactivationService:
    """
    Service for reactivation workflows and automation rules.

    Handles:
    - Listing and retrieving pre-configured reactivation workflow templates
    - Pre-populating campaign config when an operator selects a workflow
    - CRUD for automation_rules table
    - Ensuring automation-generated campaigns are created as drafts (require approval)
    - Tracking reactivation success (status change within 30 days)

    Parameters
    ----------
    get_connection : callable
        A zero-argument function that returns a MySQL connection object.
    """

    def __init__(self, get_connection: Callable):
        self._get_conn = get_connection

    # ------------------------------------------------------------------
    # Reactivation Workflow Templates (Requirement 6.1, 6.2)
    # ------------------------------------------------------------------

    def list_workflows(self) -> List[Dict]:
        """
        List all available reactivation workflow templates.

        Returns a list of workflow definitions with their suggested
        segment filters and template names.
        """
        return [
            {
                "workflow_id": wf.workflow_id,
                "name": wf.name,
                "description": wf.description,
                "campaign_type": wf.campaign_type,
                "suggested_segment_filters": wf.suggested_segment_filters,
                "suggested_template_name": wf.suggested_template_name,
                "cooldown_days": wf.cooldown_days,
                "success_window_days": wf.success_window_days,
            }
            for wf in REACTIVATION_WORKFLOWS.values()
        ]

    def get_workflow(self, workflow_id: str) -> Dict:
        """
        Get a specific reactivation workflow by ID.

        Parameters
        ----------
        workflow_id : str
            One of: expired_recovery, inactive_comeback,
            disconnected_reengagement, speed_upgrade, festive_promotions

        Returns
        -------
        dict with workflow configuration.

        Raises
        ------
        ValueError
            If the workflow_id is not recognized.
        """
        wf = REACTIVATION_WORKFLOWS.get(workflow_id)
        if not wf:
            valid_ids = list(REACTIVATION_WORKFLOWS.keys())
            raise ValueError(
                f"Unknown workflow_id '{workflow_id}'. "
                f"Valid options: {valid_ids}"
            )
        return {
            "workflow_id": wf.workflow_id,
            "name": wf.name,
            "description": wf.description,
            "campaign_type": wf.campaign_type,
            "suggested_segment_filters": wf.suggested_segment_filters,
            "suggested_template_name": wf.suggested_template_name,
            "cooldown_days": wf.cooldown_days,
            "success_window_days": wf.success_window_days,
        }

    def prepare_campaign_from_workflow(
        self, workflow_id: str, operator_name: str
    ) -> Dict:
        """
        Pre-populate campaign configuration when an operator selects a
        reactivation workflow. Creates a campaign draft with the workflow's
        suggested segment and template. (Requirement 6.2)

        The campaign is created in 'draft' state and must go through
        operator approval before sending. (Requirement 2.5 / Property 4)

        Parameters
        ----------
        workflow_id : str
            The reactivation workflow to use.
        operator_name : str
            The operator initiating the campaign.

        Returns
        -------
        dict with campaign_id, segment_id (or None if segment doesn't exist
        yet), and suggested template info.
        """
        workflow = REACTIVATION_WORKFLOWS.get(workflow_id)
        if not workflow:
            raise ValueError(f"Unknown workflow_id '{workflow_id}'.")

        conn = self._get_conn()
        cursor = None
        try:
            cursor = conn.cursor(dictionary=True)
            conn.start_transaction()

            # Find or create an audience segment matching the workflow filters
            segment_id = self._find_or_create_segment(
                cursor, workflow, operator_name
            )

            # Find a template matching the suggested name (if it exists)
            template_id = self._find_template(cursor, workflow.suggested_template_name)

            # Create the campaign in draft state
            sql = """
                INSERT INTO campaigns
                    (organization_id, branch_id, name, description, campaign_type,
                     status, segment_id, template_id, channel, priority,
                     created_by)
                VALUES
                    (1, 1, %s, %s, 'reactivation',
                     'draft', %s, %s, 'whatsapp', 5,
                     %s)
            """
            campaign_name = f"{workflow.name} — {datetime.now().strftime('%Y-%m-%d')}"
            cursor.execute(sql, (
                campaign_name,
                workflow.description,
                segment_id,
                template_id,
                operator_name,
            ))
            campaign_id = cursor.lastrowid

            # Audit log
            self._log_action(
                cursor, operator_name, "create_reactivation_campaign", campaign_id,
                {
                    "workflow_id": workflow_id,
                    "segment_id": segment_id,
                    "template_id": template_id,
                }
            )

            conn.commit()

            return {
                "campaign_id": campaign_id,
                "campaign_name": campaign_name,
                "status": "draft",
                "workflow_id": workflow_id,
                "segment_id": segment_id,
                "template_id": template_id,
                "suggested_segment_filters": workflow.suggested_segment_filters,
                "suggested_template_name": workflow.suggested_template_name,
                "cooldown_days": workflow.cooldown_days,
                "success_window_days": workflow.success_window_days,
            }
        except Exception:
            conn.rollback()
            raise
        finally:
            if cursor:
                cursor.close()
            conn.close()

    def _find_or_create_segment(
        self, cursor, workflow: ReactivationWorkflow, operator_name: str
    ) -> Optional[int]:
        """Find an existing segment matching the workflow or create one."""
        segment_name = f"Reactivation — {workflow.name}"
        filter_json = json.dumps(workflow.suggested_segment_filters)

        # Check if a segment with this name already exists
        cursor.execute(
            "SELECT id FROM audience_segments WHERE name = %s AND organization_id = 1",
            (segment_name,),
        )
        existing = cursor.fetchone()
        if existing:
            return existing["id"]

        # Create a new segment
        cursor.execute(
            """
            INSERT INTO audience_segments
                (organization_id, name, description, filter_criteria, created_by)
            VALUES (1, %s, %s, %s, %s)
            """,
            (
                segment_name,
                f"Auto-created segment for {workflow.name} workflow",
                filter_json,
                operator_name,
            ),
        )
        return cursor.lastrowid

    def _find_template(self, cursor, template_name: str) -> Optional[int]:
        """Find a template by name, return ID or None."""
        cursor.execute(
            "SELECT id FROM campaign_templates WHERE template_name = %s "
            "AND status = 'approved' ORDER BY id DESC LIMIT 1",
            (template_name,),
        )
        row = cursor.fetchone()
        return row["id"] if row else None

    # ------------------------------------------------------------------
    # Automation Rules CRUD (Requirement 6.1 implied, design table)
    # ------------------------------------------------------------------

    def create_automation_rule(self, data: Dict, operator_name: str) -> Dict:
        """
        Create a new automation rule.

        Trigger types: schedule, event, threshold
        Action types: create_campaign_draft, notify_operator

        All automation-generated campaigns are created as drafts requiring
        operator approval. (Requirement 2.5)

        Parameters
        ----------
        data : dict
            Required keys: name, trigger_type, trigger_config, action_type, action_config
            Optional: condition_config, is_active
        operator_name : str
            Creator of the rule.

        Returns
        -------
        dict with the created automation rule record.
        """
        # Validate required fields
        required = ["name", "trigger_type", "trigger_config", "action_type", "action_config"]
        for field_name in required:
            if field_name not in data:
                raise ValueError(f"Missing required field: '{field_name}'")

        # Validate trigger_type
        valid_triggers = ("schedule", "event", "threshold")
        if data["trigger_type"] not in valid_triggers:
            raise ValueError(
                f"Invalid trigger_type '{data['trigger_type']}'. "
                f"Must be one of: {valid_triggers}"
            )

        # Validate action_type
        valid_actions = ("create_campaign_draft", "notify_operator")
        if data["action_type"] not in valid_actions:
            raise ValueError(
                f"Invalid action_type '{data['action_type']}'. "
                f"Must be one of: {valid_actions}"
            )

        conn = self._get_conn()
        cursor = None
        try:
            cursor = conn.cursor(dictionary=True)
            conn.start_transaction()

            trigger_config = (
                json.dumps(data["trigger_config"])
                if isinstance(data["trigger_config"], dict)
                else data["trigger_config"]
            )
            condition_config = None
            if data.get("condition_config"):
                condition_config = (
                    json.dumps(data["condition_config"])
                    if isinstance(data["condition_config"], dict)
                    else data["condition_config"]
                )
            action_config = (
                json.dumps(data["action_config"])
                if isinstance(data["action_config"], dict)
                else data["action_config"]
            )

            sql = """
                INSERT INTO automation_rules
                    (organization_id, tenant_id, name, trigger_type, trigger_config,
                     condition_config, action_type, action_config, is_active, created_by)
                VALUES (1, 1, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(sql, (
                data["name"],
                data["trigger_type"],
                trigger_config,
                condition_config,
                data["action_type"],
                action_config,
                data.get("is_active", 1),
                operator_name,
            ))
            rule_id = cursor.lastrowid
            conn.commit()

            return self.get_automation_rule(rule_id)
        except Exception:
            conn.rollback()
            raise
        finally:
            if cursor:
                cursor.close()
            conn.close()

    def get_automation_rule(self, rule_id: int) -> Dict:
        """Retrieve a single automation rule by ID."""
        conn = self._get_conn()
        cursor = None
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                "SELECT * FROM automation_rules WHERE id = %s", (rule_id,)
            )
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Automation rule {rule_id} not found")

            # Parse JSON fields
            row = self._parse_rule_json_fields(row)
            return row
        finally:
            if cursor:
                cursor.close()
            conn.close()

    def list_automation_rules(
        self, active_only: bool = False, page: int = 1, per_page: int = 20
    ) -> Dict:
        """
        List automation rules with optional filtering and pagination.

        Parameters
        ----------
        active_only : bool
            If True, only return active rules.
        page : int
            Page number (1-based).
        per_page : int
            Results per page.

        Returns
        -------
        dict with rules list, total, page info.
        """
        conn = self._get_conn()
        cursor = None
        try:
            cursor = conn.cursor(dictionary=True)

            where_sql = "WHERE is_active = 1" if active_only else ""

            # Count total
            cursor.execute(
                f"SELECT COUNT(*) as total FROM automation_rules {where_sql}"
            )
            total = cursor.fetchone()["total"]

            # Fetch page
            offset = (page - 1) * per_page
            cursor.execute(
                f"SELECT * FROM automation_rules {where_sql} "
                f"ORDER BY created_at DESC LIMIT %s OFFSET %s",
                (per_page, offset),
            )
            rules = cursor.fetchall()

            # Parse JSON fields
            rules = [self._parse_rule_json_fields(r) for r in rules]

            return {
                "rules": rules,
                "total": total,
                "page": page,
                "per_page": per_page,
                "total_pages": (total + per_page - 1) // per_page if per_page > 0 else 0,
            }
        finally:
            if cursor:
                cursor.close()
            conn.close()

    def update_automation_rule(self, rule_id: int, data: Dict, operator_name: str) -> Dict:
        """
        Update an existing automation rule.

        Parameters
        ----------
        rule_id : int
            The automation rule to update.
        data : dict
            Fields to update: name, trigger_type, trigger_config,
            condition_config, action_type, action_config, is_active
        operator_name : str
            Operator making the change.

        Returns
        -------
        dict with the updated rule record.
        """
        conn = self._get_conn()
        cursor = None
        try:
            cursor = conn.cursor(dictionary=True)
            conn.start_transaction()

            # Verify rule exists
            cursor.execute(
                "SELECT id FROM automation_rules WHERE id = %s", (rule_id,)
            )
            if not cursor.fetchone():
                raise ValueError(f"Automation rule {rule_id} not found")

            # Validate trigger_type if provided
            if "trigger_type" in data:
                valid_triggers = ("schedule", "event", "threshold")
                if data["trigger_type"] not in valid_triggers:
                    raise ValueError(
                        f"Invalid trigger_type '{data['trigger_type']}'. "
                        f"Must be one of: {valid_triggers}"
                    )

            # Validate action_type if provided
            if "action_type" in data:
                valid_actions = ("create_campaign_draft", "notify_operator")
                if data["action_type"] not in valid_actions:
                    raise ValueError(
                        f"Invalid action_type '{data['action_type']}'. "
                        f"Must be one of: {valid_actions}"
                    )

            # Build SET clause
            allowed_fields = [
                "name", "trigger_type", "trigger_config", "condition_config",
                "action_type", "action_config", "is_active",
            ]
            set_parts = []
            values = []
            for fld in allowed_fields:
                if fld in data:
                    val = data[fld]
                    # Serialize dicts to JSON for config fields
                    if fld in ("trigger_config", "condition_config", "action_config"):
                        if isinstance(val, dict):
                            val = json.dumps(val)
                    set_parts.append(f"{fld} = %s")
                    values.append(val)

            if not set_parts:
                conn.commit()
                return self.get_automation_rule(rule_id)

            values.append(rule_id)
            sql = f"UPDATE automation_rules SET {', '.join(set_parts)} WHERE id = %s"
            cursor.execute(sql, tuple(values))

            conn.commit()
            return self.get_automation_rule(rule_id)
        except Exception:
            conn.rollback()
            raise
        finally:
            if cursor:
                cursor.close()
            conn.close()

    def delete_automation_rule(self, rule_id: int) -> bool:
        """
        Delete an automation rule.

        Parameters
        ----------
        rule_id : int
            The rule to delete.

        Returns
        -------
        bool - True if deleted.

        Raises
        ------
        ValueError if rule not found.
        """
        conn = self._get_conn()
        cursor = None
        try:
            cursor = conn.cursor(dictionary=True)
            conn.start_transaction()

            cursor.execute(
                "SELECT id FROM automation_rules WHERE id = %s", (rule_id,)
            )
            if not cursor.fetchone():
                raise ValueError(f"Automation rule {rule_id} not found")

            cursor.execute(
                "DELETE FROM automation_rules WHERE id = %s", (rule_id,)
            )
            conn.commit()
            return True
        except Exception:
            conn.rollback()
            raise
        finally:
            if cursor:
                cursor.close()
            conn.close()

    # ------------------------------------------------------------------
    # Automation Rule Execution
    # ------------------------------------------------------------------

    def execute_rule(self, rule_id: int) -> Dict:
        """
        Execute an automation rule — creates a campaign draft or sends a notification.

        All automation-generated campaigns are created in 'draft' status
        requiring explicit operator approval before any messages are dispatched.
        (Requirement 2.5 / Property 4)

        Parameters
        ----------
        rule_id : int
            The automation rule to execute.

        Returns
        -------
        dict with execution result (campaign_id or notification_id).
        """
        rule = self.get_automation_rule(rule_id)

        if rule["action_type"] == "create_campaign_draft":
            return self._execute_create_campaign_draft(rule)
        elif rule["action_type"] == "notify_operator":
            return self._execute_notify_operator(rule)
        else:
            raise ValueError(f"Unknown action_type: {rule['action_type']}")

    def _execute_create_campaign_draft(self, rule: Dict) -> Dict:
        """
        Create a campaign draft from an automation rule.

        The campaign is ALWAYS created in 'draft' status, ensuring
        operator approval is required before any messages are dispatched.
        """
        action_config = rule.get("action_config", {})
        if isinstance(action_config, str):
            action_config = json.loads(action_config)

        conn = self._get_conn()
        cursor = None
        try:
            cursor = conn.cursor(dictionary=True)
            conn.start_transaction()

            campaign_name = action_config.get(
                "campaign_name",
                f"Auto: {rule['name']} — {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            )

            sql = """
                INSERT INTO campaigns
                    (organization_id, branch_id, name, description, campaign_type,
                     status, segment_id, template_id, channel, priority,
                     created_by)
                VALUES
                    (1, 1, %s, %s, %s,
                     'draft', %s, %s, 'whatsapp', 5,
                     %s)
            """
            cursor.execute(sql, (
                campaign_name,
                f"Auto-generated by automation rule: {rule['name']}",
                action_config.get("campaign_type", "reactivation"),
                action_config.get("segment_id"),
                action_config.get("template_id"),
                f"automation_rule_{rule['id']}",
            ))
            campaign_id = cursor.lastrowid

            # Update last_triggered_at on the rule
            cursor.execute(
                "UPDATE automation_rules SET last_triggered_at = %s WHERE id = %s",
                (datetime.now(), rule["id"]),
            )

            # Audit log
            self._log_action(
                cursor,
                f"automation_rule_{rule['id']}",
                "automation_create_draft",
                campaign_id,
                {
                    "rule_id": rule["id"],
                    "rule_name": rule["name"],
                    "requires_approval": True,
                },
            )

            conn.commit()

            return {
                "action": "create_campaign_draft",
                "campaign_id": campaign_id,
                "campaign_name": campaign_name,
                "status": "draft",
                "requires_approval": True,
                "rule_id": rule["id"],
            }
        except Exception:
            conn.rollback()
            raise
        finally:
            if cursor:
                cursor.close()
            conn.close()

    def _execute_notify_operator(self, rule: Dict) -> Dict:
        """Send a notification to the operator based on the rule's action config."""
        action_config = rule.get("action_config", {})
        if isinstance(action_config, str):
            action_config = json.loads(action_config)

        conn = self._get_conn()
        cursor = None
        try:
            cursor = conn.cursor(dictionary=True)
            conn.start_transaction()

            title = action_config.get("title", f"Automation Alert: {rule['name']}")
            severity = action_config.get("severity", "info")
            target_operators = action_config.get("target_operators", [])

            cursor.execute(
                """
                INSERT INTO system_notifications
                    (organization_id, alert_type, severity, title, details, target_operators)
                VALUES (1, %s, %s, %s, %s, %s)
                """,
                (
                    "automation_trigger",
                    severity,
                    title,
                    json.dumps({
                        "rule_id": rule["id"],
                        "rule_name": rule["name"],
                        "trigger_type": rule["trigger_type"],
                    }),
                    json.dumps(target_operators) if target_operators else None,
                ),
            )
            notification_id = cursor.lastrowid

            # Update last_triggered_at
            cursor.execute(
                "UPDATE automation_rules SET last_triggered_at = %s WHERE id = %s",
                (datetime.now(), rule["id"]),
            )

            conn.commit()

            return {
                "action": "notify_operator",
                "notification_id": notification_id,
                "title": title,
                "severity": severity,
                "rule_id": rule["id"],
            }
        except Exception:
            conn.rollback()
            raise
        finally:
            if cursor:
                cursor.close()
            conn.close()

    # ------------------------------------------------------------------
    # Reactivation Success Tracking (Requirement 6.5)
    # ------------------------------------------------------------------

    def track_reactivation_success(self, days_window: int = 30) -> Dict:
        """
        Track reactivation success by checking if customers who received
        a reactivation campaign message have changed status from
        'expired'/'inactive' to 'active' within the specified window.

        Parameters
        ----------
        days_window : int
            Number of days after campaign message to check for reactivation.
            Default is 30 days per Requirement 6.5.

        Returns
        -------
        dict with reactivation metrics:
            total_targeted, reactivated_count, reactivation_rate, by_workflow
        """
        conn = self._get_conn()
        cursor = None
        try:
            cursor = conn.cursor(dictionary=True)

            # Find reactivation campaign messages sent within the tracking window
            window_start = datetime.now() - timedelta(days=days_window)

            cursor.execute(
                """
                SELECT
                    cm.campaign_id,
                    cm.customer_mobile,
                    cm.sent_at,
                    c.name AS campaign_name,
                    c.created_by
                FROM campaign_messages cm
                JOIN campaigns c ON cm.campaign_id = c.id
                WHERE c.campaign_type = 'reactivation'
                  AND cm.status IN ('sent', 'delivered', 'read')
                  AND cm.sent_at >= %s
                  AND cm.is_test_send = 0
                """,
                (window_start,),
            )
            targeted_messages = cursor.fetchall()

            if not targeted_messages:
                return {
                    "total_targeted": 0,
                    "reactivated_count": 0,
                    "reactivation_rate": 0.0,
                    "tracking_window_days": days_window,
                }

            # Check which of those customers now have 'active' status
            mobiles = list(set(m["customer_mobile"] for m in targeted_messages))
            total_targeted = len(mobiles)

            # Query current status from renewal_records
            placeholders = ", ".join(["%s"] * len(mobiles))
            cursor.execute(
                f"""
                SELECT mobile, status
                FROM renewal_records
                WHERE mobile IN ({placeholders})
                  AND status = 'active'
                """,
                tuple(mobiles),
            )
            active_records = cursor.fetchall()
            reactivated_mobiles = set(r["mobile"] for r in active_records)
            reactivated_count = len(reactivated_mobiles)

            reactivation_rate = (
                reactivated_count / total_targeted if total_targeted > 0 else 0.0
            )

            return {
                "total_targeted": total_targeted,
                "reactivated_count": reactivated_count,
                "reactivation_rate": round(reactivation_rate, 4),
                "tracking_window_days": days_window,
            }
        finally:
            if cursor:
                cursor.close()
            conn.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_rule_json_fields(self, row: Dict) -> Dict:
        """Parse JSON string fields in an automation rule row."""
        for field_name in ("trigger_config", "condition_config", "action_config"):
            val = row.get(field_name)
            if isinstance(val, str):
                try:
                    row[field_name] = json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    pass
        return row

    def _log_action(
        self, cursor, operator_name: str, action_type: str,
        campaign_id: int, details: dict
    ):
        """Insert an audit record into operator_actions."""
        cursor.execute(
            """
            INSERT INTO operator_actions
                (operator_name, action_type, target_id, campaign_id, details)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (operator_name, action_type, campaign_id, campaign_id, json.dumps(details)),
        )
