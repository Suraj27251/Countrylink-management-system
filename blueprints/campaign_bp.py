"""
Campaign Manager Blueprint — state machine, CRUD, and lifecycle management.

Flask Blueprint registered at /api/campaigns/
Provides CampaignService class with pure business logic (testable without Flask context).
"""

import json
import logging
from datetime import datetime
from functools import wraps

from flask import Blueprint, jsonify, request, session

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Valid state transitions per the design state diagram
# ---------------------------------------------------------------------------
VALID_TRANSITIONS = {
    "draft": {"scheduled", "pending_approval"},
    "scheduled": {"pending_approval"},
    "pending_approval": {"approved", "cancelled"},
    "approved": {"sending"},
    "sending": {"paused", "completed", "failed"},
    "paused": {"sending"},
}

# Actions that require the "campaign_send" permission
PRIVILEGED_ACTIONS = {"approved", "sending"}


# ---------------------------------------------------------------------------
# CampaignService — pure business logic, testable with a mock connection fn
# ---------------------------------------------------------------------------
class CampaignService:
    """
    Campaign management service with state machine enforcement.

    Parameters
    ----------
    get_connection : callable
        A zero-argument function that returns a MySQL connection object.
        This allows dependency injection for testing.
    """

    def __init__(self, get_connection):
        self._get_conn = get_connection

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create_campaign(self, data: dict, operator_name: str) -> dict:
        """Create a new campaign in draft state."""
        conn = self._get_conn()
        cursor = None
        try:
            cursor = conn.cursor(dictionary=True)
            conn.start_transaction()

            sql = """
                INSERT INTO campaigns
                    (organization_id, branch_id, name, description, campaign_type,
                     status, segment_id, template_id, channel, priority,
                     recurring_frequency, recurring_end_date, created_by)
                VALUES
                    (%s, %s, %s, %s, %s,
                     'draft', %s, %s, %s, %s,
                     %s, %s, %s)
            """
            params = (
                data.get("organization_id", 1),
                data.get("branch_id", 1),
                data["name"],
                data.get("description", ""),
                data.get("campaign_type", "promotional"),
                data.get("segment_id"),
                data.get("template_id"),
                data.get("channel", "whatsapp"),
                data.get("priority", 5),
                data.get("recurring_frequency", "none"),
                data.get("recurring_end_date"),
                operator_name,
            )
            cursor.execute(sql, params)
            campaign_id = cursor.lastrowid

            # Audit log
            self._log_action(cursor, operator_name, "create_campaign", campaign_id, {
                "name": data["name"],
                "campaign_type": data.get("campaign_type", "promotional"),
            })

            conn.commit()
            return self.get_campaign(campaign_id)
        except Exception:
            conn.rollback()
            raise
        finally:
            if cursor:
                cursor.close()
            conn.close()

    def update_campaign(self, campaign_id: int, data: dict, operator_name: str) -> dict:
        """Update a campaign (only allowed in draft state)."""
        conn = self._get_conn()
        cursor = None
        try:
            cursor = conn.cursor(dictionary=True)
            conn.start_transaction()

            # Verify campaign is in draft state
            cursor.execute("SELECT status FROM campaigns WHERE id = %s", (campaign_id,))
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Campaign {campaign_id} not found")
            if row["status"] != "draft":
                raise ValueError(f"Cannot update campaign in '{row['status']}' state; must be 'draft'")

            # Build dynamic SET clause from allowed fields
            allowed_fields = [
                "name", "description", "campaign_type", "segment_id",
                "template_id", "channel", "priority", "recurring_frequency",
                "recurring_end_date",
            ]
            set_parts = []
            values = []
            for field in allowed_fields:
                if field in data:
                    set_parts.append(f"{field} = %s")
                    values.append(data[field])

            if not set_parts:
                conn.commit()
                return self.get_campaign(campaign_id)

            values.append(campaign_id)
            sql = f"UPDATE campaigns SET {', '.join(set_parts)} WHERE id = %s"
            cursor.execute(sql, tuple(values))

            # Audit log
            self._log_action(cursor, operator_name, "update_campaign", campaign_id, {
                "updated_fields": list(data.keys()),
            })

            conn.commit()
            return self.get_campaign(campaign_id)
        except Exception:
            conn.rollback()
            raise
        finally:
            if cursor:
                cursor.close()
            conn.close()

    def get_campaign(self, campaign_id: int) -> dict:
        """Retrieve a single campaign by ID."""
        conn = self._get_conn()
        cursor = None
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM campaigns WHERE id = %s", (campaign_id,))
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Campaign {campaign_id} not found")
            return row
        finally:
            if cursor:
                cursor.close()
            conn.close()

    def list_campaigns(self, filters: dict = None, page: int = 1, per_page: int = 20) -> dict:
        """List campaigns with optional filtering and pagination."""
        conn = self._get_conn()
        cursor = None
        try:
            cursor = conn.cursor(dictionary=True)
            where_clauses = []
            params = []

            if filters:
                if filters.get("status"):
                    where_clauses.append("status = %s")
                    params.append(filters["status"])
                if filters.get("campaign_type"):
                    where_clauses.append("campaign_type = %s")
                    params.append(filters["campaign_type"])
                if filters.get("organization_id"):
                    where_clauses.append("organization_id = %s")
                    params.append(filters["organization_id"])
                if filters.get("branch_id"):
                    where_clauses.append("branch_id = %s")
                    params.append(filters["branch_id"])
                if filters.get("created_by"):
                    where_clauses.append("created_by = %s")
                    params.append(filters["created_by"])

            where_sql = ""
            if where_clauses:
                where_sql = "WHERE " + " AND ".join(where_clauses)

            # Count total
            count_sql = f"SELECT COUNT(*) as total FROM campaigns {where_sql}"
            cursor.execute(count_sql, tuple(params))
            total = cursor.fetchone()["total"]

            # Paginated fetch
            offset = (page - 1) * per_page
            data_sql = f"""
                SELECT * FROM campaigns {where_sql}
                ORDER BY updated_at DESC
                LIMIT %s OFFSET %s
            """
            cursor.execute(data_sql, tuple(params) + (per_page, offset))
            campaigns = cursor.fetchall()

            return {
                "campaigns": campaigns,
                "total": total,
                "page": page,
                "per_page": per_page,
                "total_pages": (total + per_page - 1) // per_page if per_page > 0 else 0,
            }
        finally:
            if cursor:
                cursor.close()
            conn.close()

    # ------------------------------------------------------------------
    # State Machine
    # ------------------------------------------------------------------

    def transition_state(self, campaign_id: int, new_state: str, operator_name: str) -> dict:
        """
        Transition a campaign to a new state with validation.

        Raises ValueError if transition is invalid.
        """
        conn = self._get_conn()
        cursor = None
        try:
            cursor = conn.cursor(dictionary=True)
            conn.start_transaction()

            cursor.execute(
                "SELECT id, status FROM campaigns WHERE id = %s FOR UPDATE",
                (campaign_id,),
            )
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Campaign {campaign_id} not found")

            current_state = row["status"]
            allowed = VALID_TRANSITIONS.get(current_state, set())
            if new_state not in allowed:
                raise ValueError(
                    f"Invalid transition: '{current_state}' → '{new_state}'. "
                    f"Allowed transitions from '{current_state}': {sorted(allowed)}"
                )

            # Build update with state-specific timestamp fields
            update_fields = ["status = %s"]
            update_values = [new_state]

            if new_state == "approved":
                update_fields.append("approved_at = %s")
                update_values.append(datetime.now())
                update_fields.append("approved_by = %s")
                update_values.append(operator_name)
            elif new_state == "sending":
                update_fields.append("started_at = %s")
                update_values.append(datetime.now())
            elif new_state in ("completed", "failed", "cancelled"):
                update_fields.append("completed_at = %s")
                update_values.append(datetime.now())

            update_values.append(campaign_id)
            sql = f"UPDATE campaigns SET {', '.join(update_fields)} WHERE id = %s"
            cursor.execute(sql, tuple(update_values))

            # Audit log
            self._log_action(cursor, operator_name, f"transition_{new_state}", campaign_id, {
                "from_state": current_state,
                "to_state": new_state,
            })

            conn.commit()
            return self.get_campaign(campaign_id)
        except Exception:
            conn.rollback()
            raise
        finally:
            if cursor:
                cursor.close()
            conn.close()

    # ------------------------------------------------------------------
    # Operator Approval Workflow
    # ------------------------------------------------------------------

    def approve_campaign(self, campaign_id: int, operator_name: str) -> dict:
        """
        Approve a pending campaign — transition to approved then sending,
        and trigger queue enqueue for recipients.

        Raises ValueError if campaign is not in 'pending_approval' state.
        """
        conn = self._get_conn()
        cursor = None
        try:
            cursor = conn.cursor(dictionary=True)
            conn.start_transaction()

            cursor.execute(
                "SELECT id, status, segment_id, template_id FROM campaigns WHERE id = %s FOR UPDATE",
                (campaign_id,),
            )
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Campaign {campaign_id} not found")

            if row["status"] != "pending_approval":
                raise ValueError(
                    f"Cannot approve campaign in '{row['status']}' state; "
                    f"must be in 'pending_approval' state"
                )

            # Transition to approved
            cursor.execute(
                "UPDATE campaigns SET status = 'approved', approved_at = %s, approved_by = %s "
                "WHERE id = %s",
                (datetime.now(), operator_name, campaign_id),
            )

            # Audit log for approval
            self._log_action(cursor, operator_name, "approve_campaign", campaign_id, {
                "from_state": "pending_approval",
                "to_state": "approved",
            })

            # Transition to sending
            cursor.execute(
                "UPDATE campaigns SET status = 'sending', started_at = %s WHERE id = %s",
                (datetime.now(), campaign_id),
            )

            # Audit log for sending transition
            self._log_action(cursor, operator_name, "start_sending", campaign_id, {
                "from_state": "approved",
                "to_state": "sending",
            })

            conn.commit()

            # Enqueue campaign messages via the sending queue
            self._enqueue_campaign_recipients(campaign_id, row.get("segment_id"))

            return self.get_campaign(campaign_id)
        except Exception:
            conn.rollback()
            raise
        finally:
            if cursor:
                cursor.close()
            conn.close()

    def reject_campaign(self, campaign_id: int, operator_name: str, reason: str = "") -> dict:
        """
        Reject a pending campaign — transition to cancelled with reason logging.

        Raises ValueError if campaign is not in 'pending_approval' state.
        """
        conn = self._get_conn()
        cursor = None
        try:
            cursor = conn.cursor(dictionary=True)
            conn.start_transaction()

            cursor.execute(
                "SELECT id, status FROM campaigns WHERE id = %s FOR UPDATE",
                (campaign_id,),
            )
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Campaign {campaign_id} not found")

            if row["status"] != "pending_approval":
                raise ValueError(
                    f"Cannot reject campaign in '{row['status']}' state; "
                    f"must be in 'pending_approval' state"
                )

            # Transition to cancelled
            cursor.execute(
                "UPDATE campaigns SET status = 'cancelled', completed_at = %s WHERE id = %s",
                (datetime.now(), campaign_id),
            )

            # Audit log with rejection reason
            self._log_action(cursor, operator_name, "reject_campaign", campaign_id, {
                "from_state": "pending_approval",
                "to_state": "cancelled",
                "reason": reason,
            })

            conn.commit()
            return self.get_campaign(campaign_id)
        except Exception:
            conn.rollback()
            raise
        finally:
            if cursor:
                cursor.close()
            conn.close()

    def get_approval_preview(self, campaign_id: int) -> dict:
        """
        Return approval preview showing recipient count, template content,
        and estimated delivery time.

        Returns
        -------
        dict with keys: campaign_id, campaign_name, recipient_count, template_content,
        template_name, estimated_time_seconds, segment_name, status
        """
        conn = self._get_conn()
        cursor = None
        try:
            cursor = conn.cursor(dictionary=True)

            # Fetch campaign details
            cursor.execute(
                "SELECT id, name, status, segment_id, template_id, campaign_type "
                "FROM campaigns WHERE id = %s",
                (campaign_id,),
            )
            campaign = cursor.fetchone()
            if not campaign:
                raise ValueError(f"Campaign {campaign_id} not found")

            # Get recipient count from segment
            recipient_count = 0
            segment_name = None
            if campaign.get("segment_id"):
                cursor.execute(
                    "SELECT name, filter_criteria FROM audience_segments WHERE id = %s",
                    (campaign["segment_id"],),
                )
                segment = cursor.fetchone()
                if segment:
                    segment_name = segment["name"]
                    # Count recipients by evaluating the segment
                    recipient_count = self._count_segment_recipients(
                        cursor, segment.get("filter_criteria")
                    )

            # Get template content
            template_content = None
            template_name = None
            if campaign.get("template_id"):
                cursor.execute(
                    "SELECT template_name, body_text, header_type, footer_text, "
                    "placeholder_count, placeholder_mappings "
                    "FROM campaign_templates WHERE id = %s",
                    (campaign["template_id"],),
                )
                template = cursor.fetchone()
                if template:
                    template_name = template["template_name"]
                    template_content = {
                        "body_text": template.get("body_text", ""),
                        "header_type": template.get("header_type", "none"),
                        "footer_text": template.get("footer_text", ""),
                        "placeholder_count": template.get("placeholder_count", 0),
                        "placeholder_mappings": template.get("placeholder_mappings"),
                    }

            # Estimate delivery time: recipient_count / throttle_rate (80/sec default)
            throttle_rate = 80
            estimated_time_seconds = (
                recipient_count / throttle_rate if recipient_count > 0 else 0
            )

            return {
                "campaign_id": campaign_id,
                "campaign_name": campaign["name"],
                "status": campaign["status"],
                "campaign_type": campaign.get("campaign_type", "promotional"),
                "recipient_count": recipient_count,
                "template_name": template_name,
                "template_content": template_content,
                "estimated_time_seconds": round(estimated_time_seconds, 1),
                "segment_name": segment_name,
            }
        finally:
            if cursor:
                cursor.close()
            conn.close()

    def submit_for_approval(self, campaign_id: int, operator_name: str) -> dict:
        """
        Submit a draft or scheduled campaign for operator approval.

        Ensures automation_rule-generated drafts go through approval before dispatch.
        """
        conn = self._get_conn()
        cursor = None
        try:
            cursor = conn.cursor(dictionary=True)
            conn.start_transaction()

            cursor.execute(
                "SELECT id, status FROM campaigns WHERE id = %s FOR UPDATE",
                (campaign_id,),
            )
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Campaign {campaign_id} not found")

            current_state = row["status"]
            allowed_from = {"draft", "scheduled"}
            if current_state not in allowed_from:
                raise ValueError(
                    f"Cannot submit campaign for approval from '{current_state}' state; "
                    f"must be in 'draft' or 'scheduled' state"
                )

            cursor.execute(
                "UPDATE campaigns SET status = 'pending_approval' WHERE id = %s",
                (campaign_id,),
            )

            self._log_action(cursor, operator_name, "submit_for_approval", campaign_id, {
                "from_state": current_state,
                "to_state": "pending_approval",
            })

            conn.commit()
            return self.get_campaign(campaign_id)
        except Exception:
            conn.rollback()
            raise
        finally:
            if cursor:
                cursor.close()
            conn.close()

    def _count_segment_recipients(self, cursor, filter_criteria) -> int:
        """Count the number of recipients matching segment filter criteria."""
        import json as _json

        if not filter_criteria:
            return 0

        if isinstance(filter_criteria, str):
            try:
                criteria = _json.loads(filter_criteria)
            except (_json.JSONDecodeError, TypeError):
                return 0
        else:
            criteria = filter_criteria

        # Build simple WHERE clause from criteria for counting
        # Use the renewal_records table (source of customer data)
        where_parts = []
        params = []

        for field, value in criteria.items():
            if value is None:
                continue
            if field in ("expiry_category", "plan_name", "plan_category",
                         "zone_name", "area", "building", "status",
                         "network_type", "connectivity_mode", "owner_tenant"):
                where_parts.append(f"{field} = %s")
                params.append(value)
            elif field == "kyc_approved":
                where_parts.append("kyc_approved = %s")
                params.append(1 if value else 0)
            elif field == "days_remaining" and isinstance(value, dict):
                if value.get("min") is not None:
                    where_parts.append("days_remaining >= %s")
                    params.append(value["min"])
                if value.get("max") is not None:
                    where_parts.append("days_remaining <= %s")
                    params.append(value["max"])

        where_sql = " AND ".join(where_parts) if where_parts else "1=1"
        count_sql = f"SELECT COUNT(*) as cnt FROM renewal_records WHERE {where_sql}"

        try:
            cursor.execute(count_sql, tuple(params))
            result = cursor.fetchone()
            return result["cnt"] if result else 0
        except Exception:
            logger.warning("Failed to count segment recipients", exc_info=True)
            return 0

    def _enqueue_campaign_recipients(self, campaign_id: int, segment_id: int) -> None:
        """
        Fetch recipients from segment and enqueue them in the sending queue.

        This triggers the actual message dispatch process.
        """
        if not segment_id:
            logger.warning("Campaign %d has no segment_id, cannot enqueue", campaign_id)
            return

        conn = self._get_conn()
        cursor = None
        try:
            cursor = conn.cursor(dictionary=True)

            # Get segment filter criteria
            cursor.execute(
                "SELECT filter_criteria FROM audience_segments WHERE id = %s",
                (segment_id,),
            )
            segment = cursor.fetchone()
            if not segment:
                logger.warning("Segment %d not found for campaign %d", segment_id, campaign_id)
                return

            import json as _json
            filter_criteria = segment.get("filter_criteria")
            if isinstance(filter_criteria, str):
                try:
                    criteria = _json.loads(filter_criteria)
                except (_json.JSONDecodeError, TypeError):
                    criteria = {}
            else:
                criteria = filter_criteria or {}

            # Build query to get recipients
            where_parts = []
            params = []

            for field, value in criteria.items():
                if value is None:
                    continue
                if field in ("expiry_category", "plan_name", "plan_category",
                             "zone_name", "area", "building", "status",
                             "network_type", "connectivity_mode", "owner_tenant"):
                    where_parts.append(f"{field} = %s")
                    params.append(value)
                elif field == "kyc_approved":
                    where_parts.append("kyc_approved = %s")
                    params.append(1 if value else 0)
                elif field == "days_remaining" and isinstance(value, dict):
                    if value.get("min") is not None:
                        where_parts.append("days_remaining >= %s")
                        params.append(value["min"])
                    if value.get("max") is not None:
                        where_parts.append("days_remaining <= %s")
                        params.append(value["max"])

            where_sql = " AND ".join(where_parts) if where_parts else "1=1"
            query = f"SELECT mobile, customer_name FROM renewal_records WHERE {where_sql}"

            cursor.execute(query, tuple(params))
            recipients = cursor.fetchall()

        except Exception:
            logger.exception("Failed to fetch recipients for campaign %d", campaign_id)
            recipients = []
        finally:
            if cursor:
                cursor.close()
            conn.close()

        # Enqueue via SendingQueue
        if recipients:
            try:
                from services.sending_queue import SendingQueue
                queue = SendingQueue(self._get_conn)
                queue.enqueue_campaign(campaign_id, recipients)
            except Exception:
                logger.exception("Failed to enqueue campaign %d", campaign_id)

    # ------------------------------------------------------------------
    # Duplicate & Schedule
    # ------------------------------------------------------------------

    def duplicate_campaign(self, campaign_id: int, operator_name: str) -> dict:
        """Duplicate a campaign — copy config to a new draft."""
        conn = self._get_conn()
        cursor = None
        try:
            cursor = conn.cursor(dictionary=True)
            conn.start_transaction()

            cursor.execute("SELECT * FROM campaigns WHERE id = %s", (campaign_id,))
            source = cursor.fetchone()
            if not source:
                raise ValueError(f"Campaign {campaign_id} not found")

            sql = """
                INSERT INTO campaigns
                    (organization_id, branch_id, name, description, campaign_type,
                     status, segment_id, template_id, channel, priority,
                     recurring_frequency, recurring_end_date, created_by)
                VALUES
                    (%s, %s, %s, %s, %s,
                     'draft', %s, %s, %s, %s,
                     %s, %s, %s)
            """
            params = (
                source["organization_id"],
                source["branch_id"],
                f"{source['name']} (Copy)",
                source.get("description", ""),
                source["campaign_type"],
                source.get("segment_id"),
                source.get("template_id"),
                source.get("channel", "whatsapp"),
                source.get("priority", 5),
                source.get("recurring_frequency", "none"),
                source.get("recurring_end_date"),
                operator_name,
            )
            cursor.execute(sql, params)
            new_id = cursor.lastrowid

            # Audit log
            self._log_action(cursor, operator_name, "duplicate_campaign", new_id, {
                "source_campaign_id": campaign_id,
            })

            conn.commit()
            return self.get_campaign(new_id)
        except Exception:
            conn.rollback()
            raise
        finally:
            if cursor:
                cursor.close()
            conn.close()

    def schedule_campaign(self, campaign_id: int, scheduled_at: datetime, operator_name: str) -> dict:
        """Set scheduled_at and transition campaign to scheduled state."""
        conn = self._get_conn()
        cursor = None
        try:
            cursor = conn.cursor(dictionary=True)
            conn.start_transaction()

            cursor.execute(
                "SELECT id, status FROM campaigns WHERE id = %s FOR UPDATE",
                (campaign_id,),
            )
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Campaign {campaign_id} not found")

            current_state = row["status"]
            if "scheduled" not in VALID_TRANSITIONS.get(current_state, set()):
                raise ValueError(
                    f"Cannot schedule campaign in '{current_state}' state. "
                    f"Must be in 'draft' state."
                )

            cursor.execute(
                "UPDATE campaigns SET status = 'scheduled', scheduled_at = %s WHERE id = %s",
                (scheduled_at, campaign_id),
            )

            # Audit log
            self._log_action(cursor, operator_name, "schedule_campaign", campaign_id, {
                "scheduled_at": scheduled_at.isoformat(),
            })

            conn.commit()
            return self.get_campaign(campaign_id)
        except Exception:
            conn.rollback()
            raise
        finally:
            if cursor:
                cursor.close()
            conn.close()

    # ------------------------------------------------------------------
    # Test Send
    # ------------------------------------------------------------------

    def test_send(self, campaign_id: int, test_numbers: list, operator_name: str) -> list:
        """
        Send a test message to 1-5 mobile numbers immediately, bypassing the queue.

        Test messages are marked distinctly in the delivery log with
        is_test_send=1 so they are excluded from campaign analytics.

        Parameters
        ----------
        campaign_id : int
            The campaign whose template to send.
        test_numbers : list of str
            1 to 5 mobile numbers to send test messages to.
        operator_name : str
            The operator initiating the test send.

        Returns
        -------
        list of dict
            Each dict has: mobile, status, message_id, error_code, error_message.

        Raises
        ------
        ValueError
            If campaign not found, has no template, or test_numbers invalid.
        """
        if not test_numbers or len(test_numbers) < 1:
            raise ValueError("At least 1 test number is required.")
        if len(test_numbers) > 5:
            raise ValueError("Maximum 5 test numbers allowed.")

        # Validate numbers are non-empty strings
        cleaned_numbers = []
        for num in test_numbers:
            cleaned = str(num).strip()
            if not cleaned:
                raise ValueError("Test numbers must be non-empty strings.")
            cleaned_numbers.append(cleaned)

        conn = self._get_conn()
        cursor = None
        try:
            cursor = conn.cursor(dictionary=True)

            # Fetch campaign and template info
            cursor.execute(
                "SELECT id, name, template_id, status FROM campaigns WHERE id = %s",
                (campaign_id,),
            )
            campaign = cursor.fetchone()
            if not campaign:
                raise ValueError(f"Campaign {campaign_id} not found")

            template_id = campaign.get("template_id")
            if not template_id:
                raise ValueError(
                    f"Campaign {campaign_id} has no template assigned. "
                    "Assign a template before sending a test."
                )

            # Fetch template details
            cursor.execute(
                "SELECT id, template_name, body_text, placeholder_mappings "
                "FROM campaign_templates WHERE id = %s",
                (template_id,),
            )
            template = cursor.fetchone()
            if not template:
                raise ValueError(f"Template {template_id} not found")

            template_name = template["template_name"]

            # Resolve template params (use sample/empty values for test send)
            params_list = self._resolve_test_params(template)

            # Dispatch immediately via channel dispatcher (bypass queue)
            from services.channel import WhatsAppDispatcher
            dispatcher = WhatsAppDispatcher()

            results = []
            for mobile in cleaned_numbers:
                dispatch_result = dispatcher.send_template(
                    recipient=mobile,
                    template_name=template_name,
                    params=params_list,
                )

                # Record in campaign_messages with is_test_send flag
                status = "sent" if dispatch_result.success else "failed"
                idempotency_key = f"test_{campaign_id}_{mobile}_{template_id}_{int(datetime.now().timestamp())}"

                cursor.execute(
                    """
                    INSERT INTO campaign_messages
                        (campaign_id, customer_mobile, customer_name,
                         template_id, template_params, channel, status,
                         whatsapp_message_id, error_code, error_message,
                         is_test_send, sent_at, failed_at, idempotency_key)
                    VALUES
                        (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        campaign_id,
                        mobile,
                        "Test Recipient",
                        template_id,
                        json.dumps(params_list) if params_list else None,
                        "whatsapp",
                        status,
                        dispatch_result.message_id,
                        dispatch_result.error_code,
                        dispatch_result.error_message,
                        1,  # is_test_send = True
                        datetime.now() if dispatch_result.success else None,
                        datetime.now() if not dispatch_result.success else None,
                        idempotency_key,
                    ),
                )

                results.append({
                    "mobile": mobile,
                    "status": status,
                    "message_id": dispatch_result.message_id,
                    "error_code": dispatch_result.error_code,
                    "error_message": dispatch_result.error_message,
                })

            conn.commit()

            # Audit log
            conn2 = self._get_conn()
            cursor2 = None
            try:
                cursor2 = conn2.cursor(dictionary=True)
                self._log_action(cursor2, operator_name, "test_send", campaign_id, {
                    "test_numbers": cleaned_numbers,
                    "results_count": len(results),
                    "success_count": sum(1 for r in results if r["status"] == "sent"),
                })
                conn2.commit()
            finally:
                if cursor2:
                    cursor2.close()
                conn2.close()

            return results
        except Exception:
            conn.rollback()
            raise
        finally:
            if cursor:
                cursor.close()
            conn.close()

    def _resolve_test_params(self, template: dict) -> list:
        """
        Resolve template placeholder params for test send using sample values.

        For test sends, we use the placeholder mappings with generic sample values.
        """
        mappings_raw = template.get("placeholder_mappings")
        if not mappings_raw:
            return []

        if isinstance(mappings_raw, str):
            mappings = json.loads(mappings_raw)
        else:
            mappings = mappings_raw

        # Sort by key to maintain positional order (1, 2, 3...)
        params = []
        for key in sorted(mappings.keys(), key=lambda k: (k.isdigit(), int(k) if k.isdigit() else k)):
            # Use the field name as sample value for test sends
            field_name = mappings[key]
            params.append(f"[{field_name}]")

        return params

    # ------------------------------------------------------------------
    # A/B Testing
    # ------------------------------------------------------------------

    def create_ab_test(self, campaign_id: int, variants: list, test_pct: float,
                       operator_name: str) -> dict:
        """
        Create an A/B test for a campaign.

        Parameters
        ----------
        campaign_id : int
            The campaign to configure as an A/B test.
        variants : list of int
            List of template_ids (2-4 variants).
        test_pct : float
            Percentage of audience to use for the test (10-50).
        operator_name : str
            Operator performing the action.

        Returns
        -------
        dict with campaign info and variant records.

        Raises
        ------
        ValueError
            If campaign not found, not in draft state, or params invalid.
        """
        # Validate variant count (2-4)
        if not variants or len(variants) < 2:
            raise ValueError("A/B test requires at least 2 template variants.")
        if len(variants) > 4:
            raise ValueError("A/B test supports a maximum of 4 template variants.")

        # Validate test percentage (10-50)
        if test_pct < 10 or test_pct > 50:
            raise ValueError(
                "Test percentage must be between 10 and 50 (inclusive)."
            )

        conn = self._get_conn()
        cursor = None
        try:
            cursor = conn.cursor(dictionary=True)
            conn.start_transaction()

            # Verify campaign is in draft state
            cursor.execute(
                "SELECT id, status, segment_id FROM campaigns WHERE id = %s FOR UPDATE",
                (campaign_id,),
            )
            campaign = cursor.fetchone()
            if not campaign:
                raise ValueError(f"Campaign {campaign_id} not found")
            if campaign["status"] != "draft":
                raise ValueError(
                    f"Cannot create A/B test for campaign in '{campaign['status']}' state; "
                    f"must be in 'draft' state."
                )

            # Update campaign to ab_test type with test percentage
            cursor.execute(
                "UPDATE campaigns SET campaign_type = 'ab_test', "
                "ab_test_percentage = %s, template_id = %s WHERE id = %s",
                (test_pct, variants[0], campaign_id),
            )

            # Remove any existing variants for this campaign
            cursor.execute(
                "DELETE FROM campaign_ab_variants WHERE campaign_id = %s",
                (campaign_id,),
            )

            # Create variant records
            variant_labels = ["A", "B", "C", "D"]
            created_variants = []
            for idx, template_id in enumerate(variants):
                label = variant_labels[idx]
                cursor.execute(
                    """
                    INSERT INTO campaign_ab_variants
                        (campaign_id, template_id, variant_label,
                         recipient_count, sent_count, delivered_count,
                         read_count, response_count, is_winner)
                    VALUES (%s, %s, %s, 0, 0, 0, 0, 0, 0)
                    """,
                    (campaign_id, template_id, label),
                )
                created_variants.append({
                    "id": cursor.lastrowid,
                    "campaign_id": campaign_id,
                    "template_id": template_id,
                    "variant_label": label,
                    "recipient_count": 0,
                    "sent_count": 0,
                    "delivered_count": 0,
                    "read_count": 0,
                    "response_count": 0,
                    "is_winner": 0,
                })

            # Audit log
            self._log_action(cursor, operator_name, "create_ab_test", campaign_id, {
                "variants": variants,
                "test_percentage": test_pct,
                "variant_count": len(variants),
            })

            conn.commit()

            return {
                "campaign_id": campaign_id,
                "campaign_type": "ab_test",
                "test_percentage": test_pct,
                "variants": created_variants,
            }
        except Exception:
            conn.rollback()
            raise
        finally:
            if cursor:
                cursor.close()
            conn.close()

    def get_ab_variants(self, campaign_id: int) -> list:
        """
        Get all A/B test variants for a campaign with their metrics.

        Returns
        -------
        list of dict
            Each variant record with metrics.
        """
        conn = self._get_conn()
        cursor = None
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                "SELECT * FROM campaign_ab_variants WHERE campaign_id = %s "
                "ORDER BY variant_label ASC",
                (campaign_id,),
            )
            return cursor.fetchall()
        finally:
            if cursor:
                cursor.close()
            conn.close()

    def select_ab_winner(self, campaign_id: int, variant_id: int,
                         operator_name: str) -> dict:
        """
        Select the winning A/B variant and create a full rollout campaign
        for the remaining audience.

        Parameters
        ----------
        campaign_id : int
            The A/B test campaign.
        variant_id : int
            The ID of the winning variant in campaign_ab_variants.
        operator_name : str
            Operator performing the action.

        Returns
        -------
        dict with rollout_campaign info.

        Raises
        ------
        ValueError
            If campaign/variant not found or campaign not in completed state.
        """
        conn = self._get_conn()
        cursor = None
        try:
            cursor = conn.cursor(dictionary=True)
            conn.start_transaction()

            # Verify campaign is ab_test type and in completed state
            cursor.execute(
                "SELECT id, name, status, campaign_type, segment_id, "
                "ab_test_percentage, organization_id, branch_id, channel, priority "
                "FROM campaigns WHERE id = %s FOR UPDATE",
                (campaign_id,),
            )
            campaign = cursor.fetchone()
            if not campaign:
                raise ValueError(f"Campaign {campaign_id} not found")
            if campaign["campaign_type"] != "ab_test":
                raise ValueError(
                    f"Campaign {campaign_id} is not an A/B test campaign."
                )
            if campaign["status"] != "completed":
                raise ValueError(
                    f"Cannot select winner until A/B test campaign is completed. "
                    f"Current status: '{campaign['status']}'"
                )

            # Verify the variant belongs to this campaign
            cursor.execute(
                "SELECT id, template_id, variant_label FROM campaign_ab_variants "
                "WHERE id = %s AND campaign_id = %s",
                (variant_id, campaign_id),
            )
            winner_variant = cursor.fetchone()
            if not winner_variant:
                raise ValueError(
                    f"Variant {variant_id} not found for campaign {campaign_id}."
                )

            # Mark variant as winner
            cursor.execute(
                "UPDATE campaign_ab_variants SET is_winner = 0 WHERE campaign_id = %s",
                (campaign_id,),
            )
            cursor.execute(
                "UPDATE campaign_ab_variants SET is_winner = 1 WHERE id = %s",
                (variant_id,),
            )

            # Create rollout campaign targeting remaining audience with winning template
            winning_template_id = winner_variant["template_id"]
            rollout_name = f"{campaign['name']} — Rollout ({winner_variant['variant_label']})"

            cursor.execute(
                """
                INSERT INTO campaigns
                    (organization_id, branch_id, name, description, campaign_type,
                     status, segment_id, template_id, channel, priority,
                     parent_campaign_id, created_by)
                VALUES
                    (%s, %s, %s, %s, 'promotional',
                     'draft', %s, %s, %s, %s,
                     %s, %s)
                """,
                (
                    campaign["organization_id"],
                    campaign["branch_id"],
                    rollout_name,
                    f"Full rollout of winning variant {winner_variant['variant_label']} "
                    f"from A/B test campaign {campaign_id}",
                    "promotional",
                    campaign["segment_id"],
                    winning_template_id,
                    campaign.get("channel", "whatsapp"),
                    campaign.get("priority", 5),
                    campaign_id,
                    operator_name,
                ),
            )
            rollout_campaign_id = cursor.lastrowid

            # Audit log
            self._log_action(cursor, operator_name, "select_ab_winner", campaign_id, {
                "winning_variant_id": variant_id,
                "winning_variant_label": winner_variant["variant_label"],
                "winning_template_id": winning_template_id,
                "rollout_campaign_id": rollout_campaign_id,
            })

            conn.commit()

            return {
                "campaign_id": campaign_id,
                "winning_variant_id": variant_id,
                "winning_variant_label": winner_variant["variant_label"],
                "winning_template_id": winning_template_id,
                "rollout_campaign_id": rollout_campaign_id,
                "rollout_campaign_name": rollout_name,
                "rollout_status": "draft",
            }
        except Exception:
            conn.rollback()
            raise
        finally:
            if cursor:
                cursor.close()
            conn.close()

    @staticmethod
    def compute_ab_split(audience_size: int, variant_count: int) -> list:
        """
        Compute even audience split for A/B testing.

        Uses floor(N/V) or ceil(N/V) per variant to ensure the maximum
        difference between any two variants is at most 1 recipient.

        Parameters
        ----------
        audience_size : int
            Total number of recipients for the test portion (N).
        variant_count : int
            Number of variants (V), must be 2-4.

        Returns
        -------
        list of int
            Number of recipients assigned to each variant.
            Sum equals audience_size. Max difference between any two is 1.

        Raises
        ------
        ValueError
            If variant_count is not between 2 and 4, or audience_size < 0.
        """
        if variant_count < 2 or variant_count > 4:
            raise ValueError("Variant count must be between 2 and 4.")
        if audience_size < 0:
            raise ValueError("Audience size must be non-negative.")

        base = audience_size // variant_count
        remainder = audience_size % variant_count

        # First `remainder` variants get ceil(N/V), rest get floor(N/V)
        split = []
        for i in range(variant_count):
            if i < remainder:
                split.append(base + 1)
            else:
                split.append(base)

        return split

    # ------------------------------------------------------------------
    # Audit logging helper
    # ------------------------------------------------------------------

    def _log_action(self, cursor, operator_name: str, action_type: str,
                    campaign_id: int, details: dict):
        """Insert an audit record into operator_actions."""
        cursor.execute(
            """
            INSERT INTO operator_actions (operator_name, action_type, target_id, campaign_id, details)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (operator_name, action_type, campaign_id, campaign_id, json.dumps(details)),
        )


# ---------------------------------------------------------------------------
# Flask Blueprint and route handlers
# ---------------------------------------------------------------------------
campaign_bp = Blueprint("campaigns", __name__, url_prefix="/api/campaigns")


def _require_auth(f):
    """Decorator: require Flask session authentication."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("user_id"):
            return jsonify({"error": "Authentication required."}), 401
        return f(*args, **kwargs)
    return decorated


def _require_campaign_send_permission(f):
    """Decorator: require 'campaign_send' permission for privileged actions."""
    @wraps(f)
    def decorated(*args, **kwargs):
        user_role = session.get("user_role", "")
        user_permissions = session.get("permissions", [])
        # Admin role always has permission; otherwise check explicit permission
        if user_role != "admin" and "campaign_send" not in user_permissions:
            return jsonify({"error": "Forbidden. 'campaign_send' permission required."}), 403
        return f(*args, **kwargs)
    return decorated


def _get_service():
    """Get CampaignService instance using app's MySQL connection factory."""
    from app import get_mysql_connection
    return CampaignService(get_mysql_connection)


# ------------------------------------------------------------------
# CRUD Routes
# ------------------------------------------------------------------

@campaign_bp.route("/", methods=["POST"])
@_require_auth
def create_campaign():
    """Create a new campaign."""
    data = request.get_json(force=True)
    if not data or not data.get("name"):
        return jsonify({"error": "Campaign name is required."}), 400

    service = _get_service()
    operator = session.get("user_name", "unknown")
    try:
        campaign = service.create_campaign(data, operator)
        return jsonify(campaign), 201
    except Exception as exc:
        logger.exception("Failed to create campaign")
        return jsonify({"error": str(exc)}), 500


@campaign_bp.route("/", methods=["GET"])
@_require_auth
def list_campaigns():
    """List campaigns with optional filtering and pagination."""
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 20, type=int)
    filters = {}
    for key in ("status", "campaign_type", "organization_id", "branch_id", "created_by"):
        val = request.args.get(key)
        if val:
            filters[key] = val

    service = _get_service()
    try:
        result = service.list_campaigns(filters=filters or None, page=page, per_page=per_page)
        return jsonify(result), 200
    except Exception as exc:
        logger.exception("Failed to list campaigns")
        return jsonify({"error": str(exc)}), 500


@campaign_bp.route("/<int:campaign_id>", methods=["GET"])
@_require_auth
def get_campaign(campaign_id):
    """Get a single campaign by ID."""
    service = _get_service()
    try:
        campaign = service.get_campaign(campaign_id)
        return jsonify(campaign), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        logger.exception("Failed to get campaign")
        return jsonify({"error": str(exc)}), 500


@campaign_bp.route("/<int:campaign_id>", methods=["PUT"])
@_require_auth
def update_campaign(campaign_id):
    """Update a campaign (draft state only)."""
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "Request body required."}), 400

    service = _get_service()
    operator = session.get("user_name", "unknown")
    try:
        campaign = service.update_campaign(campaign_id, data, operator)
        return jsonify(campaign), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.exception("Failed to update campaign")
        return jsonify({"error": str(exc)}), 500


# ------------------------------------------------------------------
# State Transition Routes
# ------------------------------------------------------------------

@campaign_bp.route("/<int:campaign_id>/transition", methods=["POST"])
@_require_auth
def transition_campaign(campaign_id):
    """Transition a campaign to a new state."""
    data = request.get_json(force=True)
    new_state = data.get("new_state") if data else None
    if not new_state:
        return jsonify({"error": "'new_state' is required."}), 400

    # Privileged actions require campaign_send permission
    if new_state in PRIVILEGED_ACTIONS:
        user_role = session.get("user_role", "")
        user_permissions = session.get("permissions", [])
        if user_role != "admin" and "campaign_send" not in user_permissions:
            return jsonify({"error": "Forbidden. 'campaign_send' permission required."}), 403

    service = _get_service()
    operator = session.get("user_name", "unknown")
    try:
        campaign = service.transition_state(campaign_id, new_state, operator)
        return jsonify(campaign), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.exception("Failed to transition campaign")
        return jsonify({"error": str(exc)}), 500


@campaign_bp.route("/<int:campaign_id>/schedule", methods=["POST"])
@_require_auth
def schedule_campaign(campaign_id):
    """Schedule a campaign for a future date/time."""
    data = request.get_json(force=True)
    scheduled_at_str = data.get("scheduled_at") if data else None
    if not scheduled_at_str:
        return jsonify({"error": "'scheduled_at' datetime is required."}), 400

    try:
        scheduled_at = datetime.fromisoformat(scheduled_at_str)
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid datetime format. Use ISO 8601."}), 400

    if scheduled_at <= datetime.now():
        return jsonify({"error": "Scheduled time must be in the future."}), 400

    service = _get_service()
    operator = session.get("user_name", "unknown")
    try:
        campaign = service.schedule_campaign(campaign_id, scheduled_at, operator)
        return jsonify(campaign), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.exception("Failed to schedule campaign")
        return jsonify({"error": str(exc)}), 500


@campaign_bp.route("/<int:campaign_id>/duplicate", methods=["POST"])
@_require_auth
def duplicate_campaign(campaign_id):
    """Duplicate a campaign to a new draft."""
    service = _get_service()
    operator = session.get("user_name", "unknown")
    try:
        campaign = service.duplicate_campaign(campaign_id, operator)
        return jsonify(campaign), 201
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        logger.exception("Failed to duplicate campaign")
        return jsonify({"error": str(exc)}), 500


# ------------------------------------------------------------------
# Operator Approval Workflow Routes
# ------------------------------------------------------------------

@campaign_bp.route("/<int:campaign_id>/approve", methods=["POST"])
@_require_auth
@_require_campaign_send_permission
def approve_campaign(campaign_id):
    """
    Approve a pending campaign — triggers queue enqueue and sending.

    Requires 'campaign_send' permission.
    """
    service = _get_service()
    operator = session.get("user_name", "unknown")
    try:
        campaign = service.approve_campaign(campaign_id, operator)
        return jsonify(campaign), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.exception("Failed to approve campaign")
        return jsonify({"error": str(exc)}), 500


@campaign_bp.route("/<int:campaign_id>/reject", methods=["POST"])
@_require_auth
@_require_campaign_send_permission
def reject_campaign(campaign_id):
    """
    Reject a pending campaign — transitions to cancelled with reason.

    Requires 'campaign_send' permission.
    """
    data = request.get_json(force=True) or {}
    reason = data.get("reason", "")

    service = _get_service()
    operator = session.get("user_name", "unknown")
    try:
        campaign = service.reject_campaign(campaign_id, operator, reason)
        return jsonify(campaign), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.exception("Failed to reject campaign")
        return jsonify({"error": str(exc)}), 500


@campaign_bp.route("/<int:campaign_id>/preview", methods=["GET"])
@_require_auth
def preview_campaign(campaign_id):
    """
    Get approval preview: recipient count, template content, estimated time.

    Available to all authenticated operators for reviewing campaigns
    before approval.
    """
    service = _get_service()
    try:
        preview = service.get_approval_preview(campaign_id)
        return jsonify(preview), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        logger.exception("Failed to generate campaign preview")
        return jsonify({"error": str(exc)}), 500


@campaign_bp.route("/<int:campaign_id>/submit-for-approval", methods=["POST"])
@_require_auth
def submit_campaign_for_approval(campaign_id):
    """
    Submit a draft/scheduled campaign for operator approval.

    This endpoint is used by automation rules and operators to move
    campaigns into the pending_approval state. Ensures that no campaign
    (including automation-generated drafts) can be dispatched without
    going through the approval gate.
    """
    service = _get_service()
    operator = session.get("user_name", "unknown")
    try:
        campaign = service.submit_for_approval(campaign_id, operator)
        return jsonify(campaign), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.exception("Failed to submit campaign for approval")
        return jsonify({"error": str(exc)}), 500


# ------------------------------------------------------------------
# Test Send Route
# ------------------------------------------------------------------

@campaign_bp.route("/<int:campaign_id>/simulate", methods=["POST"])
@_require_auth
def simulate_campaign(campaign_id):
    """
    Run a pre-send simulation for a campaign.

    Returns projected audience count after exclusions, estimated send time,
    estimated cost, duplicate count, exclusion breakdown, and warnings.

    Requirements: 22.1, 22.2, 22.3, 22.4, 22.5, 22.6, 22.7
    """
    from services.simulation import SimulationEngine

    from app import get_mysql_connection

    engine = SimulationEngine(get_mysql_connection)
    try:
        result = engine.simulate(campaign_id)
        return jsonify({
            "campaign_id": campaign_id,
            "original_audience_count": result.original_audience_count,
            "final_audience_count": result.final_audience_count,
            "estimated_send_time_seconds": result.estimated_send_time_seconds,
            "estimated_cost_inr": result.estimated_cost_inr,
            "exclusions": {
                "cooldown": result.exclusions.cooldown,
                "opted_out": result.exclusions.opted_out,
                "dnd": result.exclusions.dnd,
                "invalid_number": result.exclusions.invalid_number,
                "incomplete_data": result.exclusions.incomplete_data,
                "total": result.exclusions.total,
            },
            "duplicate_count": result.duplicate_count,
            "warnings": result.warnings,
        }), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        logger.exception("Failed to simulate campaign %d", campaign_id)
        return jsonify({"error": str(exc)}), 500


@campaign_bp.route("/<int:campaign_id>/test-send", methods=["POST"])
@_require_auth
def test_send_campaign(campaign_id):
    """
    Send a test message to 1-5 mobile numbers immediately.

    Bypasses the regular queue and marks messages as test sends
    (excluded from campaign analytics).

    Request body:
        {"test_numbers": ["919876543210", "919876543211"]}

    Returns delivery status for each test number.
    """
    data = request.get_json(force=True)
    if not data or not data.get("test_numbers"):
        return jsonify({"error": "'test_numbers' array is required (1-5 numbers)."}), 400

    test_numbers = data["test_numbers"]

    if not isinstance(test_numbers, list):
        return jsonify({"error": "'test_numbers' must be an array."}), 400

    if len(test_numbers) < 1 or len(test_numbers) > 5:
        return jsonify({
            "error": "Must provide between 1 and 5 test numbers."
        }), 400

    service = _get_service()
    operator = session.get("user_name", "unknown")
    try:
        results = service.test_send(campaign_id, test_numbers, operator)
        return jsonify({
            "campaign_id": campaign_id,
            "test_results": results,
            "total": len(results),
            "success_count": sum(1 for r in results if r["status"] == "sent"),
            "failed_count": sum(1 for r in results if r["status"] == "failed"),
        }), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.exception("Failed to execute test send for campaign %d", campaign_id)
        return jsonify({"error": str(exc)}), 500


# ------------------------------------------------------------------
# A/B Testing Routes
# ------------------------------------------------------------------

@campaign_bp.route("/<int:campaign_id>/ab-test", methods=["POST"])
@_require_auth
def create_ab_test(campaign_id):
    """
    Create an A/B test for a campaign.

    Request body:
        {
            "variants": [1, 2, 3],   // template_ids (2-4)
            "test_percentage": 25    // 10-50
        }

    Returns A/B test configuration with variant records.
    """
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "Request body required."}), 400

    variants = data.get("variants")
    test_pct = data.get("test_percentage")

    if not variants or not isinstance(variants, list):
        return jsonify({"error": "'variants' must be a list of template IDs (2-4)."}), 400

    if test_pct is None:
        return jsonify({"error": "'test_percentage' is required (10-50)."}), 400

    try:
        test_pct = float(test_pct)
    except (ValueError, TypeError):
        return jsonify({"error": "'test_percentage' must be a number."}), 400

    service = _get_service()
    operator = session.get("user_name", "unknown")
    try:
        result = service.create_ab_test(campaign_id, variants, test_pct, operator)
        return jsonify(result), 201
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.exception("Failed to create A/B test for campaign %d", campaign_id)
        return jsonify({"error": str(exc)}), 500


@campaign_bp.route("/<int:campaign_id>/ab-test/variants", methods=["GET"])
@_require_auth
def get_ab_variants(campaign_id):
    """
    Get A/B test variants and their metrics for a campaign.

    Returns list of variant records with per-variant delivery stats.
    """
    service = _get_service()
    try:
        variants = service.get_ab_variants(campaign_id)
        return jsonify({
            "campaign_id": campaign_id,
            "variants": variants,
            "variant_count": len(variants),
        }), 200
    except Exception as exc:
        logger.exception("Failed to get A/B variants for campaign %d", campaign_id)
        return jsonify({"error": str(exc)}), 500


@campaign_bp.route("/<int:campaign_id>/ab-test/select-winner", methods=["POST"])
@_require_auth
@_require_campaign_send_permission
def select_ab_winner(campaign_id):
    """
    Select the winning A/B variant and create a full rollout campaign.

    Request body:
        {"variant_id": 5}

    Creates a new draft campaign targeting the remaining audience with the
    winning template. Requires 'campaign_send' permission.

    The full rollout campaign must still go through the approval workflow
    before any messages are dispatched — prevents full send until operator
    explicitly approves the rollout.
    """
    data = request.get_json(force=True)
    if not data or not data.get("variant_id"):
        return jsonify({"error": "'variant_id' is required."}), 400

    variant_id = data["variant_id"]
    try:
        variant_id = int(variant_id)
    except (ValueError, TypeError):
        return jsonify({"error": "'variant_id' must be an integer."}), 400

    service = _get_service()
    operator = session.get("user_name", "unknown")
    try:
        result = service.select_ab_winner(campaign_id, variant_id, operator)
        return jsonify(result), 201
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.exception("Failed to select A/B winner for campaign %d", campaign_id)
        return jsonify({"error": str(exc)}), 500


# ------------------------------------------------------------------
# Reactivation Workflow Routes (Requirements 6.1, 6.2, 6.3, 6.4, 6.5)
# ------------------------------------------------------------------

def _get_reactivation_service():
    """Get ReactivationService instance using app's MySQL connection factory."""
    from app import get_mysql_connection
    from services.reactivation import ReactivationService
    return ReactivationService(get_mysql_connection)


@campaign_bp.route("/reactivation/workflows", methods=["GET"])
@_require_auth
def list_reactivation_workflows():
    """
    List all available reactivation workflow templates.

    Returns the 5 built-in workflow templates with their suggested
    segment filters and template names.
    """
    service = _get_reactivation_service()
    workflows = service.list_workflows()
    return jsonify({"workflows": workflows, "total": len(workflows)}), 200


@campaign_bp.route("/reactivation/workflows/<workflow_id>", methods=["GET"])
@_require_auth
def get_reactivation_workflow(workflow_id):
    """
    Get a specific reactivation workflow by ID.

    Returns the workflow configuration including suggested segment
    filters and template name.
    """
    service = _get_reactivation_service()
    try:
        workflow = service.get_workflow(workflow_id)
        return jsonify(workflow), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404


@campaign_bp.route("/reactivation/workflows/<workflow_id>/prepare", methods=["POST"])
@_require_auth
def prepare_reactivation_campaign(workflow_id):
    """
    Pre-populate a campaign from a reactivation workflow.

    Creates a campaign draft with the workflow's suggested segment and template.
    The campaign MUST be approved by an operator before any messages are dispatched.
    """
    service = _get_reactivation_service()
    operator = session.get("user_name", "unknown")
    try:
        result = service.prepare_campaign_from_workflow(workflow_id, operator)
        return jsonify(result), 201
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.exception("Failed to prepare reactivation campaign for workflow %s", workflow_id)
        return jsonify({"error": str(exc)}), 500


@campaign_bp.route("/reactivation/success", methods=["GET"])
@_require_auth
def get_reactivation_success():
    """
    Track reactivation success metrics.

    Monitors if customers who received reactivation campaigns have
    changed status to 'active' within 30 days.
    """
    days = request.args.get("days", 30, type=int)
    service = _get_reactivation_service()
    try:
        metrics = service.track_reactivation_success(days_window=days)
        return jsonify(metrics), 200
    except Exception as exc:
        logger.exception("Failed to compute reactivation success metrics")
        return jsonify({"error": str(exc)}), 500


# ------------------------------------------------------------------
# Automation Rules CRUD Routes
# ------------------------------------------------------------------

@campaign_bp.route("/automation-rules", methods=["POST"])
@_require_auth
def create_automation_rule():
    """
    Create a new automation rule.

    Request body:
        {
            "name": "Weekly Expired Recovery",
            "trigger_type": "schedule",  // schedule | event | threshold
            "trigger_config": {"cron": "0 9 * * 1"},
            "condition_config": {"min_expired_count": 10},  // optional
            "action_type": "create_campaign_draft",  // create_campaign_draft | notify_operator
            "action_config": {
                "campaign_name": "Auto Recovery",
                "campaign_type": "reactivation",
                "segment_id": 5,
                "template_id": 3
            },
            "is_active": true
        }

    All automation-generated campaigns require operator approval.
    """
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "Request body required."}), 400

    service = _get_reactivation_service()
    operator = session.get("user_name", "unknown")
    try:
        rule = service.create_automation_rule(data, operator)
        return jsonify(rule), 201
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.exception("Failed to create automation rule")
        return jsonify({"error": str(exc)}), 500


@campaign_bp.route("/automation-rules", methods=["GET"])
@_require_auth
def list_automation_rules():
    """List automation rules with optional filtering and pagination."""
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 20, type=int)
    active_only = request.args.get("active_only", "false").lower() == "true"

    service = _get_reactivation_service()
    try:
        result = service.list_automation_rules(
            active_only=active_only, page=page, per_page=per_page
        )
        return jsonify(result), 200
    except Exception as exc:
        logger.exception("Failed to list automation rules")
        return jsonify({"error": str(exc)}), 500


@campaign_bp.route("/automation-rules/<int:rule_id>", methods=["GET"])
@_require_auth
def get_automation_rule(rule_id):
    """Get a single automation rule by ID."""
    service = _get_reactivation_service()
    try:
        rule = service.get_automation_rule(rule_id)
        return jsonify(rule), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        logger.exception("Failed to get automation rule %d", rule_id)
        return jsonify({"error": str(exc)}), 500


@campaign_bp.route("/automation-rules/<int:rule_id>", methods=["PUT"])
@_require_auth
def update_automation_rule(rule_id):
    """Update an existing automation rule."""
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "Request body required."}), 400

    service = _get_reactivation_service()
    operator = session.get("user_name", "unknown")
    try:
        rule = service.update_automation_rule(rule_id, data, operator)
        return jsonify(rule), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.exception("Failed to update automation rule %d", rule_id)
        return jsonify({"error": str(exc)}), 500


@campaign_bp.route("/automation-rules/<int:rule_id>", methods=["DELETE"])
@_require_auth
def delete_automation_rule(rule_id):
    """Delete an automation rule."""
    service = _get_reactivation_service()
    try:
        service.delete_automation_rule(rule_id)
        return jsonify({"deleted": True, "rule_id": rule_id}), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        logger.exception("Failed to delete automation rule %d", rule_id)
        return jsonify({"error": str(exc)}), 500


@campaign_bp.route("/automation-rules/<int:rule_id>/execute", methods=["POST"])
@_require_auth
@_require_campaign_send_permission
def execute_automation_rule(rule_id):
    """
    Manually execute an automation rule.

    For 'create_campaign_draft' action: creates a campaign draft that
    requires operator approval before sending.
    For 'notify_operator' action: creates a system notification.

    Requires 'campaign_send' permission.
    """
    service = _get_reactivation_service()
    try:
        result = service.execute_rule(rule_id)
        return jsonify(result), 201
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.exception("Failed to execute automation rule %d", rule_id)
        return jsonify({"error": str(exc)}), 500
