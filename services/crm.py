"""
CRM Service — customer profile, notes, tags, interaction timeline.

Provides CRMService class with pure business logic.
Accepts a connection factory (get_connection callable) for testability.

Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 19.6, 23.7
"""

import json
import logging
from datetime import datetime
from typing import List, Optional

logger = logging.getLogger(__name__)


class CRMService:
    """
    Customer Relationship Management service.

    Provides customer profile views, interaction timeline, notes, tags,
    campaign history, opt-out/DND status, and engagement scores.

    Parameters
    ----------
    get_connection : callable
        A zero-argument function that returns a MySQL connection object.
    """

    def __init__(self, get_connection):
        self._get_conn = get_connection

    # ------------------------------------------------------------------
    # Customer Profile (Requirement 7.1, 19.6, 23.7)
    # ------------------------------------------------------------------

    def get_customer_profile(self, customer_id: int = None, mobile: str = None) -> dict:
        """
        Get full customer profile by joining customers with
        suppression/engagement data.

        Accepts either customer_id (customers.id) or mobile number.
        Returns profile dict including opt-out/DND status and engagement score.

        Raises ValueError if customer not found.
        """
        if customer_id is None and mobile is None:
            raise ValueError("Either customer_id or mobile must be provided")

        conn = self._get_conn()
        cursor = None
        try:
            cursor = conn.cursor(dictionary=True)

            # Fetch base customer data from customers
            if customer_id is not None:
                cursor.execute(
                    "SELECT * FROM customers WHERE id = %s", (customer_id,)
                )
            else:
                cursor.execute(
                    "SELECT * FROM customers WHERE mobile = %s LIMIT 1", (mobile,)
                )

            customer = cursor.fetchone()
            if not customer:
                identifier = customer_id if customer_id is not None else mobile
                raise ValueError(f"Customer not found: {identifier}")

            customer_mobile = customer.get("mobile", "")

            # Fetch opt-out / DND suppression status (Requirement 19.6)
            cursor.execute(
                """
                SELECT reason, source_keyword, created_at
                FROM suppression_list
                WHERE customer_mobile = %s AND is_active = 1
                ORDER BY created_at DESC
                """,
                (customer_mobile,),
            )
            suppression_records = cursor.fetchall()

            opt_out_status = None
            dnd_status = None
            for record in suppression_records:
                if record["reason"] == "opt_out_keyword":
                    opt_out_status = {
                        "opted_out": True,
                        "source": "keyword",
                        "keyword": record.get("source_keyword"),
                        "date": record["created_at"],
                    }
                elif record["reason"] == "manual_dnd":
                    dnd_status = {
                        "dnd_active": True,
                        "source": "manual",
                        "date": record["created_at"],
                    }
                elif record["reason"] in ("user_blocked", "spam_reported"):
                    if not opt_out_status:
                        opt_out_status = {
                            "opted_out": True,
                            "source": record["reason"],
                            "keyword": record.get("source_keyword"),
                            "date": record["created_at"],
                        }

            if not opt_out_status:
                opt_out_status = {"opted_out": False, "source": None, "date": None}
            if not dnd_status:
                dnd_status = {"dnd_active": False, "source": None, "date": None}

            # Fetch engagement score and trend (Requirement 23.7)
            cursor.execute(
                """
                SELECT interaction_score, engagement_trend, last_interaction_at,
                       messages_received_count, messages_read_count, response_count,
                       avg_time_to_read_seconds, preferred_time_window,
                       avg_response_time_seconds
                FROM customer_engagement
                WHERE customer_mobile = %s
                """,
                (customer_mobile,),
            )
            engagement = cursor.fetchone()

            if engagement:
                engagement_data = {
                    "score": engagement["interaction_score"],
                    "trend": engagement["engagement_trend"],
                    "last_interaction_at": engagement["last_interaction_at"],
                    "messages_received": engagement["messages_received_count"],
                    "messages_read": engagement["messages_read_count"],
                    "response_count": engagement["response_count"],
                    "avg_time_to_read_seconds": engagement["avg_time_to_read_seconds"],
                    "preferred_time_window": engagement["preferred_time_window"],
                    "avg_response_time_seconds": engagement["avg_response_time_seconds"],
                }
            else:
                engagement_data = {
                    "score": 0,
                    "trend": "stable",
                    "last_interaction_at": None,
                    "messages_received": 0,
                    "messages_read": 0,
                    "response_count": 0,
                    "avg_time_to_read_seconds": None,
                    "preferred_time_window": None,
                    "avg_response_time_seconds": None,
                }

            # Fetch customer tags
            cursor.execute(
                """
                SELECT tag_name, added_by, created_at
                FROM customer_tags
                WHERE customer_mobile = %s
                ORDER BY created_at DESC
                """,
                (customer_mobile,),
            )
            tags = cursor.fetchall()

            # Build profile response
            profile = {
                "id": customer.get("id"),
                "mobile": customer_mobile,
                "customer_name": customer.get("customer_name"),
                "plan_name": customer.get("plan_name"),
                "plan_category": customer.get("plan_category"),
                "validity": customer.get("validity"),
                "days_remaining": customer.get("days_remaining"),
                "status": customer.get("status"),
                "zone_name": customer.get("zone_name"),
                "area": customer.get("area"),
                "building": customer.get("building"),
                "network_type": customer.get("network_type"),
                "connectivity_mode": customer.get("connectivity_mode"),
                "activation_date": customer.get("activation_date"),
                "expiry_date": customer.get("expiry_date"),
                "kyc_approved": customer.get("kyc_approved"),
                "owner_tenant": customer.get("owner_tenant"),
                "opt_out_status": opt_out_status,
                "dnd_status": dnd_status,
                "engagement": engagement_data,
                "tags": [t["tag_name"] for t in tags],
                "tags_detail": tags,
            }

            return profile
        finally:
            if cursor:
                cursor.close()
            conn.close()

    # ------------------------------------------------------------------
    # Interaction Timeline (Requirement 7.2, 7.6)
    # ------------------------------------------------------------------

    def get_interaction_timeline(
        self, customer_id: int = None, mobile: str = None, page: int = 1, per_page: int = 50
    ) -> dict:
        """
        Get merged interaction timeline for a customer.

        Merges: WhatsApp messages, campaign messages, operator notes,
        tag changes, and status changes — sorted reverse chronological.

        Returns paginated result with timeline events.
        """
        customer_mobile = self._resolve_mobile(customer_id, mobile)

        conn = self._get_conn()
        cursor = None
        try:
            cursor = conn.cursor(dictionary=True)

            # Fetch from customer_activity table (stores all activity types)
            offset = (page - 1) * per_page

            # Count total activities
            cursor.execute(
                "SELECT COUNT(*) as total FROM customer_activity WHERE customer_mobile = %s",
                (customer_mobile,),
            )
            total = cursor.fetchone()["total"]

            # Fetch paginated activities in reverse chronological order
            cursor.execute(
                """
                SELECT id, customer_mobile, activity_type, channel, reference_id,
                       details, created_at
                FROM customer_activity
                WHERE customer_mobile = %s
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
                """,
                (customer_mobile, per_page, offset),
            )
            activities = cursor.fetchall()

            # Parse JSON details field
            timeline = []
            for activity in activities:
                details = activity.get("details")
                if details and isinstance(details, str):
                    try:
                        details = json.loads(details)
                    except (json.JSONDecodeError, TypeError):
                        pass

                timeline.append({
                    "id": activity["id"],
                    "type": activity["activity_type"],
                    "channel": activity.get("channel", "whatsapp"),
                    "reference_id": activity.get("reference_id"),
                    "details": details,
                    "timestamp": activity["created_at"],
                })

            return {
                "mobile": customer_mobile,
                "timeline": timeline,
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
    # Notes (Requirement 7.4)
    # ------------------------------------------------------------------

    def add_note(
        self, customer_id: int = None, mobile: str = None,
        note: str = "", operator: str = ""
    ) -> dict:
        """
        Add a note to a customer profile.

        Persists with operator name and timestamp. Records in customer_activity.
        Returns the created note record.
        """
        if not note or not note.strip():
            raise ValueError("Note text cannot be empty")
        if not operator or not operator.strip():
            raise ValueError("Operator name is required")

        customer_mobile = self._resolve_mobile(customer_id, mobile)

        conn = self._get_conn()
        cursor = None
        try:
            cursor = conn.cursor(dictionary=True)
            conn.start_transaction()

            # Insert into customer_notes
            cursor.execute(
                """
                INSERT INTO customer_notes
                    (organization_id, branch_id, customer_mobile, note_text, added_by)
                VALUES (1, 1, %s, %s, %s)
                """,
                (customer_mobile, note.strip(), operator.strip()),
            )
            note_id = cursor.lastrowid

            # Record activity in customer_activity timeline
            details = json.dumps({
                "note_text": note.strip(),
                "added_by": operator.strip(),
            })
            cursor.execute(
                """
                INSERT INTO customer_activity
                    (customer_mobile, activity_type, channel, reference_id, details)
                VALUES (%s, 'note_added', 'system', %s, %s)
                """,
                (customer_mobile, note_id, details),
            )

            conn.commit()

            # Return the created note
            cursor.execute(
                "SELECT * FROM customer_notes WHERE id = %s", (note_id,)
            )
            created_note = cursor.fetchone()
            return created_note
        except Exception:
            conn.rollback()
            raise
        finally:
            if cursor:
                cursor.close()
            conn.close()

    # ------------------------------------------------------------------
    # Tags (Requirement 7.5)
    # ------------------------------------------------------------------

    def add_tags(
        self, customer_id: int = None, mobile: str = None,
        tags: List[str] = None, operator: str = ""
    ) -> List[str]:
        """
        Add one or more tags to a customer.

        Tags are persisted in customer_tags and made available for segmentation.
        Duplicate tags are silently ignored (UNIQUE constraint).
        Returns the list of tags successfully added.
        """
        if not tags:
            raise ValueError("At least one tag must be provided")
        if not operator or not operator.strip():
            raise ValueError("Operator name is required")

        customer_mobile = self._resolve_mobile(customer_id, mobile)

        conn = self._get_conn()
        cursor = None
        added_tags = []
        try:
            cursor = conn.cursor(dictionary=True)
            conn.start_transaction()

            for tag in tags:
                tag_name = tag.strip()
                if not tag_name:
                    continue

                # Use INSERT IGNORE to handle duplicate (customer_mobile, tag_name)
                cursor.execute(
                    """
                    INSERT IGNORE INTO customer_tags
                        (organization_id, branch_id, customer_mobile, tag_name, added_by)
                    VALUES (1, 1, %s, %s, %s)
                    """,
                    (customer_mobile, tag_name, operator.strip()),
                )

                if cursor.rowcount > 0:
                    added_tags.append(tag_name)
                    tag_id = cursor.lastrowid

                    # Record activity
                    details = json.dumps({
                        "tag_name": tag_name,
                        "added_by": operator.strip(),
                    })
                    cursor.execute(
                        """
                        INSERT INTO customer_activity
                            (customer_mobile, activity_type, channel, reference_id, details)
                        VALUES (%s, 'tag_added', 'system', %s, %s)
                        """,
                        (customer_mobile, tag_id, details),
                    )

            conn.commit()
            return added_tags
        except Exception:
            conn.rollback()
            raise
        finally:
            if cursor:
                cursor.close()
            conn.close()

    def remove_tag(
        self, customer_id: int = None, mobile: str = None,
        tag: str = "", operator: str = ""
    ) -> bool:
        """
        Remove a tag from a customer.

        Returns True if the tag was removed, False if it didn't exist.
        Records the removal in customer_activity.
        """
        if not tag or not tag.strip():
            raise ValueError("Tag name cannot be empty")
        if not operator or not operator.strip():
            raise ValueError("Operator name is required")

        customer_mobile = self._resolve_mobile(customer_id, mobile)
        tag_name = tag.strip()

        conn = self._get_conn()
        cursor = None
        try:
            cursor = conn.cursor(dictionary=True)
            conn.start_transaction()

            # Delete the tag
            cursor.execute(
                """
                DELETE FROM customer_tags
                WHERE customer_mobile = %s AND tag_name = %s
                """,
                (customer_mobile, tag_name),
            )

            removed = cursor.rowcount > 0

            if removed:
                # Record activity
                details = json.dumps({
                    "tag_name": tag_name,
                    "removed_by": operator.strip(),
                })
                cursor.execute(
                    """
                    INSERT INTO customer_activity
                        (customer_mobile, activity_type, channel, reference_id, details)
                    VALUES (%s, 'tag_removed', 'system', NULL, %s)
                    """,
                    (customer_mobile, details),
                )

            conn.commit()
            return removed
        except Exception:
            conn.rollback()
            raise
        finally:
            if cursor:
                cursor.close()
            conn.close()

    # ------------------------------------------------------------------
    # Campaign History (Requirement 7.3)
    # ------------------------------------------------------------------

    def get_campaign_history(self, customer_id: int = None, mobile: str = None) -> list:
        """
        Get all campaigns that targeted this customer with delivery status.

        Returns list of campaign delivery records with send date, template,
        and delivery status.
        """
        customer_mobile = self._resolve_mobile(customer_id, mobile)

        conn = self._get_conn()
        cursor = None
        try:
            cursor = conn.cursor(dictionary=True)

            cursor.execute(
                """
                SELECT
                    cm.id as message_id,
                    cm.campaign_id,
                    c.name as campaign_name,
                    c.campaign_type,
                    ct.template_name,
                    cm.status as delivery_status,
                    cm.sent_at,
                    cm.delivered_at,
                    cm.read_at,
                    cm.failed_at,
                    cm.error_message,
                    cm.channel
                FROM campaign_messages cm
                JOIN campaigns c ON c.id = cm.campaign_id
                LEFT JOIN campaign_templates ct ON ct.id = cm.template_id
                WHERE cm.customer_mobile = %s
                ORDER BY cm.sent_at DESC, cm.created_at DESC
                """,
                (customer_mobile,),
            )
            history = cursor.fetchall()
            return history
        finally:
            if cursor:
                cursor.close()
            conn.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_mobile(self, customer_id: int = None, mobile: str = None) -> str:
        """
        Resolve customer mobile number from either customer_id or mobile param.

        Raises ValueError if neither is provided or customer not found.
        """
        if mobile:
            return mobile

        if customer_id is None:
            raise ValueError("Either customer_id or mobile must be provided")

        conn = self._get_conn()
        cursor = None
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                "SELECT mobile FROM customers WHERE id = %s", (customer_id,)
            )
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Customer not found: {customer_id}")
            return row["mobile"]
        finally:
            if cursor:
                cursor.close()
            conn.close()
