"""
Simulation Engine for Enterprise WhatsApp CRM.

Performs pre-send campaign analysis showing:
- Final audience count after all exclusions
- Estimated send time based on throttle rate
- Estimated WhatsApp API cost based on per-message pricing
- Duplicate recipient detection
- Exclusion breakdown by reason
- Warnings when exclusions exceed threshold

Requirements: 22.1, 22.2, 22.3, 22.4, 22.5, 22.6, 22.7
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

# Default throttle rate (messages per second) per WhatsApp Business API
DEFAULT_THROTTLE_RATE = 80

# Default per-message cost rates (INR) by template category
DEFAULT_COST_RATES = {
    "marketing": 0.76,
    "utility": 0.35,
    "authentication": 0.28,
}

# Warning threshold: generate warning when exclusions > 30% of original segment
EXCLUSION_WARNING_THRESHOLD = 0.30


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ExclusionBreakdown:
    """Breakdown of excluded customers by reason."""

    cooldown: int = 0
    opted_out: int = 0
    dnd: int = 0
    invalid_number: int = 0
    incomplete_data: int = 0

    @property
    def total(self) -> int:
        """Total number of exclusions across all reasons."""
        return (
            self.cooldown
            + self.opted_out
            + self.dnd
            + self.invalid_number
            + self.incomplete_data
        )


@dataclass
class SimulationResult:
    """Result of a campaign simulation."""

    original_audience_count: int = 0
    final_audience_count: int = 0
    estimated_send_time_seconds: float = 0.0
    estimated_cost_inr: float = 0.0
    exclusions: ExclusionBreakdown = field(default_factory=ExclusionBreakdown)
    duplicate_count: int = 0
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# SimulationEngine
# ---------------------------------------------------------------------------


class SimulationEngine:
    """
    Campaign simulation engine for pre-send analysis.

    Computes final audience after applying all exclusion rules, estimates
    send time and cost, detects duplicates, and generates warnings.

    Parameters
    ----------
    get_connection : callable
        A zero-argument function that returns a MySQL connection object.
    throttle_rate : int, optional
        Messages per second dispatch rate. Default: 80.
    cost_rates : dict, optional
        Per-message cost rates (INR) keyed by template category.
        Default: marketing=0.76, utility=0.35, authentication=0.28.
    """

    def __init__(
        self,
        get_connection: Callable,
        throttle_rate: int = DEFAULT_THROTTLE_RATE,
        cost_rates: Optional[dict] = None,
    ):
        self._get_conn = get_connection
        self._throttle_rate = throttle_rate
        self._cost_rates = cost_rates or DEFAULT_COST_RATES.copy()

    @property
    def throttle_rate(self) -> int:
        """Return the configured throttle rate."""
        return self._throttle_rate

    @property
    def cost_rates(self) -> dict:
        """Return the configured cost rates."""
        return self._cost_rates

    def simulate(self, campaign_id: int) -> SimulationResult:
        """
        Run a full simulation for a campaign.

        Performs the following computations:
        1. Load campaign and resolve segment to get original audience
        2. Apply exclusions: cooldown, opt-out, DND, invalid numbers,
           incomplete template parameters
        3. Detect duplicate recipients
        4. Calculate estimated send time = final_audience / throttle_rate
        5. Calculate estimated cost = final_audience * per_message_rate
        6. Generate warnings if exclusions > 30% of original segment

        Parameters
        ----------
        campaign_id : int
            The campaign to simulate.

        Returns
        -------
        SimulationResult
            Complete simulation analysis.

        Raises
        ------
        ValueError
            If campaign or segment not found.
        """
        conn = self._get_conn()
        cursor = None
        try:
            cursor = conn.cursor(dictionary=True)

            # 1. Load campaign
            campaign = self._load_campaign(cursor, campaign_id)
            segment_id = campaign.get("segment_id")
            template_id = campaign.get("template_id")
            campaign_type = campaign.get("campaign_type", "promotional")

            # 2. Load segment and get recipients
            recipients = self._get_segment_recipients(cursor, segment_id)
            original_count = len(recipients)

            # 3. Load template mappings for incomplete data check
            template_mappings = self._load_template_mappings(cursor, template_id)

            # 4. Apply exclusions
            exclusions = ExclusionBreakdown()
            eligible_mobiles = []
            seen_mobiles = set()
            duplicate_count = 0

            for recipient in recipients:
                mobile = recipient.get("mobile", "")

                # Duplicate detection
                if mobile in seen_mobiles:
                    duplicate_count += 1
                    continue
                seen_mobiles.add(mobile)

                # Invalid number check
                if not self._is_valid_mobile(mobile):
                    exclusions.invalid_number += 1
                    continue

                # Opt-out / DND check (from suppression_list)
                suppression_reason = self._check_suppression(cursor, mobile)
                if suppression_reason == "opted_out":
                    exclusions.opted_out += 1
                    continue
                elif suppression_reason == "dnd":
                    exclusions.dnd += 1
                    continue

                # Cooldown check
                if self._is_in_cooldown(cursor, mobile, campaign_type):
                    exclusions.cooldown += 1
                    continue

                # Incomplete template parameters check
                if template_mappings and not self._has_complete_params(
                    recipient, template_mappings
                ):
                    exclusions.incomplete_data += 1
                    continue

                eligible_mobiles.append(mobile)

            # 5. Calculate final metrics
            final_audience = len(eligible_mobiles)

            # Estimated send time (seconds)
            estimated_send_time = (
                final_audience / self._throttle_rate
                if self._throttle_rate > 0
                else 0.0
            )

            # Estimated cost
            category = self._get_template_category(cursor, template_id)
            per_message_rate = self._cost_rates.get(
                category, self._cost_rates.get("marketing", 0.76)
            )
            estimated_cost = final_audience * per_message_rate

            # 6. Generate warnings
            warnings = []
            if original_count > 0:
                exclusion_ratio = exclusions.total / original_count
                if exclusion_ratio > EXCLUSION_WARNING_THRESHOLD:
                    pct = round(exclusion_ratio * 100, 1)
                    warnings.append(
                        f"High exclusion rate: {pct}% of the original segment "
                        f"({exclusions.total} of {original_count} recipients) "
                        f"would be excluded. Consider reviewing your segment criteria."
                    )

            if duplicate_count > 0:
                warnings.append(
                    f"Detected {duplicate_count} duplicate recipient(s) in the audience segment."
                )

            return SimulationResult(
                original_audience_count=original_count,
                final_audience_count=final_audience,
                estimated_send_time_seconds=round(estimated_send_time, 2),
                estimated_cost_inr=round(estimated_cost, 2),
                exclusions=exclusions,
                duplicate_count=duplicate_count,
                warnings=warnings,
            )

        finally:
            if cursor:
                cursor.close()
            conn.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_campaign(self, cursor, campaign_id: int) -> dict:
        """Load campaign record by ID."""
        cursor.execute(
            "SELECT id, name, segment_id, template_id, campaign_type, status "
            "FROM campaigns WHERE id = %s",
            (campaign_id,),
        )
        campaign = cursor.fetchone()
        if not campaign:
            raise ValueError(f"Campaign {campaign_id} not found")
        return campaign

    def _get_segment_recipients(self, cursor, segment_id: Optional[int]) -> list:
        """
        Fetch all recipients for a given segment.

        Returns list of dicts with at least 'mobile' and 'customer_name' keys.
        """
        if not segment_id:
            return []

        # Load segment filter criteria
        cursor.execute(
            "SELECT filter_criteria FROM audience_segments WHERE id = %s",
            (segment_id,),
        )
        segment = cursor.fetchone()
        if not segment:
            raise ValueError(f"Segment {segment_id} not found")

        filter_criteria = segment.get("filter_criteria")
        if isinstance(filter_criteria, str):
            try:
                criteria = json.loads(filter_criteria)
            except (json.JSONDecodeError, TypeError):
                criteria = {}
        else:
            criteria = filter_criteria or {}

        # Build query using the same logic as segmentation service
        where_parts = []
        params = []

        for field_name, value in criteria.items():
            if value is None:
                continue
            if field_name in (
                "expiry_category",
                "plan_name",
                "plan_category",
                "zone_name",
                "area",
                "building",
                "status",
                "network_type",
                "connectivity_mode",
                "owner_tenant",
            ):
                where_parts.append(f"{field_name} = %s")
                params.append(value)
            elif field_name == "kyc_approved":
                where_parts.append("kyc_approved = %s")
                params.append(1 if value else 0)
            elif field_name == "days_remaining" and isinstance(value, dict):
                if value.get("min") is not None:
                    where_parts.append("days_remaining >= %s")
                    params.append(value["min"])
                if value.get("max") is not None:
                    where_parts.append("days_remaining <= %s")
                    params.append(value["max"])

        where_sql = " AND ".join(where_parts) if where_parts else "1=1"
        query = f"SELECT mobile, customer_name, plan_name, zone_name, area, expiry_date FROM customers WHERE {where_sql}"

        cursor.execute(query, tuple(params))
        return cursor.fetchall()

    def _load_template_mappings(self, cursor, template_id: Optional[int]) -> dict:
        """Load placeholder mappings for a template."""
        if not template_id:
            return {}

        cursor.execute(
            "SELECT placeholder_mappings FROM campaign_templates WHERE id = %s",
            (template_id,),
        )
        template = cursor.fetchone()
        if not template:
            return {}

        mappings_raw = template.get("placeholder_mappings")
        if not mappings_raw:
            return {}

        if isinstance(mappings_raw, str):
            try:
                return json.loads(mappings_raw)
            except (json.JSONDecodeError, TypeError):
                return {}
        return mappings_raw if isinstance(mappings_raw, dict) else {}

    def _is_valid_mobile(self, mobile: str) -> bool:
        """
        Check if a mobile number is valid.

        A valid mobile must be non-empty, contain only digits (optionally
        prefixed with +), and be between 7 and 15 digits long.
        """
        if not mobile or not isinstance(mobile, str):
            return False
        cleaned = mobile.strip().lstrip("+")
        if not cleaned:
            return False
        if not cleaned.isdigit():
            return False
        if len(cleaned) < 7 or len(cleaned) > 15:
            return False
        return True

    def _check_suppression(self, cursor, mobile: str) -> Optional[str]:
        """
        Check if a mobile is on the suppression list.

        Returns:
            'opted_out' if suppressed due to opt-out keyword
            'dnd' if suppressed due to manual DND or other reasons
            None if not suppressed
        """
        cursor.execute(
            """
            SELECT reason FROM suppression_list
            WHERE customer_mobile = %s AND is_active = 1
            LIMIT 1
            """,
            (mobile,),
        )
        row = cursor.fetchone()
        if not row:
            return None

        reason = row.get("reason", "")
        if reason == "opt_out_keyword":
            return "opted_out"
        # manual_dnd, user_blocked, spam_reported, invalid_number all count as DND
        return "dnd"

    def _is_in_cooldown(self, cursor, mobile: str, campaign_type: str) -> bool:
        """
        Check if a customer is in cooldown for the given campaign type.

        Transactional messages bypass cooldown checks.
        """
        # Transactional messages bypass cooldown
        if campaign_type.lower() == "transactional":
            return False

        from datetime import datetime, timedelta

        # Check for promotional message within 72 hours (default window)
        window_cutoff = datetime.now() - timedelta(hours=72)

        cursor.execute(
            """
            SELECT COUNT(*) AS cnt
            FROM message_cooldowns
            WHERE customer_mobile = %s
              AND campaign_type IN ('promotional', 'reactivation', 'ab_test')
              AND sent_at >= %s
            """,
            (mobile, window_cutoff),
        )
        row = cursor.fetchone()
        return (row.get("cnt", 0) if row else 0) > 0

    def _has_complete_params(self, recipient: dict, mappings: dict) -> bool:
        """
        Check if a recipient has all required template parameters.

        Returns True if all mapped fields exist and are non-empty in the
        recipient data.
        """
        for placeholder_name, field_name in mappings.items():
            value = recipient.get(field_name)
            if value is None:
                return False
            if isinstance(value, str) and value.strip() == "":
                return False
        return True

    def _get_template_category(self, cursor, template_id: Optional[int]) -> str:
        """Get the template category for cost calculation."""
        if not template_id:
            return "marketing"

        cursor.execute(
            "SELECT category FROM campaign_templates WHERE id = %s",
            (template_id,),
        )
        row = cursor.fetchone()
        if not row:
            return "marketing"
        return row.get("category", "marketing")
