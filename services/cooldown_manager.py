"""
Cooldown Manager for enterprise WhatsApp CRM.

Enforces message frequency limits per customer to protect WhatsApp
Business API quality ratings:
- 72-hour promotional window (120h when quality tier is Yellow)
- 2 promotional campaigns per rolling 7-day period
- Transactional messages bypass all cooldown checks
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Optional


@dataclass
class CooldownResult:
    """Result of a cooldown check for a customer/campaign combination."""

    allowed: bool
    reason: Optional[str] = None
    excluded_count: int = 0


class CooldownManager:
    """
    Manages message cooldown enforcement per customer.

    Queries the message_cooldowns table to determine whether a customer
    is eligible to receive a new campaign message based on:
    - Time-based window (72h default, 120h for Yellow quality tier)
    - Rolling frequency limit (max 2 promotional per 7 days)
    - Campaign type bypass (transactional messages skip all checks)

    Args:
        get_connection: Callable that returns a MySQL connection.
        quality_tier: Current WhatsApp quality tier ('green', 'yellow', 'red').
                      When 'yellow', the promotional window increases from 72h to 120h.
    """

    # Campaign types that bypass all cooldown checks
    TRANSACTIONAL_TYPES = ("transactional",)

    # Default promotional cooldown window in hours
    DEFAULT_PROMO_WINDOW_HOURS = 72

    # Extended promotional cooldown window when quality tier is Yellow
    YELLOW_TIER_PROMO_WINDOW_HOURS = 120

    # Maximum promotional campaigns per rolling 7-day period
    MAX_PROMO_PER_7_DAYS = 2

    def __init__(
        self,
        get_connection: Callable,
        quality_tier: str = "green",
    ):
        self._get_connection = get_connection
        self._quality_tier = quality_tier.lower()

    @property
    def quality_tier(self) -> str:
        """Return the current quality tier."""
        return self._quality_tier

    @quality_tier.setter
    def quality_tier(self, value: str) -> None:
        """Update the quality tier."""
        self._quality_tier = value.lower()

    def _get_promo_window_hours(self) -> int:
        """
        Return the promotional cooldown window in hours based on quality tier.

        Returns 120h for Yellow tier, 72h otherwise.
        """
        if self._quality_tier == "yellow":
            return self.YELLOW_TIER_PROMO_WINDOW_HOURS
        return self.DEFAULT_PROMO_WINDOW_HOURS

    def check_cooldown(
        self,
        mobile: str,
        campaign_type: str,
    ) -> CooldownResult:
        """
        Check if a customer is allowed to receive a message based on cooldown rules.

        Args:
            mobile: Customer mobile number.
            campaign_type: Type of the campaign ('promotional', 'transactional',
                          'reactivation', 'ab_test').

        Returns:
            CooldownResult with allowed=True if the message can be sent,
            or allowed=False with the exclusion reason.
        """
        # Transactional messages always bypass cooldown
        if campaign_type.lower() in self.TRANSACTIONAL_TYPES:
            return CooldownResult(allowed=True, reason=None, excluded_count=0)

        conn = self._get_connection()
        cursor = conn.cursor(dictionary=True)
        try:
            # Check 1: Time-based promotional window
            window_hours = self._get_promo_window_hours()
            window_cutoff = datetime.now() - timedelta(hours=window_hours)

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
            window_count = row["cnt"] if row else 0

            if window_count > 0:
                return CooldownResult(
                    allowed=False,
                    reason="cooldown_active",
                    excluded_count=window_count,
                )

            # Check 2: Rolling 7-day frequency limit (max 2 promotional campaigns)
            seven_day_cutoff = datetime.now() - timedelta(days=7)

            cursor.execute(
                """
                SELECT COUNT(*) AS cnt
                FROM message_cooldowns
                WHERE customer_mobile = %s
                  AND campaign_type IN ('promotional', 'reactivation', 'ab_test')
                  AND sent_at >= %s
                """,
                (mobile, seven_day_cutoff),
            )
            row = cursor.fetchone()
            weekly_count = row["cnt"] if row else 0

            if weekly_count >= self.MAX_PROMO_PER_7_DAYS:
                return CooldownResult(
                    allowed=False,
                    reason="weekly_limit_exceeded",
                    excluded_count=weekly_count,
                )

            # All checks passed
            return CooldownResult(allowed=True, reason=None, excluded_count=0)

        finally:
            cursor.close()

    def record_send(
        self,
        mobile: str,
        campaign_id: int,
        campaign_type: str,
    ) -> None:
        """
        Record a message send in the cooldown tracking table.

        Called after a message is successfully dispatched to update
        the customer's cooldown state.

        Args:
            mobile: Customer mobile number.
            campaign_id: ID of the campaign that sent the message.
            campaign_type: Type of the campaign.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                INSERT INTO message_cooldowns
                    (customer_mobile, campaign_id, campaign_type, sent_at)
                VALUES (%s, %s, %s, %s)
                """,
                (mobile, campaign_id, campaign_type, datetime.now()),
            )
            conn.commit()
        finally:
            cursor.close()
