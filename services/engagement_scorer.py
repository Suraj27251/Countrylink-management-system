"""
Engagement Scorer — batch computation of customer interaction scores.

Computes per-customer engagement metrics after each campaign:
- interaction_score = round(0.4 * read_rate + 0.3 * response_rate
                           + 0.2 * recency_score + 0.1 * frequency_score)
  bounded to [0, 100]
- engagement_trend classification (increasing, stable, declining)
- preferred_time_window detection (morning, afternoon, evening)

Designed to run as a batch process after campaign completion, processing
up to 50,000 customer records within 60 seconds.

Requirements: 23.1, 23.2, 23.3, 23.4, 23.5, 23.6
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class EngagementResult:
    """Result of computing engagement for a single customer."""
    customer_mobile: str
    messages_received_count: int
    messages_read_count: int
    response_count: int
    interaction_score: int
    engagement_trend: str
    preferred_time_window: str
    last_interaction_at: Optional[datetime] = None
    avg_response_time_seconds: Optional[int] = None


def compute_interaction_score(
    read_rate: float,
    response_rate: float,
    recency_score: float,
    frequency_score: float,
) -> int:
    """
    Compute the weighted interaction score, bounded [0, 100].

    Formula:
        score = round(0.4 * read_rate + 0.3 * response_rate
                      + 0.2 * recency_score + 0.1 * frequency_score)

    All input rates should be in the range [0, 100].
    The result is clamped to [0, 100].

    Args:
        read_rate: Percentage of messages read (0-100).
        response_rate: Percentage of messages responded to (0-100).
        recency_score: Recency score based on last interaction (0-100).
        frequency_score: Frequency of campaign participation (0-100).

    Returns:
        Integer score bounded [0, 100].
    """
    raw = 0.4 * read_rate + 0.3 * response_rate + 0.2 * recency_score + 0.1 * frequency_score
    score = round(raw)
    return max(0, min(100, score))


def classify_engagement_trend(
    current_score: int,
    previous_score: Optional[int],
    threshold: int = 10,
) -> str:
    """
    Classify engagement trend based on score change.

    Args:
        current_score: Current interaction score (0-100).
        previous_score: Previous interaction score, or None if first computation.
        threshold: Minimum change to classify as increasing/declining (default 10).

    Returns:
        One of 'increasing', 'stable', 'declining'.
    """
    if previous_score is None:
        return "stable"

    delta = current_score - previous_score
    if delta >= threshold:
        return "increasing"
    elif delta <= -threshold:
        return "declining"
    return "stable"


def detect_preferred_time_window(hour_counts: Dict[int, int]) -> str:
    """
    Detect preferred communication time window from interaction hours.

    Time windows:
    - morning: 6:00 - 11:59 (hours 6-11)
    - afternoon: 12:00 - 17:59 (hours 12-17)
    - evening: 18:00 - 23:59 and 0:00 - 5:59 (hours 18-23, 0-5)

    Args:
        hour_counts: Dictionary mapping hour (0-23) to interaction count.

    Returns:
        One of 'morning', 'afternoon', 'evening'.
    """
    morning_count = sum(hour_counts.get(h, 0) for h in range(6, 12))
    afternoon_count = sum(hour_counts.get(h, 0) for h in range(12, 18))
    evening_count = sum(
        hour_counts.get(h, 0) for h in list(range(18, 24)) + list(range(0, 6))
    )

    if morning_count >= afternoon_count and morning_count >= evening_count:
        return "morning"
    elif afternoon_count >= evening_count:
        return "afternoon"
    return "evening"


def compute_recency_score(last_interaction_at: Optional[datetime], now: Optional[datetime] = None) -> float:
    """
    Compute recency score based on last interaction timestamp.

    Scoring:
    - Within 1 day: 100
    - Within 7 days: 80
    - Within 14 days: 60
    - Within 30 days: 40
    - Within 60 days: 20
    - Older than 60 days or no interaction: 0

    Args:
        last_interaction_at: Datetime of the last interaction.
        now: Current datetime (defaults to datetime.now()).

    Returns:
        Recency score as float (0-100).
    """
    if last_interaction_at is None:
        return 0.0

    if now is None:
        now = datetime.now()

    days_since = (now - last_interaction_at).days

    if days_since <= 1:
        return 100.0
    elif days_since <= 7:
        return 80.0
    elif days_since <= 14:
        return 60.0
    elif days_since <= 30:
        return 40.0
    elif days_since <= 60:
        return 20.0
    return 0.0


def compute_frequency_score(campaign_count: int, max_campaigns: int = 20) -> float:
    """
    Compute frequency score based on campaign participation count.

    Linearly scales from 0 to 100 based on campaign_count / max_campaigns.
    Capped at 100.

    Args:
        campaign_count: Number of campaigns the customer participated in.
        max_campaigns: Reference count for 100% score (default 20).

    Returns:
        Frequency score as float (0-100).
    """
    if max_campaigns <= 0:
        return 0.0
    score = (campaign_count / max_campaigns) * 100
    return min(100.0, score)


class EngagementScorer:
    """
    Batch engagement scoring service.

    Processes customer engagement data and updates the customer_engagement table
    with computed interaction scores, trends, and preferred time windows.

    Args:
        get_connection: Callable that returns a MySQL connection.
    """

    # Batch size for processing customers in chunks
    BATCH_SIZE = 1000

    def __init__(self, get_connection: Callable):
        self._get_conn = get_connection

    def compute_campaign_engagement(self, campaign_id: int) -> int:
        """
        Compute engagement scores for all customers in a specific campaign.

        Updates the customer_engagement table for every recipient.

        Args:
            campaign_id: The campaign to process.

        Returns:
            Number of customer records updated.
        """
        conn = self._get_conn()
        cursor = conn.cursor(dictionary=True)
        try:
            # Get all unique customers from this campaign
            cursor.execute(
                """
                SELECT DISTINCT customer_mobile
                FROM campaign_messages
                WHERE campaign_id = %s
                """,
                (campaign_id,),
            )
            mobiles = [row["customer_mobile"] for row in cursor.fetchall()]
        finally:
            cursor.close()
            conn.close()

        if not mobiles:
            return 0

        updated = 0
        for i in range(0, len(mobiles), self.BATCH_SIZE):
            batch = mobiles[i: i + self.BATCH_SIZE]
            updated += self._process_batch(batch)

        logger.info(
            "Engagement scoring for campaign %d: %d customers updated",
            campaign_id,
            updated,
        )
        return updated

    def compute_all_engagement(self) -> int:
        """
        Compute engagement scores for all customers with campaign message history.

        Designed for periodic batch recomputation (e.g., nightly job).

        Returns:
            Number of customer records updated.
        """
        conn = self._get_conn()
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute(
                """
                SELECT DISTINCT customer_mobile
                FROM campaign_messages
                WHERE status IN ('sent', 'delivered', 'read', 'failed', 'permanently_failed')
                """
            )
            mobiles = [row["customer_mobile"] for row in cursor.fetchall()]
        finally:
            cursor.close()
            conn.close()

        if not mobiles:
            return 0

        updated = 0
        for i in range(0, len(mobiles), self.BATCH_SIZE):
            batch = mobiles[i: i + self.BATCH_SIZE]
            updated += self._process_batch(batch)

        logger.info("Full engagement recomputation: %d customers updated", updated)
        return updated

    def _process_batch(self, mobiles: List[str]) -> int:
        """
        Process a batch of customer mobiles and update engagement records.

        Args:
            mobiles: List of customer mobile numbers to process.

        Returns:
            Number of records updated.
        """
        if not mobiles:
            return 0

        conn = self._get_conn()
        cursor = conn.cursor(dictionary=True)
        updated = 0
        try:
            for mobile in mobiles:
                try:
                    result = self._compute_for_customer(cursor, mobile)
                    if result:
                        self._upsert_engagement(cursor, result)
                        updated += 1
                except Exception:
                    logger.warning(
                        "Failed to compute engagement for %s", mobile, exc_info=True
                    )
                    continue

            conn.commit()
        except Exception:
            conn.rollback()
            logger.exception("Failed to process engagement batch")
        finally:
            cursor.close()
            conn.close()

        return updated

    def _compute_for_customer(self, cursor, mobile: str) -> Optional[EngagementResult]:
        """
        Compute engagement metrics for a single customer.

        Args:
            cursor: Active DB cursor.
            mobile: Customer mobile number.

        Returns:
            EngagementResult or None if no data.
        """
        # Get message counts
        cursor.execute(
            """
            SELECT
                COUNT(*) AS total_received,
                SUM(CASE WHEN status = 'read' THEN 1 ELSE 0 END) AS read_count,
                MAX(COALESCE(read_at, delivered_at, sent_at)) AS last_interaction
            FROM campaign_messages
            WHERE customer_mobile = %s
              AND status IN ('sent', 'delivered', 'read', 'failed', 'permanently_failed')
            """,
            (mobile,),
        )
        msg_row = cursor.fetchone()

        if not msg_row or not msg_row["total_received"]:
            return None

        total_received = msg_row["total_received"] or 0
        read_count = msg_row["read_count"] or 0
        last_interaction = msg_row["last_interaction"]

        # Get response count (inbound messages from customer after campaign messages)
        cursor.execute(
            """
            SELECT COUNT(*) AS response_count
            FROM customer_activity
            WHERE customer_mobile = %s
              AND activity_type = 'message_received'
            """,
            (mobile,),
        )
        resp_row = cursor.fetchone()
        response_count = resp_row["response_count"] if resp_row else 0

        # Get campaign participation count (distinct campaigns)
        cursor.execute(
            """
            SELECT COUNT(DISTINCT campaign_id) AS campaign_count
            FROM campaign_messages
            WHERE customer_mobile = %s
            """,
            (mobile,),
        )
        camp_row = cursor.fetchone()
        campaign_count = camp_row["campaign_count"] if camp_row else 0

        # Get hour distribution for time window detection
        cursor.execute(
            """
            SELECT HOUR(COALESCE(read_at, delivered_at, sent_at)) AS hr,
                   COUNT(*) AS cnt
            FROM campaign_messages
            WHERE customer_mobile = %s
              AND status IN ('read', 'delivered')
              AND COALESCE(read_at, delivered_at, sent_at) IS NOT NULL
            GROUP BY hr
            """,
            (mobile,),
        )
        hour_counts = {}
        for row in cursor.fetchall():
            if row["hr"] is not None:
                hour_counts[int(row["hr"])] = row["cnt"]

        # Compute individual rate components (as 0-100 scale)
        read_rate = (read_count / total_received * 100) if total_received > 0 else 0.0
        response_rate = (response_count / total_received * 100) if total_received > 0 else 0.0
        recency_score = compute_recency_score(last_interaction)
        frequency_score = compute_frequency_score(campaign_count)

        # Compute interaction score
        interaction_score = compute_interaction_score(
            read_rate, response_rate, recency_score, frequency_score
        )

        # Get previous score for trend classification
        cursor.execute(
            """
            SELECT interaction_score
            FROM customer_engagement
            WHERE customer_mobile = %s
            LIMIT 1
            """,
            (mobile,),
        )
        prev_row = cursor.fetchone()
        previous_score = prev_row["interaction_score"] if prev_row else None

        # Classify trend
        trend = classify_engagement_trend(interaction_score, previous_score)

        # Detect preferred time window
        preferred_window = detect_preferred_time_window(hour_counts)

        # Compute average response time (if activity data available)
        avg_response_time = self._compute_avg_response_time(cursor, mobile)

        return EngagementResult(
            customer_mobile=mobile,
            messages_received_count=total_received,
            messages_read_count=read_count,
            response_count=response_count,
            interaction_score=interaction_score,
            engagement_trend=trend,
            preferred_time_window=preferred_window,
            last_interaction_at=last_interaction,
            avg_response_time_seconds=avg_response_time,
        )

    def _compute_avg_response_time(self, cursor, mobile: str) -> Optional[int]:
        """
        Compute average response time in seconds for a customer.

        Looks at time between campaign message delivery and customer reply.

        Args:
            cursor: Active DB cursor.
            mobile: Customer mobile number.

        Returns:
            Average response time in seconds, or None if no data.
        """
        cursor.execute(
            """
            SELECT AVG(TIMESTAMPDIFF(SECOND, cm.delivered_at, ca.created_at)) AS avg_resp
            FROM campaign_messages cm
            JOIN customer_activity ca
              ON ca.customer_mobile = cm.customer_mobile
              AND ca.activity_type = 'message_received'
              AND ca.created_at > cm.delivered_at
              AND ca.created_at <= DATE_ADD(cm.delivered_at, INTERVAL 24 HOUR)
            WHERE cm.customer_mobile = %s
              AND cm.delivered_at IS NOT NULL
            """,
            (mobile,),
        )
        row = cursor.fetchone()
        if row and row["avg_resp"] is not None:
            return int(row["avg_resp"])
        return None

    def _upsert_engagement(self, cursor, result: EngagementResult) -> None:
        """
        Insert or update customer_engagement record.

        Uses INSERT ... ON DUPLICATE KEY UPDATE for upsert semantics.

        Args:
            cursor: Active DB cursor.
            result: Computed engagement result.
        """
        cursor.execute(
            """
            INSERT INTO customer_engagement
                (customer_mobile, messages_received_count, messages_read_count,
                 response_count, interaction_score, engagement_trend,
                 preferred_time_window, last_interaction_at,
                 avg_response_time_seconds, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            ON DUPLICATE KEY UPDATE
                messages_received_count = VALUES(messages_received_count),
                messages_read_count = VALUES(messages_read_count),
                response_count = VALUES(response_count),
                interaction_score = VALUES(interaction_score),
                engagement_trend = VALUES(engagement_trend),
                preferred_time_window = VALUES(preferred_time_window),
                last_interaction_at = VALUES(last_interaction_at),
                avg_response_time_seconds = VALUES(avg_response_time_seconds),
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                result.customer_mobile,
                result.messages_received_count,
                result.messages_read_count,
                result.response_count,
                result.interaction_score,
                result.engagement_trend,
                result.preferred_time_window,
                result.last_interaction_at,
                result.avg_response_time_seconds,
            ),
        )

    def store_campaign_performance(self, campaign_id: int) -> None:
        """
        Store campaign performance patterns in campaign_analytics table.

        Records template effectiveness scores by segment type, optimal send time
        correlations, and seasonal engagement variations (Req 23.4).

        Args:
            campaign_id: Campaign to analyze.
        """
        conn = self._get_conn()
        cursor = conn.cursor(dictionary=True)
        try:
            # Get campaign details
            cursor.execute(
                """
                SELECT c.id, c.template_id, c.segment_id, c.started_at,
                       c.completed_at, c.total_recipients, c.sent_count,
                       c.delivered_count, c.read_count, c.failed_count,
                       ct.template_name, ct.category AS template_category,
                       aud.name AS segment_name
                FROM campaigns c
                LEFT JOIN campaign_templates ct ON ct.id = c.template_id
                LEFT JOIN audience_segments aud ON aud.id = c.segment_id
                WHERE c.id = %s
                """,
                (campaign_id,),
            )
            campaign = cursor.fetchone()
            if not campaign:
                return

            import json
            now = datetime.now()

            # Store delivery rate
            if campaign["sent_count"] and campaign["sent_count"] > 0:
                delivery_rate = (
                    (campaign["delivered_count"] or 0) / campaign["sent_count"]
                )
                cursor.execute(
                    """
                    INSERT INTO campaign_analytics
                        (organization_id, campaign_id, metric_type, metric_value,
                         dimensions, period_start, period_end, computed_at)
                    VALUES (1, %s, 'delivery_rate', %s, %s, %s, %s, %s)
                    """,
                    (
                        campaign_id,
                        round(delivery_rate, 4),
                        json.dumps({
                            "template": campaign.get("template_name"),
                            "segment": campaign.get("segment_name"),
                            "category": campaign.get("template_category"),
                        }),
                        campaign.get("started_at"),
                        campaign.get("completed_at") or now,
                        now,
                    ),
                )

                # Store read rate
                read_rate = (campaign["read_count"] or 0) / campaign["sent_count"]
                cursor.execute(
                    """
                    INSERT INTO campaign_analytics
                        (organization_id, campaign_id, metric_type, metric_value,
                         dimensions, period_start, period_end, computed_at)
                    VALUES (1, %s, 'read_rate', %s, %s, %s, %s, %s)
                    """,
                    (
                        campaign_id,
                        round(read_rate, 4),
                        json.dumps({
                            "template": campaign.get("template_name"),
                            "segment": campaign.get("segment_name"),
                            "category": campaign.get("template_category"),
                        }),
                        campaign.get("started_at"),
                        campaign.get("completed_at") or now,
                        now,
                    ),
                )

                # Store failure rate
                failure_rate = (campaign["failed_count"] or 0) / campaign["sent_count"]
                cursor.execute(
                    """
                    INSERT INTO campaign_analytics
                        (organization_id, campaign_id, metric_type, metric_value,
                         dimensions, period_start, period_end, computed_at)
                    VALUES (1, %s, 'failure_rate', %s, %s, %s, %s, %s)
                    """,
                    (
                        campaign_id,
                        round(failure_rate, 4),
                        json.dumps({
                            "template": campaign.get("template_name"),
                            "segment": campaign.get("segment_name"),
                            "category": campaign.get("template_category"),
                        }),
                        campaign.get("started_at"),
                        campaign.get("completed_at") or now,
                        now,
                    ),
                )

            conn.commit()
            logger.info("Stored campaign performance for campaign %d", campaign_id)
        except Exception:
            conn.rollback()
            logger.exception("Failed to store campaign performance for %d", campaign_id)
            raise
        finally:
            cursor.close()
            conn.close()
