"""
Analytics Blueprint — campaign metrics, aggregate reporting, and engagement data.

Flask Blueprint registered at /api/analytics/
Uses pre-computed summary tables (campaign_analytics) for dashboard queries
to avoid expensive aggregate queries on each page load (Req 12.4).

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 12.4
"""

import json
import logging
from datetime import datetime, timedelta
from functools import wraps

from flask import Blueprint, jsonify, request, session

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Flask Blueprint
# ---------------------------------------------------------------------------
analytics_bp = Blueprint("analytics", __name__, url_prefix="/api/analytics")


def _require_auth(f):
    """Decorator: require Flask session authentication."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("user_id"):
            return jsonify({"error": "Authentication required."}), 401
        return f(*args, **kwargs)
    return decorated


def _get_connection():
    """Get MySQL connection from app context."""
    from app import get_mysql_connection
    return get_mysql_connection()


def _parse_date_range():
    """
    Parse date range from query parameters.

    Query Parameters:
        start_date (str): Start date in ISO format (YYYY-MM-DD).
        end_date (str): End date in ISO format (YYYY-MM-DD).

    Returns:
        Tuple of (start_datetime, end_datetime) or (None, None).
    """
    start_str = request.args.get("start_date")
    end_str = request.args.get("end_date")

    start_date = None
    end_date = None

    if start_str:
        try:
            start_date = datetime.strptime(start_str, "%Y-%m-%d")
        except ValueError:
            pass

    if end_str:
        try:
            end_date = datetime.strptime(end_str, "%Y-%m-%d")
            # Set to end of day
            end_date = end_date.replace(hour=23, minute=59, second=59)
        except ValueError:
            pass

    return start_date, end_date


# ------------------------------------------------------------------
# Per-campaign metrics (Requirement 8.1)
# ------------------------------------------------------------------

@analytics_bp.route("/campaigns/<int:campaign_id>", methods=["GET"])
@_require_auth
def get_campaign_metrics(campaign_id):
    """
    Get per-campaign metrics: delivery_rate, read_rate, failure_rate, response_rate.

    Uses pre-computed summary from campaign_analytics table.
    Falls back to computing from campaign_messages if not pre-computed.

    Returns JSON with campaign metrics as percentage values.
    """
    conn = _get_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        # Try pre-computed metrics first (Req 12.4)
        cursor.execute(
            """
            SELECT metric_type, metric_value, dimensions, computed_at
            FROM campaign_analytics
            WHERE campaign_id = %s
              AND metric_type IN ('delivery_rate', 'read_rate', 'failure_rate', 'response_rate')
            ORDER BY computed_at DESC
            """,
            (campaign_id,),
        )
        precomputed = cursor.fetchall()

        if precomputed:
            metrics = {}
            for row in precomputed:
                # Take the most recently computed value for each metric type
                if row["metric_type"] not in metrics:
                    metrics[row["metric_type"]] = {
                        "value": float(row["metric_value"]) if row["metric_value"] else 0.0,
                        "percentage": round(float(row["metric_value"]) * 100, 2) if row["metric_value"] else 0.0,
                        "computed_at": row["computed_at"].isoformat() if row["computed_at"] else None,
                    }

            return jsonify({
                "campaign_id": campaign_id,
                "metrics": metrics,
                "source": "precomputed",
            }), 200

        # Fallback: compute from campaign_messages
        cursor.execute(
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN status IN ('sent', 'delivered', 'read') THEN 1 ELSE 0 END) AS sent_count,
                SUM(CASE WHEN status IN ('delivered', 'read') THEN 1 ELSE 0 END) AS delivered_count,
                SUM(CASE WHEN status = 'read' THEN 1 ELSE 0 END) AS read_count,
                SUM(CASE WHEN status IN ('failed', 'permanently_failed') THEN 1 ELSE 0 END) AS failed_count
            FROM campaign_messages
            WHERE campaign_id = %s
              AND status != 'queued'
              AND status != 'skipped'
            """,
            (campaign_id,),
        )
        row = cursor.fetchone()

        if not row or not row["total"]:
            return jsonify({
                "campaign_id": campaign_id,
                "metrics": {},
                "message": "No message data found for this campaign.",
            }), 200

        total = row["total"]
        sent_count = row["sent_count"] or 0
        delivered_count = row["delivered_count"] or 0
        read_count = row["read_count"] or 0
        failed_count = row["failed_count"] or 0

        # Compute response rate from customer_activity
        cursor.execute(
            """
            SELECT COUNT(DISTINCT ca.customer_mobile) AS responders
            FROM customer_activity ca
            JOIN campaign_messages cm
              ON cm.customer_mobile = ca.customer_mobile
              AND cm.campaign_id = %s
            WHERE ca.activity_type = 'message_received'
              AND ca.created_at >= cm.sent_at
              AND ca.created_at <= DATE_ADD(cm.sent_at, INTERVAL 48 HOUR)
            """,
            (campaign_id,),
        )
        resp_row = cursor.fetchone()
        responders = resp_row["responders"] if resp_row else 0

        metrics = {
            "delivery_rate": {
                "value": round(delivered_count / total, 4) if total > 0 else 0.0,
                "percentage": round(delivered_count / total * 100, 2) if total > 0 else 0.0,
            },
            "read_rate": {
                "value": round(read_count / total, 4) if total > 0 else 0.0,
                "percentage": round(read_count / total * 100, 2) if total > 0 else 0.0,
            },
            "failure_rate": {
                "value": round(failed_count / total, 4) if total > 0 else 0.0,
                "percentage": round(failed_count / total * 100, 2) if total > 0 else 0.0,
            },
            "response_rate": {
                "value": round(responders / total, 4) if total > 0 else 0.0,
                "percentage": round(responders / total * 100, 2) if total > 0 else 0.0,
            },
        }

        return jsonify({
            "campaign_id": campaign_id,
            "metrics": metrics,
            "totals": {
                "total_messages": total,
                "sent": sent_count,
                "delivered": delivered_count,
                "read": read_count,
                "failed": failed_count,
                "responded": responders,
            },
            "source": "computed",
        }), 200

    except Exception as e:
        logger.exception("Error fetching campaign metrics for %d", campaign_id)
        return jsonify({"error": "Failed to fetch campaign metrics."}), 500
    finally:
        cursor.close()
        conn.close()


# ------------------------------------------------------------------
# Aggregate metrics with date range filtering (Requirements 8.2, 8.4)
# ------------------------------------------------------------------

@analytics_bp.route("/aggregate", methods=["GET"])
@_require_auth
def get_aggregate_metrics():
    """
    Get aggregate metrics across all campaigns with date range filtering.

    Query Parameters:
        start_date (str): Start of date range (YYYY-MM-DD).
        end_date (str): End of date range (YYYY-MM-DD).
        period (str): Aggregation period - 'daily', 'weekly', 'monthly' (default: 'daily').

    Returns:
        Total messages sent, average delivery rate, average read rate,
        and top-performing templates.
    """
    start_date, end_date = _parse_date_range()
    period = request.args.get("period", "daily")

    # Default to last 30 days if no date range specified
    if not end_date:
        end_date = datetime.now()
    if not start_date:
        start_date = end_date - timedelta(days=30)

    conn = _get_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        # Get aggregate totals from pre-computed summary tables (Req 12.4)
        cursor.execute(
            """
            SELECT
                COUNT(DISTINCT campaign_id) AS total_campaigns,
                AVG(CASE WHEN metric_type = 'delivery_rate' THEN metric_value END) AS avg_delivery_rate,
                AVG(CASE WHEN metric_type = 'read_rate' THEN metric_value END) AS avg_read_rate,
                AVG(CASE WHEN metric_type = 'failure_rate' THEN metric_value END) AS avg_failure_rate
            FROM campaign_analytics
            WHERE period_start >= %s
              AND period_end <= %s
              AND metric_type IN ('delivery_rate', 'read_rate', 'failure_rate')
            """,
            (start_date, end_date),
        )
        agg_row = cursor.fetchone()

        # Get total messages sent in period
        cursor.execute(
            """
            SELECT COUNT(*) AS total_messages
            FROM campaign_messages
            WHERE sent_at >= %s
              AND sent_at <= %s
              AND status NOT IN ('queued', 'skipped')
            """,
            (start_date, end_date),
        )
        msg_row = cursor.fetchone()
        total_messages = msg_row["total_messages"] if msg_row else 0

        # Get period breakdown
        if period == "daily":
            date_format = "%Y-%m-%d"
            group_expr = "DATE(sent_at)"
        elif period == "weekly":
            date_format = "%Y-W%u"
            group_expr = "YEARWEEK(sent_at, 1)"
        else:  # monthly
            date_format = "%Y-%m"
            group_expr = "DATE_FORMAT(sent_at, '%Y-%m')"

        cursor.execute(
            f"""
            SELECT
                DATE_FORMAT(sent_at, %s) AS period_label,
                COUNT(*) AS messages_sent,
                SUM(CASE WHEN status IN ('delivered', 'read') THEN 1 ELSE 0 END) AS delivered,
                SUM(CASE WHEN status = 'read' THEN 1 ELSE 0 END) AS read_count
            FROM campaign_messages
            WHERE sent_at >= %s
              AND sent_at <= %s
              AND status NOT IN ('queued', 'skipped')
            GROUP BY period_label
            ORDER BY period_label ASC
            """,
            (date_format, start_date, end_date),
        )
        period_breakdown = cursor.fetchall()

        result = {
            "date_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "period": period,
            "summary": {
                "total_campaigns": agg_row["total_campaigns"] if agg_row and agg_row["total_campaigns"] else 0,
                "total_messages": total_messages,
                "avg_delivery_rate": round(float(agg_row["avg_delivery_rate"]) * 100, 2) if agg_row and agg_row["avg_delivery_rate"] else 0.0,
                "avg_read_rate": round(float(agg_row["avg_read_rate"]) * 100, 2) if agg_row and agg_row["avg_read_rate"] else 0.0,
                "avg_failure_rate": round(float(agg_row["avg_failure_rate"]) * 100, 2) if agg_row and agg_row["avg_failure_rate"] else 0.0,
            },
            "period_breakdown": period_breakdown,
        }

        return jsonify(result), 200

    except Exception as e:
        logger.exception("Error fetching aggregate metrics")
        return jsonify({"error": "Failed to fetch aggregate metrics."}), 500
    finally:
        cursor.close()
        conn.close()


# ------------------------------------------------------------------
# Zone-wise engagement breakdown (Requirement 8.5)
# ------------------------------------------------------------------

@analytics_bp.route("/zones", methods=["GET"])
@_require_auth
def get_zone_breakdown():
    """
    Get zone-wise engagement breakdown showing delivery and read rates per zone.

    Query Parameters:
        start_date (str): Start of date range (YYYY-MM-DD).
        end_date (str): End of date range (YYYY-MM-DD).

    Returns delivery_rate and read_rate per zone_name.
    """
    start_date, end_date = _parse_date_range()

    # Default to last 30 days
    if not end_date:
        end_date = datetime.now()
    if not start_date:
        start_date = end_date - timedelta(days=30)

    conn = _get_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        # Try pre-computed zone breakdown from campaign_analytics
        cursor.execute(
            """
            SELECT
                JSON_UNQUOTE(JSON_EXTRACT(dimensions, '$.zone')) AS zone_name,
                AVG(CASE WHEN metric_type = 'delivery_rate' THEN metric_value END) AS avg_delivery_rate,
                AVG(CASE WHEN metric_type = 'read_rate' THEN metric_value END) AS avg_read_rate,
                COUNT(DISTINCT campaign_id) AS campaign_count
            FROM campaign_analytics
            WHERE period_start >= %s
              AND period_end <= %s
              AND JSON_EXTRACT(dimensions, '$.zone') IS NOT NULL
            GROUP BY zone_name
            HAVING zone_name IS NOT NULL
            ORDER BY avg_read_rate DESC
            """,
            (start_date, end_date),
        )
        zone_data = cursor.fetchall()

        if zone_data:
            zones = []
            for row in zone_data:
                zones.append({
                    "zone_name": row["zone_name"],
                    "delivery_rate": round(float(row["avg_delivery_rate"]) * 100, 2) if row["avg_delivery_rate"] else 0.0,
                    "read_rate": round(float(row["avg_read_rate"]) * 100, 2) if row["avg_read_rate"] else 0.0,
                    "campaign_count": row["campaign_count"],
                })

            return jsonify({
                "date_range": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat(),
                },
                "zones": zones,
                "source": "precomputed",
            }), 200

        # Fallback: compute from campaign_messages joined with customers
        cursor.execute(
            """
            SELECT
                rr.zone_name,
                COUNT(*) AS total_sent,
                SUM(CASE WHEN cm.status IN ('delivered', 'read') THEN 1 ELSE 0 END) AS delivered_count,
                SUM(CASE WHEN cm.status = 'read' THEN 1 ELSE 0 END) AS read_count
            FROM campaign_messages cm
            JOIN customers rr ON rr.mobile = cm.customer_mobile
            WHERE cm.sent_at >= %s
              AND cm.sent_at <= %s
              AND cm.status NOT IN ('queued', 'skipped')
              AND rr.zone_name IS NOT NULL
              AND rr.zone_name != ''
            GROUP BY rr.zone_name
            ORDER BY read_count DESC
            """,
            (start_date, end_date),
        )
        zone_rows = cursor.fetchall()

        zones = []
        for row in zone_rows:
            total = row["total_sent"] or 1
            zones.append({
                "zone_name": row["zone_name"],
                "total_sent": row["total_sent"],
                "delivered_count": row["delivered_count"] or 0,
                "read_count": row["read_count"] or 0,
                "delivery_rate": round((row["delivered_count"] or 0) / total * 100, 2),
                "read_rate": round((row["read_count"] or 0) / total * 100, 2),
            })

        return jsonify({
            "date_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "zones": zones,
            "source": "computed",
        }), 200

    except Exception as e:
        logger.exception("Error fetching zone breakdown")
        return jsonify({"error": "Failed to fetch zone breakdown."}), 500
    finally:
        cursor.close()
        conn.close()


# ------------------------------------------------------------------
# Top-performing templates (Requirement 8.6)
# ------------------------------------------------------------------

@analytics_bp.route("/templates/top", methods=["GET"])
@_require_auth
def get_top_templates():
    """
    Get top 5 performing campaign templates by read rate over the last 30 days.

    Query Parameters:
        limit (int): Number of templates to return (default 5).
        days (int): Number of days to look back (default 30).

    Returns templates ranked by read rate.
    """
    limit = request.args.get("limit", 5, type=int)
    days = request.args.get("days", 30, type=int)

    cutoff_date = datetime.now() - timedelta(days=days)

    conn = _get_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        # Use pre-computed metrics from campaign_analytics (Req 12.4)
        cursor.execute(
            """
            SELECT
                JSON_UNQUOTE(JSON_EXTRACT(ca.dimensions, '$.template')) AS template_name,
                AVG(ca.metric_value) AS avg_read_rate,
                COUNT(DISTINCT ca.campaign_id) AS usage_count
            FROM campaign_analytics ca
            WHERE ca.metric_type = 'read_rate'
              AND ca.period_start >= %s
              AND JSON_EXTRACT(ca.dimensions, '$.template') IS NOT NULL
            GROUP BY template_name
            HAVING template_name IS NOT NULL
            ORDER BY avg_read_rate DESC
            LIMIT %s
            """,
            (cutoff_date, limit),
        )
        templates = cursor.fetchall()

        if templates:
            results = []
            for row in templates:
                results.append({
                    "template_name": row["template_name"],
                    "avg_read_rate": round(float(row["avg_read_rate"]) * 100, 2) if row["avg_read_rate"] else 0.0,
                    "usage_count": row["usage_count"],
                })

            return jsonify({
                "top_templates": results,
                "period_days": days,
                "source": "precomputed",
            }), 200

        # Fallback: compute from campaign_messages
        cursor.execute(
            """
            SELECT
                ct.template_name,
                ct.id AS template_id,
                COUNT(cm.id) AS total_sent,
                SUM(CASE WHEN cm.status = 'read' THEN 1 ELSE 0 END) AS read_count,
                COUNT(DISTINCT cm.campaign_id) AS usage_count
            FROM campaign_messages cm
            JOIN campaign_templates ct ON ct.id = cm.template_id
            WHERE cm.sent_at >= %s
              AND cm.status NOT IN ('queued', 'skipped')
            GROUP BY ct.id, ct.template_name
            HAVING total_sent > 0
            ORDER BY (read_count / total_sent) DESC
            LIMIT %s
            """,
            (cutoff_date, limit),
        )
        template_rows = cursor.fetchall()

        results = []
        for row in template_rows:
            total = row["total_sent"] or 1
            results.append({
                "template_name": row["template_name"],
                "template_id": row["template_id"],
                "total_sent": row["total_sent"],
                "read_count": row["read_count"] or 0,
                "avg_read_rate": round((row["read_count"] or 0) / total * 100, 2),
                "usage_count": row["usage_count"],
            })

        return jsonify({
            "top_templates": results,
            "period_days": days,
            "source": "computed",
        }), 200

    except Exception as e:
        logger.exception("Error fetching top templates")
        return jsonify({"error": "Failed to fetch top templates."}), 500
    finally:
        cursor.close()
        conn.close()


# ------------------------------------------------------------------
# Customer retention metrics (Requirement 8.3)
# ------------------------------------------------------------------

@analytics_bp.route("/retention", methods=["GET"])
@_require_auth
def get_retention_metrics():
    """
    Get customer retention metrics: reactivation conversion rate,
    churn rate by zone, and renewal rate trends.

    Query Parameters:
        start_date (str): Start of date range (YYYY-MM-DD).
        end_date (str): End of date range (YYYY-MM-DD).

    Returns retention analytics data.
    """
    start_date, end_date = _parse_date_range()

    if not end_date:
        end_date = datetime.now()
    if not start_date:
        start_date = end_date - timedelta(days=30)

    conn = _get_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        # Reactivation conversion: customers who renewed after campaign message
        cursor.execute(
            """
            SELECT COUNT(DISTINCT cm.customer_mobile) AS reactivated
            FROM campaign_messages cm
            JOIN campaigns c ON c.id = cm.campaign_id
            JOIN customers rr ON rr.mobile = cm.customer_mobile
            WHERE c.campaign_type = 'reactivation'
              AND cm.sent_at >= %s
              AND cm.sent_at <= %s
              AND cm.status IN ('delivered', 'read')
              AND rr.status = 'active'
              AND rr.activation_date >= DATE(cm.sent_at)
              AND rr.activation_date <= DATE_ADD(DATE(cm.sent_at), INTERVAL 30 DAY)
            """,
            (start_date, end_date),
        )
        react_row = cursor.fetchone()
        reactivated = react_row["reactivated"] if react_row else 0

        # Total reactivation campaign recipients
        cursor.execute(
            """
            SELECT COUNT(DISTINCT cm.customer_mobile) AS total_targeted
            FROM campaign_messages cm
            JOIN campaigns c ON c.id = cm.campaign_id
            WHERE c.campaign_type = 'reactivation'
              AND cm.sent_at >= %s
              AND cm.sent_at <= %s
              AND cm.status IN ('delivered', 'read')
            """,
            (start_date, end_date),
        )
        target_row = cursor.fetchone()
        total_targeted = target_row["total_targeted"] if target_row else 0

        reactivation_rate = (
            round(reactivated / total_targeted * 100, 2)
            if total_targeted > 0
            else 0.0
        )

        # Churn rate by zone
        cursor.execute(
            """
            SELECT
                zone_name,
                COUNT(*) AS total_customers,
                SUM(CASE WHEN status IN ('expired', 'inactive', 'disconnected')
                    THEN 1 ELSE 0 END) AS churned
            FROM customers
            WHERE zone_name IS NOT NULL AND zone_name != ''
            GROUP BY zone_name
            ORDER BY churned DESC
            """,
        )
        zone_churn = []
        for row in cursor.fetchall():
            total = row["total_customers"] or 1
            zone_churn.append({
                "zone_name": row["zone_name"],
                "total_customers": row["total_customers"],
                "churned": row["churned"] or 0,
                "churn_rate": round((row["churned"] or 0) / total * 100, 2),
            })

        return jsonify({
            "date_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "reactivation": {
                "reactivated_count": reactivated,
                "total_targeted": total_targeted,
                "conversion_rate": reactivation_rate,
            },
            "churn_by_zone": zone_churn,
        }), 200

    except Exception as e:
        logger.exception("Error fetching retention metrics")
        return jsonify({"error": "Failed to fetch retention metrics."}), 500
    finally:
        cursor.close()
        conn.close()


# ------------------------------------------------------------------
# Engagement scores endpoint (Requirement 23.6)
# ------------------------------------------------------------------

@analytics_bp.route("/engagement", methods=["GET"])
@_require_auth
def get_engagement_data():
    """
    Expose engagement data through queryable views for AI modules (Req 23.6).

    Query Parameters:
        min_score (int): Minimum interaction score filter.
        max_score (int): Maximum interaction score filter.
        trend (str): Filter by engagement_trend (increasing, stable, declining).
        limit (int): Number of records (default 100).
        offset (int): Pagination offset (default 0).

    Returns customer engagement records.
    """
    min_score = request.args.get("min_score", type=int)
    max_score = request.args.get("max_score", type=int)
    trend = request.args.get("trend")
    limit = request.args.get("limit", 100, type=int)
    offset = request.args.get("offset", 0, type=int)

    conn = _get_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        where_parts = []
        params = []

        if min_score is not None:
            where_parts.append("interaction_score >= %s")
            params.append(min_score)
        if max_score is not None:
            where_parts.append("interaction_score <= %s")
            params.append(max_score)
        if trend and trend in ("increasing", "stable", "declining"):
            where_parts.append("engagement_trend = %s")
            params.append(trend)

        where_sql = "WHERE " + " AND ".join(where_parts) if where_parts else ""

        # Get total count
        count_sql = f"SELECT COUNT(*) AS total FROM customer_engagement {where_sql}"
        cursor.execute(count_sql, tuple(params))
        total = cursor.fetchone()["total"]

        # Fetch paginated records
        data_sql = f"""
            SELECT customer_mobile, messages_received_count, messages_read_count,
                   response_count, interaction_score, engagement_trend,
                   preferred_time_window, last_interaction_at,
                   avg_response_time_seconds, updated_at
            FROM customer_engagement
            {where_sql}
            ORDER BY interaction_score DESC
            LIMIT %s OFFSET %s
        """
        cursor.execute(data_sql, tuple(params) + (limit, offset))
        records = cursor.fetchall()

        # Serialize datetime fields
        for rec in records:
            if rec.get("last_interaction_at") and hasattr(rec["last_interaction_at"], "isoformat"):
                rec["last_interaction_at"] = rec["last_interaction_at"].isoformat()
            if rec.get("updated_at") and hasattr(rec["updated_at"], "isoformat"):
                rec["updated_at"] = rec["updated_at"].isoformat()

        return jsonify({
            "engagement_records": records,
            "total": total,
            "limit": limit,
            "offset": offset,
        }), 200

    except Exception as e:
        logger.exception("Error fetching engagement data")
        return jsonify({"error": "Failed to fetch engagement data."}), 500
    finally:
        cursor.close()
        conn.close()


# ------------------------------------------------------------------
# Trigger engagement recomputation (Requirement 23.5)
# ------------------------------------------------------------------

@analytics_bp.route("/engagement/recompute", methods=["POST"])
@_require_auth
def trigger_engagement_recompute():
    """
    Trigger batch engagement score recomputation.

    Body (optional):
        campaign_id (int): Recompute for a specific campaign only.

    Returns count of updated records.
    """
    from app import get_mysql_connection
    from services.engagement_scorer import EngagementScorer

    scorer = EngagementScorer(get_mysql_connection)

    data = request.get_json(silent=True) or {}
    campaign_id = data.get("campaign_id")

    try:
        if campaign_id:
            updated = scorer.compute_campaign_engagement(int(campaign_id))
            scorer.store_campaign_performance(int(campaign_id))
        else:
            updated = scorer.compute_all_engagement()

        return jsonify({
            "success": True,
            "updated_count": updated,
            "campaign_id": campaign_id,
        }), 200

    except Exception as e:
        logger.exception("Error during engagement recomputation")
        return jsonify({"error": "Failed to recompute engagement scores."}), 500


# ------------------------------------------------------------------
# Quality Monitor Dashboard (Requirement 18.4)
# ------------------------------------------------------------------

@analytics_bp.route("/quality", methods=["GET"])
@_require_auth
def get_quality_dashboard():
    """
    Get quality monitor dashboard data: current tier (Green/Yellow/Red),
    24h and 7d metrics, and active alerts.

    Returns quality tier, metrics, and alerts for the dashboard.
    """
    from services.quality_monitor import QualityMonitor

    try:
        conn_factory = _get_connection
        monitor = QualityMonitor(conn_factory)
        dashboard = monitor.get_dashboard_data()

        def _metrics_to_dict(m):
            return {
                "period_hours": m.period_hours,
                "period_start": m.period_start.isoformat() if m.period_start else None,
                "period_end": m.period_end.isoformat() if m.period_end else None,
                "blocked_count": m.blocked_count,
                "failure_rate": round(m.failure_rate * 100, 2),
                "opt_out_rate": round(m.opt_out_rate * 100, 2),
                "read_rate": round(m.read_rate * 100, 2),
                "total_sent": m.total_sent,
                "total_failed": m.total_failed,
                "total_opt_outs": m.total_opt_outs,
                "total_read": m.total_read,
            }

        alerts_list = []
        for alert in dashboard.active_alerts:
            alerts_list.append({
                "alert_type": alert.alert_type,
                "severity": alert.severity,
                "title": alert.title,
                "details": alert.details,
                "created_at": alert.created_at.isoformat() if alert.created_at else None,
            })

        return jsonify({
            "current_tier": dashboard.current_tier.value,
            "metrics_24h": _metrics_to_dict(dashboard.metrics_24h),
            "metrics_7d": _metrics_to_dict(dashboard.metrics_7d),
            "active_alerts": alerts_list,
        }), 200

    except Exception as e:
        logger.exception("Error fetching quality dashboard")
        return jsonify({"error": "Failed to fetch quality dashboard data."}), 500


# ------------------------------------------------------------------
# Opt-out trends (Requirement 19.7)
# ------------------------------------------------------------------

@analytics_bp.route("/optout-trends", methods=["GET"])
@_require_auth
def get_optout_trends():
    """
    Get opt-out trends and failure breakdown by category.

    Query Parameters:
        start_date (str): Start of date range (YYYY-MM-DD).
        end_date (str): End of date range (YYYY-MM-DD).

    Returns daily opt-out counts and failure category breakdown.
    """
    start_date, end_date = _parse_date_range()

    if not end_date:
        end_date = datetime.now()
    if not start_date:
        start_date = end_date - timedelta(days=30)

    conn = _get_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        # Daily opt-out trend
        cursor.execute(
            """
            SELECT DATE(created_at) AS date_label, COUNT(*) AS opt_out_count
            FROM suppression_list
            WHERE reason = 'opt_out_keyword'
              AND created_at >= %s
              AND created_at <= %s
            GROUP BY date_label
            ORDER BY date_label ASC
            """,
            (start_date, end_date),
        )
        opt_out_daily = cursor.fetchall()

        # Serialize date objects
        for row in opt_out_daily:
            if row.get("date_label") and hasattr(row["date_label"], "isoformat"):
                row["date_label"] = row["date_label"].isoformat()

        # Failure breakdown by category
        cursor.execute(
            """
            SELECT
                COALESCE(error_category, 'unknown') AS category,
                COUNT(*) AS count
            FROM campaign_messages
            WHERE status IN ('failed', 'permanently_failed')
              AND sent_at >= %s
              AND sent_at <= %s
            GROUP BY category
            ORDER BY count DESC
            """,
            (start_date, end_date),
        )
        failure_breakdown = cursor.fetchall()

        return jsonify({
            "date_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "opt_out_daily": opt_out_daily,
            "failure_breakdown": failure_breakdown,
        }), 200

    except Exception as e:
        logger.exception("Error fetching opt-out trends")
        return jsonify({"error": "Failed to fetch opt-out trends."}), 500
    finally:
        cursor.close()
        conn.close()
