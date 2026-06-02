"""
Segmentation Engine — builds parameterized SQL queries from filter criteria.

Provides SegmentationService class with pure business logic.
Accepts a connection factory (get_connection callable) for testability.
"""

import json
import logging
from datetime import date, datetime

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported exact-match and derived filter fields
# ---------------------------------------------------------------------------
EXACT_MATCH_FIELDS = {
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
}

BOOLEAN_FIELDS = {
    "kyc_approved",
}

# Range filter fields: provided as {"min": ..., "max": ...}
RANGE_FIELDS = {
    "days_remaining",
}

# Derived range filters computed from dates
DERIVED_RANGE_FIELDS = {
    "days_since_last_recharge",  # (current_date - activation_date).days
    "days_inactive",             # (current_date - expiry_date).days for expired records
}

# Tag-based filter
TAG_FIELD = "tags"


class SegmentationService:
    """
    Segmentation service for building dynamic audience queries.

    Parameters
    ----------
    get_connection : callable
        A zero-argument function that returns a MySQL connection object.
    """

    def __init__(self, get_connection):
        self._get_conn = get_connection

    # ------------------------------------------------------------------
    # Core query building
    # ------------------------------------------------------------------

    def build_query(self, filters: dict, *, reference_date: date = None) -> tuple:
        """
        Build a parameterized SQL WHERE clause from filter criteria.

        Parameters
        ----------
        filters : dict
            Filter criteria. Keys correspond to field names; values are the
            filter values (exact match, range dict, or list for tags).
        reference_date : date, optional
            Reference date for derived filters. Defaults to today.

        Returns
        -------
        tuple of (where_clause: str, params: list)
            The WHERE clause (without the WHERE keyword) and parameter list.
            Returns ("1=1", []) if no filters are provided.
        """
        if reference_date is None:
            reference_date = date.today()

        clauses = []
        params = []

        if not filters:
            return "1=1", []

        for field, value in filters.items():
            if value is None:
                continue

            # Exact match fields
            if field in EXACT_MATCH_FIELDS:
                clauses.append(f"r.{field} = %s")
                params.append(value)

            # Boolean fields
            elif field in BOOLEAN_FIELDS:
                clauses.append(f"r.{field} = %s")
                params.append(1 if value else 0)

            # Range fields (days_remaining)
            elif field in RANGE_FIELDS:
                if isinstance(value, dict):
                    if "min" in value and value["min"] is not None:
                        clauses.append(f"r.{field} >= %s")
                        params.append(value["min"])
                    if "max" in value and value["max"] is not None:
                        clauses.append(f"r.{field} <= %s")
                        params.append(value["max"])
                else:
                    # Treat scalar as exact match
                    clauses.append(f"r.{field} = %s")
                    params.append(value)

            # Derived: days since last recharge = (ref_date - activation_date)
            elif field == "days_since_last_recharge":
                if isinstance(value, dict):
                    if "min" in value and value["min"] is not None:
                        clauses.append("DATEDIFF(%s, r.activation_date) >= %s")
                        params.append(reference_date)
                        params.append(value["min"])
                    if "max" in value and value["max"] is not None:
                        clauses.append("DATEDIFF(%s, r.activation_date) <= %s")
                        params.append(reference_date)
                        params.append(value["max"])
                else:
                    clauses.append("DATEDIFF(%s, r.activation_date) = %s")
                    params.append(reference_date)
                    params.append(value)

            # Derived: days inactive = (ref_date - expiry_date) for expired
            elif field == "days_inactive":
                if isinstance(value, dict):
                    if "min" in value and value["min"] is not None:
                        clauses.append(
                            "DATEDIFF(%s, r.expiry_date) >= %s"
                        )
                        params.append(reference_date)
                        params.append(value["min"])
                    if "max" in value and value["max"] is not None:
                        clauses.append(
                            "DATEDIFF(%s, r.expiry_date) <= %s"
                        )
                        params.append(reference_date)
                        params.append(value["max"])
                else:
                    clauses.append("DATEDIFF(%s, r.expiry_date) = %s")
                    params.append(reference_date)
                    params.append(value)
                # Only apply to expired records
                clauses.append("r.expiry_date < %s")
                params.append(reference_date)

            # Tags filter (requires JOIN to customer_tags)
            elif field == TAG_FIELD:
                if isinstance(value, dict):
                    tag_list = value.get("values", [])
                    mode = value.get("mode", "ANY").upper()
                else:
                    tag_list = value if isinstance(value, list) else [value]
                    mode = "ANY"

                if tag_list:
                    placeholders = ", ".join(["%s"] * len(tag_list))
                    if mode == "ALL":
                        # Must have ALL specified tags
                        clauses.append(
                            f"(SELECT COUNT(DISTINCT ct.tag_name) FROM customer_tags ct "
                            f"WHERE ct.customer_mobile = r.mobile "
                            f"AND ct.tag_name IN ({placeholders})) = %s"
                        )
                        params.extend(tag_list)
                        params.append(len(tag_list))
                    else:
                        # ANY of the specified tags
                        clauses.append(
                            f"EXISTS (SELECT 1 FROM customer_tags ct "
                            f"WHERE ct.customer_mobile = r.mobile "
                            f"AND ct.tag_name IN ({placeholders}))"
                        )
                        params.extend(tag_list)

        if not clauses:
            return "1=1", []

        where_clause = " AND ".join(clauses)
        return where_clause, params

    # ------------------------------------------------------------------
    # Count estimation
    # ------------------------------------------------------------------

    def estimate_count(self, filters: dict, *, reference_date: date = None) -> int:
        """
        Return the count of customers matching the given filters.

        Uses a COUNT(*) query for real-time estimation.
        """
        where_clause, params = self.build_query(filters, reference_date=reference_date)
        sql = f"SELECT COUNT(*) as cnt FROM renewal_records r WHERE {where_clause}"

        conn = self._get_conn()
        cursor = None
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(sql, params)
            row = cursor.fetchone()
            return row["cnt"] if row else 0
        finally:
            if cursor:
                cursor.close()
            conn.close()

    # ------------------------------------------------------------------
    # Evaluate segment (paginated)
    # ------------------------------------------------------------------

    def evaluate_segment(self, filters: dict, *, page: int = 1, per_page: int = 50,
                         reference_date: date = None) -> dict:
        """
        Evaluate filter criteria and return paginated customer results.

        Parameters
        ----------
        filters : dict
            Filter criteria.
        page : int
            Page number (1-based).
        per_page : int
            Records per page (default 50).
        reference_date : date, optional
            Reference date for derived filters.

        Returns
        -------
        dict with keys: customers, total, page, per_page, total_pages, warning
        """
        where_clause, params = self.build_query(filters, reference_date=reference_date)

        conn = self._get_conn()
        cursor = None
        try:
            cursor = conn.cursor(dictionary=True)

            # Get total count
            count_sql = f"SELECT COUNT(*) as cnt FROM renewal_records r WHERE {where_clause}"
            cursor.execute(count_sql, params)
            total = cursor.fetchone()["cnt"]

            # Zero-result prevention check
            warning = None
            if total == 0:
                warning = "No customers match the specified filter criteria. Please adjust your filters."

            # Paginated fetch
            offset = (page - 1) * per_page
            data_sql = (
                f"SELECT r.* FROM renewal_records r WHERE {where_clause} "
                f"ORDER BY r.mobile ASC LIMIT %s OFFSET %s"
            )
            cursor.execute(data_sql, params + [per_page, offset])
            customers = cursor.fetchall()

            total_pages = (total + per_page - 1) // per_page if per_page > 0 else 0

            result = {
                "customers": customers,
                "total": total,
                "page": page,
                "per_page": per_page,
                "total_pages": total_pages,
            }
            if warning:
                result["warning"] = warning

            return result
        finally:
            if cursor:
                cursor.close()
            conn.close()

    # ------------------------------------------------------------------
    # Save & Load segments
    # ------------------------------------------------------------------

    def save_segment(self, name: str, filters: dict, *, description: str = "",
                     created_by: str = "system", organization_id: int = 1) -> dict:
        """
        Save a segment definition (name + filter criteria) to the database.

        Returns the saved segment record.
        """
        # Estimate count at save time
        estimated_count = self.estimate_count(filters)

        conn = self._get_conn()
        cursor = None
        try:
            cursor = conn.cursor(dictionary=True)
            conn.start_transaction()

            sql = """
                INSERT INTO audience_segments
                    (organization_id, name, description, filter_criteria, estimated_count, created_by)
                VALUES
                    (%s, %s, %s, %s, %s, %s)
            """
            params = (
                organization_id,
                name,
                description,
                json.dumps(filters),
                estimated_count,
                created_by,
            )
            cursor.execute(sql, params)
            segment_id = cursor.lastrowid

            conn.commit()
            return self.load_segment(segment_id)
        except Exception:
            conn.rollback()
            raise
        finally:
            if cursor:
                cursor.close()
            conn.close()

    def load_segment(self, segment_id: int) -> dict:
        """
        Load a segment definition by ID.

        Returns dict with id, name, description, filter_criteria (parsed), estimated_count, etc.
        """
        conn = self._get_conn()
        cursor = None
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM audience_segments WHERE id = %s", (segment_id,))
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Segment {segment_id} not found")

            # Parse filter_criteria from JSON string to dict
            if isinstance(row.get("filter_criteria"), str):
                row["filter_criteria"] = json.loads(row["filter_criteria"])

            return row
        finally:
            if cursor:
                cursor.close()
            conn.close()

    def list_segments(self, *, page: int = 1, per_page: int = 20,
                      organization_id: int = 1) -> dict:
        """
        List saved segments with pagination.
        """
        conn = self._get_conn()
        cursor = None
        try:
            cursor = conn.cursor(dictionary=True)

            # Count total
            cursor.execute(
                "SELECT COUNT(*) as cnt FROM audience_segments WHERE organization_id = %s",
                (organization_id,),
            )
            total = cursor.fetchone()["cnt"]

            # Paginated fetch
            offset = (page - 1) * per_page
            cursor.execute(
                """
                SELECT * FROM audience_segments
                WHERE organization_id = %s
                ORDER BY updated_at DESC
                LIMIT %s OFFSET %s
                """,
                (organization_id, per_page, offset),
            )
            segments = cursor.fetchall()

            # Parse JSON filter_criteria for each segment
            for seg in segments:
                if isinstance(seg.get("filter_criteria"), str):
                    seg["filter_criteria"] = json.loads(seg["filter_criteria"])

            return {
                "segments": segments,
                "total": total,
                "page": page,
                "per_page": per_page,
                "total_pages": (total + per_page - 1) // per_page if per_page > 0 else 0,
            }
        finally:
            if cursor:
                cursor.close()
            conn.close()
