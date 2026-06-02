"""
Tests for ensure_multi_tenant_columns() migration function.

Uses an in-memory SQLite database to simulate MySQL behavior.
Tests verify idempotency and correctness of column/index additions.
"""

import sys
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class FakeCursor:
    """Wraps a sqlite3 cursor to behave like mysql.connector cursor."""

    def __init__(self, conn):
        self._conn = conn
        self._cursor = conn.cursor()

    def execute(self, sql, params=None):
        # Translate MySQL information_schema queries to SQLite pragmas
        if 'information_schema.COLUMNS' in sql:
            # Params are (table_name, column_name)
            table_name = params[0] if params else ''
            column_name = params[1] if params and len(params) > 1 else ''
            # Use pragma to check column existence
            self._cursor.execute(f"PRAGMA table_info(`{table_name}`)")
            cols = [row[1] for row in self._cursor.fetchall()]
            if column_name in cols:
                self._result = [(1,)]
            else:
                self._result = [(0,)]
            return

        if 'information_schema.STATISTICS' in sql:
            # Params are (table_name, index_name)
            table_name = params[0] if params else ''
            index_name = params[1] if params and len(params) > 1 else ''
            # Check if index exists using sqlite_master
            self._cursor.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name=? AND tbl_name=?",
                (index_name, table_name)
            )
            self._result = [self._cursor.fetchone()]
            return

        if 'information_schema.TABLES' in sql:
            table_name = params[0] if params else ''
            self._cursor.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,)
            )
            self._result = [self._cursor.fetchone()]
            return

        # Handle ALTER TABLE ADD COLUMN and CREATE INDEX
        # These don't use params in our implementation
        if params:
            self._cursor.execute(sql, params)
        else:
            # For ALTER TABLE and CREATE INDEX, catch duplicate errors to
            # simulate the MySQL behavior where our code uses try/except
            try:
                self._cursor.execute(sql)
            except Exception as e:
                # Re-raise so the calling code's except block can handle it
                raise
        self._result = None

    def fetchone(self):
        if self._result is not None:
            return self._result[0] if self._result else None
        return self._cursor.fetchone()

    def fetchall(self):
        if self._result is not None:
            return self._result
        return self._cursor.fetchall()

    def close(self):
        self._cursor.close()


class FakeConnection:
    """Wraps a sqlite3 connection to behave like mysql.connector connection."""

    def __init__(self, conn):
        self._conn = conn

    def cursor(self):
        return FakeCursor(self._conn)

    def commit(self):
        self._conn.commit()

    def rollback(self):
        self._conn.rollback()

    def is_connected(self):
        return True

    def close(self):
        pass  # Don't close the shared sqlite connection


def _create_test_db():
    """Create an in-memory SQLite database with CRM tables for testing."""
    conn = sqlite3.connect(':memory:')

    # Tables that have organization_id + branch_id (per design CREATE TABLE)
    conn.execute("""
        CREATE TABLE campaigns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            organization_id INTEGER NOT NULL DEFAULT 1,
            branch_id INTEGER NOT NULL DEFAULT 1,
            name TEXT NOT NULL,
            status TEXT DEFAULT 'draft'
        )
    """)
    conn.execute("""
        CREATE TABLE customer_tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            organization_id INTEGER NOT NULL DEFAULT 1,
            branch_id INTEGER NOT NULL DEFAULT 1,
            customer_mobile TEXT NOT NULL,
            tag_name TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE customer_notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            organization_id INTEGER NOT NULL DEFAULT 1,
            branch_id INTEGER NOT NULL DEFAULT 1,
            customer_mobile TEXT NOT NULL,
            note_text TEXT NOT NULL
        )
    """)

    # Tables with organization_id only
    conn.execute("""
        CREATE TABLE audience_segments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            organization_id INTEGER NOT NULL DEFAULT 1,
            name TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE campaign_templates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            organization_id INTEGER NOT NULL DEFAULT 1,
            template_name TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE campaign_analytics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            organization_id INTEGER NOT NULL DEFAULT 1,
            campaign_id INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE media_assets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            organization_id INTEGER NOT NULL DEFAULT 1,
            filename TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE suppression_list (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            organization_id INTEGER NOT NULL DEFAULT 1,
            customer_mobile TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE system_notifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            organization_id INTEGER NOT NULL DEFAULT 1,
            alert_type TEXT NOT NULL
        )
    """)

    # Table with organization_id + tenant_id
    conn.execute("""
        CREATE TABLE automation_rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            organization_id INTEGER NOT NULL DEFAULT 1,
            tenant_id INTEGER NOT NULL DEFAULT 1,
            name TEXT NOT NULL
        )
    """)

    # Tables without any multi-tenant columns
    conn.execute("""
        CREATE TABLE campaign_ab_variants (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            campaign_id INTEGER NOT NULL,
            template_id INTEGER NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE campaign_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            campaign_id INTEGER NOT NULL,
            customer_mobile TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE customer_activity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_mobile TEXT NOT NULL,
            activity_type TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE customer_engagement (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_mobile TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE message_cooldowns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_mobile TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE quality_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            period_start TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE error_classifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            error_code INTEGER NOT NULL
        )
    """)

    conn.commit()
    return conn


def _get_columns(conn, table_name):
    """Get list of column names for a table."""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info(`{table_name}`)")
    return [row[1] for row in cursor.fetchall()]


def _get_indexes(conn, table_name):
    """Get list of index names for a table."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name=?",
        (table_name,)
    )
    return [row[0] for row in cursor.fetchall()]


class TestEnsureMultiTenantColumns:
    """Tests for the ensure_multi_tenant_columns function."""

    def setup_method(self):
        """Create fresh test database for each test."""
        self.conn = _create_test_db()

        def get_fake_connection():
            return FakeConnection(self.conn)

        self.get_connection = get_fake_connection

    def teardown_method(self):
        """Close database connection."""
        self.conn.close()

    def test_adds_tenant_id_to_tables_with_org_and_branch(self):
        """Tables with org_id+branch_id should get tenant_id added."""
        from migrations.enterprise_crm_schema import ensure_multi_tenant_columns

        ensure_multi_tenant_columns(self.get_connection)

        for table in ['campaigns', 'customer_tags', 'customer_notes']:
            cols = _get_columns(self.conn, table)
            assert 'tenant_id' in cols, f"tenant_id missing from {table}"

    def test_adds_branch_and_tenant_to_tables_with_org_only(self):
        """Tables with only org_id should get branch_id and tenant_id added."""
        from migrations.enterprise_crm_schema import ensure_multi_tenant_columns

        ensure_multi_tenant_columns(self.get_connection)

        for table in ['audience_segments', 'campaign_templates', 'campaign_analytics',
                      'media_assets', 'suppression_list', 'system_notifications']:
            cols = _get_columns(self.conn, table)
            assert 'branch_id' in cols, f"branch_id missing from {table}"
            assert 'tenant_id' in cols, f"tenant_id missing from {table}"

    def test_adds_branch_to_tables_with_org_and_tenant(self):
        """automation_rules has org+tenant, should get branch_id added."""
        from migrations.enterprise_crm_schema import ensure_multi_tenant_columns

        ensure_multi_tenant_columns(self.get_connection)

        cols = _get_columns(self.conn, 'automation_rules')
        assert 'branch_id' in cols, "branch_id missing from automation_rules"

    def test_adds_all_three_to_tables_without_multi_tenant(self):
        """Tables without any multi-tenant columns should get all three."""
        from migrations.enterprise_crm_schema import ensure_multi_tenant_columns

        ensure_multi_tenant_columns(self.get_connection)

        for table in ['campaign_ab_variants', 'campaign_messages', 'customer_activity',
                      'customer_engagement', 'message_cooldowns', 'quality_metrics',
                      'error_classifications']:
            cols = _get_columns(self.conn, table)
            assert 'organization_id' in cols, f"organization_id missing from {table}"
            assert 'branch_id' in cols, f"branch_id missing from {table}"
            assert 'tenant_id' in cols, f"tenant_id missing from {table}"

    def test_creates_composite_tenant_indexes(self):
        """Composite indexes should be created on all tables."""
        from migrations.enterprise_crm_schema import ensure_multi_tenant_columns

        ensure_multi_tenant_columns(self.get_connection)

        all_tables = [
            'campaigns', 'customer_tags', 'customer_notes',
            'audience_segments', 'campaign_templates', 'campaign_analytics',
            'media_assets', 'suppression_list', 'system_notifications',
            'automation_rules',
            'campaign_ab_variants', 'campaign_messages', 'customer_activity',
            'customer_engagement', 'message_cooldowns', 'quality_metrics',
            'error_classifications',
        ]
        for table in all_tables:
            indexes = _get_indexes(self.conn, table)
            expected_index = f"idx_{table}_tenant"
            assert expected_index in indexes, f"Index {expected_index} missing on {table}"

    def test_idempotent_multiple_calls(self):
        """Calling ensure_multi_tenant_columns multiple times should not error."""
        from migrations.enterprise_crm_schema import ensure_multi_tenant_columns

        # Call it three times - should work without errors each time
        result1 = ensure_multi_tenant_columns(self.get_connection)
        result2 = ensure_multi_tenant_columns(self.get_connection)
        result3 = ensure_multi_tenant_columns(self.get_connection)

        # First call should add columns/indexes
        assert result1['columns_added'] > 0 or result1['indexes_created'] > 0

        # Subsequent calls should add nothing (idempotent)
        assert result2['columns_added'] == 0
        assert result2['indexes_created'] == 0
        assert result3['columns_added'] == 0
        assert result3['indexes_created'] == 0

    def test_skips_nonexistent_tables(self):
        """Should not error if some tables don't exist yet."""
        from migrations.enterprise_crm_schema import ensure_multi_tenant_columns

        # Drop one table to simulate it not existing yet
        self.conn.execute("DROP TABLE IF EXISTS campaign_ab_variants")
        self.conn.commit()

        # Should still work for all other tables
        result = ensure_multi_tenant_columns(self.get_connection)
        assert result['columns_added'] > 0

    def test_all_tables_have_all_three_columns_after_migration(self):
        """After migration, every CRM table should have all three multi-tenant columns."""
        from migrations.enterprise_crm_schema import ensure_multi_tenant_columns

        ensure_multi_tenant_columns(self.get_connection)

        all_tables = [
            'campaigns', 'customer_tags', 'customer_notes',
            'audience_segments', 'campaign_templates', 'campaign_analytics',
            'media_assets', 'suppression_list', 'system_notifications',
            'automation_rules',
            'campaign_ab_variants', 'campaign_messages', 'customer_activity',
            'customer_engagement', 'message_cooldowns', 'quality_metrics',
            'error_classifications',
        ]
        for table in all_tables:
            cols = _get_columns(self.conn, table)
            assert 'organization_id' in cols, f"organization_id missing from {table}"
            assert 'branch_id' in cols, f"branch_id missing from {table}"
            assert 'tenant_id' in cols, f"tenant_id missing from {table}"
