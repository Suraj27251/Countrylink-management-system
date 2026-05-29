"""
Fix duplicate WhatsApp conversations caused by inconsistent phone normalization.

Merges conversations where the same customer appears with both '91XXXXXXXXXX' 
and 'XXXXXXXXXX' phone formats. Keeps the most recently active conversation,
reassigns messages from the duplicate, and normalizes all phone numbers.

Usage:
    python database/fix_duplicates.py

Set environment variables for MySQL connection:
    MYSQL_DB_HOST, MYSQL_DB_NAME, MYSQL_DB_USER, MYSQL_DB_PASSWORD
"""

import os
import sys

try:
    import mysql.connector
except ImportError:
    print("ERROR: mysql-connector-python not installed. Run: pip install mysql-connector-python")
    sys.exit(1)


def get_connection():
    return mysql.connector.connect(
        host=os.environ.get("MYSQL_DB_HOST", "localhost"),
        database=os.environ.get("MYSQL_DB_NAME", "countrylinks_user_database"),
        user=os.environ.get("MYSQL_DB_USER", "countrylinks_Suraj27251"),
        password=os.environ.get("MYSQL_DB_PASSWORD", ""),
        charset='utf8mb4',
        collation='utf8mb4_unicode_ci',
        use_unicode=True,
    )


def normalize_phone(phone):
    """Normalize to 10-digit Indian mobile number."""
    if not phone:
        return ''
    digits = ''.join(ch for ch in str(phone).strip() if ch.isdigit())
    if len(digits) == 12 and digits.startswith('91'):
        digits = digits[2:]
    elif len(digits) == 13 and digits.startswith('091'):
        digits = digits[3:]
    elif len(digits) > 10 and digits.startswith('91'):
        candidate = digits[2:]
        if len(candidate) == 10:
            digits = candidate
    return digits


def find_duplicates(cursor):
    """Find conversations that are duplicates based on last 10 digits of phone."""
    cursor.execute("""
        SELECT id, phone, customer_name, unread_count, updated_at,
               (SELECT COUNT(*) FROM whatsapp_messages WHERE conversation_id = c.id) AS msg_count
        FROM whatsapp_conversations c
        ORDER BY phone
    """)
    all_convos = cursor.fetchall()

    # Group by normalized phone
    groups = {}
    for convo in all_convos:
        normalized = normalize_phone(convo['phone'])
        if normalized not in groups:
            groups[normalized] = []
        groups[normalized].append(convo)

    # Find groups with more than one conversation
    duplicates = {k: v for k, v in groups.items() if len(v) > 1}
    return duplicates


def merge_duplicates(conn, cursor, duplicates, dry_run=True):
    """Merge duplicate conversations. Keep the one with most recent activity."""
    merged_count = 0

    for normalized_phone, convos in duplicates.items():
        # Sort by updated_at DESC — keep the most recent
        convos.sort(key=lambda c: c['updated_at'] or '', reverse=True)
        keep = convos[0]
        to_delete = convos[1:]

        print(f"\n{'[DRY RUN] ' if dry_run else ''}Phone: {normalized_phone}")
        print(f"  KEEP: id={keep['id']}, phone='{keep['phone']}', "
              f"name='{keep['customer_name']}', msgs={keep['msg_count']}, "
              f"unread={keep['unread_count']}")

        for dup in to_delete:
            print(f"  DELETE: id={dup['id']}, phone='{dup['phone']}', "
                  f"name='{dup['customer_name']}', msgs={dup['msg_count']}, "
                  f"unread={dup['unread_count']}")

            if not dry_run:
                # Reassign messages from duplicate to keep
                cursor.execute(
                    "UPDATE whatsapp_messages SET conversation_id = %s WHERE conversation_id = %s",
                    (keep['id'], dup['id'])
                )
                reassigned = cursor.rowcount
                print(f"    → Reassigned {reassigned} messages")

                # Add unread count from duplicate
                cursor.execute(
                    "UPDATE whatsapp_conversations SET unread_count = unread_count + %s WHERE id = %s",
                    (dup['unread_count'], keep['id'])
                )

                # Delete the duplicate conversation
                cursor.execute(
                    "DELETE FROM whatsapp_conversations WHERE id = %s",
                    (dup['id'],)
                )
                print(f"    → Deleted conversation id={dup['id']}")

        # Normalize the kept conversation's phone number
        if keep['phone'] != normalized_phone:
            if not dry_run:
                cursor.execute(
                    "UPDATE whatsapp_conversations SET phone = %s WHERE id = %s",
                    (normalized_phone, keep['id'])
                )
            print(f"  NORMALIZE: '{keep['phone']}' → '{normalized_phone}'")

        merged_count += 1

    return merged_count


def normalize_all_phones(conn, cursor, dry_run=True):
    """Normalize all phone numbers in both tables to 10-digit format."""
    # Fix conversations table
    cursor.execute("""
        SELECT id, phone FROM whatsapp_conversations
        WHERE LENGTH(phone) = 12 AND phone LIKE '91%'
    """)
    to_fix_convos = cursor.fetchall()

    if to_fix_convos:
        print(f"\n{'[DRY RUN] ' if dry_run else ''}Normalizing {len(to_fix_convos)} conversation phone numbers...")
        if not dry_run:
            cursor.execute("""
                UPDATE whatsapp_conversations 
                SET phone = SUBSTRING(phone, 3)
                WHERE LENGTH(phone) = 12 AND phone LIKE '91%'
            """)
            print(f"  → Updated {cursor.rowcount} rows in whatsapp_conversations")

    # Fix messages table
    cursor.execute("""
        SELECT COUNT(*) AS cnt FROM whatsapp_messages
        WHERE LENGTH(phone) = 12 AND phone LIKE '91%'
    """)
    msg_count = cursor.fetchone()['cnt']

    if msg_count:
        print(f"{'[DRY RUN] ' if dry_run else ''}Normalizing {msg_count} message phone numbers...")
        if not dry_run:
            cursor.execute("""
                UPDATE whatsapp_messages 
                SET phone = SUBSTRING(phone, 3)
                WHERE LENGTH(phone) = 12 AND phone LIKE '91%'
            """)
            print(f"  → Updated {cursor.rowcount} rows in whatsapp_messages")


def main():
    dry_run = '--execute' not in sys.argv

    if dry_run:
        print("=" * 60)
        print("DRY RUN MODE — No changes will be made.")
        print("Run with --execute to apply changes.")
        print("=" * 60)
    else:
        print("=" * 60)
        print("⚠️  EXECUTING — Changes will be committed to the database!")
        print("=" * 60)
        confirm = input("Type 'yes' to confirm: ")
        if confirm.lower() != 'yes':
            print("Aborted.")
            sys.exit(0)

    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    try:
        # Step 1: Find duplicates
        print("\n--- Finding duplicate conversations ---")
        duplicates = find_duplicates(cursor)

        if not duplicates:
            print("✅ No duplicate conversations found!")
        else:
            print(f"Found {len(duplicates)} phone numbers with duplicate conversations.")

            # Step 2: Merge duplicates
            print("\n--- Merging duplicates ---")
            if not dry_run:
                conn.start_transaction()
            merged = merge_duplicates(conn, cursor, duplicates, dry_run=dry_run)

        # Step 3: Normalize remaining phone numbers
        print("\n--- Normalizing phone numbers ---")
        normalize_all_phones(conn, cursor, dry_run=dry_run)

        if not dry_run:
            conn.commit()
            print("\n✅ All changes committed successfully!")

            # Step 4: Try to add unique index
            print("\n--- Adding unique index ---")
            try:
                cursor.execute("""
                    SELECT COUNT(*) AS cnt FROM (
                        SELECT phone FROM whatsapp_conversations 
                        GROUP BY phone HAVING COUNT(*) > 1
                    ) AS dups
                """)
                remaining_dups = cursor.fetchone()['cnt']
                if remaining_dups == 0:
                    cursor.execute("SHOW INDEX FROM whatsapp_conversations WHERE Key_name = 'uniq_whatsapp_conversations_phone'")
                    if not cursor.fetchone():
                        cursor.execute("CREATE UNIQUE INDEX uniq_whatsapp_conversations_phone ON whatsapp_conversations(phone)")
                        conn.commit()
                        print("✅ Unique index created on whatsapp_conversations.phone")
                    else:
                        print("✅ Unique index already exists.")
                else:
                    print(f"⚠️  Still {remaining_dups} duplicates remaining. Skipping unique index.")
            except Exception as e:
                print(f"⚠️  Could not create unique index: {e}")
        else:
            print("\n--- DRY RUN COMPLETE ---")
            print("Run with --execute to apply these changes.")

    except Exception as e:
        if not dry_run:
            conn.rollback()
        print(f"\n❌ Error: {e}")
        raise
    finally:
        cursor.close()
        conn.close()


if __name__ == '__main__':
    main()
