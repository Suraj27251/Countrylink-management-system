# Agent Toggle Patch

Add these changes to your agent repo's `app.py`:

## Step 1: Add this function AFTER `get_db_connection()`

```python
def get_conversation_mode(phone: str) -> str:
    """Check human_takeover flag. Returns 'human' or 'ai'."""
    try:
        # Normalize to 10 digits (same as management system)
        digits = re.sub(r'\D', '', str(phone))
        if len(digits) >= 12 and digits.startswith('91'):
            digits = digits[2:]
        normalized = digits[-10:] if len(digits) >= 10 else digits

        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT human_takeover FROM whatsapp_conversations WHERE phone = %s OR phone = %s LIMIT 1",
                (phone, normalized)
            )
            row = cursor.fetchone()
        conn.close()
        if row and int(row.get('human_takeover', 0)) == 1:
            return 'human'
        return 'ai'
    except Exception as e:
        logging.error(f"get_conversation_mode error: {e}")
        return 'ai'
```

## Step 2: In `handle_webhook()`, add 4 lines after `from_number = message.get("from")`

Find this line:
```python
                    from_number = message.get("from")
                    msg_type    = message.get("type")
```

Add RIGHT AFTER `from_number = message.get("from")`:
```python
                    # ── HUMAN/AI TOGGLE CHECK ──
                    if get_conversation_mode(from_number) == 'human':
                        logging.info(f"Human mode for {from_number}, skipping AI")
                        continue
```

So it becomes:
```python
                    from_number = message.get("from")

                    # ── HUMAN/AI TOGGLE CHECK ──
                    if get_conversation_mode(from_number) == 'human':
                        logging.info(f"Human mode for {from_number}, skipping AI")
                        continue

                    msg_type    = message.get("type")
```

That's it. 2 changes, the agent respects the toggle.
