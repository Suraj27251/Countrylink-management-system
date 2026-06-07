# WhatsApp Template Sending Fixes - Complete ✅

## Issues Fixed

### 1. Missing Customer Fields in Recipient Query (Campaign Flow)
**File:** `blueprints/campaign_bp.py` → `_enqueue_campaign_recipients()`
- **Root Cause:** The SQL query `SELECT mobile, customer_name FROM customers` only fetched 2 fields. Template `placeholder_mappings` often reference other fields like `plan_name`, `total`, `due_date`. These resolved to empty strings, causing Meta to reject the template.
- **Fix:** Now dynamically reads the template's `placeholder_mappings` JSON, discovers all referenced field names, and builds the SELECT with ALL needed columns.

### 2. Template Language Code Not Used During Dispatch
**Files:** `services/channel.py`, `services/sending_queue.py`
- **Root Cause:** When dispatching campaign messages via batch worker, `_dispatch_message()` was NOT fetching the language code from the DB. The WhatsAppDispatcher always used `"en"` (default), even when templates were registered with `en_US`.
- **Fix:** 
  - `_dispatch_message()` now queries `template_language` from `campaign_templates` and passes it to `send_template()`.
  - `WhatsAppDispatcher.send_template()` now accepts an optional `language` parameter. If provided, it uses that language code instead of the default.
  - `MessageDispatcher` ABC interface also updated to accept `language`.

### 3. Missing Language in Test Send
**File:** `blueprints/campaign_bp.py` → `test_send()`
- **Root Cause:** Test sends via the campaign test-send endpoint also didn't pass language to the dispatcher.
- **Fix:** Now fetches `template_language` from the template record and passes it.

## Files Modified
- `blueprints/campaign_bp.py` - Recipient field discovery + test send language
- `services/channel.py` - Language parameter in send_template
- `services/sending_queue.py` - Language fetching + passing during dispatch