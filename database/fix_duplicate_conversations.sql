-- ============================================================
-- Fix Duplicate WhatsApp Conversations
-- ============================================================
-- Problem: Some phone numbers stored as '918149912379' (with country code)
-- and others as '8149912379' (without). This causes duplicate conversations.
--
-- Solution: Merge duplicates by keeping the conversation with the most recent
-- activity, reassigning messages from the duplicate, then deleting the duplicate.
-- Finally, normalize all phone numbers to 10-digit format.
--
-- RUN THIS MANUALLY after backing up the database!
-- ============================================================

-- Step 1: Identify duplicates (phone numbers that exist in both 91XXXXXXXXXX and XXXXXXXXXX format)
SELECT 
    c1.id AS keep_id, c1.phone AS keep_phone,
    c2.id AS duplicate_id, c2.phone AS duplicate_phone,
    c1.unread_count AS keep_unread, c2.unread_count AS dup_unread
FROM whatsapp_conversations c1
JOIN whatsapp_conversations c2 
    ON RIGHT(c1.phone, 10) = RIGHT(c2.phone, 10)
    AND c1.id != c2.id
    AND c1.updated_at >= c2.updated_at
WHERE LENGTH(c1.phone) <= LENGTH(c2.phone);

-- Step 2: Reassign messages from duplicate conversations to the primary one
-- (Run this for each duplicate pair found in Step 1)
-- Replace {KEEP_ID} and {DUPLICATE_ID} with actual values from Step 1

-- UPDATE whatsapp_messages 
-- SET conversation_id = {KEEP_ID}
-- WHERE conversation_id = {DUPLICATE_ID};

-- Step 3: Merge unread counts
-- UPDATE whatsapp_conversations 
-- SET unread_count = unread_count + (SELECT unread_count FROM whatsapp_conversations WHERE id = {DUPLICATE_ID})
-- WHERE id = {KEEP_ID};

-- Step 4: Delete the duplicate conversation
-- DELETE FROM whatsapp_conversations WHERE id = {DUPLICATE_ID};

-- Step 5: Normalize all phone numbers to 10-digit format (strip 91 prefix)
UPDATE whatsapp_conversations 
SET phone = SUBSTRING(phone, 3)
WHERE LENGTH(phone) = 12 AND phone LIKE '91%';

UPDATE whatsapp_messages 
SET phone = SUBSTRING(phone, 3)
WHERE LENGTH(phone) = 12 AND phone LIKE '91%';

-- Step 6: Add unique constraint to prevent future duplicates
-- (Only run after all duplicates are resolved)
-- CREATE UNIQUE INDEX uniq_whatsapp_conversations_phone ON whatsapp_conversations(phone);
