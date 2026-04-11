ALTER TABLE invoices
    ADD COLUMN zoho_contact_id VARCHAR(100) NULL AFTER customer_name,
    ADD INDEX idx_invoices_zoho_contact_id (zoho_contact_id);
