# Implementation Plan: Enterprise WhatsApp CRM

## Overview

This implementation plan builds the enterprise WhatsApp CRM platform on top of the existing Flask application. The work is organized into incremental steps: database schema first, then core services (segmentation, queue, campaign), followed by supporting modules (CRM, analytics, media, notifications), and finally frontend integration. Each task builds on previous steps, with property-based tests placed close to the implementations they validate.

**Backend:** Python (Flask Blueprints, MySQL)
**Frontend:** JavaScript (existing Jinja2 + vanilla JS workspace panel system)
**Testing:** Jest + fast-check (JS), Python unittest + Hypothesis (backend)

## Tasks

- [x] 1. Database schema and migration setup
  - [x] 1.1 Create database migration module with auto-create tables
    - Create `migrations/enterprise_crm_schema.py` with all CREATE TABLE IF NOT EXISTS statements from the design (campaigns, campaign_ab_variants, audience_segments, campaign_messages, campaign_templates, customer_tags, customer_notes, customer_activity, campaign_analytics, automation_rules, media_assets, suppression_list, customer_engagement, system_notifications, error_classifications, message_cooldowns, quality_metrics)
    - Add ALTER TABLE statements for existing tables (whatsapp_campaign_logs, operator_actions, renewal_records) using ADD COLUMN IF NOT EXISTS
    - Create `ensure_crm_tables()` function callable at app startup
    - Register startup hook in `app.py` to call migration on boot
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

  - [x] 1.2 Seed error classification lookup data
    - Populate `error_classifications` table with known WhatsApp API error codes mapped to categories (transient, permanent, suppression)
    - Include common error codes: 131047 (rate limit → transient), 131026 (invalid number → permanent), 131056 (blocked → suppression), etc.
    - _Requirements: 21.5_

  - [x] 1.3 Add multi-tenant default columns and indexes
    - Ensure organization_id (default 1), branch_id (default 1), tenant_id (default 1) columns exist on all new tables per design
    - Add composite indexes for organization_id scoping
    - _Requirements: 24.1, 24.2, 24.3, 24.6_

- [x] 2. Core services foundation — Channel abstraction and template validation
  - [x] 2.1 Implement channel abstraction layer
    - Create `services/channel.py` with `MessageDispatcher` abstract base class
    - Implement `WhatsAppDispatcher` using existing `get_whatsapp_headers()`, `WHATSAPP_API_VERSION`, and `get_whatsapp_phone_number_id()` from `app.py`
    - Implement `send_template()` method that calls Meta WhatsApp Business API
    - Implement `get_channel_name()` returning "whatsapp"
    - _Requirements: 15.1, 15.3_

  - [x] 2.2 Implement template validator service
    - Create `services/template_validator.py` with `TemplateValidator` class
    - Implement `parse_placeholders()` — extract {{1}}, {{2}}, {{name}} style variables from template body
    - Implement `validate_mappings()` — check all placeholders have corresponding customer field mappings
    - Implement `validate_customer_params()` — verify resolved values are non-null, non-empty, ≤ 1024 chars
    - Implement `sanitize_param()` — remove control characters (U+0000–U+001F, U+007F–U+009F) while preserving Unicode
    - Implement `render_preview()` — substitute sample customer data into template
    - _Requirements: 20.1, 20.2, 20.3, 20.4, 20.5, 20.6, 20.7, 11.6_

  - [x] 2.3 Write property test for template placeholder parsing (Property 24)
    - **Property 24: Template placeholder parsing and mapping validation**
    - Test that for any template body with N placeholders, exactly N are identified; block approval if fewer than N mappings or any resolved value is null/empty/>1024 chars
    - **Validates: Requirements 20.1, 20.2, 20.5, 20.6**

  - [x] 2.4 Write property test for template parameter sanitization (Property 17)
    - **Property 17: Template parameter sanitization**
    - Test that sanitization removes control characters while preserving printable Unicode (Hindi/Marathi), and output length ≤ input length
    - **Validates: Requirements 11.6**

- [x] 3. Campaign Manager Blueprint — state machine and CRUD
  - [x] 3.1 Create campaign Blueprint with state machine
    - Create `blueprints/campaign_bp.py` with Flask Blueprint registered at `/api/campaigns/`
    - Implement `CampaignService` class with `create_campaign()`, `update_campaign()`, `get_campaign()`, `list_campaigns()`
    - Implement `transition_state()` with valid transition enforcement per design state diagram (draft→scheduled, draft→pending_approval, scheduled→pending_approval, pending_approval→approved, pending_approval→cancelled, approved→sending, sending→paused, paused→sending, sending→completed, sending→failed)
    - Implement `duplicate_campaign()` — copy config to new draft
    - Implement `schedule_campaign()` — set scheduled_at, transition to scheduled
    - Add Flask session auth checks and role-based permission on state-change endpoints
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.10, 2.1, 2.2, 2.3, 2.6, 11.1, 11.2, 11.3, 11.4, 11.5_

  - [x] 3.2 Write property test for campaign state machine transitions (Property 1)
    - **Property 1: Campaign state machine transition validity**
    - Test that only valid transitions are allowed and all invalid transitions are rejected with error
    - **Validates: Requirements 1.2**

  - [x] 3.3 Write property test for campaign duplication (Property 3)
    - **Property 3: Campaign duplication preserves configuration**
    - Test that duplicating a campaign produces a new draft with identical segment_id, template_id, campaign_type, and config
    - **Validates: Requirements 1.8**

  - [x] 3.4 Write property test for audit logging (Property 5)
    - **Property 5: Every campaign action produces an audit log entry**
    - Test that any state-changing action creates a record in operator_actions with correct fields within 1 second
    - **Validates: Requirements 2.6, 11.2**

  - [x] 3.5 Write property test for role-based permissions (Property 16)
    - **Property 16: Role-based permission enforcement**
    - Test that operators without "campaign_send" permission get HTTP 403 on approve/send, and operators with permission succeed
    - **Validates: Requirements 11.1**

- [x] 4. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Segmentation Engine
  - [x] 5.1 Implement segmentation service and Blueprint
    - Create `blueprints/segment_bp.py` with Flask Blueprint at `/api/segments/`
    - Create `services/segmentation.py` with `SegmentationService` class
    - Implement `build_query()` — construct parameterized MySQL WHERE clauses from filter criteria JSON (expiry_category, days_remaining, plan_name, plan_category, zone_name, area, building, status, network_type, connectivity_mode, kyc_approved, owner_tenant, tags)
    - Implement AND logic for combining multiple filters
    - Implement `estimate_count()` with COUNT query for real-time estimation
    - Implement `evaluate_segment()` with pagination (default 50/page)
    - Implement `save_segment()` and `load_segment()` for persistent segment definitions
    - Implement derived filters: "days since last recharge" = (current_date - activation_date), "days inactive" = (current_date - expiry_date) for expired records
    - Add zero-result prevention check with warning response
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 12.1, 12.6_

  - [x] 5.2 Write property test for AND-filter correctness (Property 6)
    - **Property 6: Segmentation AND-filter returns only matching customers**
    - Test that every customer returned satisfies ALL filter criteria, and no matching customer is omitted
    - **Validates: Requirements 3.1, 3.2**

  - [x] 5.3 Write property test for segment save/load round-trip (Property 7)
    - **Property 7: Segment save/load round-trip**
    - Test that saving and loading a segment preserves semantically equivalent filter criteria
    - **Validates: Requirements 3.3**

  - [x] 5.4 Write property test for derived filter computation (Property 8)
    - **Property 8: Derived filter computation correctness**
    - Test that "days since last recharge" = (current_date - activation_date).days and "days inactive" = (current_date - expiry_date).days for expired customers
    - **Validates: Requirements 3.6, 3.7**

- [x] 6. Sending Queue, Cooldown, and Opt-Out services
  - [x] 6.1 Implement sending queue with background workers
    - Create `services/sending_queue.py` with `SendingQueue` class
    - Implement `enqueue_campaign()` — create campaign_messages records with idempotency_key (campaign_id + mobile + template_id)
    - Implement `process_batch()` — dispatch messages via `WhatsAppDispatcher` with configurable throttle rate (default 80/sec)
    - Implement `pause_campaign()`, `resume_campaign()`, `cancel_campaign()` — update message statuses
    - Implement background worker using `threading` + `concurrent.futures` (matching existing pattern)
    - Implement real-time progress tracking (sent_count, failed_count, remaining_count) with DB updates
    - Integrate with `TemplateValidator` for per-message param validation before dispatch
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 12.3_

  - [x] 6.2 Implement retry categorizer service
    - Create `services/retry_categorizer.py` with `RetryCategorizerService` class
    - Implement `classify_error()` — lookup error_code in error_classifications table, return category
    - Implement `should_retry()` — check retry_count < max_retries and category == "transient"
    - Implement exponential backoff: 5 * 3^(N-1) seconds for attempt N (5s, 15s, 45s)
    - Handle "permanent" errors: mark as permanently_failed, flag customer
    - Handle "suppression" errors: add to suppression list
    - Trigger automatic campaign pause when suppression_rate > 20%
    - _Requirements: 21.1, 21.2, 21.3, 21.4, 21.5, 21.6, 21.7_

  - [x] 6.3 Implement cooldown manager
    - Create `services/cooldown_manager.py` with `CooldownManager` class
    - Implement `check_cooldown()` — query message_cooldowns for 72-hour promotional window (120h when Yellow tier)
    - Implement 2-per-7-day rolling limit check
    - Implement `record_send()` — insert cooldown record after dispatch
    - Allow transactional messages to bypass all cooldown checks
    - Integrate with `QualityMonitor` to read current quality tier
    - _Requirements: 17.1, 17.2, 17.3, 17.4, 17.5, 17.6, 17.7, 6.4_

  - [x] 6.4 Implement opt-out manager
    - Create `services/opt_out_manager.py` with `OptOutManager` class
    - Implement keyword recognition: STOP, UNSUBSCRIBE, OPT OUT, CANCEL, DND (case-insensitive)
    - Implement `process_opt_out()` — add to suppression_list, send confirmation
    - Implement `process_opt_in()` — handle START, SUBSCRIBE keywords, remove from suppression_list
    - Implement `is_suppressed()` — check active suppression_list for recipient before dispatch
    - Implement `add_to_dnd()` — manual operator DND addition with reason
    - Integrate opt-out keyword detection into existing webhook processing in `app.py`
    - _Requirements: 19.1, 19.2, 19.3, 19.4, 19.5, 19.6_

  - [x] 6.5 Write property test for queue manipulation invariants (Property 2)
    - **Property 2: Queue manipulation preserves message invariants**
    - Test that pausing stops dispatch, resuming only dispatches "queued" messages, cancelling transitions all queued to cancelled
    - **Validates: Requirements 1.5, 1.6, 1.7**

  - [x] 6.6 Write property test for retry backoff computation (Property 9)
    - **Property 9: Retry backoff computation**
    - Test that delay = 5 * 3^(N-1) for N in {1,2,3}, and no message retried more than 3 times
    - **Validates: Requirements 4.3**

  - [x] 6.7 Write property test for idempotency (Property 10)
    - **Property 10: Idempotency prevents duplicate message delivery**
    - Test that duplicate (campaign_id, mobile, template_id) combinations produce exactly one record
    - **Validates: Requirements 4.5, 26.3**

  - [x] 6.8 Write property test for cooldown enforcement (Property 20)
    - **Property 20: Cooldown enforcement correctness**
    - Test 72h/120h window, 2-per-7-day limit, and 7-day reactivation cooldown
    - **Validates: Requirements 17.1, 17.3, 6.4, 18.5**

  - [x] 6.9 Write property test for transactional bypass (Property 21)
    - **Property 21: Transactional messages bypass cooldown**
    - Test that transactional messages are never blocked by cooldown state
    - **Validates: Requirements 17.6**

  - [x] 6.10 Write property test for opt-out keywords (Property 22)
    - **Property 22: Opt-out keyword recognition and round-trip**
    - Test case-insensitive matching of opt-out keywords and opt-in round-trip with START/SUBSCRIBE
    - **Validates: Requirements 19.1, 19.2, 19.5**

  - [x] 6.11 Write property test for suppression enforcement (Property 23)
    - **Property 23: Suppression list enforcement at dispatch time**
    - Test that no campaign message is dispatched to a customer on the active suppression list
    - **Validates: Requirements 19.3**

  - [x] 6.12 Write property test for error classification (Property 25)
    - **Property 25: Error code classification determinism**
    - Test that each error code maps to exactly one deterministic category
    - **Validates: Requirements 21.1**

- [x] 7. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Operator approval workflow and campaign test send
  - [x] 8.1 Implement operator approval workflow endpoints
    - Add `/api/campaigns/<id>/approve` and `/api/campaigns/<id>/reject` endpoints to campaign Blueprint
    - Implement approval → sending transition triggering queue enqueue
    - Implement rejection → cancelled with reason logging
    - Implement approval preview endpoint returning recipient count, template content, estimated time
    - Ensure Automation_Rule drafts require approval before dispatch (Property 4)
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

  - [x] 8.2 Implement campaign test send
    - Add `/api/campaigns/<id>/test-send` endpoint accepting 1-5 test mobile numbers
    - Dispatch template immediately bypassing regular queue
    - Mark test messages distinctly in delivery log (excluded from analytics)
    - Return delivery status for each test number
    - _Requirements: 16.1, 16.2, 16.3, 16.4_

  - [x] 8.3 Write property test for approval gate (Property 4)
    - **Property 4: No messages dispatched without explicit operator approval**
    - Test that automation-generated or scheduled campaigns have zero dispatched messages until operator approves
    - **Validates: Requirements 2.5**

- [x] 9. Delivery tracking and webhook integration
  - [x] 9.1 Implement delivery tracking and webhook status updates
    - Extend existing webhook handler in `app.py` to detect campaign message status callbacks
    - Update campaign_messages records when webhook delivers: delivered, read, failed statuses with timestamps
    - Mark permanent failures and add to suppression list via Retry_Categorizer
    - Update campaign aggregate counts (delivered_count, read_count, failed_count) on each status update
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

  - [x] 9.2 Write property test for webhook status updates (Property 11)
    - **Property 11: Webhook status updates apply to correct records**
    - Test that a webhook with whatsapp_message_id updates exactly the matching campaign_message record
    - **Validates: Requirements 5.2**

  - [x] 9.3 Write property test for delivery rate computation (Property 12)
    - **Property 12: Delivery rate computation correctness**
    - Test that delivery_rate = delivered/sent, read_rate = read/sent, failure_rate = failed/sent, all in [0,1], test-sends excluded
    - **Validates: Requirements 5.3, 8.1, 16.3**

- [x] 10. Simulation engine and A/B testing
  - [x] 10.1 Implement simulation engine
    - Create `services/simulation.py` with `SimulationEngine` class
    - Implement `simulate()` — compute final_audience after exclusions (cooldown, opt-out, DND, invalid, incomplete params)
    - Calculate estimated_send_time = final_audience / throttle_rate
    - Calculate estimated_cost = final_audience * per_message_rate (configurable by category)
    - Detect duplicate recipients
    - Generate warning when exclusions > 30% of original segment
    - Ensure computation within 5 seconds for 50k customers
    - Add `/api/campaigns/<id>/simulate` endpoint
    - _Requirements: 22.1, 22.2, 22.3, 22.4, 22.5, 22.6, 22.7_

  - [x] 10.2 Implement A/B testing
    - Implement `create_ab_test()` in CampaignService — allow 2-4 variants with test percentage (10-50%)
    - Implement even audience split algorithm: floor(N/V) or ceil(N/V) per variant
    - Create campaign_ab_variants records tracking per-variant metrics
    - Implement winner selection endpoint that creates full rollout campaign
    - Prevent full send until operator selects winner
    - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.5_

  - [x] 10.3 Write property test for simulation computation (Property 27)
    - **Property 27: Simulation computation correctness**
    - Test final_audience = N - exclusions, estimated_time = final_audience / rate, cost = final_audience * price, exclusions non-negative
    - **Validates: Requirements 22.1, 22.2, 22.4**

  - [x] 10.4 Write property test for simulation warning threshold (Property 28)
    - **Property 28: Simulation exclusion warning threshold**
    - Test warning when excluded > 30% and no warning when ≤ 30%
    - **Validates: Requirements 22.5**

  - [x] 10.5 Write property test for A/B test even split (Property 18)
    - **Property 18: A/B test even audience split**
    - Test that for audience N and V variants, each variant gets floor(N/V) or ceil(N/V) with max difference of 1
    - **Validates: Requirements 13.2**

- [x] 11. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 12. CRM Panel and customer profile services
  - [x] 12.1 Implement CRM service and Blueprint
    - Create `blueprints/crm_bp.py` with Flask Blueprint at `/api/crm/`
    - Create `services/crm.py` with `CRMService` class
    - Implement `get_customer_profile()` — join renewal_records + customer data for full profile
    - Implement `get_interaction_timeline()` — merge messages, campaigns, notes, tags, status changes sorted reverse-chronological
    - Implement `add_note()` — persist with operator name, timestamp; record in customer_activity
    - Implement `add_tags()` / `remove_tag()` — manage customer_tags; make tags available for segmentation
    - Implement `get_campaign_history()` — campaigns targeting customer with delivery status
    - Display opt-out/DND status, engagement score, and engagement trend on profile
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 19.6, 23.7_

  - [x] 12.2 Write property test for timeline ordering (Property 13)
    - **Property 13: Timeline ordering consistency**
    - Test that all interaction records merge in strictly reverse chronological order with no records omitted
    - **Validates: Requirements 7.2, 7.6**

  - [x] 12.3 Write property test for tag queryability in segmentation (Property 14)
    - **Property 14: Tags assigned to customers are queryable in segmentation**
    - Test that tagged customers appear in segment results when filtering by that tag
    - **Validates: Requirements 7.5**

- [x] 13. Quality monitor, notification engine, and recovery manager
  - [x] 13.1 Implement quality monitor service
    - Create `services/quality_monitor.py` with `QualityMonitor` class
    - Implement `compute_metrics()` — aggregate blocked_count, failure_rate, opt_out_rate, read_rate for 24h and 7d windows
    - Implement `get_quality_tier()` — determine GREEN/YELLOW/RED based on metrics thresholds
    - Implement `check_alerts()` — generate alerts when thresholds exceeded (failure > 5%, blocks > 10/day)
    - Implement `record_block()` — add blocked customer to permanent suppression
    - Implement hourly metric logging to quality_metrics table
    - Integrate adaptive cooldown: increase to 120h when tier is Yellow
    - _Requirements: 18.1, 18.2, 18.3, 18.4, 18.5, 18.6, 18.7_

  - [x] 13.2 Implement notification engine
    - Create `services/notification_engine.py` with `NotificationEngine` class
    - Implement `send_alert()` — create system_notifications record with type, severity, details
    - Implement alert delivery: in-app notification (persistent), browser push (optional), WhatsApp to operator (optional)
    - Implement `get_unacknowledged()` / `acknowledge()` endpoints
    - Generate alerts for: campaign_degraded (>10% failures), queue_overloaded (>10k backlog), webhook_connectivity (5min gap), template_rejected, quality_drop
    - _Requirements: 25.1, 25.2, 25.3, 25.4, 25.5, 25.6, 25.7_

  - [x] 13.3 Implement recovery manager
    - Create `services/recovery_manager.py` with `RecoveryManager` class
    - Implement `recover_on_startup()` — query campaign_messages in "queued" or "sending" status, re-enqueue within 30s
    - Implement `identify_stale_messages()` — find records in "sending" with updated_at > 5 minutes ago
    - Reset stale messages to "queued" for re-dispatch
    - Use idempotency keys to prevent duplicates during recovery
    - Log recovery actions: re-queued count, duplicates prevented, campaigns resumed, total recovery time
    - Register startup hook in app.py to call `recover_on_startup()`
    - _Requirements: 26.1, 26.2, 26.3, 26.4, 26.5, 26.6, 26.7_

  - [x] 13.4 Write property test for threshold alerts (Property 26)
    - **Property 26: Threshold-triggered alert correctness**
    - Test that failure_rate > 10% triggers campaign_degraded, blocked_count > 10 triggers block_spike, suppression > 20% triggers pause
    - **Validates: Requirements 18.2, 18.3, 21.7, 25.1**

  - [x] 13.5 Write property test for stale message recovery (Property 30)
    - **Property 30: Stale message recovery reset**
    - Test that messages in "sending" state for >5 minutes get reset to "queued"
    - **Validates: Requirements 26.6**

- [x] 14. Engagement scoring and analytics
  - [x] 14.1 Implement engagement scorer and analytics Blueprint
    - Create `services/engagement_scorer.py` with engagement batch computation
    - Implement interaction_score = round(0.4 * read_rate + 0.3 * response_rate + 0.2 * recency_score + 0.1 * frequency_score), bounded [0, 100]
    - Implement engagement trend classification (increasing, stable, declining)
    - Implement preferred time window detection
    - Create `blueprints/analytics_bp.py` with Flask Blueprint at `/api/analytics/`
    - Implement per-campaign metrics: delivery_rate, read_rate, failure_rate, response_rate
    - Implement aggregate metrics with date range filtering
    - Implement zone-wise breakdown and top-performing templates
    - Use pre-computed summary tables (campaign_analytics) for dashboard queries
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 12.4, 23.1, 23.2, 23.3, 23.4, 23.5, 23.6_

  - [x] 14.2 Write property test for engagement score bounds (Property 29)
    - **Property 29: Engagement score bounded computation**
    - Test weighted score formula is bounded to [0, 100] for all valid inputs
    - **Validates: Requirements 23.2**

  - [x] 14.3 Write property test for date range filtering (Property 15)
    - **Property 15: Date range filtering includes only records within bounds**
    - Test that all returned analytics records have timestamps within [start, end]
    - **Validates: Requirements 8.4**

- [x] 15. Media library
  - [x] 15.1 Implement media library Blueprint
    - Create `blueprints/media_bp.py` with Flask Blueprint at `/api/media/`
    - Implement file upload with size validation: image ≤ 5MB, video ≤ 16MB, document ≤ 100MB
    - Store metadata in media_assets table (filename, mime_type, file_size, storage_path, uploaded_by)
    - Implement grid view endpoint with search, type filtering, thumbnail paths
    - Implement usage count tracking
    - Reject unsupported mime types with specific error messages
    - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5_

  - [x] 15.2 Write property test for media file validation (Property 19)
    - **Property 19: Media file size validation**
    - Test acceptance matrix: image ≤ 5MB, video ≤ 16MB, document ≤ 100MB; reject all others with specific constraint
    - **Validates: Requirements 14.2**

- [x] 16. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 17. Reactivation workflows and automation rules
  - [x] 17.1 Implement reactivation workflows and automation rules
    - Add reactivation workflow templates in CampaignService: expired recovery, inactive comeback, disconnected re-engagement, speed upgrade, festive promotions
    - Pre-populate segment + template suggestions when operator selects workflow
    - Implement automation_rules CRUD — trigger types (schedule, event, threshold), condition config, action (create_campaign_draft, notify_operator)
    - Ensure all automation-generated campaigns require operator approval
    - Track reactivation success: monitor status changes within 30 days of campaign
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [x] 18. Frontend — Campaign Manager UI
  - [x] 18.1 Create Campaign Manager workspace panel HTML/JS
    - Create `templates/campaigns/` Jinja2 templates for campaign manager
    - Add "Campaigns" workspace-switcher tab alongside existing tabs (Inbox, Templates, etc.)
    - Implement dashboard summary cards (total campaigns, active sends, delivery rates)
    - Implement campaign list table with server-side pagination and status filtering
    - Use existing CSS variables (--surface, --surface-2, --border, --text-1, --green, etc.)
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 12.2, 12.5_

  - [x] 18.2 Create Campaign creation wizard and audience builder UI
    - Implement campaign creation form: name, description, segment, template, schedule, priority
    - Implement audience builder with multi-filter interface and real-time count estimation
    - Implement template selection with preview and parameter mapping UI
    - Implement scheduling modal with date/time picker and recurring options
    - Implement test send modal (1-5 numbers)
    - Add simulation results display panel
    - _Requirements: 1.1, 3.5, 16.1, 22.1_

  - [x] 18.3 Create approval workflow UI and campaign detail view
    - Implement pending approval queue view for operators
    - Implement approval/rejection buttons with preview summary
    - Implement real-time campaign progress view (sent, delivered, failed counts)
    - Implement A/B test results comparison view and winner selection
    - Implement campaign pause/resume/cancel controls
    - _Requirements: 2.1, 2.4, 4.6, 13.3, 13.4_

- [x] 19. Frontend — CRM Panel and Analytics Dashboard
  - [x] 19.1 Create CRM Panel slide-out UI
    - Create `templates/crm/` Jinja2 templates
    - Implement slide-out panel within existing chat view (wa-main area)
    - Display customer profile: name, mobile, plan, validity, status, zone, area, building
    - Display interaction timeline with lazy loading (100 records initial)
    - Implement notes panel with add-note form
    - Implement tags management (add/remove) UI
    - Display engagement score indicator and opt-out status
    - Integrate within existing dark theme layout (wa-rail, wa-sidebar, wa-main)
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 10.5, 12.5_

  - [x] 19.2 Create Analytics Dashboard UI
    - Create `templates/analytics/` Jinja2 templates
    - Add analytics workspace panel tab
    - Implement per-campaign metrics display with percentage breakdowns
    - Implement aggregate metrics view: daily/weekly/monthly totals, averages
    - Implement date range picker for filtering
    - Implement zone-wise engagement breakdown
    - Implement top-5 performing templates highlight
    - Implement customer retention metrics (reactivation rate, churn by zone)
    - Implement quality monitor dashboard (Green/Yellow/Red tier display)
    - Use lightweight charting library compatible with existing Bootstrap/Font Awesome
    - Display opt-out trends and failure breakdown by category
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 10.6, 18.4, 19.7, 21.6_

- [x] 20. Frontend — Media Library and Notifications UI
  - [x] 20.1 Create Media Library and Notification panel UI
    - Create media upload form with drag-and-drop support
    - Implement media grid view with thumbnails, search, and type filter
    - Display file metadata and usage count
    - Create notification bell indicator with unread count
    - Create notification panel with alert list, severity badges, and acknowledge button
    - Implement theme-aware styling using existing CSS variables
    - _Requirements: 14.3, 14.4, 25.6, 10.3_

- [x] 21. Integration wiring — Blueprint registration and app.py updates
  - [x] 21.1 Register all Blueprints and wire startup hooks
    - Register `campaign_bp`, `segment_bp`, `crm_bp`, `analytics_bp`, `media_bp` in `app.py`
    - Wire `ensure_crm_tables()` migration call on app startup
    - Wire `RecoveryManager.recover_on_startup()` on app startup
    - Wire opt-out keyword detection into existing webhook handler
    - Wire campaign message status updates into existing webhook handler
    - Add CSRF protection on all new state-changing endpoints
    - Add quality monitor hourly scheduler (using existing threading pattern)
    - Add engagement scorer batch scheduler post-campaign
    - _Requirements: 9.1, 11.5, 15.1, 26.2_

- [x] 22. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- Backend uses Python (Flask + MySQL), frontend uses JavaScript with existing Jinja2 templates
- All new Blueprints follow the existing Flask session auth pattern from auth.py
- Background workers use Python threading + concurrent.futures matching the existing WEBHOOK_ASYNC_PROCESSING pattern
- Property-based tests use Hypothesis (Python) for backend logic and fast-check (JS) for frontend computation

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2", "1.3"] },
    { "id": 1, "tasks": ["2.1", "2.2"] },
    { "id": 2, "tasks": ["2.3", "2.4", "3.1"] },
    { "id": 3, "tasks": ["3.2", "3.3", "3.4", "3.5", "5.1"] },
    { "id": 4, "tasks": ["5.2", "5.3", "5.4", "6.1", "6.2", "6.3", "6.4"] },
    { "id": 5, "tasks": ["6.5", "6.6", "6.7", "6.8", "6.9", "6.10", "6.11", "6.12", "8.1", "8.2"] },
    { "id": 6, "tasks": ["8.3", "9.1"] },
    { "id": 7, "tasks": ["9.2", "9.3", "10.1", "10.2"] },
    { "id": 8, "tasks": ["10.3", "10.4", "10.5", "12.1"] },
    { "id": 9, "tasks": ["12.2", "12.3", "13.1", "13.2", "13.3"] },
    { "id": 10, "tasks": ["13.4", "13.5", "14.1"] },
    { "id": 11, "tasks": ["14.2", "14.3", "15.1"] },
    { "id": 12, "tasks": ["15.2", "17.1"] },
    { "id": 13, "tasks": ["18.1", "19.1", "19.2", "20.1"] },
    { "id": 14, "tasks": ["18.2", "18.3"] },
    { "id": 15, "tasks": ["21.1"] }
  ]
}
```
