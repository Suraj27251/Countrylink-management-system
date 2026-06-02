# Requirements Document

## Introduction

This specification defines the requirements for extending the existing CountryLink WhatsApp Inbox System into an enterprise-grade CRM + WhatsApp Marketing + Campaign Automation platform. The system will be developed within the CountryLink Management System repository, enhancing the current WhatsApp Inbox (which serves as the communication engine) with enterprise campaign management, smart customer segmentation, marketing automation, customer relationship management, and analytics capabilities. The existing operator-controlled workflow is preserved: the system prepares and recommends, but operators approve and trigger all outbound communications.

## Glossary

- **Campaign_Manager**: The module responsible for creating, scheduling, managing, and monitoring WhatsApp broadcast campaigns
- **Segmentation_Engine**: The module that builds dynamic audience segments by filtering the customer database on multiple attributes
- **Sending_Queue**: The queue-based processing system that handles WhatsApp message dispatch with throttling, retry, and deduplication
- **CRM_Panel**: The customer relationship management interface displaying profile, history, interactions, notes, and tags for each customer
- **Analytics_Dashboard**: The reporting module that aggregates and visualizes campaign performance, engagement metrics, and customer retention data
- **Operator**: A human user of the CountryLink Management System who reviews, approves, and triggers campaign sends
- **Customer**: An ISP subscriber record in the countrylinks_user_database containing plan, billing, and contact information
- **Template**: A pre-approved WhatsApp Business API message template registered with Meta
- **Audience_Segment**: A saved set of filter criteria that dynamically resolves to a list of customers matching those criteria
- **Campaign**: A planned or executed bulk WhatsApp message broadcast targeting an Audience_Segment with a specific Template
- **Delivery_Log**: A record tracking the send status (queued, sent, delivered, read, failed) of each individual message within a Campaign
- **Automation_Rule**: A configurable trigger-condition-action definition that prepares campaign drafts for operator approval
- **Reactivation_Workflow**: A sequence of templated messages designed to recover inactive, expired, or disconnected customers
- **Media_Library**: A centralized repository of images, videos, and documents used across campaigns and templates
- **Omnichannel_Engine**: The future-ready abstraction layer supporting WhatsApp today and additional messaging channels (SMS, Email, Telegram, Facebook) later
- **Cooldown_Manager**: The module responsible for enforcing message frequency limits per customer and per campaign to protect WhatsApp quality ratings
- **Quality_Monitor**: The module that tracks WhatsApp Business API health indicators including blocked users, failed sends, opt-outs, and engagement metrics to protect the Meta reputation score
- **Opt_Out_Manager**: The module that processes unsubscribe requests (STOP keyword), maintains the suppression list, and enforces Do Not Disturb (DND) exclusions for all outbound campaigns
- **Template_Validator**: The module that validates WhatsApp message template placeholders against available customer data fields before campaign dispatch to prevent rendering errors
- **Retry_Categorizer**: The module that classifies message send failures by type and applies differentiated retry strategies (retry, suppress, or permanently exclude) based on error classification
- **Simulation_Engine**: The module that performs pre-send campaign analysis showing audience count, estimated send time, API cost projection, potential duplicates, and excluded customers before operator approval
- **Engagement_Store**: The data layer that persists campaign engagement metrics, customer interaction scores, and response patterns to enable future AI-driven campaign optimization and predictive analytics
- **Tenant_Framework**: The architectural layer that supports multi-tenant isolation through organization_id, branch_id, and tenant_id scoping for future expansion to multiple ISPs, franchise operators, and reseller panels
- **Notification_Engine**: The internal alerting module that notifies operators of critical system events including campaign failures, queue overloads, Meta webhook issues, template rejections, and quality score drops
- **Recovery_Manager**: The module responsible for persisting queue state to durable storage and automatically resuming pending campaign messages after server restart or crash recovery

## Requirements

### Requirement 1: Campaign Creation and Lifecycle Management

**User Story:** As an Operator, I want to create, configure, and manage WhatsApp broadcast campaigns through a professional campaign manager interface, so that I can plan and execute bulk customer communications efficiently.

#### Acceptance Criteria

1. WHEN an Operator selects "Create Campaign" from the Campaign_Manager, THE Campaign_Manager SHALL present a form capturing campaign name, description, target Audience_Segment, Template selection, scheduling options, and send priority
2. THE Campaign_Manager SHALL support the following campaign states: draft, scheduled, pending_approval, approved, sending, paused, completed, cancelled, and failed
3. WHEN an Operator saves a campaign without scheduling, THE Campaign_Manager SHALL persist the campaign in "draft" state with all configured parameters
4. WHEN an Operator schedules a campaign for a future date and time, THE Campaign_Manager SHALL transition the campaign to "scheduled" state and dispatch it at the specified time only after operator approval
5. WHEN an Operator pauses an active sending campaign, THE Sending_Queue SHALL stop dispatching pending messages within 10 seconds and transition the campaign to "paused" state
6. WHEN an Operator resumes a paused campaign, THE Sending_Queue SHALL continue dispatching remaining unsent messages from where it stopped
7. WHEN an Operator cancels a campaign, THE Campaign_Manager SHALL transition the campaign to "cancelled" state and discard all unsent messages from the Sending_Queue
8. WHEN an Operator duplicates an existing campaign, THE Campaign_Manager SHALL create a new draft campaign copying the original audience segment, template, and configuration
9. THE Campaign_Manager SHALL support recurring campaign scheduling with configurable frequency (daily, weekly, monthly) and end date
10. WHEN a campaign completes sending to all recipients, THE Campaign_Manager SHALL transition the campaign to "completed" state and generate a delivery summary

### Requirement 2: Operator Approval Workflow

**User Story:** As an Operator, I want all campaign sends to require my explicit approval before execution, so that no messages are sent to customers without human oversight.

#### Acceptance Criteria

1. WHEN a scheduled campaign reaches its dispatch time, THE Campaign_Manager SHALL transition the campaign to "pending_approval" state and notify the assigned Operator
2. WHEN an Operator approves a pending campaign, THE Sending_Queue SHALL begin dispatching messages to the target audience
3. WHEN an Operator rejects a pending campaign, THE Campaign_Manager SHALL transition the campaign to "cancelled" state and log the rejection reason
4. THE Campaign_Manager SHALL provide a preview showing estimated recipient count, template content with sample parameters, and estimated delivery time before approval
5. WHEN an Automation_Rule generates a campaign draft, THE Campaign_Manager SHALL require operator approval before any messages are dispatched
6. THE Campaign_Manager SHALL log every approval, rejection, and send action with operator name, timestamp, and action details in the operator_actions table

### Requirement 3: Smart Customer Segmentation

**User Story:** As an Operator, I want to build dynamic audience segments using multiple customer attributes, so that I can target precise groups of customers for campaigns.

#### Acceptance Criteria

1. THE Segmentation_Engine SHALL support filtering customers by: expiry category (expired, today, upcoming), days_remaining range, plan_name, plan_category, zone_name, area, building, status (active, inactive, disconnected), network_type, connectivity_mode, KYC approval status, owner_tenant classification, and validity range
2. WHEN an Operator combines multiple filter criteria, THE Segmentation_Engine SHALL apply all criteria using logical AND and return only customers matching every specified condition
3. WHEN an Operator saves a segment with a name, THE Segmentation_Engine SHALL persist the filter criteria and make the segment available for reuse in future campaigns
4. WHEN an Operator loads a saved Audience_Segment, THE Segmentation_Engine SHALL re-evaluate the filter criteria against current customer data and return the dynamically refreshed audience list
5. WHEN an Operator modifies filter criteria in the audience builder, THE Segmentation_Engine SHALL display the estimated recipient count within 2 seconds
6. THE Segmentation_Engine SHALL support a "days since last recharge" derived filter calculated from the activation_date and expiry_date fields
7. THE Segmentation_Engine SHALL support a "days inactive" derived filter for customers whose status is inactive or expired
8. THE Segmentation_Engine SHALL prevent creating segments that resolve to zero customers and display a warning to the Operator

### Requirement 4: Queue-Based Message Sending

**User Story:** As an Operator, I want campaign messages to be sent through a managed queue with throttling and retry logic, so that the system respects WhatsApp API rate limits and handles failures gracefully.

#### Acceptance Criteria

1. WHEN a campaign is approved for sending, THE Sending_Queue SHALL enqueue individual messages for each recipient in the target Audience_Segment
2. THE Sending_Queue SHALL throttle outbound message dispatch to a configurable rate limit (default: 80 messages per second) to comply with WhatsApp Business API throughput limits
3. IF a message send fails due to a transient error (network timeout, rate limit response, server error), THEN THE Sending_Queue SHALL retry the message up to 3 times with exponential backoff (5s, 15s, 45s)
4. IF a message send fails after all retry attempts, THEN THE Sending_Queue SHALL mark the message as "failed" in the Delivery_Log and record the error details
5. THE Sending_Queue SHALL prevent duplicate message delivery by checking if a message to the same mobile with the same template was already sent within the same campaign
6. WHEN the Sending_Queue processes messages, THE Sending_Queue SHALL update real-time progress (sent count, failed count, remaining count) visible to the Operator
7. THE Sending_Queue SHALL process campaign messages using background async workers that do not block the main application request handling
8. WHEN a campaign is paused or cancelled, THE Sending_Queue SHALL immediately stop dequeuing new messages for that campaign

### Requirement 5: Delivery Tracking and Status Updates

**User Story:** As an Operator, I want to track the delivery status of every message sent through campaigns, so that I can monitor message reach and identify delivery issues.

#### Acceptance Criteria

1. WHEN a message is dispatched via the WhatsApp Business API, THE Delivery_Log SHALL record the initial status as "sent" with the Meta message_id
2. WHEN a webhook status update is received from Meta (delivered, read, failed), THE Delivery_Log SHALL update the corresponding message record with the new status and timestamp
3. THE Campaign_Manager SHALL display aggregated delivery statistics per campaign: total sent, delivered, read, and failed counts with percentage breakdowns
4. WHEN a message delivery fails with a permanent error (invalid number, blocked contact), THE Delivery_Log SHALL mark the message as "permanently_failed" and exclude the number from retry attempts
5. THE Delivery_Log SHALL retain message status history for a minimum of 90 days for audit and analytics purposes

### Requirement 6: Reactivation Marketing Workflows

**User Story:** As an Operator, I want pre-configured reactivation campaign workflows for inactive customers, so that I can quickly launch recovery campaigns with proven messaging sequences.

#### Acceptance Criteria

1. THE Campaign_Manager SHALL provide reactivation workflow templates for: expired plan recovery, inactive customer comeback, disconnected customer re-engagement, speed upgrade offers, and festive promotional campaigns
2. WHEN an Operator selects a reactivation workflow, THE Campaign_Manager SHALL pre-populate the campaign with the recommended audience segment (customers matching the workflow criteria) and suggested template
3. THE Segmentation_Engine SHALL identify reactivation-eligible customers by filtering for: status equals "expired" or "inactive", days_remaining less than zero, and expiry_date older than a configurable threshold
4. WHEN an Operator launches a reactivation campaign, THE Campaign_Manager SHALL enforce a minimum cooldown period of 7 days between repeat messages to the same customer for the same workflow type
5. THE Campaign_Manager SHALL track reactivation success by monitoring if a customer's status changes from "expired" or "inactive" to "active" within 30 days of receiving a reactivation message

### Requirement 7: Customer CRM Panel

**User Story:** As an Operator, I want a comprehensive customer profile panel with complete interaction history, so that I can understand each customer's relationship with CountryLink before communicating with them.

#### Acceptance Criteria

1. WHEN an Operator selects a customer from the inbox or search, THE CRM_Panel SHALL display the customer's profile information: name, mobile, plan details, validity, status, zone, area, building, network type, activation date, and expiry date
2. THE CRM_Panel SHALL display the complete WhatsApp conversation history for the selected customer in chronological order
3. THE CRM_Panel SHALL display all campaigns that targeted the selected customer with send date, template used, and delivery status
4. WHEN an Operator adds a note to a customer profile, THE CRM_Panel SHALL persist the note with operator name and timestamp and display it in the customer's activity timeline
5. WHEN an Operator assigns tags to a customer (e.g., "VIP", "complaint_pending", "upgrade_interested"), THE CRM_Panel SHALL persist the tags and make them available as segmentation filter criteria
6. THE CRM_Panel SHALL display an interaction timeline combining: WhatsApp messages, campaign messages received, operator notes, tag changes, and status changes in reverse chronological order
7. THE CRM_Panel SHALL integrate within the existing WhatsApp Inbox dark theme UI as a slide-out or tabbed panel alongside the conversation view

### Requirement 8: Campaign Analytics and Reporting

**User Story:** As an Operator, I want detailed analytics on campaign performance and customer engagement, so that I can optimize messaging strategy and measure business impact.

#### Acceptance Criteria

1. THE Analytics_Dashboard SHALL display per-campaign metrics: delivery rate, read rate, failure rate, and response rate as percentage values
2. THE Analytics_Dashboard SHALL display aggregate metrics across all campaigns: total messages sent (daily, weekly, monthly), average delivery rate, average read rate, and top-performing templates
3. THE Analytics_Dashboard SHALL display customer retention metrics: reactivation conversion rate (customers who renewed after receiving a campaign message), churn rate by zone, and renewal rate trends
4. WHEN an Operator selects a date range, THE Analytics_Dashboard SHALL filter all displayed metrics to the specified period
5. THE Analytics_Dashboard SHALL display zone-wise engagement breakdown showing delivery and read rates per zone_name
6. THE Analytics_Dashboard SHALL identify and highlight the top 5 performing campaign templates by read rate over the last 30 days

### Requirement 9: Database Schema Extension

**User Story:** As a developer, I want all new database tables to be created automatically through migrations, so that deployments do not require manual SQL execution.

#### Acceptance Criteria

1. WHEN the application starts, THE Campaign_Manager SHALL verify that required database tables exist and create any missing tables automatically using CREATE TABLE IF NOT EXISTS statements
2. THE database migration SHALL create the following tables: campaigns, campaign_audiences, campaign_messages, campaign_templates, customer_tags, customer_notes, customer_activity, campaign_analytics, automation_rules, and media_assets
3. THE database migration SHALL safely ALTER existing tables (whatsapp_campaign_logs, renewal_records, operator_actions) by adding new columns only if they do not already exist
4. THE Campaign_Manager SHALL use the existing MySQL connection configuration (countrylinks_user_database) and maintain utf8mb4 charset compatibility
5. THE database migration SHALL create appropriate indexes on foreign key columns, status columns, and date columns used in frequent query patterns

### Requirement 10: UI Enhancement Within Existing Theme

**User Story:** As an Operator, I want the new CRM and campaign features to integrate seamlessly with the existing WhatsApp Inbox dark theme, so that the user experience remains cohesive and professional.

#### Acceptance Criteria

1. THE Campaign_Manager UI SHALL use the existing CSS custom property system (--surface, --surface-2, --surface-3, --border, --text-1, --text-2, --text-3, --green, --green-dk, --green-lt variables) defined in whatsapp.css
2. THE Campaign_Manager UI SHALL render within the existing workspace panel system (workspace-switcher buttons) as new workspace tabs alongside Inbox, Templates, Campaigns, Flows, and Renewals
3. WHEN the Operator switches between dark and light themes, THE Campaign_Manager UI SHALL adapt using the existing :root[data-theme="dark"] CSS variable overrides
4. THE Campaign_Manager UI SHALL include: dashboard summary cards, data tables with pagination, filter panels, scheduling modals, audience builder interface, and timeline views
5. THE CRM_Panel SHALL render as a slide-out panel or tabbed section within the existing chat view area, maintaining the existing layout structure (wa-rail, wa-sidebar, wa-main)
6. THE Analytics_Dashboard SHALL display charts and visualizations using a lightweight charting library that does not conflict with the existing Bootstrap and Font Awesome dependencies

### Requirement 11: Security and Access Control

**User Story:** As an administrator, I want role-based access control and audit logging for all campaign operations, so that the system maintains accountability and prevents unauthorized actions.

#### Acceptance Criteria

1. THE Campaign_Manager SHALL enforce role-based permissions: only operators with "campaign_send" permission can approve and trigger campaign sends
2. THE Campaign_Manager SHALL log every campaign action (create, edit, approve, reject, pause, cancel, send) in the operator_actions table with operator identity, action type, target campaign ID, and timestamp
3. THE Campaign_Manager SHALL validate all API requests using the existing Flask session authentication mechanism
4. IF an unauthenticated request is made to any Campaign_Manager API endpoint, THEN THE Campaign_Manager SHALL return HTTP 401 and redirect to the login page
5. THE Campaign_Manager SHALL implement CSRF protection on all state-changing endpoints using Flask's existing session-based CSRF mechanism
6. THE Sending_Queue SHALL sanitize all template parameter values to prevent injection of control characters or malformed WhatsApp API payloads

### Requirement 12: Performance and Scalability

**User Story:** As an Operator, I want the campaign system to handle large audience segments efficiently, so that bulk operations do not degrade the system's responsiveness.

#### Acceptance Criteria

1. WHEN the Segmentation_Engine queries customers matching filter criteria, THE Segmentation_Engine SHALL use indexed database queries and return results with pagination (default 50 records per page)
2. WHEN the Campaign_Manager displays campaign lists, THE Campaign_Manager SHALL load data with server-side pagination and return results within 500 milliseconds for up to 10,000 campaign records
3. THE Sending_Queue SHALL process campaign messages in background workers separate from the main Flask application process to prevent blocking HTTP request handling
4. WHEN the Analytics_Dashboard computes aggregate metrics, THE Analytics_Dashboard SHALL use pre-computed summary tables updated periodically rather than running expensive aggregate queries on each page load
5. THE Campaign_Manager UI SHALL implement lazy loading for campaign message lists and customer activity timelines to prevent loading more than 100 records at initial render
6. THE Segmentation_Engine SHALL support audience segments containing up to 50,000 customers without query timeout

### Requirement 13: A/B Template Testing

**User Story:** As an Operator, I want to test different message templates with subsets of an audience, so that I can identify the most effective messaging before full rollout.

#### Acceptance Criteria

1. WHEN an Operator creates an A/B test campaign, THE Campaign_Manager SHALL allow selection of 2 to 4 template variants and a test audience percentage (10% to 50% of the total segment)
2. WHEN an A/B test campaign is approved, THE Sending_Queue SHALL randomly split the test audience evenly across the selected template variants and dispatch messages accordingly
3. WHEN an A/B test completes sending, THE Analytics_Dashboard SHALL display per-variant metrics: delivery rate, read rate, and response rate with statistical comparison
4. WHEN an Operator selects the winning variant after reviewing A/B results, THE Campaign_Manager SHALL create a new campaign targeting the remaining audience with the selected template
5. THE Campaign_Manager SHALL prevent sending the full campaign to the remaining audience until the Operator explicitly selects a winning variant

### Requirement 14: Media Library Management

**User Story:** As an Operator, I want a centralized media library to manage images, videos, and documents used in campaigns, so that I can reuse media assets across multiple campaigns efficiently.

#### Acceptance Criteria

1. WHEN an Operator uploads a media file (image, video, document), THE Media_Library SHALL store the file and record metadata (filename, mime type, file size, upload date, uploader name) in the media_assets table
2. THE Media_Library SHALL enforce file size limits: images up to 5MB, videos up to 16MB, documents up to 100MB in compliance with WhatsApp Business API media limits
3. WHEN an Operator selects a template that requires media in a campaign, THE Media_Library SHALL present available compatible media files filtered by the required media type
4. THE Media_Library SHALL display uploaded assets in a searchable grid view with thumbnail previews, filename, upload date, and usage count
5. IF an Operator attempts to upload a file exceeding the size limit or with an unsupported mime type, THEN THE Media_Library SHALL reject the upload and display the specific constraint violated

### Requirement 15: Omnichannel Foundation Architecture

**User Story:** As a developer, I want the campaign and CRM modules to be built with channel-agnostic abstractions, so that future messaging channels (SMS, Email, Telegram, Facebook) can be added without rewriting core logic.

#### Acceptance Criteria

1. THE Campaign_Manager SHALL structure the sending logic with a channel abstraction layer where WhatsApp is one implementation of a generic message dispatch interface
2. THE Audience_Segment definitions SHALL be stored independently of any specific messaging channel, containing only customer identifiers and filter criteria
3. THE Delivery_Log schema SHALL include a "channel" column to distinguish message delivery records by communication channel (defaulting to "whatsapp")
4. THE Analytics_Dashboard SHALL group metrics by channel, enabling future per-channel and cross-channel reporting
5. THE CRM_Panel interaction timeline SHALL label each interaction with its source channel to support a unified customer view across future messaging channels

### Requirement 16: Campaign Test Send

**User Story:** As an Operator, I want to send a test message to myself or a small group before launching a campaign, so that I can verify the template renders correctly with actual parameters.

#### Acceptance Criteria

1. WHEN an Operator clicks "Test Send" on a campaign draft, THE Campaign_Manager SHALL present a form to enter 1 to 5 test mobile numbers
2. WHEN an Operator confirms a test send, THE Sending_Queue SHALL dispatch the campaign template with sample parameters to the specified test numbers immediately, bypassing the regular queue
3. THE Campaign_Manager SHALL mark test send messages distinctly in the Delivery_Log so they are excluded from campaign analytics
4. WHEN a test send completes, THE Campaign_Manager SHALL display the delivery status of each test message to the Operator within the campaign editor view

### Requirement 17: Message Cooldown Protection

**User Story:** As an Operator, I want the system to enforce message frequency limits per customer, so that customers are not spammed and the WhatsApp Business API quality rating remains healthy.

#### Acceptance Criteria

1. WHEN the Sending_Queue prepares to dispatch a promotional message to a customer, THE Cooldown_Manager SHALL verify that the customer has not received a promotional campaign message within the preceding 72 hours
2. IF a customer has received a promotional message within the last 72 hours, THEN THE Cooldown_Manager SHALL exclude that customer from the current campaign send and log the exclusion reason as "cooldown_active"
3. THE Cooldown_Manager SHALL enforce a maximum of 2 promotional campaigns targeting the same customer within any rolling 7-day period
4. IF a campaign would exceed the 2-per-week limit for a customer, THEN THE Cooldown_Manager SHALL exclude that customer and add them to the campaign exclusion report visible to the Operator
5. WHEN the Cooldown_Manager excludes customers from a campaign due to frequency limits, THE Campaign_Manager SHALL display the excluded count and reasons in the campaign summary before and after sending
6. THE Cooldown_Manager SHALL allow transactional messages (payment confirmations, service alerts) to bypass cooldown restrictions while promotional messages remain subject to frequency enforcement
7. WHEN an Operator configures a campaign, THE Campaign_Manager SHALL display a warning if the estimated audience after cooldown exclusions drops below 50% of the original segment size

### Requirement 18: WhatsApp Quality Monitoring

**User Story:** As an Operator, I want continuous monitoring of WhatsApp Business API quality indicators, so that I can proactively protect the Meta reputation score and prevent account restrictions.

#### Acceptance Criteria

1. THE Quality_Monitor SHALL track and aggregate the following metrics per rolling 24-hour and 7-day periods: blocked user count, message send failure rate, opt-out rate, and read rate (as engagement proxy)
2. WHEN the message failure rate exceeds 5% over a rolling 24-hour period, THE Quality_Monitor SHALL generate a "quality_warning" alert and notify the assigned Operator via the Notification_Engine
3. WHEN the blocked user count increases by more than 10 within a 24-hour period, THE Quality_Monitor SHALL generate a "block_spike" alert and recommend pausing active campaigns
4. THE Quality_Monitor SHALL maintain a quality score dashboard displaying current Meta quality tier (Green, Yellow, Red) based on tracked metrics and historical trends
5. WHEN the quality score drops to "Yellow" tier, THE Quality_Monitor SHALL automatically increase the Cooldown_Manager minimum interval from 72 hours to 120 hours until the score recovers to "Green"
6. THE Quality_Monitor SHALL log all quality metric snapshots to the campaign_analytics table hourly for historical trend analysis
7. WHEN a webhook callback indicates a customer has blocked the WhatsApp Business number, THE Quality_Monitor SHALL add that customer to a permanent suppression list excluded from all future campaigns

### Requirement 19: Opt-Out System

**User Story:** As an Operator, I want an automated opt-out handling system that processes unsubscribe requests and maintains a suppression list, so that the system complies with messaging regulations and respects customer communication preferences.

#### Acceptance Criteria

1. WHEN a customer sends a message containing the keyword "STOP" (case-insensitive), THE Opt_Out_Manager SHALL add the customer's mobile number to the suppression list and send a confirmation message acknowledging the opt-out
2. THE Opt_Out_Manager SHALL recognize the following opt-out keywords: "STOP", "UNSUBSCRIBE", "OPT OUT", "CANCEL", and "DND" (all case-insensitive)
3. WHEN the Sending_Queue prepares a campaign message for dispatch, THE Opt_Out_Manager SHALL verify the recipient is not on the suppression list and exclude suppressed customers before sending
4. THE Opt_Out_Manager SHALL maintain a DND (Do Not Disturb) list that Operators can manually add customers to, with a reason field and date of addition
5. WHEN a customer sends "START" or "SUBSCRIBE" (case-insensitive), THE Opt_Out_Manager SHALL remove the customer from the suppression list and send a confirmation message acknowledging the re-subscription
6. THE CRM_Panel SHALL display the opt-out status and DND status prominently on the customer profile with the date of opt-out and source (keyword, manual, or complaint)
7. THE Analytics_Dashboard SHALL display opt-out trends: daily opt-out count, opt-out rate per campaign, and net subscriber growth or decline over configurable time periods

### Requirement 20: Template Variable Validator

**User Story:** As an Operator, I want the system to validate all template placeholder variables before sending, so that messages do not fail due to missing or mismatched parameter values.

#### Acceptance Criteria

1. WHEN a campaign is created with a selected Template, THE Template_Validator SHALL parse the template body to identify all placeholder variables (e.g., {{1}}, {{2}}, {{name}}, {{plan}})
2. WHEN a campaign reaches the approval stage, THE Template_Validator SHALL verify that every identified placeholder has a corresponding data mapping to a customer field or static value
3. IF one or more placeholders lack a data mapping, THEN THE Template_Validator SHALL block the campaign from proceeding to approval and display the specific unmapped placeholders to the Operator
4. WHEN the Template_Validator detects a mismatch between the template placeholder count and the configured parameter mappings, THE Template_Validator SHALL display an error specifying which placeholders are missing or extra
5. THE Template_Validator SHALL validate parameter data types and lengths against WhatsApp Business API constraints (maximum 1024 characters per parameter value) before dispatch
6. WHEN the Sending_Queue processes each individual message, THE Template_Validator SHALL verify that the resolved parameter values for that customer are non-null and non-empty, skipping customers with incomplete data and logging them as "skipped_invalid_params"
7. THE Campaign_Manager SHALL display a template preview with sample customer data populated into all placeholders so the Operator can visually verify the rendered message before approval

### Requirement 21: Smart Retry Categorization

**User Story:** As an Operator, I want the system to apply different retry strategies based on failure type, so that transient errors are recovered while permanent failures are handled appropriately without wasting resources.

#### Acceptance Criteria

1. WHEN a message send fails, THE Retry_Categorizer SHALL classify the error into one of the following categories: transient (timeout, rate_limit, server_error), permanent (invalid_number, unregistered_number), or suppression (user_blocked, spam_reported)
2. WHEN a failure is classified as "transient", THE Sending_Queue SHALL retry the message up to 3 times with exponential backoff (5 seconds, 15 seconds, 45 seconds)
3. WHEN a failure is classified as "permanent" (invalid or unregistered number), THE Retry_Categorizer SHALL mark the message as "permanently_failed", flag the customer record with "invalid_whatsapp_number", and exclude the customer from all future WhatsApp campaigns
4. WHEN a failure is classified as "suppression" (user blocked or reported spam), THE Retry_Categorizer SHALL add the customer to the permanent suppression list maintained by the Quality_Monitor and exclude them from all future campaign sends
5. THE Retry_Categorizer SHALL maintain an error classification lookup table mapping WhatsApp Business API error codes to failure categories, updatable by administrators without code changes
6. THE Analytics_Dashboard SHALL display failure breakdown by category per campaign: transient failures recovered, permanent failures, and suppression events with trend analysis
7. WHEN more than 20% of messages in an active campaign fail with "suppression" classification, THE Retry_Categorizer SHALL trigger an automatic campaign pause and generate a critical alert to the Operator

### Requirement 22: Campaign Simulation Mode

**User Story:** As an Operator, I want to simulate a campaign before approval to see projected audience count, estimated costs, and potential issues, so that I can make informed decisions before committing to a send.

#### Acceptance Criteria

1. WHEN an Operator clicks "Simulate Campaign" on a draft or scheduled campaign, THE Simulation_Engine SHALL compute and display: final audience count after all exclusions, estimated total send time based on current throttle rate, and estimated WhatsApp API cost based on per-message pricing
2. THE Simulation_Engine SHALL calculate exclusions by applying: cooldown-protected customers, opt-out/DND suppressed customers, customers with invalid numbers, and customers with incomplete template parameters
3. THE Simulation_Engine SHALL detect and display the count of potential duplicate recipients (customers appearing in overlapping segments or previously messaged with the same template within 7 days)
4. THE Simulation_Engine SHALL display a breakdown of excluded customers by exclusion reason: cooldown (count), opted-out (count), DND (count), invalid number (count), and incomplete data (count)
5. WHEN the simulation reveals that more than 30% of the original segment would be excluded, THE Simulation_Engine SHALL display a prominent warning to the Operator recommending segment review
6. THE Simulation_Engine SHALL estimate API cost using configurable per-message rates (utility, marketing, authentication categories) stored in system configuration
7. THE Simulation_Engine SHALL complete simulation computation within 5 seconds for audience segments containing up to 50,000 customers

### Requirement 23: AI Layer Preparation

**User Story:** As a developer, I want the system to store engagement data and interaction patterns in a structured format, so that future AI features (smart segmentation, campaign suggestions, predictive churn, auto-retention scoring) can be built on a rich data foundation.

#### Acceptance Criteria

1. THE Engagement_Store SHALL persist per-customer engagement metrics after each campaign: messages received count, messages read count, response count, average time-to-read, and last interaction timestamp
2. THE Engagement_Store SHALL compute and store a customer interaction score (0-100) based on weighted factors: message read rate (40%), response rate (30%), recency of last interaction (20%), and campaign participation frequency (10%)
3. THE Engagement_Store SHALL record response pattern data: average response time, common response keywords, preferred communication time windows (morning, afternoon, evening), and engagement trend (increasing, stable, declining)
4. THE Engagement_Store SHALL store campaign performance patterns: template effectiveness scores by segment type, optimal send time correlations, and seasonal engagement variations
5. THE Engagement_Store SHALL update customer interaction scores after each campaign completion using a batch computation process that runs within 60 seconds for up to 50,000 customer records
6. THE Engagement_Store SHALL expose engagement data through queryable views that enable future AI modules to retrieve training datasets without modifying the core schema
7. THE CRM_Panel SHALL display the customer interaction score and engagement trend on the customer profile as a visual health indicator

### Requirement 24: Multi-Tenant Preparedness

**User Story:** As a developer, I want the database schema and application architecture to support multi-tenant isolation, so that the platform can later serve multiple ISPs, franchise operators, and reseller panels without structural rewrites.

#### Acceptance Criteria

1. THE database migration SHALL include an organization_id column (with default value for current single-tenant operation) on all campaign, segment, template, and analytics tables
2. THE database migration SHALL include a branch_id column on customer-facing tables (campaigns, campaign_messages, customer_tags, customer_notes) to support future franchise-level data isolation
3. THE database migration SHALL include a tenant_id column on system configuration and access control tables to enable future reseller panel separation
4. THE Campaign_Manager SHALL scope all database queries with the current organization_id to ensure data isolation, defaulting to the single organization until multi-tenant activation
5. THE Campaign_Manager SHALL structure API routes and service layers to accept tenant context as a parameter, defaulting to the current organization context for backward compatibility
6. THE Tenant_Framework columns (organization_id, branch_id, tenant_id) SHALL use indexed BIGINT fields with NOT NULL constraints and default values that maintain full backward compatibility with existing single-tenant operation

### Requirement 25: Internal Notification System

**User Story:** As an Operator, I want to receive immediate alerts when critical system events occur (campaign failures, queue overloads, Meta issues), so that I can take corrective action before customers are impacted.

#### Acceptance Criteria

1. WHEN a campaign send fails for more than 10% of recipients, THE Notification_Engine SHALL generate a "campaign_degraded" alert and deliver it to all operators with "campaign_send" permission
2. WHEN the Sending_Queue backlog exceeds 10,000 pending messages and processing rate drops below 50% of configured throughput, THE Notification_Engine SHALL generate a "queue_overloaded" alert
3. WHEN the system fails to receive Meta webhook callbacks for more than 5 minutes during an active campaign send, THE Notification_Engine SHALL generate a "webhook_connectivity" alert
4. WHEN a Template submission is rejected by Meta's template review process, THE Notification_Engine SHALL generate a "template_rejected" alert including the template name and rejection reason
5. WHEN the Quality_Monitor detects a quality tier drop (Green to Yellow, or Yellow to Red), THE Notification_Engine SHALL generate a "quality_drop" alert with current metrics and recommended corrective actions
6. THE Notification_Engine SHALL deliver alerts through: in-app notification panel (persistent until acknowledged), browser push notification, and optionally a WhatsApp message to the operator's personal number
7. THE Notification_Engine SHALL maintain a notification log with: event type, severity (info, warning, critical), timestamp, delivery status, and acknowledgment status per operator

### Requirement 26: Campaign Recovery System

**User Story:** As an Operator, I want campaign queue state to persist through server restarts, so that active campaigns resume automatically without message loss or duplicate sends.

#### Acceptance Criteria

1. THE Recovery_Manager SHALL persist all queued message records to the database (campaign_messages table) with status tracking before attempting dispatch, ensuring queue state survives process termination
2. WHEN the application restarts after an unexpected shutdown, THE Recovery_Manager SHALL query the campaign_messages table for records in "queued" or "sending" status and re-enqueue them for processing within 30 seconds of application startup
3. THE Recovery_Manager SHALL use message-level idempotency keys (campaign_id + customer_mobile + template_id) to prevent duplicate sends during recovery by checking delivery status before re-dispatching
4. WHEN a campaign was in "sending" state at the time of shutdown, THE Recovery_Manager SHALL transition the campaign back to "sending" state and resume dispatching from the last confirmed sent position
5. THE Recovery_Manager SHALL log all recovery actions: messages re-queued count, duplicates prevented count, campaigns resumed count, and total recovery time in the system audit log
6. IF the Recovery_Manager detects messages that were in "sending" status for more than 5 minutes without a delivery confirmation, THEN THE Recovery_Manager SHALL reset those messages to "queued" status for re-dispatch
7. THE Recovery_Manager SHALL complete the full recovery process (identify pending work, deduplicate, resume sending) within 60 seconds of application startup for up to 100,000 pending messages

