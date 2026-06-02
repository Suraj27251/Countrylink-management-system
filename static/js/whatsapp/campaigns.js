/**
 * Campaign Manager Panel - Full Campaign Management UI
 * Fetches data from /api/campaigns/ and /api/segments/ endpoints.
 * Provides dashboard, campaign creation, segments management, and A/B testing.
 */

const CAMP_API = '/api/campaigns';
const SEG_API = '/api/segments';

const campState = {
  view: 'dashboard', // dashboard | create | segments | abtest | analytics
  campaigns: [],
  segments: [],
  total: 0,
  page: 1,
  perPage: 20,
  totalPages: 0,
  loaded: false,
  loading: false,
};

// ─── Initialization ────────────────────────────────────────────────────────────

function initCampaignsPanel() {
  if (!campState.loaded) {
    campState.loaded = true;
  }
  campShowDashboard();
}

// ─── Views ─────────────────────────────────────────────────────────────────────

function campShowDashboard() {
  campState.view = 'dashboard';
  const container = document.getElementById('campaignsPanelContent');
  if (!container) return;

  container.innerHTML = `
    <div class="camp-header">
      <h3><i class="fas fa-bullhorn"></i> Campaign Manager</h3>
      <div class="camp-actions">
        <button class="camp-btn camp-btn-primary" onclick="campShowCreate()">
          <i class="fas fa-plus"></i> New Campaign
        </button>
        <button class="camp-btn camp-btn-outline" onclick="campShowApprovalQueue()">
          <i class="fas fa-clipboard-check"></i> Approval Queue
        </button>
        <button class="camp-btn camp-btn-outline" onclick="campShowSegments()">
          <i class="fas fa-users"></i> Manage Segments
        </button>
        <button class="camp-btn camp-btn-outline" onclick="campShowABTest()">
          <i class="fas fa-flask"></i> A/B Test
        </button>
        <button class="camp-btn camp-btn-outline" onclick="campShowAnalytics()">
          <i class="fas fa-chart-line"></i> Analytics
        </button>
      </div>
    </div>

    <!-- Summary Cards -->
    <div class="camp-stats" id="campStatsRow">
      <div class="camp-stat-card">
        <div class="camp-stat-value" id="campStatTotal">-</div>
        <div class="camp-stat-label">Total Campaigns</div>
      </div>
      <div class="camp-stat-card camp-stat-active">
        <div class="camp-stat-value" id="campStatActive">-</div>
        <div class="camp-stat-label">Active Sends</div>
      </div>
      <div class="camp-stat-card camp-stat-delivery">
        <div class="camp-stat-value" id="campStatDelivery">-</div>
        <div class="camp-stat-label">Avg Delivery Rate</div>
      </div>
      <div class="camp-stat-card camp-stat-pending">
        <div class="camp-stat-value" id="campStatPending">-</div>
        <div class="camp-stat-label">Pending Approval</div>
      </div>
    </div>

    <!-- Campaign List Table -->
    <div class="camp-table-wrap">
      <table class="camp-table">
        <thead>
          <tr>
            <th>Name</th>
            <th>Type</th>
            <th>Status</th>
            <th>Recipients</th>
            <th>Sent</th>
            <th>Delivered</th>
            <th>Created</th>
            <th></th>
          </tr>
        </thead>
        <tbody id="campTableBody">
          <tr><td colspan="8" class="camp-loading">Loading campaigns...</td></tr>
        </tbody>
      </table>
    </div>

    <!-- Pagination -->
    <div class="camp-pagination" id="campPagination"></div>
  `;

  campLoadDashboardData();
}

async function campLoadDashboardData() {
  campState.loading = true;
  try {
    // Fetch all campaigns for stats
    const statsRes = await fetch(`${CAMP_API}/?per_page=100&page=1`, { credentials: 'same-origin' });
    if (!statsRes.ok) {
      const errText = await statsRes.text().catch(() => '');
      console.error('[CAMPAIGNS] API error:', statsRes.status, errText);
      const tbody = document.getElementById('campTableBody');
      if (tbody) tbody.innerHTML = `<tr><td colspan="8" class="camp-loading">API Error (${statsRes.status}): ${statsRes.status === 401 ? 'Please log in again.' : 'Server error — check if database migration ran.'}</td></tr>`;
      campState.loading = false;
      return;
    }
    const statsData = statsRes.ok ? await statsRes.json() : { campaigns: [], total: 0 };

    const allCampaigns = statsData.campaigns || [];
    const total = statsData.total || allCampaigns.length;
    const activeSends = allCampaigns.filter(c => c.status === 'sending').length;
    const pendingApproval = allCampaigns.filter(c => c.status === 'pending_approval').length;

    // Calculate avg delivery rate from completed campaigns
    const completed = allCampaigns.filter(c => c.status === 'completed');
    let avgDelivery = 0;
    if (completed.length > 0) {
      const rates = completed.map(c => {
        const sent = c.sent_count || c.total_recipients || 1;
        const delivered = c.delivered_count || 0;
        return sent > 0 ? (delivered / sent) * 100 : 0;
      });
      avgDelivery = rates.reduce((a, b) => a + b, 0) / rates.length;
    }

    document.getElementById('campStatTotal').textContent = total;
    document.getElementById('campStatActive').textContent = activeSends;
    document.getElementById('campStatDelivery').textContent = avgDelivery > 0 ? avgDelivery.toFixed(1) + '%' : 'N/A';
    document.getElementById('campStatPending').textContent = pendingApproval;

    // Fetch paginated list for table
    const listRes = await fetch(`${CAMP_API}/?page=${campState.page}&per_page=${campState.perPage}`, { credentials: 'same-origin' });
    const listData = listRes.ok ? await listRes.json() : { campaigns: [], total: 0, total_pages: 0 };

    campState.campaigns = listData.campaigns || [];
    campState.total = listData.total || 0;
    campState.totalPages = listData.total_pages || 0;

    campRenderTable();
    campRenderPagination();
  } catch (err) {
    console.error('[CAMPAIGNS] Failed to load dashboard data:', err);
    const tbody = document.getElementById('campTableBody');
    if (tbody) tbody.innerHTML = `<tr><td colspan="8" class="camp-loading">
      <div style="text-align:center;">
        <p style="font-weight:600;margin-bottom:8px;">Unable to load campaigns</p>
        <p style="font-size:12px;color:var(--text-3);">
          ${err.message || 'Network or server error'}. 
          Check that the server has restarted after deployment (database migration runs on startup).
        </p>
        <button class="camp-btn camp-btn-outline" onclick="campLoadDashboardData()" style="margin-top:12px;">
          <i class="fas fa-sync-alt"></i> Retry
        </button>
      </div>
    </td></tr>`;
  }
  campState.loading = false;
}

function campRenderTable() {
  const tbody = document.getElementById('campTableBody');
  if (!tbody) return;

  if (campState.campaigns.length === 0) {
    tbody.innerHTML = '<tr><td colspan="8" class="camp-loading">No campaigns found. Create your first campaign!</td></tr>';
    return;
  }

  tbody.innerHTML = campState.campaigns.map(c => {
    const statusBadge = campStatusBadge(c.status);
    const created = c.created_at ? new Date(c.created_at).toLocaleDateString() : '-';
    const recipients = c.total_recipients || c.recipient_count || '-';
    const sent = c.sent_count || '-';
    const delivered = c.delivered_count || '-';
    const type = c.campaign_type || 'promotional';

    return `
      <tr>
        <td class="camp-cell-name">${escHtml(c.name || 'Untitled')}</td>
        <td><span class="camp-type-badge">${escHtml(type)}</span></td>
        <td>${statusBadge}</td>
        <td>${recipients}</td>
        <td>${sent}</td>
        <td>${delivered}</td>
        <td>${created}</td>
        <td class="camp-cell-actions">
          <button class="camp-action-btn" onclick="campViewDetails(${c.id})" title="View Details">
            <i class="fas fa-eye"></i>
          </button>
          <button class="camp-action-btn" onclick="campDuplicate(${c.id})" title="Duplicate">
            <i class="fas fa-copy"></i>
          </button>
          ${c.status === 'draft' || c.status === 'scheduled' ? `
            <button class="camp-action-btn camp-action-approve" onclick="campSubmitForApproval(${c.id})" title="Submit for Approval">
              <i class="fas fa-paper-plane"></i>
            </button>
          ` : ''}
          ${c.status === 'pending_approval' ? `
            <button class="camp-action-btn camp-action-approve" onclick="campShowApprovalPreview(${c.id})" title="Review & Approve">
              <i class="fas fa-clipboard-check"></i>
            </button>
          ` : ''}
          ${c.status === 'sending' || c.status === 'paused' ? `
            <button class="camp-action-btn" onclick="campShowCampaignProgress(${c.id})" title="View Progress">
              <i class="fas fa-satellite-dish"></i>
            </button>
          ` : ''}
          ${c.campaign_type === 'ab_test' ? `
            <button class="camp-action-btn" onclick="campShowABResults(${c.id})" title="A/B Results">
              <i class="fas fa-flask"></i>
            </button>
          ` : ''}
        </td>
      </tr>
    `;
  }).join('');
}

function campRenderPagination() {
  const el = document.getElementById('campPagination');
  if (!el) return;

  if (campState.totalPages <= 1) {
    el.innerHTML = '';
    return;
  }

  let html = '';
  html += `<button class="camp-page-btn" ${campState.page <= 1 ? 'disabled' : ''} onclick="campGoToPage(${campState.page - 1})"><i class="fas fa-chevron-left"></i></button>`;

  for (let i = 1; i <= campState.totalPages; i++) {
    if (i === campState.page) {
      html += `<button class="camp-page-btn active">${i}</button>`;
    } else if (i <= 3 || i >= campState.totalPages - 1 || Math.abs(i - campState.page) <= 1) {
      html += `<button class="camp-page-btn" onclick="campGoToPage(${i})">${i}</button>`;
    } else if (i === 4 && campState.page > 5) {
      html += `<span class="camp-page-ellipsis">...</span>`;
    }
  }

  html += `<button class="camp-page-btn" ${campState.page >= campState.totalPages ? 'disabled' : ''} onclick="campGoToPage(${campState.page + 1})"><i class="fas fa-chevron-right"></i></button>`;
  el.innerHTML = html;
}

function campGoToPage(page) {
  if (page < 1 || page > campState.totalPages) return;
  campState.page = page;
  campLoadDashboardData();
}

// ─── Status Badge ──────────────────────────────────────────────────────────────

function campStatusBadge(status) {
  const colors = {
    draft: { bg: 'rgba(156,163,175,0.15)', color: 'var(--text-3)' },
    scheduled: { bg: 'rgba(99,102,241,0.15)', color: '#818cf8' },
    pending_approval: { bg: 'rgba(245,158,11,0.15)', color: 'var(--amber)' },
    approved: { bg: 'rgba(16,185,129,0.15)', color: '#10b981' },
    sending: { bg: 'rgba(59,130,246,0.15)', color: 'var(--blue)' },
    paused: { bg: 'rgba(245,158,11,0.15)', color: 'var(--amber)' },
    completed: { bg: 'rgba(16,185,129,0.15)', color: 'var(--green)' },
    failed: { bg: 'rgba(239,68,68,0.15)', color: 'var(--red)' },
    cancelled: { bg: 'rgba(156,163,175,0.15)', color: 'var(--text-3)' },
  };
  const s = colors[status] || colors.draft;
  const label = (status || 'unknown').replace(/_/g, ' ');
  return `<span class="camp-status-badge" style="background:${s.bg};color:${s.color}">${label}</span>`;
}

// ─── Campaign Actions ──────────────────────────────────────────────────────────

async function campViewDetails(id) {
  try {
    const res = await fetch(`${CAMP_API}/${id}`);
    if (!res.ok) throw new Error('Failed to load campaign');
    const campaign = await res.json();

    const container = document.getElementById('campaignsPanelContent');
    if (!container) return;

    const isActive = campaign.status === 'sending';
    const isPaused = campaign.status === 'paused';
    const isPending = campaign.status === 'pending_approval';
    const isABTest = campaign.campaign_type === 'ab_test';
    const isFinished = ['completed', 'cancelled', 'failed'].includes(campaign.status);

    container.innerHTML = `
      <div class="camp-header">
        <button class="camp-btn camp-btn-ghost" onclick="campShowDashboard()">
          <i class="fas fa-arrow-left"></i> Back
        </button>
        <h3>${escHtml(campaign.name || 'Campaign Details')}</h3>
      </div>
      <div class="camp-detail-grid">
        <div class="camp-detail-card">
          <h5>Status</h5>
          <p>${campStatusBadge(campaign.status)}</p>
        </div>
        <div class="camp-detail-card">
          <h5>Type</h5>
          <p>${escHtml(campaign.campaign_type || 'promotional')}</p>
        </div>
        <div class="camp-detail-card">
          <h5>Channel</h5>
          <p>${escHtml(campaign.channel || 'whatsapp')}</p>
        </div>
        <div class="camp-detail-card">
          <h5>Priority</h5>
          <p>${campaign.priority || 5}</p>
        </div>
        <div class="camp-detail-card">
          <h5>Created By</h5>
          <p>${escHtml(campaign.created_by || '-')}</p>
        </div>
        <div class="camp-detail-card">
          <h5>Created At</h5>
          <p>${campaign.created_at ? new Date(campaign.created_at).toLocaleString() : '-'}</p>
        </div>
        <div class="camp-detail-card full-width">
          <h5>Description</h5>
          <p>${escHtml(campaign.description || 'No description')}</p>
        </div>
      </div>

      <div class="camp-detail-actions">
        ${campaign.status === 'draft' || campaign.status === 'scheduled' ? `
          <button class="camp-btn camp-btn-primary" onclick="campSubmitForApproval(${campaign.id})">
            <i class="fas fa-paper-plane"></i> Submit for Approval
          </button>
          <button class="camp-btn camp-btn-outline" onclick="campDuplicate(${campaign.id})">
            <i class="fas fa-copy"></i> Duplicate
          </button>
        ` : ''}
        ${isPending ? `
          <button class="camp-btn camp-btn-success" onclick="campShowApprovalPreview(${campaign.id})">
            <i class="fas fa-clipboard-check"></i> Review & Approve
          </button>
        ` : ''}
        ${(isActive || isPaused) ? `
          <button class="camp-btn camp-btn-primary" onclick="campShowCampaignProgress(${campaign.id})">
            <i class="fas fa-satellite-dish"></i> View Progress
          </button>
        ` : ''}
        ${isActive ? `
          <button class="camp-btn camp-btn-warning" onclick="campPauseCampaign(${campaign.id})">
            <i class="fas fa-pause"></i> Pause
          </button>
        ` : ''}
        ${isPaused ? `
          <button class="camp-btn camp-btn-primary" onclick="campResumeCampaign(${campaign.id})">
            <i class="fas fa-play"></i> Resume
          </button>
        ` : ''}
        ${(isActive || isPaused) ? `
          <button class="camp-btn camp-btn-danger" onclick="campCancelCampaign(${campaign.id})">
            <i class="fas fa-ban"></i> Cancel
          </button>
        ` : ''}
        ${isABTest ? `
          <button class="camp-btn camp-btn-outline" onclick="campShowABResults(${campaign.id})">
            <i class="fas fa-flask"></i> A/B Results
          </button>
        ` : ''}
        ${isFinished ? `
          <button class="camp-btn camp-btn-outline" onclick="campShowCampaignProgress(${campaign.id})">
            <i class="fas fa-chart-bar"></i> View Final Stats
          </button>
        ` : ''}
      </div>
    `;
  } catch (err) {
    console.error('[CAMPAIGNS] Failed to load campaign details:', err);
    campShowToast('Failed to load campaign details', 'error');
  }
}

async function campDuplicate(id) {
  try {
    const res = await fetch(`${CAMP_API}/${id}/duplicate`, { method: 'POST' });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.error || 'Failed to duplicate');
    }
    campShowToast('Campaign duplicated successfully', 'success');
    campShowDashboard();
  } catch (err) {
    console.error('[CAMPAIGNS] Duplicate failed:', err);
    campShowToast(err.message || 'Failed to duplicate campaign', 'error');
  }
}

async function campSubmitForApproval(id) {
  if (!confirm('Submit this campaign for approval?')) return;
  try {
    const res = await fetch(`${CAMP_API}/${id}/submit-for-approval`, { method: 'POST' });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.error || 'Failed to submit');
    }
    campShowToast('Campaign submitted for approval', 'success');
    campShowDashboard();
  } catch (err) {
    console.error('[CAMPAIGNS] Submit for approval failed:', err);
    campShowToast(err.message || 'Failed to submit for approval', 'error');
  }
}

// ─── Create Campaign Form ──────────────────────────────────────────────────────

function campShowCreate() {
  campState.view = 'create';
  const container = document.getElementById('campaignsPanelContent');
  if (!container) return;

  container.innerHTML = `
    <div class="camp-header">
      <button class="camp-btn camp-btn-ghost" onclick="campShowDashboard()">
        <i class="fas fa-arrow-left"></i> Back
      </button>
      <h3>Create New Campaign</h3>
    </div>
    <form class="camp-form" id="campCreateForm" onsubmit="campHandleCreate(event)">
      <div class="camp-form-group">
        <label for="campName">Campaign Name *</label>
        <input type="text" id="campName" required placeholder="e.g., March Recharge Reminder" class="camp-input">
      </div>
      <div class="camp-form-row">
        <div class="camp-form-group">
          <label for="campType">Campaign Type</label>
          <select id="campType" class="camp-input">
            <option value="promotional">Promotional</option>
            <option value="transactional">Transactional</option>
            <option value="reactivation">Reactivation</option>
            <option value="informational">Informational</option>
          </select>
        </div>
        <div class="camp-form-group">
          <label for="campPriority">Priority</label>
          <select id="campPriority" class="camp-input">
            <option value="1">High (1)</option>
            <option value="5" selected>Normal (5)</option>
            <option value="10">Low (10)</option>
          </select>
        </div>
      </div>
      <div class="camp-form-group">
        <label for="campSegment">Target Audience Segment</label>
        <select id="campSegment" class="camp-input">
          <option value="">-- Select Segment --</option>
        </select>
        <small class="camp-hint" id="campSegmentHint">Or build a custom audience below</small>
      </div>

      <!-- Audience Builder Filters -->
      <div class="camp-audience-builder">
        <label class="camp-audience-builder-label"><i class="fas fa-filter"></i> Audience Builder</label>
        <div class="camp-filter-grid">
          <div class="camp-form-group">
            <label for="campFilterStatus">Category</label>
            <select id="campFilterStatus" class="camp-input camp-filter-input" onchange="campOnFiltersChanged()">
              <option value="">Any</option>
              <option value="expired">Expired</option>
              <option value="today">Expiring Today</option>
              <option value="upcoming">Upcoming</option>
            </select>
              <option value="inactive">Inactive</option>
              <option value="disconnected">Disconnected</option>
            </select>
          </div>
          <div class="camp-form-group">
            <label for="campFilterExpiry">Expiry Category</label>
            <select id="campFilterExpiry" class="camp-input camp-filter-input" onchange="campOnFiltersChanged()">
              <option value="">Any</option>
              <option value="expired">Expired</option>
              <option value="today">Expiring Today</option>
              <option value="upcoming">Upcoming</option>
            </select>
          </div>
          <div class="camp-form-group">
            <label for="campFilterZone">Zone</label>
            <input type="text" id="campFilterZone" class="camp-input camp-filter-input" placeholder="e.g., North Zone" oninput="campOnFiltersChanged()">
          </div>
          <div class="camp-form-group">
            <label for="campFilterPlan">Plan Name</label>
            <input type="text" id="campFilterPlan" class="camp-input camp-filter-input" placeholder="e.g., 100Mbps" oninput="campOnFiltersChanged()">
          </div>
          <div class="camp-form-group">
            <label for="campFilterNetwork">Network Type</label>
            <input type="text" id="campFilterNetwork" class="camp-input camp-filter-input" placeholder="e.g., fiber" oninput="campOnFiltersChanged()">
          </div>
          <div class="camp-form-group">
            <label>Days Remaining</label>
            <div class="camp-filter-range">
              <input type="number" id="campFilterDaysMin" class="camp-input camp-filter-input" placeholder="Min" oninput="campOnFiltersChanged()">
              <span class="camp-filter-range-sep">–</span>
              <input type="number" id="campFilterDaysMax" class="camp-input camp-filter-input" placeholder="Max" oninput="campOnFiltersChanged()">
            </div>
          </div>
        </div>
        <div class="camp-audience-estimate" id="campAudienceEstimate">Add filters to see audience estimate</div>
      </div>

      <div class="camp-form-group">
        <label for="campTemplate">Message Template</label>
        <select id="campTemplate" class="camp-input">
          <option value="">-- Select Template --</option>
        </select>
      </div>
      <div class="camp-form-group">
        <label for="campDesc">Description</label>
        <textarea id="campDesc" class="camp-input" rows="3" placeholder="Optional campaign description"></textarea>
      </div>

      <!-- Scheduling -->
      <div class="camp-form-group">
        <label>Schedule</label>
        <button type="button" class="camp-btn camp-btn-outline" onclick="campShowScheduleModal(null)">
          <i class="fas fa-clock"></i> Set Schedule
        </button>
        <div class="camp-schedule-display" id="campScheduleDisplay" style="display:none;"></div>
        <input type="hidden" id="campScheduledAt" value="">
        <input type="hidden" id="campRecurringFrequency" value="none">
        <input type="hidden" id="campRecurringEndDate" value="">
      </div>

      <div class="camp-form-actions">
        <button type="submit" class="camp-btn camp-btn-primary">
          <i class="fas fa-save"></i> Create Campaign
        </button>
        <button type="button" class="camp-btn camp-btn-outline" onclick="campShowDashboard()">Cancel</button>
      </div>
    </form>

    <!-- Simulation Panel (visible after campaign is created) -->
    <div id="campSimulationPanel" class="camp-simulation-panel" style="display:none;"></div>
  `;

  campLoadSegmentsForSelect();
  campLoadTemplatesForSelect();
}

async function campLoadSegmentsForSelect() {
  try {
    const res = await fetch(`${SEG_API}/?per_page=100`);
    if (!res.ok) return;
    const data = await res.json();
    const segments = data.segments || data.items || [];
    const select = document.getElementById('campSegment');
    if (!select) return;

    segments.forEach(seg => {
      const opt = document.createElement('option');
      opt.value = seg.id;
      opt.textContent = seg.name || `Segment #${seg.id}`;
      select.appendChild(opt);
    });
  } catch (err) {
    console.error('[CAMPAIGNS] Failed to load segments:', err);
  }
}

async function campLoadTemplatesForSelect() {
  try {
    const res = await fetch('/api/whatsapp/templates');
    if (!res.ok) return;
    const data = await res.json();
    const templates = data.data || data.templates || [];
    const select = document.getElementById('campTemplate');
    if (!select) return;

    templates.forEach(tpl => {
      const opt = document.createElement('option');
      opt.value = tpl.id || tpl.name;
      opt.textContent = tpl.name || tpl.template_name || `Template #${tpl.id}`;
      if (tpl.status && tpl.status !== 'APPROVED') {
        opt.textContent += ` (${tpl.status})`;
      }
      select.appendChild(opt);
    });
  } catch (err) {
    console.error('[CAMPAIGNS] Failed to load templates:', err);
  }
}

async function campHandleCreate(e) {
  e.preventDefault();
  const name = document.getElementById('campName')?.value?.trim();
  if (!name) return campShowToast('Campaign name is required', 'error');

  const payload = {
    name,
    campaign_type: document.getElementById('campType')?.value || 'promotional',
    priority: parseInt(document.getElementById('campPriority')?.value || '5', 10),
    segment_id: document.getElementById('campSegment')?.value || null,
    template_id: document.getElementById('campTemplate')?.value || null,
    description: document.getElementById('campDesc')?.value?.trim() || '',
  };

  // Convert empty strings to null
  if (!payload.segment_id) payload.segment_id = null;
  if (!payload.template_id) payload.template_id = null;

  // Include scheduling if set
  const scheduledAt = document.getElementById('campScheduledAt')?.value;
  if (scheduledAt) {
    payload.scheduled_at = scheduledAt;
    payload.recurring_frequency = document.getElementById('campRecurringFrequency')?.value || 'none';
    const recurEnd = document.getElementById('campRecurringEndDate')?.value;
    if (recurEnd) payload.recurring_end_date = recurEnd;
  }

  try {
    const res = await fetch(`${CAMP_API}/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.error || 'Failed to create campaign');
    }
    const created = await res.json();
    const campaignId = created.id || created.campaign_id;

    campShowToast('Campaign created successfully!', 'success');

    // Show post-creation actions (test send & simulation)
    if (campaignId) {
      campShowPostCreateActions(campaignId);
    } else {
      campShowDashboard();
    }
  } catch (err) {
    console.error('[CAMPAIGNS] Create failed:', err);
    campShowToast(err.message || 'Failed to create campaign', 'error');
  }
}

/**
 * After campaign creation, show test send & simulation action buttons.
 */
function campShowPostCreateActions(campaignId) {
  const form = document.getElementById('campCreateForm');
  if (form) {
    form.style.display = 'none';
  }

  const container = document.getElementById('campaignsPanelContent');
  if (!container) return;

  // Insert post-create panel after the form
  const actionsPanel = document.createElement('div');
  actionsPanel.className = 'camp-post-create';
  actionsPanel.innerHTML = `
    <div class="camp-post-create-header">
      <i class="fas fa-check-circle" style="color:var(--green);font-size:20px;"></i>
      <h4>Campaign Created</h4>
    </div>
    <p class="camp-post-create-desc">Your campaign has been saved as a draft. You can now test it or run a simulation.</p>
    <div class="camp-post-create-actions">
      <button class="camp-btn camp-btn-primary" onclick="campShowTestSendModal(${campaignId})">
        <i class="fas fa-paper-plane"></i> Test Send
      </button>
      <button class="camp-btn camp-btn-outline" onclick="campRunSimulation(${campaignId})">
        <i class="fas fa-chart-bar"></i> Run Simulation
      </button>
      <button class="camp-btn camp-btn-outline" onclick="campShowScheduleModal(${campaignId})">
        <i class="fas fa-clock"></i> Schedule
      </button>
      <button class="camp-btn camp-btn-ghost" onclick="campShowDashboard()">
        <i class="fas fa-arrow-left"></i> Back to Dashboard
      </button>
    </div>
    <div id="campSimulationPanel" class="camp-simulation-panel" style="display:none;"></div>
  `;
  container.appendChild(actionsPanel);
}

// ─── Segments View ─────────────────────────────────────────────────────────────

function campShowSegments() {
  campState.view = 'segments';
  const container = document.getElementById('campaignsPanelContent');
  if (!container) return;

  container.innerHTML = `
    <div class="camp-header">
      <button class="camp-btn camp-btn-ghost" onclick="campShowDashboard()">
        <i class="fas fa-arrow-left"></i> Back
      </button>
      <h3><i class="fas fa-users"></i> Audience Segments</h3>
    </div>
    <div class="camp-table-wrap">
      <table class="camp-table">
        <thead>
          <tr>
            <th>Name</th>
            <th>Description</th>
            <th>Created By</th>
            <th>Created At</th>
          </tr>
        </thead>
        <tbody id="campSegmentsBody">
          <tr><td colspan="4" class="camp-loading">Loading segments...</td></tr>
        </tbody>
      </table>
    </div>
  `;

  campLoadSegmentsList();
}

async function campLoadSegmentsList() {
  try {
    const res = await fetch(`${SEG_API}/?per_page=50`);
    if (!res.ok) throw new Error('Failed to load segments');
    const data = await res.json();
    const segments = data.segments || data.items || [];

    const tbody = document.getElementById('campSegmentsBody');
    if (!tbody) return;

    if (segments.length === 0) {
      tbody.innerHTML = '<tr><td colspan="4" class="camp-loading">No segments saved yet.</td></tr>';
      return;
    }

    tbody.innerHTML = segments.map(seg => `
      <tr>
        <td class="camp-cell-name">${escHtml(seg.name || 'Unnamed')}</td>
        <td>${escHtml(seg.description || '-')}</td>
        <td>${escHtml(seg.created_by || '-')}</td>
        <td>${seg.created_at ? new Date(seg.created_at).toLocaleDateString() : '-'}</td>
      </tr>
    `).join('');
  } catch (err) {
    console.error('[CAMPAIGNS] Failed to load segments:', err);
    const tbody = document.getElementById('campSegmentsBody');
    if (tbody) tbody.innerHTML = '<tr><td colspan="4" class="camp-loading">Failed to load segments.</td></tr>';
  }
}

// ─── A/B Test View ─────────────────────────────────────────────────────────────

function campShowABTest() {
  campState.view = 'abtest';
  const container = document.getElementById('campaignsPanelContent');
  if (!container) return;

  container.innerHTML = `
    <div class="camp-header">
      <button class="camp-btn camp-btn-ghost" onclick="campShowDashboard()">
        <i class="fas fa-arrow-left"></i> Back
      </button>
      <h3><i class="fas fa-flask"></i> A/B Template Test</h3>
    </div>
    <div class="camp-ab-info">
      <p>A/B testing lets you send different template variants to subsets of your audience and compare performance metrics.</p>
    </div>
    <form class="camp-form" id="campABForm" onsubmit="campHandleABTest(event)">
      <div class="camp-form-group">
        <label for="campABCampaign">Select Campaign (must be in draft)</label>
        <select id="campABCampaign" class="camp-input" required>
          <option value="">-- Select Campaign --</option>
        </select>
      </div>
      <div class="camp-form-row">
        <div class="camp-form-group">
          <label for="campABVariantA">Variant A Template</label>
          <select id="campABVariantA" class="camp-input">
            <option value="">-- Select Template --</option>
          </select>
        </div>
        <div class="camp-form-group">
          <label for="campABVariantB">Variant B Template</label>
          <select id="campABVariantB" class="camp-input">
            <option value="">-- Select Template --</option>
          </select>
        </div>
      </div>
      <div class="camp-form-group">
        <label for="campABPercent">Test Audience Percentage</label>
        <input type="range" id="campABPercent" min="10" max="50" value="20" class="camp-range"
               oninput="document.getElementById('campABPercentVal').textContent = this.value + '%'">
        <span id="campABPercentVal" class="camp-hint">20%</span>
      </div>
      <div class="camp-form-actions">
        <button type="submit" class="camp-btn camp-btn-primary">
          <i class="fas fa-play"></i> Create A/B Test
        </button>
        <button type="button" class="camp-btn camp-btn-outline" onclick="campShowDashboard()">Cancel</button>
      </div>
    </form>
  `;

  campLoadDraftCampaigns();
  campLoadABTemplates();
}

async function campLoadDraftCampaigns() {
  try {
    const res = await fetch(`${CAMP_API}/?per_page=50`);
    if (!res.ok) return;
    const data = await res.json();
    const drafts = (data.campaigns || []).filter(c => c.status === 'draft');
    const select = document.getElementById('campABCampaign');
    if (!select) return;

    drafts.forEach(c => {
      const opt = document.createElement('option');
      opt.value = c.id;
      opt.textContent = c.name || `Campaign #${c.id}`;
      select.appendChild(opt);
    });
  } catch (err) {
    console.error('[CAMPAIGNS] Failed to load drafts for AB:', err);
  }
}

async function campLoadABTemplates() {
  try {
    const res = await fetch('/api/whatsapp/templates');
    if (!res.ok) return;
    const data = await res.json();
    const templates = data.data || data.templates || [];

    ['campABVariantA', 'campABVariantB'].forEach(selectId => {
      const select = document.getElementById(selectId);
      if (!select) return;
      templates.forEach(tpl => {
        const opt = document.createElement('option');
        opt.value = tpl.id || tpl.name;
        opt.textContent = tpl.name || tpl.template_name || `Template #${tpl.id}`;
        select.appendChild(opt);
      });
    });
  } catch (err) {
    console.error('[CAMPAIGNS] Failed to load AB templates:', err);
  }
}

async function campHandleABTest(e) {
  e.preventDefault();
  const campaignId = document.getElementById('campABCampaign')?.value;
  const variantA = document.getElementById('campABVariantA')?.value;
  const variantB = document.getElementById('campABVariantB')?.value;
  const percent = parseInt(document.getElementById('campABPercent')?.value || '20', 10);

  if (!campaignId) return campShowToast('Select a campaign', 'error');
  if (!variantA || !variantB) return campShowToast('Select both template variants', 'error');
  if (variantA === variantB) return campShowToast('Variants must be different templates', 'error');

  try {
    const res = await fetch(`${CAMP_API}/${campaignId}/ab-test`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        template_variants: [parseInt(variantA, 10), parseInt(variantB, 10)],
        test_percentage: percent,
      }),
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.error || 'Failed to create A/B test');
    }
    campShowToast('A/B test created successfully!', 'success');
    campShowDashboard();
  } catch (err) {
    console.error('[CAMPAIGNS] AB test creation failed:', err);
    campShowToast(err.message || 'Failed to create A/B test', 'error');
  }
}

// ─── Analytics View ────────────────────────────────────────────────────────────

function campShowAnalytics() {
  campState.view = 'analytics';
  const container = document.getElementById('campaignsPanelContent');
  if (!container) return;

  container.innerHTML = `
    <div class="camp-header">
      <button class="camp-btn camp-btn-ghost" onclick="campShowDashboard()">
        <i class="fas fa-arrow-left"></i> Back
      </button>
      <h3><i class="fas fa-chart-line"></i> Campaign Analytics</h3>
    </div>
    <div class="camp-analytics-grid" id="campAnalyticsGrid">
      <div class="camp-loading">Loading analytics...</div>
    </div>
  `;

  campLoadAnalytics();
}

async function campLoadAnalytics() {
  try {
    const res = await fetch(`${CAMP_API}/?per_page=100`);
    if (!res.ok) throw new Error('Failed to load data');
    const data = await res.json();
    const campaigns = data.campaigns || [];

    const completed = campaigns.filter(c => c.status === 'completed');
    const totalSent = campaigns.reduce((acc, c) => acc + (c.sent_count || 0), 0);
    const totalDelivered = campaigns.reduce((acc, c) => acc + (c.delivered_count || 0), 0);
    const totalFailed = campaigns.reduce((acc, c) => acc + (c.failed_count || 0), 0);
    const deliveryRate = totalSent > 0 ? ((totalDelivered / totalSent) * 100).toFixed(1) : '0';
    const failureRate = totalSent > 0 ? ((totalFailed / totalSent) * 100).toFixed(1) : '0';

    const grid = document.getElementById('campAnalyticsGrid');
    if (!grid) return;

    grid.innerHTML = `
      <div class="camp-analytics-card">
        <div class="camp-analytics-value">${campaigns.length}</div>
        <div class="camp-analytics-label">Total Campaigns</div>
      </div>
      <div class="camp-analytics-card">
        <div class="camp-analytics-value">${completed.length}</div>
        <div class="camp-analytics-label">Completed</div>
      </div>
      <div class="camp-analytics-card">
        <div class="camp-analytics-value">${totalSent.toLocaleString()}</div>
        <div class="camp-analytics-label">Messages Sent</div>
      </div>
      <div class="camp-analytics-card">
        <div class="camp-analytics-value">${totalDelivered.toLocaleString()}</div>
        <div class="camp-analytics-label">Messages Delivered</div>
      </div>
      <div class="camp-analytics-card">
        <div class="camp-analytics-value">${deliveryRate}%</div>
        <div class="camp-analytics-label">Delivery Rate</div>
      </div>
      <div class="camp-analytics-card">
        <div class="camp-analytics-value">${failureRate}%</div>
        <div class="camp-analytics-label">Failure Rate</div>
      </div>

      <div class="camp-analytics-section full-width">
        <h4>Recent Completed Campaigns</h4>
        <table class="camp-table camp-table-sm">
          <thead>
            <tr>
              <th>Campaign</th>
              <th>Sent</th>
              <th>Delivered</th>
              <th>Failed</th>
              <th>Delivery %</th>
            </tr>
          </thead>
          <tbody>
            ${completed.length === 0 ? '<tr><td colspan="5" class="camp-loading">No completed campaigns yet.</td></tr>' :
              completed.slice(0, 10).map(c => {
                const s = c.sent_count || 0;
                const d = c.delivered_count || 0;
                const f = c.failed_count || 0;
                const rate = s > 0 ? ((d / s) * 100).toFixed(1) : '0';
                return `<tr>
                  <td>${escHtml(c.name || 'Untitled')}</td>
                  <td>${s}</td>
                  <td>${d}</td>
                  <td>${f}</td>
                  <td>${rate}%</td>
                </tr>`;
              }).join('')
            }
          </tbody>
        </table>
      </div>
    `;
  } catch (err) {
    console.error('[CAMPAIGNS] Failed to load analytics:', err);
    const grid = document.getElementById('campAnalyticsGrid');
    if (grid) grid.innerHTML = '<div class="camp-loading">Failed to load analytics data.</div>';
  }
}

// ─── Audience Estimation ────────────────────────────────────────────────────────

/**
 * Debounce timer for audience estimation requests.
 */
let _campEstimateTimer = null;

/**
 * Fetch real-time audience count estimate from POST /api/segments/estimate.
 * Called when audience filters change in the campaign creation form.
 * Debounced to avoid rapid fire requests (waits 400ms after last change).
 *
 * @param {Object} filters - Filter criteria matching the segmentation engine format
 */
function campEstimateAudience(filters) {
  clearTimeout(_campEstimateTimer);
  const display = document.getElementById('campAudienceEstimate');
  if (display) {
    display.textContent = 'Estimating...';
    display.className = 'camp-audience-estimate camp-audience-loading';
  }

  _campEstimateTimer = setTimeout(async () => {
    try {
      const res = await fetch(`${SEG_API}/estimate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(filters),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.error || 'Estimation failed');
      }
      const data = await res.json();
      const count = data.count || 0;
      if (display) {
        display.textContent = `${count.toLocaleString()} recipient${count !== 1 ? 's' : ''}`;
        display.className = 'camp-audience-estimate' + (count === 0 ? ' camp-audience-zero' : ' camp-audience-ok');
      }
      if (data.warning && display) {
        display.textContent += ` — ${data.warning}`;
      }
    } catch (err) {
      console.error('[CAMPAIGNS] Audience estimation failed:', err);
      if (display) {
        display.textContent = 'Unable to estimate';
        display.className = 'camp-audience-estimate camp-audience-error';
      }
    }
  }, 400);
}

/**
 * Build filter criteria from the audience builder form fields and trigger estimation.
 */
function campOnFiltersChanged() {
  const filters = {};

  const status = document.getElementById('campFilterStatus')?.value;
  if (status) filters.category = status;

  const zone = document.getElementById('campFilterZone')?.value?.trim();
  if (zone) filters.zone_name = zone;

  const plan = document.getElementById('campFilterPlan')?.value?.trim();
  if (plan) filters.plan_name = plan;

  const expiry = document.getElementById('campFilterExpiry')?.value;
  if (expiry) filters.expiry_category = expiry;

  const daysMin = document.getElementById('campFilterDaysMin')?.value;
  const daysMax = document.getElementById('campFilterDaysMax')?.value;
  if (daysMin || daysMax) {
    filters.days_remaining = {};
    if (daysMin) filters.days_remaining.min = parseInt(daysMin, 10);
    if (daysMax) filters.days_remaining.max = parseInt(daysMax, 10);
  }

  const networkType = document.getElementById('campFilterNetwork')?.value?.trim();
  if (networkType) filters.network_type = networkType;

  // Only call estimate if at least one filter is set
  if (Object.keys(filters).length > 0) {
    campEstimateAudience(filters);
  } else {
    const display = document.getElementById('campAudienceEstimate');
    if (display) {
      display.textContent = 'Add filters to see audience estimate';
      display.className = 'camp-audience-estimate';
    }
  }
}

// ─── Scheduling Modal ──────────────────────────────────────────────────────────

/**
 * Show the scheduling modal for a campaign.
 * Allows picking a date/time and optional recurring frequency.
 *
 * @param {number|null} campaignId - If provided, schedules an existing campaign. Otherwise stores for creation.
 */
function campShowScheduleModal(campaignId = null) {
  // Remove existing modal if present
  document.getElementById('campScheduleModal')?.remove();

  const now = new Date();
  const minDate = now.toISOString().slice(0, 16); // YYYY-MM-DDTHH:MM

  const modal = document.createElement('div');
  modal.id = 'campScheduleModal';
  modal.className = 'camp-modal-overlay';
  modal.innerHTML = `
    <div class="camp-modal">
      <div class="camp-modal-header">
        <h4><i class="fas fa-clock"></i> Schedule Campaign</h4>
        <button class="camp-modal-close" onclick="campCloseScheduleModal()">&times;</button>
      </div>
      <div class="camp-modal-body">
        <div class="camp-form-group">
          <label for="campScheduleDate">Send Date & Time *</label>
          <input type="datetime-local" id="campScheduleDate" class="camp-input" min="${minDate}" required>
        </div>
        <div class="camp-form-group">
          <label for="campScheduleRecurring">Recurring Frequency</label>
          <select id="campScheduleRecurring" class="camp-input">
            <option value="none">One-time (no repeat)</option>
            <option value="daily">Daily</option>
            <option value="weekly">Weekly</option>
            <option value="monthly">Monthly</option>
          </select>
        </div>
        <div class="camp-form-group" id="campScheduleEndGroup" style="display:none;">
          <label for="campScheduleEnd">Recurring End Date</label>
          <input type="date" id="campScheduleEnd" class="camp-input">
        </div>
      </div>
      <div class="camp-modal-footer">
        <button class="camp-btn camp-btn-primary" onclick="campConfirmSchedule(${campaignId})">
          <i class="fas fa-check"></i> Confirm Schedule
        </button>
        <button class="camp-btn camp-btn-outline" onclick="campCloseScheduleModal()">Cancel</button>
      </div>
    </div>
  `;
  document.body.appendChild(modal);

  // Show/hide end date based on recurring selection
  document.getElementById('campScheduleRecurring').addEventListener('change', (e) => {
    const endGroup = document.getElementById('campScheduleEndGroup');
    if (endGroup) {
      endGroup.style.display = e.target.value !== 'none' ? 'flex' : 'none';
    }
  });
}

function campCloseScheduleModal() {
  document.getElementById('campScheduleModal')?.remove();
}

/**
 * Confirm schedule. If campaignId is given, call the schedule API.
 * Otherwise, populate the hidden fields in the create form.
 */
async function campConfirmSchedule(campaignId) {
  const dateVal = document.getElementById('campScheduleDate')?.value;
  if (!dateVal) {
    campShowToast('Please select a date and time', 'error');
    return;
  }

  const scheduledAt = new Date(dateVal);
  if (scheduledAt <= new Date()) {
    campShowToast('Schedule time must be in the future', 'error');
    return;
  }

  const recurring = document.getElementById('campScheduleRecurring')?.value || 'none';
  const endDate = document.getElementById('campScheduleEnd')?.value || null;

  if (campaignId) {
    // Schedule an existing campaign via API
    try {
      const res = await fetch(`${CAMP_API}/${campaignId}/schedule`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          scheduled_at: dateVal,
          recurring_frequency: recurring,
          recurring_end_date: endDate,
        }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error || 'Failed to schedule');
      }
      campShowToast('Campaign scheduled successfully', 'success');
      campCloseScheduleModal();
      campShowDashboard();
    } catch (err) {
      console.error('[CAMPAIGNS] Schedule failed:', err);
      campShowToast(err.message || 'Failed to schedule campaign', 'error');
    }
  } else {
    // Store schedule data for the create form
    const schedDisplay = document.getElementById('campScheduleDisplay');
    if (schedDisplay) {
      const formatted = scheduledAt.toLocaleString();
      const recurLabel = recurring !== 'none' ? ` (${recurring})` : '';
      schedDisplay.textContent = `Scheduled: ${formatted}${recurLabel}`;
      schedDisplay.style.display = 'block';
    }
    // Store values in hidden inputs
    const hiddenDate = document.getElementById('campScheduledAt');
    if (hiddenDate) hiddenDate.value = dateVal;
    const hiddenRecurring = document.getElementById('campRecurringFrequency');
    if (hiddenRecurring) hiddenRecurring.value = recurring;
    const hiddenEnd = document.getElementById('campRecurringEndDate');
    if (hiddenEnd) hiddenEnd.value = endDate || '';

    campCloseScheduleModal();
  }
}

// ─── Test Send Modal ───────────────────────────────────────────────────────────

/**
 * Show a modal to send test messages to 1-5 mobile numbers.
 *
 * @param {number} campaignId - The campaign ID to test send
 */
function campShowTestSendModal(campaignId) {
  if (!campaignId) {
    campShowToast('Save the campaign first before sending a test', 'error');
    return;
  }

  // Remove existing modal
  document.getElementById('campTestSendModal')?.remove();

  const modal = document.createElement('div');
  modal.id = 'campTestSendModal';
  modal.className = 'camp-modal-overlay';
  modal.innerHTML = `
    <div class="camp-modal">
      <div class="camp-modal-header">
        <h4><i class="fas fa-paper-plane"></i> Test Send</h4>
        <button class="camp-modal-close" onclick="campCloseTestSendModal()">&times;</button>
      </div>
      <div class="camp-modal-body">
        <p class="camp-modal-desc">Send a test message to 1–5 mobile numbers to verify the template renders correctly.</p>
        <div class="camp-form-group">
          <label>Test Numbers (one per line)</label>
          <textarea id="campTestNumbers" class="camp-input" rows="5" placeholder="919876543210&#10;919876543211&#10;919876543212"></textarea>
          <small class="camp-hint">Enter up to 5 numbers, including country code (e.g. 91xxxxxxxxxx)</small>
        </div>
        <div id="campTestResults" class="camp-test-results" style="display:none;"></div>
      </div>
      <div class="camp-modal-footer">
        <button class="camp-btn camp-btn-primary" id="campTestSendBtn" onclick="campExecuteTestSend(${campaignId})">
          <i class="fas fa-paper-plane"></i> Send Test
        </button>
        <button class="camp-btn camp-btn-outline" onclick="campCloseTestSendModal()">Close</button>
      </div>
    </div>
  `;
  document.body.appendChild(modal);
}

function campCloseTestSendModal() {
  document.getElementById('campTestSendModal')?.remove();
}

/**
 * Execute the test send by calling POST /api/campaigns/<id>/test-send.
 *
 * @param {number} campaignId - The campaign to test
 */
async function campExecuteTestSend(campaignId) {
  const textarea = document.getElementById('campTestNumbers');
  const btn = document.getElementById('campTestSendBtn');
  const resultsDiv = document.getElementById('campTestResults');

  if (!textarea) return;

  const numbers = textarea.value
    .split(/[\n,;]+/)
    .map(n => n.trim())
    .filter(n => n.length > 0);

  if (numbers.length === 0) {
    campShowToast('Enter at least one phone number', 'error');
    return;
  }
  if (numbers.length > 5) {
    campShowToast('Maximum 5 test numbers allowed', 'error');
    return;
  }

  // Disable button during send
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Sending...';
  }

  try {
    const res = await fetch(`${CAMP_API}/${campaignId}/test-send`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ test_numbers: numbers }),
    });

    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.error || 'Test send failed');
    }

    // Show results
    if (resultsDiv) {
      resultsDiv.style.display = 'block';
      const results = data.test_results || [];
      resultsDiv.innerHTML = `
        <h5 class="camp-test-results-title">
          Results: ${data.success_count || 0} sent, ${data.failed_count || 0} failed
        </h5>
        <div class="camp-test-results-list">
          ${results.map(r => `
            <div class="camp-test-result-item camp-test-${r.status}">
              <span class="camp-test-number">${escHtml(r.mobile || r.number || '')}</span>
              <span class="camp-test-status">${r.status === 'sent' ? '<i class="fas fa-check-circle"></i> Sent' : '<i class="fas fa-times-circle"></i> Failed'}</span>
              ${r.error ? `<span class="camp-test-error">${escHtml(r.error)}</span>` : ''}
            </div>
          `).join('')}
        </div>
      `;
    }

    campShowToast(`Test sent: ${data.success_count} success, ${data.failed_count} failed`, data.failed_count > 0 ? 'error' : 'success');
  } catch (err) {
    console.error('[CAMPAIGNS] Test send failed:', err);
    campShowToast(err.message || 'Test send failed', 'error');
    if (resultsDiv) {
      resultsDiv.style.display = 'block';
      resultsDiv.innerHTML = `<p class="camp-test-error-msg">${escHtml(err.message)}</p>`;
    }
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = '<i class="fas fa-paper-plane"></i> Send Test';
    }
  }
}

// ─── Simulation Results ────────────────────────────────────────────────────────

/**
 * Run a pre-send simulation for a campaign and display the results.
 *
 * @param {number} campaignId - The campaign to simulate
 */
async function campRunSimulation(campaignId) {
  if (!campaignId) {
    campShowToast('Save the campaign first before running a simulation', 'error');
    return;
  }

  const panel = document.getElementById('campSimulationPanel');
  if (panel) {
    panel.style.display = 'block';
    panel.innerHTML = `
      <div class="camp-sim-loading">
        <i class="fas fa-spinner fa-spin"></i> Running simulation...
      </div>
    `;
  }

  try {
    const res = await fetch(`${CAMP_API}/${campaignId}/simulate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    });

    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.error || 'Simulation failed');
    }

    campRenderSimulationResults(data);
  } catch (err) {
    console.error('[CAMPAIGNS] Simulation failed:', err);
    if (panel) {
      panel.innerHTML = `
        <div class="camp-sim-error">
          <i class="fas fa-exclamation-triangle"></i> ${escHtml(err.message || 'Simulation failed')}
        </div>
      `;
    }
    campShowToast(err.message || 'Simulation failed', 'error');
  }
}

/**
 * Render simulation results in the simulation panel.
 *
 * @param {Object} data - The simulation response from the API
 */
function campRenderSimulationResults(data) {
  const panel = document.getElementById('campSimulationPanel');
  if (!panel) return;

  const sendTimeMins = Math.ceil((data.estimated_send_time_seconds || 0) / 60);
  const exclusions = data.exclusions || {};
  const warnings = data.warnings || [];
  const cost = data.estimated_cost_inr || 0;

  panel.innerHTML = `
    <div class="camp-sim-results">
      <h5 class="camp-sim-title"><i class="fas fa-chart-bar"></i> Simulation Results</h5>
      <div class="camp-sim-stats">
        <div class="camp-sim-stat">
          <span class="camp-sim-stat-value">${(data.final_audience_count || 0).toLocaleString()}</span>
          <span class="camp-sim-stat-label">Final Audience</span>
        </div>
        <div class="camp-sim-stat">
          <span class="camp-sim-stat-value">${sendTimeMins} min</span>
          <span class="camp-sim-stat-label">Estimated Time</span>
        </div>
        <div class="camp-sim-stat">
          <span class="camp-sim-stat-value">₹${cost.toFixed(2)}</span>
          <span class="camp-sim-stat-label">Estimated Cost</span>
        </div>
        <div class="camp-sim-stat">
          <span class="camp-sim-stat-value">${data.duplicate_count || 0}</span>
          <span class="camp-sim-stat-label">Duplicates</span>
        </div>
      </div>

      ${exclusions.total > 0 ? `
        <div class="camp-sim-exclusions">
          <h6>Exclusions Breakdown</h6>
          <div class="camp-sim-excl-grid">
            ${exclusions.cooldown ? `<div class="camp-sim-excl-item"><span>${exclusions.cooldown}</span> Cooldown</div>` : ''}
            ${exclusions.opted_out ? `<div class="camp-sim-excl-item"><span>${exclusions.opted_out}</span> Opted Out</div>` : ''}
            ${exclusions.dnd ? `<div class="camp-sim-excl-item"><span>${exclusions.dnd}</span> DND</div>` : ''}
            ${exclusions.invalid_number ? `<div class="camp-sim-excl-item"><span>${exclusions.invalid_number}</span> Invalid Number</div>` : ''}
            ${exclusions.incomplete_data ? `<div class="camp-sim-excl-item"><span>${exclusions.incomplete_data}</span> Incomplete Data</div>` : ''}
          </div>
        </div>
      ` : ''}

      ${warnings.length > 0 ? `
        <div class="camp-sim-warnings">
          ${warnings.map(w => `<div class="camp-sim-warning"><i class="fas fa-exclamation-triangle"></i> ${escHtml(w)}</div>`).join('')}
        </div>
      ` : ''}

      <div class="camp-sim-original">
        Original audience: ${(data.original_audience_count || 0).toLocaleString()} &rarr; Final: ${(data.final_audience_count || 0).toLocaleString()}
      </div>
    </div>
  `;
}

// ─── Approval Workflow ──────────────────────────────────────────────────────────

/**
 * Show the pending approval queue view listing campaigns awaiting operator approval.
 * Requirements: 2.1, 2.4
 */
function campShowApprovalQueue() {
  campState.view = 'approval';
  const container = document.getElementById('campaignsPanelContent');
  if (!container) return;

  container.innerHTML = `
    <div class="camp-header">
      <button class="camp-btn camp-btn-ghost" onclick="campShowDashboard()">
        <i class="fas fa-arrow-left"></i> Back
      </button>
      <h3><i class="fas fa-clipboard-check"></i> Pending Approval Queue</h3>
    </div>
    <div class="camp-table-wrap">
      <table class="camp-table">
        <thead>
          <tr>
            <th>Campaign</th>
            <th>Type</th>
            <th>Recipients</th>
            <th>Created By</th>
            <th>Submitted</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody id="campApprovalBody">
          <tr><td colspan="6" class="camp-loading">Loading pending campaigns...</td></tr>
        </tbody>
      </table>
    </div>
  `;

  campLoadApprovalQueue();
}

async function campLoadApprovalQueue() {
  try {
    const res = await fetch(`${CAMP_API}/?per_page=100&page=1`);
    if (!res.ok) throw new Error('Failed to load campaigns');
    const data = await res.json();
    const pending = (data.campaigns || []).filter(c => c.status === 'pending_approval');

    const tbody = document.getElementById('campApprovalBody');
    if (!tbody) return;

    if (pending.length === 0) {
      tbody.innerHTML = '<tr><td colspan="6" class="camp-loading">No campaigns pending approval.</td></tr>';
      return;
    }

    tbody.innerHTML = pending.map(c => {
      const created = c.created_at ? new Date(c.created_at).toLocaleString() : '-';
      const recipients = c.total_recipients || '-';
      const type = c.campaign_type || 'promotional';
      return `
        <tr>
          <td class="camp-cell-name">${escHtml(c.name || 'Untitled')}</td>
          <td><span class="camp-type-badge">${escHtml(type)}</span></td>
          <td>${recipients}</td>
          <td>${escHtml(c.created_by || '-')}</td>
          <td>${created}</td>
          <td class="camp-cell-actions">
            <button class="camp-btn camp-btn-sm camp-btn-success" onclick="campShowApprovalPreview(${c.id})" title="Review & Approve">
              <i class="fas fa-check"></i> Review
            </button>
          </td>
        </tr>
      `;
    }).join('');
  } catch (err) {
    console.error('[CAMPAIGNS] Failed to load approval queue:', err);
    const tbody = document.getElementById('campApprovalBody');
    if (tbody) tbody.innerHTML = '<tr><td colspan="6" class="camp-loading">Failed to load pending campaigns.</td></tr>';
  }
}

/**
 * Show approval preview with campaign summary and approve/reject buttons.
 * Calls GET /api/campaigns/<id>/preview for preview data.
 * Requirements: 2.4
 */
async function campShowApprovalPreview(campaignId) {
  const container = document.getElementById('campaignsPanelContent');
  if (!container) return;

  container.innerHTML = `
    <div class="camp-header">
      <button class="camp-btn camp-btn-ghost" onclick="campShowApprovalQueue()">
        <i class="fas fa-arrow-left"></i> Back to Queue
      </button>
      <h3><i class="fas fa-eye"></i> Campaign Approval Review</h3>
    </div>
    <div class="camp-loading" id="campApprovalPreviewContent">Loading preview...</div>
  `;

  try {
    // Fetch campaign details and preview in parallel
    const [campRes, previewRes] = await Promise.all([
      fetch(`${CAMP_API}/${campaignId}`),
      fetch(`${CAMP_API}/${campaignId}/preview`)
    ]);

    const campaign = campRes.ok ? await campRes.json() : null;
    const preview = previewRes.ok ? await previewRes.json() : null;

    if (!campaign) throw new Error('Failed to load campaign');

    const previewContent = document.getElementById('campApprovalPreviewContent');
    if (!previewContent) return;

    const recipientCount = preview?.recipient_count || campaign.total_recipients || '-';
    const templateContent = preview?.template_content || preview?.template_body || 'N/A';
    const estimatedTime = preview?.estimated_time || '-';
    const sampleParams = preview?.sample_params || null;

    previewContent.innerHTML = `
      <div class="camp-detail-grid">
        <div class="camp-detail-card">
          <h5>Campaign Name</h5>
          <p>${escHtml(campaign.name || 'Untitled')}</p>
        </div>
        <div class="camp-detail-card">
          <h5>Type</h5>
          <p>${escHtml(campaign.campaign_type || 'promotional')}</p>
        </div>
        <div class="camp-detail-card">
          <h5>Estimated Recipients</h5>
          <p><strong>${recipientCount}</strong></p>
        </div>
        <div class="camp-detail-card">
          <h5>Estimated Delivery Time</h5>
          <p>${estimatedTime}</p>
        </div>
        <div class="camp-detail-card">
          <h5>Channel</h5>
          <p>${escHtml(campaign.channel || 'whatsapp')}</p>
        </div>
        <div class="camp-detail-card">
          <h5>Priority</h5>
          <p>${campaign.priority || 5}</p>
        </div>
        <div class="camp-detail-card full-width">
          <h5>Description</h5>
          <p>${escHtml(campaign.description || 'No description provided')}</p>
        </div>
        <div class="camp-detail-card full-width">
          <h5>Template Content Preview</h5>
          <div class="camp-template-preview">${escHtml(templateContent)}</div>
        </div>
        ${sampleParams ? `
          <div class="camp-detail-card full-width">
            <h5>Sample Parameters</h5>
            <pre class="camp-code-preview">${escHtml(JSON.stringify(sampleParams, null, 2))}</pre>
          </div>
        ` : ''}
      </div>

      <div class="camp-approval-actions">
        <button class="camp-btn camp-btn-success camp-btn-lg" onclick="campApproveCampaign(${campaignId})">
          <i class="fas fa-check-circle"></i> Approve & Send
        </button>
        <button class="camp-btn camp-btn-danger camp-btn-lg" onclick="campRejectCampaign(${campaignId})">
          <i class="fas fa-times-circle"></i> Reject
        </button>
      </div>
    `;
    previewContent.classList.remove('camp-loading');
  } catch (err) {
    console.error('[CAMPAIGNS] Failed to load approval preview:', err);
    const previewContent = document.getElementById('campApprovalPreviewContent');
    if (previewContent) {
      previewContent.innerHTML = '<p class="camp-error">Failed to load campaign preview. Please try again.</p>';
      previewContent.classList.remove('camp-loading');
    }
  }
}

/**
 * Approve a campaign — POST /api/campaigns/<id>/approve
 * Requirements: 2.2
 */
async function campApproveCampaign(campaignId) {
  if (!confirm('Approve this campaign and begin sending messages?')) return;
  try {
    const res = await fetch(`${CAMP_API}/${campaignId}/approve`, { method: 'POST' });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.error || 'Failed to approve campaign');
    }
    campShowToast('Campaign approved! Sending has begun.', 'success');
    campShowCampaignProgress(campaignId);
  } catch (err) {
    console.error('[CAMPAIGNS] Approve failed:', err);
    campShowToast(err.message || 'Failed to approve campaign', 'error');
  }
}

/**
 * Reject a campaign — POST /api/campaigns/<id>/reject
 * Requirements: 2.3
 */
async function campRejectCampaign(campaignId) {
  const reason = prompt('Please provide a rejection reason:');
  if (reason === null) return; // User cancelled
  try {
    const res = await fetch(`${CAMP_API}/${campaignId}/reject`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reason: reason || 'No reason provided' }),
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.error || 'Failed to reject campaign');
    }
    campShowToast('Campaign rejected.', 'info');
    campShowApprovalQueue();
  } catch (err) {
    console.error('[CAMPAIGNS] Reject failed:', err);
    campShowToast(err.message || 'Failed to reject campaign', 'error');
  }
}

// ─── Real-Time Campaign Progress ───────────────────────────────────────────────

let _campProgressInterval = null;

/**
 * Show real-time campaign progress view with sent/delivered/failed counts.
 * Polls GET /api/campaigns/<id> every 5 seconds to update progress.
 * Requirements: 4.6
 */
function campShowCampaignProgress(campaignId) {
  campState.view = 'progress';
  campStopProgressPolling();

  const container = document.getElementById('campaignsPanelContent');
  if (!container) return;

  container.innerHTML = `
    <div class="camp-header">
      <button class="camp-btn camp-btn-ghost" onclick="campStopProgressPolling(); campShowDashboard()">
        <i class="fas fa-arrow-left"></i> Back
      </button>
      <h3><i class="fas fa-satellite-dish"></i> Campaign Progress</h3>
      <span class="camp-live-badge"><i class="fas fa-circle camp-pulse"></i> Live</span>
    </div>
    <div id="campProgressContent" class="camp-loading">Loading campaign progress...</div>
  `;

  // Initial load and then poll every 5 seconds
  campUpdateProgress(campaignId);
  _campProgressInterval = setInterval(() => campUpdateProgress(campaignId), 5000);
}

async function campUpdateProgress(campaignId) {
  try {
    const res = await fetch(`${CAMP_API}/${campaignId}`);
    if (!res.ok) throw new Error('Failed to fetch campaign');
    const campaign = await res.json();

    const content = document.getElementById('campProgressContent');
    if (!content) { campStopProgressPolling(); return; }

    const total = campaign.total_recipients || 0;
    const sent = campaign.sent_count || 0;
    const delivered = campaign.delivered_count || 0;
    const failed = campaign.failed_count || 0;
    const read = campaign.read_count || 0;
    const remaining = Math.max(0, total - sent - failed);
    const progressPct = total > 0 ? Math.round(((sent + failed) / total) * 100) : 0;

    const isActive = campaign.status === 'sending';
    const isPaused = campaign.status === 'paused';
    const isFinished = ['completed', 'cancelled', 'failed'].includes(campaign.status);

    // Stop polling if campaign is finished
    if (isFinished) {
      campStopProgressPolling();
    }

    content.innerHTML = `
      <div class="camp-progress-header">
        <h4>${escHtml(campaign.name || 'Campaign')}</h4>
        ${campStatusBadge(campaign.status)}
      </div>

      <div class="camp-progress-bar-wrap">
        <div class="camp-progress-bar">
          <div class="camp-progress-fill" style="width:${progressPct}%"></div>
        </div>
        <span class="camp-progress-pct">${progressPct}%</span>
      </div>

      <div class="camp-progress-stats">
        <div class="camp-progress-stat">
          <div class="camp-progress-stat-value">${total.toLocaleString()}</div>
          <div class="camp-progress-stat-label">Total Recipients</div>
        </div>
        <div class="camp-progress-stat camp-stat-sent">
          <div class="camp-progress-stat-value">${sent.toLocaleString()}</div>
          <div class="camp-progress-stat-label">Sent</div>
        </div>
        <div class="camp-progress-stat camp-stat-delivered">
          <div class="camp-progress-stat-value">${delivered.toLocaleString()}</div>
          <div class="camp-progress-stat-label">Delivered</div>
        </div>
        <div class="camp-progress-stat camp-stat-read">
          <div class="camp-progress-stat-value">${read.toLocaleString()}</div>
          <div class="camp-progress-stat-label">Read</div>
        </div>
        <div class="camp-progress-stat camp-stat-failed">
          <div class="camp-progress-stat-value">${failed.toLocaleString()}</div>
          <div class="camp-progress-stat-label">Failed</div>
        </div>
        <div class="camp-progress-stat">
          <div class="camp-progress-stat-value">${remaining.toLocaleString()}</div>
          <div class="camp-progress-stat-label">Remaining</div>
        </div>
      </div>

      ${!isFinished ? `
        <div class="camp-progress-controls">
          ${isActive ? `
            <button class="camp-btn camp-btn-warning" onclick="campPauseCampaign(${campaignId})">
              <i class="fas fa-pause"></i> Pause
            </button>
          ` : ''}
          ${isPaused ? `
            <button class="camp-btn camp-btn-primary" onclick="campResumeCampaign(${campaignId})">
              <i class="fas fa-play"></i> Resume
            </button>
          ` : ''}
          ${(isActive || isPaused) ? `
            <button class="camp-btn camp-btn-danger" onclick="campCancelCampaign(${campaignId})">
              <i class="fas fa-ban"></i> Cancel
            </button>
          ` : ''}
        </div>
      ` : `
        <div class="camp-progress-controls">
          <button class="camp-btn camp-btn-outline" onclick="campShowDashboard()">
            <i class="fas fa-arrow-left"></i> Back to Dashboard
          </button>
        </div>
      `}
    `;
    content.classList.remove('camp-loading');
  } catch (err) {
    console.error('[CAMPAIGNS] Progress fetch failed:', err);
  }
}

function campStopProgressPolling() {
  if (_campProgressInterval) {
    clearInterval(_campProgressInterval);
    _campProgressInterval = null;
  }
}

// ─── Campaign Controls (Pause / Resume / Cancel) ───────────────────────────────

/**
 * Pause a sending campaign — POST /api/campaigns/<id>/transition {new_state: 'paused'}
 * Requirements: 1.5
 */
async function campPauseCampaign(campaignId) {
  if (!confirm('Pause this campaign? No new messages will be sent until resumed.')) return;
  try {
    const res = await fetch(`${CAMP_API}/${campaignId}/transition`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ new_state: 'paused' }),
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.error || 'Failed to pause campaign');
    }
    campShowToast('Campaign paused.', 'info');
    campUpdateProgress(campaignId);
  } catch (err) {
    console.error('[CAMPAIGNS] Pause failed:', err);
    campShowToast(err.message || 'Failed to pause campaign', 'error');
  }
}

/**
 * Resume a paused campaign — POST /api/campaigns/<id>/transition {new_state: 'sending'}
 * Requirements: 1.6
 */
async function campResumeCampaign(campaignId) {
  if (!confirm('Resume this campaign? Message sending will continue.')) return;
  try {
    const res = await fetch(`${CAMP_API}/${campaignId}/transition`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ new_state: 'sending' }),
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.error || 'Failed to resume campaign');
    }
    campShowToast('Campaign resumed!', 'success');
    campUpdateProgress(campaignId);
  } catch (err) {
    console.error('[CAMPAIGNS] Resume failed:', err);
    campShowToast(err.message || 'Failed to resume campaign', 'error');
  }
}

/**
 * Cancel a campaign — POST /api/campaigns/<id>/transition {new_state: 'cancelled'}
 * Requirements: 1.7
 */
async function campCancelCampaign(campaignId) {
  if (!confirm('Cancel this campaign? All unsent messages will be discarded. This cannot be undone.')) return;
  try {
    const res = await fetch(`${CAMP_API}/${campaignId}/transition`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ new_state: 'cancelled' }),
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.error || 'Failed to cancel campaign');
    }
    campShowToast('Campaign cancelled.', 'info');
    campUpdateProgress(campaignId);
  } catch (err) {
    console.error('[CAMPAIGNS] Cancel failed:', err);
    campShowToast(err.message || 'Failed to cancel campaign', 'error');
  }
}

// ─── A/B Test Results Comparison View ──────────────────────────────────────────

/**
 * Show A/B test results comparison view for a campaign.
 * Fetches variant metrics from GET /api/campaigns/<id>/ab-test/variants
 * Requirements: 13.3, 13.4
 */
async function campShowABResults(campaignId) {
  campState.view = 'ab_results';
  const container = document.getElementById('campaignsPanelContent');
  if (!container) return;

  container.innerHTML = `
    <div class="camp-header">
      <button class="camp-btn camp-btn-ghost" onclick="campShowDashboard()">
        <i class="fas fa-arrow-left"></i> Back
      </button>
      <h3><i class="fas fa-flask"></i> A/B Test Results</h3>
    </div>
    <div id="campABResultsContent" class="camp-loading">Loading A/B test results...</div>
  `;

  try {
    const [campRes, variantsRes] = await Promise.all([
      fetch(`${CAMP_API}/${campaignId}`),
      fetch(`${CAMP_API}/${campaignId}/ab-test/variants`)
    ]);

    const campaign = campRes.ok ? await campRes.json() : null;
    const variantsData = variantsRes.ok ? await variantsRes.json() : null;
    const variants = variantsData?.variants || variantsData || [];

    const content = document.getElementById('campABResultsContent');
    if (!content) return;

    if (!Array.isArray(variants) || variants.length === 0) {
      content.innerHTML = '<p class="camp-error">No A/B test variants found for this campaign.</p>';
      content.classList.remove('camp-loading');
      return;
    }

    // Determine if a winner has been selected
    const winnerVariant = variants.find(v => v.is_winner);

    content.innerHTML = `
      <div class="camp-ab-campaign-info">
        <h4>${escHtml(campaign?.name || 'Campaign')}</h4>
        ${campStatusBadge(campaign?.status || 'draft')}
        ${winnerVariant ? `<span class="camp-ab-winner-badge"><i class="fas fa-trophy"></i> Winner: Variant ${escHtml(winnerVariant.variant_label)}</span>` : ''}
      </div>

      <div class="camp-ab-comparison">
        ${variants.map(v => {
          const sentCount = v.sent_count || 0;
          const deliveredCount = v.delivered_count || 0;
          const readCount = v.read_count || 0;
          const responseCount = v.response_count || 0;
          const deliveryRate = sentCount > 0 ? ((deliveredCount / sentCount) * 100).toFixed(1) : '0';
          const readRate = sentCount > 0 ? ((readCount / sentCount) * 100).toFixed(1) : '0';
          const responseRate = sentCount > 0 ? ((responseCount / sentCount) * 100).toFixed(1) : '0';

          return `
            <div class="camp-ab-variant-card ${v.is_winner ? 'camp-ab-variant-winner' : ''}">
              <div class="camp-ab-variant-header">
                <span class="camp-ab-variant-label">Variant ${escHtml(v.variant_label || '?')}</span>
                ${v.is_winner ? '<span class="camp-ab-trophy"><i class="fas fa-trophy"></i></span>' : ''}
              </div>
              <div class="camp-ab-variant-metrics">
                <div class="camp-ab-metric">
                  <div class="camp-ab-metric-value">${v.recipient_count || 0}</div>
                  <div class="camp-ab-metric-label">Recipients</div>
                </div>
                <div class="camp-ab-metric">
                  <div class="camp-ab-metric-value">${sentCount}</div>
                  <div class="camp-ab-metric-label">Sent</div>
                </div>
                <div class="camp-ab-metric">
                  <div class="camp-ab-metric-value">${deliveryRate}%</div>
                  <div class="camp-ab-metric-label">Delivery Rate</div>
                </div>
                <div class="camp-ab-metric">
                  <div class="camp-ab-metric-value">${readRate}%</div>
                  <div class="camp-ab-metric-label">Read Rate</div>
                </div>
                <div class="camp-ab-metric">
                  <div class="camp-ab-metric-value">${responseRate}%</div>
                  <div class="camp-ab-metric-label">Response Rate</div>
                </div>
              </div>
              ${!winnerVariant ? `
                <button class="camp-btn camp-btn-primary camp-btn-sm camp-ab-select-btn" onclick="campSelectABWinner(${campaignId}, ${v.id})">
                  <i class="fas fa-crown"></i> Select as Winner
                </button>
              ` : ''}
            </div>
          `;
        }).join('')}
      </div>

      ${!winnerVariant ? `
        <div class="camp-ab-note">
          <i class="fas fa-info-circle"></i>
          Select a winning variant to send the winning template to the remaining audience.
          The full campaign will not proceed until a winner is selected.
        </div>
      ` : ''}
    `;
    content.classList.remove('camp-loading');
  } catch (err) {
    console.error('[CAMPAIGNS] Failed to load A/B results:', err);
    const content = document.getElementById('campABResultsContent');
    if (content) {
      content.innerHTML = '<p class="camp-error">Failed to load A/B test results. Please try again.</p>';
      content.classList.remove('camp-loading');
    }
  }
}

/**
 * Select the winning A/B variant — POST /api/campaigns/<id>/ab-test/select-winner
 * Requirements: 13.4
 */
async function campSelectABWinner(campaignId, variantId) {
  if (!confirm('Select this variant as the winner? A new campaign will be created for the remaining audience.')) return;
  try {
    const res = await fetch(`${CAMP_API}/${campaignId}/ab-test/select-winner`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ variant_id: variantId }),
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.error || 'Failed to select winner');
    }
    campShowToast('Winner selected! Full rollout campaign created.', 'success');
    campShowABResults(campaignId);
  } catch (err) {
    console.error('[CAMPAIGNS] Select winner failed:', err);
    campShowToast(err.message || 'Failed to select winner', 'error');
  }
}

// ─── Utility ───────────────────────────────────────────────────────────────────

// escHtml is defined in renewals.js (loaded before this file)
if (typeof escHtml === 'undefined') {
  function escHtml(str) {
    const div = document.createElement('div');
    div.textContent = str || '';
    return div.innerHTML;
  }
}

function campShowToast(message, type = 'info') {
  // Remove existing toast
  document.getElementById('campToast')?.remove();

  const colors = {
    success: 'var(--green)',
    error: 'var(--red)',
    info: 'var(--blue)',
  };

  const toast = document.createElement('div');
  toast.id = 'campToast';
  toast.style.cssText = `
    position: fixed; bottom: 24px; right: 24px; z-index: 10000;
    padding: 12px 20px; border-radius: var(--radius);
    background: var(--surface-2); color: var(--text-1);
    border-left: 4px solid ${colors[type] || colors.info};
    box-shadow: var(--shadow-lg); font-size: 13px;
    animation: campToastIn .25s ease;
  `;
  toast.textContent = message;
  document.body.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(10px)';
    setTimeout(() => toast.remove(), 200);
  }, 3500);
}
