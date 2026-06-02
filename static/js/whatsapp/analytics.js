/**
 * Analytics Dashboard Panel — Full Campaign Analytics & Quality Monitor UI
 * Fetches data from /api/analytics/ endpoints.
 * Provides per-campaign metrics, aggregate views, zone breakdown,
 * top templates, quality monitor, and opt-out trends.
 *
 * Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 10.6, 18.4, 19.7, 21.6
 */

const ANALYTICS_API = '/api/analytics';

const analyticsState = {
  view: 'overview', // overview | campaign-detail
  startDate: '',
  endDate: '',
  period: 'daily',
  selectedCampaignId: null,
  loaded: false,
};

// ─── Initialization ────────────────────────────────────────────────────────────

function initAnalyticsPanel() {
  // Set default date range: last 30 days
  const end = new Date();
  const start = new Date();
  start.setDate(start.getDate() - 30);
  analyticsState.startDate = start.toISOString().split('T')[0];
  analyticsState.endDate = end.toISOString().split('T')[0];
  analyticsState.loaded = true;
  analyticsShowOverview();
}

// ─── Overview View ─────────────────────────────────────────────────────────────

function analyticsShowOverview() {
  analyticsState.view = 'overview';
  const container = document.getElementById('analyticsPanelContent');
  if (!container) return;

  container.innerHTML = `
    <div class="ana-header">
      <h3><i class="fas fa-chart-line"></i> Analytics Dashboard</h3>
    </div>

    <!-- Date Range Picker -->
    <div class="ana-filters">
      <div class="ana-date-group">
        <label for="anaStartDate">From</label>
        <input type="date" id="anaStartDate" class="ana-input"
               value="${analyticsState.startDate}"
               onchange="analyticsState.startDate = this.value">
      </div>
      <div class="ana-date-group">
        <label for="anaEndDate">To</label>
        <input type="date" id="anaEndDate" class="ana-input"
               value="${analyticsState.endDate}"
               onchange="analyticsState.endDate = this.value">
      </div>
      <div class="ana-date-group">
        <label for="anaPeriod">Period</label>
        <select id="anaPeriod" class="ana-input" onchange="analyticsState.period = this.value">
          <option value="daily" ${analyticsState.period === 'daily' ? 'selected' : ''}>Daily</option>
          <option value="weekly" ${analyticsState.period === 'weekly' ? 'selected' : ''}>Weekly</option>
          <option value="monthly" ${analyticsState.period === 'monthly' ? 'selected' : ''}>Monthly</option>
        </select>
      </div>
      <button class="ana-btn ana-btn-primary" onclick="analyticsRefreshAll()">
        <i class="fas fa-sync-alt"></i> Refresh
      </button>
    </div>

    <!-- Quality Monitor Tier -->
    <div class="ana-section" id="anaQualitySection">
      <div class="ana-section-header">
        <h4><i class="fas fa-shield-alt"></i> WhatsApp Quality Monitor</h4>
      </div>
      <div class="ana-quality-wrap" id="anaQualityContent">
        <div class="ana-loading">Loading quality data...</div>
      </div>
    </div>

    <!-- Aggregate Metrics -->
    <div class="ana-section" id="anaAggregateSection">
      <div class="ana-section-header">
        <h4><i class="fas fa-chart-bar"></i> Aggregate Metrics</h4>
      </div>
      <div class="ana-metrics-grid" id="anaAggregateContent">
        <div class="ana-loading">Loading aggregate metrics...</div>
      </div>
    </div>

    <!-- Period Breakdown -->
    <div class="ana-section" id="anaPeriodSection">
      <div class="ana-section-header">
        <h4><i class="fas fa-calendar-alt"></i> Message Volume (${analyticsState.period})</h4>
      </div>
      <div class="ana-period-wrap" id="anaPeriodContent">
        <div class="ana-loading">Loading...</div>
      </div>
    </div>

    <!-- Zone-wise Breakdown -->
    <div class="ana-section" id="anaZoneSection">
      <div class="ana-section-header">
        <h4><i class="fas fa-map-marker-alt"></i> Zone-wise Engagement</h4>
      </div>
      <div class="ana-zone-wrap" id="anaZoneContent">
        <div class="ana-loading">Loading zone data...</div>
      </div>
    </div>

    <!-- Top 5 Templates -->
    <div class="ana-section" id="anaTemplatesSection">
      <div class="ana-section-header">
        <h4><i class="fas fa-trophy"></i> Top 5 Performing Templates</h4>
      </div>
      <div class="ana-templates-wrap" id="anaTemplatesContent">
        <div class="ana-loading">Loading top templates...</div>
      </div>
    </div>

    <!-- Opt-Out Trends & Failure Breakdown -->
    <div class="ana-section" id="anaOptOutSection">
      <div class="ana-section-header">
        <h4><i class="fas fa-ban"></i> Opt-Out Trends & Failure Breakdown</h4>
      </div>
      <div class="ana-optout-wrap" id="anaOptOutContent">
        <div class="ana-loading">Loading opt-out data...</div>
      </div>
    </div>

    <!-- Per-Campaign Lookup -->
    <div class="ana-section" id="anaCampaignLookup">
      <div class="ana-section-header">
        <h4><i class="fas fa-search"></i> Campaign Metrics Lookup</h4>
      </div>
      <div class="ana-campaign-lookup-wrap">
        <div class="ana-lookup-form">
          <input type="number" id="anaCampaignId" class="ana-input"
                 placeholder="Enter Campaign ID" min="1">
          <button class="ana-btn ana-btn-primary" onclick="analyticsLoadCampaignMetrics()">
            <i class="fas fa-search"></i> View Metrics
          </button>
        </div>
        <div id="anaCampaignMetricsContent"></div>
      </div>
    </div>
  `;

  analyticsRefreshAll();
}

// ─── Refresh All Sections ──────────────────────────────────────────────────────

function analyticsRefreshAll() {
  analyticsLoadQuality();
  analyticsLoadAggregate();
  analyticsLoadZones();
  analyticsLoadTopTemplates();
  analyticsLoadOptOutTrends();
}

// ─── Quality Monitor ───────────────────────────────────────────────────────────

async function analyticsLoadQuality() {
  const container = document.getElementById('anaQualityContent');
  if (!container) return;

  try {
    const res = await fetch(`${ANALYTICS_API}/quality`);
    if (!res.ok) throw new Error('Failed to load quality data');
    const data = await res.json();

    const tier = data.current_tier || 'green';
    const m24 = data.metrics_24h || {};
    const m7d = data.metrics_7d || {};
    const alerts = data.active_alerts || [];

    const tierColors = {
      green: { bg: 'rgba(16,185,129,0.15)', color: '#10b981', icon: 'fa-check-circle', label: 'Green — Healthy' },
      yellow: { bg: 'rgba(245,158,11,0.15)', color: '#f59e0b', icon: 'fa-exclamation-triangle', label: 'Yellow — At Risk' },
      red: { bg: 'rgba(239,68,68,0.15)', color: '#ef4444', icon: 'fa-times-circle', label: 'Red — Critical' },
    };
    const tc = tierColors[tier] || tierColors.green;

    container.innerHTML = `
      <div class="ana-quality-tier" style="background:${tc.bg};border-color:${tc.color}">
        <i class="fas ${tc.icon}" style="color:${tc.color};font-size:32px"></i>
        <div class="ana-tier-info">
          <div class="ana-tier-label" style="color:${tc.color}">${tc.label}</div>
          <div class="ana-tier-desc">Current WhatsApp Business API quality tier</div>
        </div>
      </div>
      <div class="ana-quality-metrics">
        <div class="ana-quality-col">
          <h5>Last 24 Hours</h5>
          <div class="ana-metric-row">
            <span class="ana-metric-name">Messages Sent</span>
            <span class="ana-metric-val">${(m24.total_sent || 0).toLocaleString()}</span>
          </div>
          <div class="ana-metric-row">
            <span class="ana-metric-name">Failure Rate</span>
            <span class="ana-metric-val ${m24.failure_rate > 5 ? 'ana-val-danger' : ''}">${m24.failure_rate || 0}%</span>
          </div>
          <div class="ana-metric-row">
            <span class="ana-metric-name">Read Rate</span>
            <span class="ana-metric-val">${m24.read_rate || 0}%</span>
          </div>
          <div class="ana-metric-row">
            <span class="ana-metric-name">Blocked Count</span>
            <span class="ana-metric-val ${m24.blocked_count > 10 ? 'ana-val-danger' : ''}">${m24.blocked_count || 0}</span>
          </div>
          <div class="ana-metric-row">
            <span class="ana-metric-name">Opt-out Rate</span>
            <span class="ana-metric-val">${m24.opt_out_rate || 0}%</span>
          </div>
        </div>
        <div class="ana-quality-col">
          <h5>Last 7 Days</h5>
          <div class="ana-metric-row">
            <span class="ana-metric-name">Messages Sent</span>
            <span class="ana-metric-val">${(m7d.total_sent || 0).toLocaleString()}</span>
          </div>
          <div class="ana-metric-row">
            <span class="ana-metric-name">Failure Rate</span>
            <span class="ana-metric-val ${m7d.failure_rate > 5 ? 'ana-val-danger' : ''}">${m7d.failure_rate || 0}%</span>
          </div>
          <div class="ana-metric-row">
            <span class="ana-metric-name">Read Rate</span>
            <span class="ana-metric-val">${m7d.read_rate || 0}%</span>
          </div>
          <div class="ana-metric-row">
            <span class="ana-metric-name">Blocked Count</span>
            <span class="ana-metric-val ${m7d.blocked_count > 10 ? 'ana-val-danger' : ''}">${m7d.blocked_count || 0}</span>
          </div>
          <div class="ana-metric-row">
            <span class="ana-metric-name">Opt-out Rate</span>
            <span class="ana-metric-val">${m7d.opt_out_rate || 0}%</span>
          </div>
        </div>
      </div>
      ${alerts.length > 0 ? `
        <div class="ana-alerts">
          <h5><i class="fas fa-bell"></i> Active Alerts (${alerts.length})</h5>
          ${alerts.map(a => `
            <div class="ana-alert ana-alert-${a.severity}">
              <i class="fas ${a.severity === 'critical' ? 'fa-exclamation-circle' : 'fa-info-circle'}"></i>
              <span>${_anaEscHtml(a.title)}</span>
            </div>
          `).join('')}
        </div>
      ` : ''}
    `;
  } catch (err) {
    console.error('[ANALYTICS] Quality load failed:', err);
    container.innerHTML = '<div class="ana-error">Failed to load quality data.</div>';
  }
}

// ─── Aggregate Metrics ─────────────────────────────────────────────────────────

async function analyticsLoadAggregate() {
  const container = document.getElementById('anaAggregateContent');
  const periodContainer = document.getElementById('anaPeriodContent');
  if (!container) return;

  const params = new URLSearchParams({
    start_date: analyticsState.startDate,
    end_date: analyticsState.endDate,
    period: analyticsState.period,
  });

  try {
    const res = await fetch(`${ANALYTICS_API}/aggregate?${params}`);
    if (!res.ok) throw new Error('Failed to load aggregate data');
    const data = await res.json();

    const summary = data.summary || {};
    const breakdown = data.period_breakdown || [];

    container.innerHTML = `
      <div class="ana-stat-card">
        <div class="ana-stat-value">${(summary.total_campaigns || 0).toLocaleString()}</div>
        <div class="ana-stat-label">Total Campaigns</div>
      </div>
      <div class="ana-stat-card">
        <div class="ana-stat-value">${(summary.total_messages || 0).toLocaleString()}</div>
        <div class="ana-stat-label">Messages Sent</div>
      </div>
      <div class="ana-stat-card ana-stat-green">
        <div class="ana-stat-value">${summary.avg_delivery_rate || 0}%</div>
        <div class="ana-stat-label">Avg Delivery Rate</div>
      </div>
      <div class="ana-stat-card ana-stat-blue">
        <div class="ana-stat-value">${summary.avg_read_rate || 0}%</div>
        <div class="ana-stat-label">Avg Read Rate</div>
      </div>
      <div class="ana-stat-card ana-stat-red">
        <div class="ana-stat-value">${summary.avg_failure_rate || 0}%</div>
        <div class="ana-stat-label">Avg Failure Rate</div>
      </div>
    `;

    // Render period breakdown as a bar visualization
    if (periodContainer && breakdown.length > 0) {
      const maxVal = Math.max(...breakdown.map(b => b.messages_sent || 0), 1);
      periodContainer.innerHTML = `
        <div class="ana-period-chart">
          ${breakdown.map(b => {
            const pct = Math.round(((b.messages_sent || 0) / maxVal) * 100);
            const deliveredPct = b.messages_sent > 0
              ? Math.round(((b.delivered || 0) / b.messages_sent) * 100) : 0;
            const readPct = b.messages_sent > 0
              ? Math.round(((b.read_count || 0) / b.messages_sent) * 100) : 0;
            return `
              <div class="ana-period-bar-group" title="${b.period_label}: ${b.messages_sent} sent, ${deliveredPct}% delivered, ${readPct}% read">
                <div class="ana-period-bar" style="height:${pct}%"></div>
                <div class="ana-period-label">${_anaShortLabel(b.period_label)}</div>
              </div>
            `;
          }).join('')}
        </div>
        <div class="ana-period-legend">
          <span><i class="fas fa-square" style="color:var(--green)"></i> Message volume per ${analyticsState.period} period</span>
        </div>
      `;
    } else if (periodContainer) {
      periodContainer.innerHTML = '<div class="ana-empty">No data for selected period.</div>';
    }
  } catch (err) {
    console.error('[ANALYTICS] Aggregate load failed:', err);
    container.innerHTML = '<div class="ana-error">Failed to load aggregate metrics.</div>';
  }
}

// ─── Zone-wise Breakdown ───────────────────────────────────────────────────────

async function analyticsLoadZones() {
  const container = document.getElementById('anaZoneContent');
  if (!container) return;

  const params = new URLSearchParams({
    start_date: analyticsState.startDate,
    end_date: analyticsState.endDate,
  });

  try {
    const res = await fetch(`${ANALYTICS_API}/zones?${params}`);
    if (!res.ok) throw new Error('Failed to load zone data');
    const data = await res.json();

    const zones = data.zones || [];

    if (zones.length === 0) {
      container.innerHTML = '<div class="ana-empty">No zone data available for this period.</div>';
      return;
    }

    container.innerHTML = `
      <table class="ana-table">
        <thead>
          <tr>
            <th>Zone</th>
            <th>Delivery Rate</th>
            <th>Read Rate</th>
            <th>Campaigns</th>
          </tr>
        </thead>
        <tbody>
          ${zones.map(z => `
            <tr>
              <td class="ana-cell-name">${_anaEscHtml(z.zone_name || 'Unknown')}</td>
              <td>
                <div class="ana-bar-cell">
                  <div class="ana-bar-fill ana-bar-green" style="width:${z.delivery_rate || 0}%"></div>
                  <span>${z.delivery_rate || 0}%</span>
                </div>
              </td>
              <td>
                <div class="ana-bar-cell">
                  <div class="ana-bar-fill ana-bar-blue" style="width:${z.read_rate || 0}%"></div>
                  <span>${z.read_rate || 0}%</span>
                </div>
              </td>
              <td>${z.campaign_count || z.total_sent || '-'}</td>
            </tr>
          `).join('')}
        </tbody>
      </table>
    `;
  } catch (err) {
    console.error('[ANALYTICS] Zone load failed:', err);
    container.innerHTML = '<div class="ana-error">Failed to load zone data.</div>';
  }
}

// ─── Top Templates ─────────────────────────────────────────────────────────────

async function analyticsLoadTopTemplates() {
  const container = document.getElementById('anaTemplatesContent');
  if (!container) return;

  try {
    const res = await fetch(`${ANALYTICS_API}/templates/top?limit=5&days=30`);
    if (!res.ok) throw new Error('Failed to load top templates');
    const data = await res.json();

    const templates = data.top_templates || [];

    if (templates.length === 0) {
      container.innerHTML = '<div class="ana-empty">No template performance data yet.</div>';
      return;
    }

    container.innerHTML = `
      <div class="ana-top-templates">
        ${templates.map((t, idx) => `
          <div class="ana-template-card">
            <div class="ana-template-rank">#${idx + 1}</div>
            <div class="ana-template-info">
              <div class="ana-template-name">${_anaEscHtml(t.template_name || 'Unknown')}</div>
              <div class="ana-template-meta">
                Used in ${t.usage_count || 0} campaign${t.usage_count !== 1 ? 's' : ''}
              </div>
            </div>
            <div class="ana-template-rate">
              <div class="ana-template-rate-value">${t.avg_read_rate || 0}%</div>
              <div class="ana-template-rate-label">Read Rate</div>
            </div>
          </div>
        `).join('')}
      </div>
    `;
  } catch (err) {
    console.error('[ANALYTICS] Top templates load failed:', err);
    container.innerHTML = '<div class="ana-error">Failed to load top templates.</div>';
  }
}

// ─── Opt-Out Trends & Failure Breakdown ────────────────────────────────────────

async function analyticsLoadOptOutTrends() {
  const container = document.getElementById('anaOptOutContent');
  if (!container) return;

  const params = new URLSearchParams({
    start_date: analyticsState.startDate,
    end_date: analyticsState.endDate,
  });

  try {
    const res = await fetch(`${ANALYTICS_API}/optout-trends?${params}`);
    if (!res.ok) throw new Error('Failed to load opt-out trends');
    const data = await res.json();

    const daily = data.opt_out_daily || [];
    const failures = data.failure_breakdown || [];

    let optOutHtml = '';
    if (daily.length > 0) {
      const maxOpt = Math.max(...daily.map(d => d.opt_out_count || 0), 1);
      optOutHtml = `
        <div class="ana-subsection">
          <h5>Daily Opt-Outs</h5>
          <div class="ana-mini-chart">
            ${daily.map(d => {
              const pct = Math.round(((d.opt_out_count || 0) / maxOpt) * 100);
              return `
                <div class="ana-mini-bar-group" title="${d.date_label}: ${d.opt_out_count} opt-outs">
                  <div class="ana-mini-bar ana-mini-bar-red" style="height:${pct}%"></div>
                  <div class="ana-mini-label">${_anaShortDate(d.date_label)}</div>
                </div>
              `;
            }).join('')}
          </div>
        </div>
      `;
    } else {
      optOutHtml = '<div class="ana-subsection"><h5>Daily Opt-Outs</h5><div class="ana-empty">No opt-outs in this period.</div></div>';
    }

    let failureHtml = '';
    if (failures.length > 0) {
      const totalFailures = failures.reduce((acc, f) => acc + (f.count || 0), 0);
      failureHtml = `
        <div class="ana-subsection">
          <h5>Failure Breakdown by Category</h5>
          <div class="ana-failure-list">
            ${failures.map(f => {
              const pct = totalFailures > 0 ? Math.round((f.count / totalFailures) * 100) : 0;
              const catLabel = _anaCapitalize(f.category || 'unknown');
              return `
                <div class="ana-failure-row">
                  <span class="ana-failure-cat">${catLabel}</span>
                  <div class="ana-failure-bar-wrap">
                    <div class="ana-failure-bar" style="width:${pct}%"></div>
                  </div>
                  <span class="ana-failure-count">${f.count} (${pct}%)</span>
                </div>
              `;
            }).join('')}
          </div>
        </div>
      `;
    } else {
      failureHtml = '<div class="ana-subsection"><h5>Failure Breakdown</h5><div class="ana-empty">No failures in this period.</div></div>';
    }

    container.innerHTML = `
      <div class="ana-optout-grid">
        ${optOutHtml}
        ${failureHtml}
      </div>
    `;
  } catch (err) {
    console.error('[ANALYTICS] Opt-out trends load failed:', err);
    container.innerHTML = '<div class="ana-error">Failed to load opt-out trends.</div>';
  }
}

// ─── Per-Campaign Metrics Lookup ───────────────────────────────────────────────

async function analyticsLoadCampaignMetrics() {
  const idInput = document.getElementById('anaCampaignId');
  const container = document.getElementById('anaCampaignMetricsContent');
  if (!idInput || !container) return;

  const campaignId = parseInt(idInput.value, 10);
  if (!campaignId || campaignId < 1) {
    container.innerHTML = '<div class="ana-error">Please enter a valid Campaign ID.</div>';
    return;
  }

  container.innerHTML = '<div class="ana-loading">Loading campaign metrics...</div>';

  try {
    const res = await fetch(`${ANALYTICS_API}/campaigns/${campaignId}`);
    if (!res.ok) {
      if (res.status === 404) throw new Error('Campaign not found');
      throw new Error('Failed to load campaign metrics');
    }
    const data = await res.json();

    const metrics = data.metrics || {};
    const totals = data.totals || {};

    if (Object.keys(metrics).length === 0) {
      container.innerHTML = `<div class="ana-empty">No metrics available for Campaign #${campaignId}. It may not have sent any messages yet.</div>`;
      return;
    }

    container.innerHTML = `
      <div class="ana-campaign-metrics">
        <div class="ana-campaign-metrics-header">
          <h5>Campaign #${campaignId} Metrics</h5>
        </div>
        <div class="ana-metrics-grid">
          ${_anaRenderMetricCard('Delivery Rate', metrics.delivery_rate, 'fa-check-double', 'green')}
          ${_anaRenderMetricCard('Read Rate', metrics.read_rate, 'fa-eye', 'blue')}
          ${_anaRenderMetricCard('Failure Rate', metrics.failure_rate, 'fa-times-circle', 'red')}
          ${_anaRenderMetricCard('Response Rate', metrics.response_rate, 'fa-reply', 'purple')}
        </div>
        ${totals.total_messages ? `
          <div class="ana-totals-row">
            <span>Total: ${totals.total_messages}</span>
            <span>Sent: ${totals.sent || 0}</span>
            <span>Delivered: ${totals.delivered || 0}</span>
            <span>Read: ${totals.read || 0}</span>
            <span>Failed: ${totals.failed || 0}</span>
            <span>Responded: ${totals.responded || 0}</span>
          </div>
        ` : ''}
      </div>
    `;
  } catch (err) {
    console.error('[ANALYTICS] Campaign metrics load failed:', err);
    container.innerHTML = `<div class="ana-error">${_anaEscHtml(err.message || 'Failed to load campaign metrics.')}</div>`;
  }
}

// ─── Helper: Render Metric Card ────────────────────────────────────────────────

function _anaRenderMetricCard(label, metricObj, icon, colorClass) {
  const pct = metricObj ? (metricObj.percentage || 0) : 0;
  return `
    <div class="ana-metric-card ana-metric-${colorClass}">
      <div class="ana-metric-icon"><i class="fas ${icon}"></i></div>
      <div class="ana-metric-pct">${pct}%</div>
      <div class="ana-metric-label">${label}</div>
    </div>
  `;
}

// ─── Utility Functions ─────────────────────────────────────────────────────────

function _anaEscHtml(str) {
  const div = document.createElement('div');
  div.textContent = str || '';
  return div.innerHTML;
}

function _anaShortLabel(label) {
  if (!label) return '';
  // For daily: show DD, for weekly: Wxx, for monthly: Mon
  if (label.length === 10) {
    // YYYY-MM-DD → DD
    return label.substring(8);
  }
  if (label.includes('-W')) {
    return label.split('-')[1];
  }
  if (label.length === 7) {
    // YYYY-MM → Mon abbreviation
    const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
    const m = parseInt(label.split('-')[1], 10) - 1;
    return months[m] || label;
  }
  return label.substring(0, 6);
}

function _anaShortDate(dateStr) {
  if (!dateStr) return '';
  // YYYY-MM-DD → MM/DD
  const parts = dateStr.split('-');
  if (parts.length === 3) return `${parts[1]}/${parts[2]}`;
  return dateStr;
}

function _anaCapitalize(str) {
  if (!str) return '';
  return str.charAt(0).toUpperCase() + str.slice(1).replace(/_/g, ' ');
}
