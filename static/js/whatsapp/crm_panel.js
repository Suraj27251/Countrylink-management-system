/**
 * CRM Panel — Slide-out customer profile, timeline, notes, tags, engagement.
 *
 * Integrates within the existing WhatsApp Inbox chat view (wa-main area).
 * Fetches data from /api/crm/customers/by-mobile/<mobile>/ endpoints.
 *
 * Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 10.5, 12.5
 */

const CRM_API = '/api/crm/customers/by-mobile';

const crmState = {
  open: false,
  mobile: null,
  profile: null,
  timeline: [],
  timelinePage: 1,
  timelineHasMore: true,
  timelineLoading: false,
  tags: [],
  loading: false,
};

// ─── Initialization ────────────────────────────────────────────────────────────

function initCrmPanel() {
  // Create the CRM panel container if it doesn't exist
  if (!document.getElementById('crmPanel')) {
    const panel = document.createElement('aside');
    panel.id = 'crmPanel';
    panel.className = 'crm-panel';
    panel.innerHTML = `
      <div class="crm-panel-inner">
        <div class="crm-panel-header">
          <h4 class="crm-panel-title"><i class="fas fa-user-circle"></i> Customer Profile</h4>
          <button class="crm-close-btn" onclick="closeCrmPanel()" title="Close panel">
            <i class="fas fa-times"></i>
          </button>
        </div>
        <div class="crm-panel-body" id="crmPanelBody">
          <div class="crm-loading"><i class="fas fa-spinner fa-spin"></i> Loading...</div>
        </div>
      </div>
    `;
    // On mobile, append to body for fixed positioning; on desktop, append to waMain for absolute
    const isMobile = window.innerWidth <= 680;
    const container = isMobile ? document.body : document.getElementById('waMain');
    if (container) {
      container.appendChild(panel);
    } else {
      document.body.appendChild(panel);
    }
  }

  // Add CRM button to chat header if active chat exists
  _addCrmTriggerButton();
}

function _addCrmTriggerButton() {
  const headerActions = document.querySelector('.chat-header-actions');
  if (!headerActions) return;
  if (document.getElementById('crmPanelBtn')) return;

  const btn = document.createElement('button');
  btn.id = 'crmPanelBtn';
  btn.className = 'hdr-btn';
  btn.title = 'Customer Profile';
  btn.innerHTML = '<i class="fas fa-address-card"></i><span>CRM</span>';
  btn.addEventListener('click', () => {
    const mobile = _getActiveMobile();
    if (mobile) {
      toggleCrmPanel(mobile);
    }
  });

  // Insert before the dropdown (last element)
  const dropdown = headerActions.querySelector('.dropdown');
  if (dropdown) {
    headerActions.insertBefore(btn, dropdown);
  } else {
    headerActions.appendChild(btn);
  }
}

function _getActiveMobile() {
  // Try inboxState first, then fallback to URL param or hidden input
  if (window.inboxState && window.inboxState.activeMobile) {
    return window.inboxState.activeMobile;
  }
  const input = document.querySelector('input[name="mobile"]');
  if (input && input.value) return input.value;
  try {
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get('mobile') || '';
  } catch (_) {
    return '';
  }
}

// ─── Panel Toggle ──────────────────────────────────────────────────────────────

function toggleCrmPanel(mobile) {
  if (crmState.open && crmState.mobile === mobile) {
    closeCrmPanel();
  } else {
    openCrmPanel(mobile);
  }
}

function openCrmPanel(mobile) {
  if (!mobile) return;
  crmState.open = true;
  crmState.mobile = mobile;
  crmState.timeline = [];
  crmState.timelinePage = 1;
  crmState.timelineHasMore = true;

  const panel = document.getElementById('crmPanel');
  if (panel) {
    panel.classList.add('open');
  }

  // Mark CRM button as active
  const btn = document.getElementById('crmPanelBtn');
  if (btn) btn.classList.add('active');

  _loadCustomerProfile(mobile);
}

function closeCrmPanel() {
  crmState.open = false;

  const panel = document.getElementById('crmPanel');
  if (panel) {
    panel.classList.remove('open');
  }

  // Unmark CRM button
  const btn = document.getElementById('crmPanelBtn');
  if (btn) btn.classList.remove('active');
}

// ─── Data Loading ──────────────────────────────────────────────────────────────

async function _loadCustomerProfile(mobile) {
  const body = document.getElementById('crmPanelBody');
  if (!body) return;

  body.innerHTML = '<div class="crm-loading"><i class="fas fa-spinner fa-spin"></i> Loading profile...</div>';
  crmState.loading = true;

  try {
    const res = await fetch(`${CRM_API}/${encodeURIComponent(mobile)}/profile`);
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || `HTTP ${res.status}`);
    }
    const profile = await res.json();
    crmState.profile = profile;
    crmState.tags = profile.tags || [];

    _renderCrmPanel(profile);
    _loadTimeline(mobile);
  } catch (err) {
    console.error('[CRM] Failed to load profile:', err);
    body.innerHTML = `
      <div class="crm-error">
        <i class="fas fa-circle-exclamation"></i>
        <p>Failed to load customer profile</p>
        <small>${_escCrm(err.message)}</small>
        <button class="crm-btn crm-btn-sm" onclick="openCrmPanel('${_escCrm(mobile)}')">Retry</button>
      </div>
    `;
  }
  crmState.loading = false;
}

async function _loadTimeline(mobile) {
  if (crmState.timelineLoading || !crmState.timelineHasMore) return;
  crmState.timelineLoading = true;

  const timelineList = document.getElementById('crmTimelineList');
  const loadMoreBtn = document.getElementById('crmLoadMoreBtn');

  try {
    const res = await fetch(
      `${CRM_API}/${encodeURIComponent(mobile)}/timeline?page=${crmState.timelinePage}&per_page=50`
    );
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    const items = data.items || data.timeline || [];
    crmState.timeline = [...crmState.timeline, ...items];
    crmState.timelineHasMore = items.length >= 50;
    crmState.timelinePage++;

    if (timelineList) {
      const html = items.map(item => _renderTimelineItem(item)).join('');
      timelineList.insertAdjacentHTML('beforeend', html);
    }

    if (loadMoreBtn) {
      loadMoreBtn.style.display = crmState.timelineHasMore ? 'block' : 'none';
    }
  } catch (err) {
    console.error('[CRM] Failed to load timeline:', err);
  }
  crmState.timelineLoading = false;
}

// ─── Render ────────────────────────────────────────────────────────────────────

function _renderCrmPanel(profile) {
  const body = document.getElementById('crmPanelBody');
  if (!body) return;

  // Extract fields with proper fallbacks for the actual DB schema
  const name = profile.customer_name || profile.name || '';
  const displayName = name || 'Customer';
  const mobile = profile.mobile || crmState.mobile || '';
  const initial = (displayName.charAt(0) || 'C').toUpperCase();
  
  // Status from DB — "Unlimited" means active plan, use category/days_remaining for actual status
  const planStatus = profile.status || '';
  const category = profile.category || '';
  const daysRemaining = profile.days_remaining;
  const isExpired = category === 'expired' || (daysRemaining !== null && daysRemaining <= 0);
  const isExpiringToday = category === 'today' || daysRemaining === 0;
  
  // Engagement data (from customer_engagement table, may be null)
  const eng = profile.engagement || {};
  const engScore = eng.score != null ? eng.score : null;
  const engTrend = eng.trend || 'stable';
  
  // Opt-out/DND status
  const optOut = profile.opt_out_status || {};
  const dnd = profile.dnd_status || {};
  const optedOut = optOut.opted_out || false;
  const dndActive = dnd.dnd_active || false;

  body.innerHTML = `
    <!-- Profile Header -->
    <div class="crm-section crm-profile-section">
      <div class="crm-profile-header">
        <div class="crm-avatar">${_escCrm(initial)}</div>
        <div class="crm-profile-info">
          <h5 class="crm-profile-name">${_escCrm(displayName)}</h5>
          <span class="crm-profile-mobile"><i class="fas fa-phone"></i> ${_escCrm(mobile)}</span>
        </div>
      </div>

      <!-- Status Badges -->
      <div class="crm-status-row">
        ${optedOut ? '<span class="crm-badge crm-badge-red"><i class="fas fa-ban"></i> Opted Out</span>' : ''}
        ${dndActive ? '<span class="crm-badge crm-badge-amber"><i class="fas fa-moon"></i> DND</span>' : ''}
        ${isExpired ? '<span class="crm-badge crm-badge-red"><i class="fas fa-clock"></i> Expired</span>' : ''}
        ${isExpiringToday ? '<span class="crm-badge crm-badge-amber"><i class="fas fa-exclamation"></i> Expiring Today</span>' : ''}
        ${!isExpired && !isExpiringToday && !optedOut ? '<span class="crm-badge crm-badge-green"><i class="fas fa-check-circle"></i> Active</span>' : ''}
        ${daysRemaining != null && daysRemaining > 0 ? `<span class="crm-badge crm-badge-outline">${daysRemaining}d left</span>` : ''}
      </div>

      <!-- Engagement Score -->
      ${engScore !== null ? `
        <div class="crm-engagement">
          <div class="crm-engagement-label">
            <span>Engagement Score</span>
            <span class="crm-engagement-value">${engScore}/100</span>
          </div>
          <div class="crm-engagement-bar">
            <div class="crm-engagement-fill" style="width:${Math.min(engScore, 100)}%;background:${_engColor(engScore)}"></div>
          </div>
          <div class="crm-engagement-trend">
            <i class="fas fa-${engTrend === 'increasing' ? 'arrow-trend-up' : engTrend === 'declining' ? 'arrow-trend-down' : 'minus'}"></i>
            ${engTrend}
          </div>
        </div>
      ` : ''}

      <!-- Customer Details -->
      <div class="crm-detail-grid">
        ${_detailItem('Plan', profile.plan_name)}
        ${_detailItem('Category', profile.plan_category)}
        ${_detailItem('Validity', profile.validity || _formatValidity(profile.expiry_date))}
        ${_detailItem('Zone', profile.zone_name)}
        ${_detailItem('Area', profile.area)}
        ${_detailItem('Building', profile.building)}
        ${_detailItem('Network', profile.network_type)}
        ${_detailItem('Mode', profile.connectivity_mode)}
        ${_detailItem('Expiry', _formatDate(profile.expiry_date))}
        ${_detailItem('Activation', _formatDate(profile.activation_date))}
      </div>
    </div>

    <!-- Tags Section -->
    <div class="crm-section crm-tags-section">
      <div class="crm-section-header">
        <h6><i class="fas fa-tags"></i> Tags</h6>
      </div>
      <div class="crm-tags-list" id="crmTagsList">
        ${_renderTags(crmState.tags)}
      </div>
      <div class="crm-tag-add">
        <input type="text" id="crmTagInput" class="crm-input" placeholder="Add tag..." 
               onkeydown="if(event.key==='Enter'){event.preventDefault();crmAddTag();}">
        <button class="crm-btn crm-btn-sm crm-btn-primary" onclick="crmAddTag()">
          <i class="fas fa-plus"></i>
        </button>
      </div>
    </div>

    <!-- Notes Section -->
    <div class="crm-section crm-notes-section">
      <div class="crm-section-header">
        <h6><i class="fas fa-sticky-note"></i> Notes</h6>
      </div>
      <form class="crm-note-form" onsubmit="crmAddNote(event)">
        <textarea id="crmNoteInput" class="crm-input crm-textarea" placeholder="Add a note..." rows="2"></textarea>
        <button type="submit" class="crm-btn crm-btn-primary crm-btn-sm">
          <i class="fas fa-plus"></i> Add Note
        </button>
      </form>
    </div>

    <!-- Timeline Section -->
    <div class="crm-section crm-timeline-section">
      <div class="crm-section-header">
        <h6><i class="fas fa-clock-rotate-left"></i> Activity Timeline</h6>
      </div>
      <div class="crm-timeline" id="crmTimelineList"></div>
      <button class="crm-btn crm-btn-outline crm-btn-block" id="crmLoadMoreBtn" 
              onclick="_loadTimeline('${_escCrm(crmState.mobile)}')" style="display:none;">
        Load more...
      </button>
    </div>
  `;
}

function _renderTimelineItem(item) {
  const icon = _timelineIcon(item.activity_type);
  const time = _formatDateTime(item.created_at);
  const channel = item.channel ? `<span class="crm-tl-channel">${_escCrm(item.channel)}</span>` : '';
  let description = '';

  switch (item.activity_type) {
    case 'message_sent':
      description = `Message sent${item.details?.preview ? ': ' + _escCrm(item.details.preview.substring(0, 80)) : ''}`;
      break;
    case 'message_received':
      description = `Message received${item.details?.preview ? ': ' + _escCrm(item.details.preview.substring(0, 80)) : ''}`;
      break;
    case 'campaign_sent':
      description = `Campaign: ${_escCrm(item.details?.campaign_name || 'Unknown')}`;
      break;
    case 'note_added':
      description = `Note by ${_escCrm(item.details?.operator || 'operator')}: ${_escCrm((item.details?.note || '').substring(0, 100))}`;
      break;
    case 'tag_added':
      description = `Tag added: <span class="crm-tl-tag">${_escCrm(item.details?.tag || '')}</span>`;
      break;
    case 'tag_removed':
      description = `Tag removed: <span class="crm-tl-tag">${_escCrm(item.details?.tag || '')}</span>`;
      break;
    case 'status_change':
      description = `Status: ${_escCrm(item.details?.from || '?')} → ${_escCrm(item.details?.to || '?')}`;
      break;
    case 'opt_out':
      description = 'Customer opted out';
      break;
    case 'opt_in':
      description = 'Customer opted back in';
      break;
    default:
      description = _escCrm(item.activity_type || 'Activity');
  }

  return `
    <div class="crm-tl-item">
      <div class="crm-tl-icon">${icon}</div>
      <div class="crm-tl-content">
        <div class="crm-tl-desc">${description} ${channel}</div>
        <div class="crm-tl-time">${time}</div>
      </div>
    </div>
  `;
}

function _renderTags(tags) {
  if (!tags || tags.length === 0) {
    return '<span class="crm-tags-empty">No tags assigned</span>';
  }
  return tags.map(tag => {
    const tagName = typeof tag === 'string' ? tag : (tag.tag_name || tag.name || '');
    return `
      <span class="crm-tag">
        ${_escCrm(tagName)}
        <button class="crm-tag-remove" onclick="crmRemoveTag('${_escCrm(tagName)}')" title="Remove tag">
          <i class="fas fa-xmark"></i>
        </button>
      </span>
    `;
  }).join('');
}

// ─── Actions ───────────────────────────────────────────────────────────────────

async function crmAddTag() {
  const input = document.getElementById('crmTagInput');
  if (!input) return;
  const tagName = input.value.trim();
  if (!tagName || !crmState.mobile) return;

  try {
    const res = await fetch(`${CRM_API}/${encodeURIComponent(crmState.mobile)}/tags`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tags: [tagName] }),
    });

    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.error || 'Failed to add tag');
    }

    // Update local state
    crmState.tags.push({ tag_name: tagName });
    const tagsList = document.getElementById('crmTagsList');
    if (tagsList) {
      tagsList.innerHTML = _renderTags(crmState.tags);
    }
    input.value = '';
    _crmToast('Tag added', 'success');
  } catch (err) {
    console.error('[CRM] Failed to add tag:', err);
    _crmToast(err.message || 'Failed to add tag', 'error');
  }
}

async function crmRemoveTag(tagName) {
  if (!tagName || !crmState.mobile) return;

  try {
    const res = await fetch(
      `${CRM_API}/${encodeURIComponent(crmState.mobile)}/tags/${encodeURIComponent(tagName)}`,
      { method: 'DELETE' }
    );

    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.error || 'Failed to remove tag');
    }

    // Update local state
    crmState.tags = crmState.tags.filter(t => {
      const name = typeof t === 'string' ? t : (t.tag_name || t.name || '');
      return name !== tagName;
    });
    const tagsList = document.getElementById('crmTagsList');
    if (tagsList) {
      tagsList.innerHTML = _renderTags(crmState.tags);
    }
    _crmToast('Tag removed', 'success');
  } catch (err) {
    console.error('[CRM] Failed to remove tag:', err);
    _crmToast(err.message || 'Failed to remove tag', 'error');
  }
}

async function crmAddNote(event) {
  if (event) event.preventDefault();
  const input = document.getElementById('crmNoteInput');
  if (!input) return;
  const noteText = input.value.trim();
  if (!noteText || !crmState.mobile) return;

  try {
    const res = await fetch(`${CRM_API}/${encodeURIComponent(crmState.mobile)}/notes`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ note: noteText }),
    });

    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.error || 'Failed to add note');
    }

    input.value = '';
    _crmToast('Note added', 'success');

    // Prepend to timeline
    const newItem = {
      activity_type: 'note_added',
      details: { note: noteText, operator: 'You' },
      channel: 'whatsapp',
      created_at: new Date().toISOString(),
    };
    const timelineList = document.getElementById('crmTimelineList');
    if (timelineList) {
      timelineList.insertAdjacentHTML('afterbegin', _renderTimelineItem(newItem));
    }
  } catch (err) {
    console.error('[CRM] Failed to add note:', err);
    _crmToast(err.message || 'Failed to add note', 'error');
  }
}

// ─── Helpers ───────────────────────────────────────────────────────────────────

function _escCrm(str) {
  if (!str) return '';
  const div = document.createElement('div');
  div.textContent = String(str);
  return div.innerHTML;
}

function _engColor(score) {
  if (score >= 70) return 'var(--green)';
  if (score >= 40) return 'var(--amber)';
  return 'var(--red)';
}

function _timelineIcon(type) {
  const icons = {
    message_sent: '<i class="fas fa-paper-plane" style="color:var(--blue)"></i>',
    message_received: '<i class="fas fa-inbox" style="color:var(--green)"></i>',
    campaign_sent: '<i class="fas fa-bullhorn" style="color:var(--amber)"></i>',
    note_added: '<i class="fas fa-sticky-note" style="color:var(--text-2)"></i>',
    tag_added: '<i class="fas fa-tag" style="color:var(--green)"></i>',
    tag_removed: '<i class="fas fa-tag" style="color:var(--red)"></i>',
    status_change: '<i class="fas fa-exchange-alt" style="color:var(--blue)"></i>',
    opt_out: '<i class="fas fa-ban" style="color:var(--red)"></i>',
    opt_in: '<i class="fas fa-check-circle" style="color:var(--green)"></i>',
  };
  return icons[type] || '<i class="fas fa-circle" style="color:var(--text-3)"></i>';
}

function _detailItem(label, value) {
  if (!value) return '';
  return `
    <div class="crm-detail-item">
      <span class="crm-detail-label">${_escCrm(label)}</span>
      <span class="crm-detail-value">${_escCrm(value)}</span>
    </div>
  `;
}

function _formatDate(dateStr) {
  if (!dateStr) return '';
  try {
    const d = new Date(dateStr);
    return d.toLocaleDateString('en-IN', { day: '2-digit', month: 'short', year: 'numeric' });
  } catch (_) {
    return dateStr;
  }
}

function _formatDateTime(dateStr) {
  if (!dateStr) return '';
  try {
    const d = new Date(dateStr);
    return d.toLocaleDateString('en-IN', { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' });
  } catch (_) {
    return dateStr;
  }
}

function _formatValidity(expiryDate) {
  if (!expiryDate) return '';
  try {
    const now = new Date();
    const exp = new Date(expiryDate);
    const diffDays = Math.ceil((exp - now) / (1000 * 60 * 60 * 24));
    if (diffDays < 0) return `Expired ${Math.abs(diffDays)}d ago`;
    if (diffDays === 0) return 'Expires today';
    return `${diffDays}d remaining`;
  } catch (_) {
    return '';
  }
}

function _crmToast(message, type) {
  // Use existing campShowToast if available, else create simple toast
  if (typeof campShowToast === 'function') {
    campShowToast(message, type);
    return;
  }

  const toast = document.createElement('div');
  toast.className = `crm-toast crm-toast-${type || 'info'}`;
  toast.textContent = message;
  document.body.appendChild(toast);

  requestAnimationFrame(() => toast.classList.add('show'));
  setTimeout(() => {
    toast.classList.remove('show');
    setTimeout(() => toast.remove(), 300);
  }, 2500);
}

// ─── Conversation click integration ────────────────────────────────────────────

function crmOnConversationChange(mobile) {
  // If CRM panel is open, refresh it with new customer
  if (crmState.open && mobile && mobile !== crmState.mobile) {
    openCrmPanel(mobile);
  }
}

// ─── Auto-init ─────────────────────────────────────────────────────────────────

// Initialize once DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initCrmPanel);
} else {
  initCrmPanel();
}
