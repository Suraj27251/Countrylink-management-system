/**
 * Renewals Panel - Embedded IMS Renewal Dashboard
 * Fetches data from /ims/api/renewals endpoints and renders inline.
 */

const REN_API = '/ims/api/renewals';

const renState = {
  category: 'all',
  search: '',
  page: 1,
  perPage: 30,
  records: [],
  selectedIds: new Set(),
  loaded: false,
};

const REN_TEMPLATES = {
  expired: 'pack_expiry_alert',
  today: 'recharge_today1',
  upcoming: 'recharge_reminder',
};

// Called when workspace switches to renewals
function initRenewalsPanel() {
  if (!renState.loaded) {
    bindRenewalsEvents();
    renState.loaded = true;
  }
  loadRenStats();
  loadRenRecords();
}

function bindRenewalsEvents() {
  // Category buttons
  document.querySelectorAll('#ren-category-btns .btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('#ren-category-btns .btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      renState.category = btn.dataset.cat;
      renState.page = 1;
      loadRenRecords();
    });
  });

  // Search
  let renSearchTimer;
  document.getElementById('ren-search').addEventListener('input', (e) => {
    clearTimeout(renSearchTimer);
    renSearchTimer = setTimeout(() => {
      renState.search = e.target.value.trim();
      renState.page = 1;
      loadRenRecords();
    }, 400);
  });

  // Select all
  document.getElementById('ren-select-all').addEventListener('change', (e) => {
    renState.records.forEach(r => {
      if (e.target.checked) renState.selectedIds.add(r.id);
      else renState.selectedIds.delete(r.id);
    });
    document.querySelectorAll('.ren-row-check').forEach(cb => cb.checked = e.target.checked);
    updateRenSelCount();
  });

  // Sync button
  document.getElementById('ren-sync-btn').addEventListener('click', renSync);

  // Bulk send
  document.getElementById('ren-bulk-btn').addEventListener('click', renBulkSend);
}

// ─── Data Loading ────────────────────────────────────────

async function loadRenStats() {
  try {
    const res = await fetch(`${REN_API}/stats`);
    const data = await res.json();
    if (data.success) {
      document.getElementById('ren-stat-total').textContent = data.stats.total;
      document.getElementById('ren-stat-expired').textContent = data.stats.expired;
      document.getElementById('ren-stat-today').textContent = data.stats.today;
      document.getElementById('ren-stat-upcoming').textContent = data.stats.upcoming;
      document.getElementById('ren-stat-sent').textContent = data.stats.sent_today;
    }
  } catch (e) {
    console.error('[Renewals] Stats load failed:', e);
  }
}

async function loadRenRecords() {
  const tbody = document.getElementById('ren-tbody');
  tbody.innerHTML = '<tr><td colspan="8" style="padding:30px;text-align:center;color:var(--text-3);"><i class="fas fa-spinner fa-spin"></i> Loading...</td></tr>';

  const params = new URLSearchParams({ page: renState.page, per_page: renState.perPage });
  if (renState.category !== 'all') params.set('category', renState.category);
  if (renState.search) params.set('search', renState.search);

  try {
    const res = await fetch(`${REN_API}/?${params}`);
    const data = await res.json();
    if (data.success) {
      renState.records = data.data;
      renderRenTable(data.data);
      renderRenPagination(data.pagination);
    } else {
      tbody.innerHTML = `<tr><td colspan="8" style="padding:30px;text-align:center;color:#dc3545;">${data.error || 'Failed to load'}</td></tr>`;
    }
  } catch (e) {
    tbody.innerHTML = '<tr><td colspan="8" style="padding:30px;text-align:center;color:#dc3545;">Cannot connect to IMS API</td></tr>';
  }
}

// ─── Table Rendering ─────────────────────────────────────

function renderRenTable(records) {
  const tbody = document.getElementById('ren-tbody');
  if (!records || records.length === 0) {
    tbody.innerHTML = '<tr><td colspan="8" style="padding:30px;text-align:center;color:var(--text-3);">No renewal records found</td></tr>';
    return;
  }

  tbody.innerHTML = records.map(r => {
    const checked = renState.selectedIds.has(r.id) ? 'checked' : '';
    const badge = renBadge(r.category);
    const days = renDaysLabel(r.days_remaining);
    const name = escHtml(r.customer_name || '--');
    const mobile = escHtml(r.mobile || '--');
    const plan = escHtml(r.plan_name || '--');
    const expiry = r.expiry_date || '--';

    return `<tr style="border-bottom:1px solid var(--border);">
      <td style="padding:6px 10px;"><input type="checkbox" class="ren-row-check" data-id="${r.id}" ${checked}></td>
      <td style="padding:6px 10px;font-weight:500;">${name}</td>
      <td style="padding:6px 10px;font-family:'DM Mono',monospace;font-size:11px;">${mobile}</td>
      <td style="padding:6px 10px;font-size:11px;max-width:160px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${plan}</td>
      <td style="padding:6px 10px;white-space:nowrap;">${expiry}</td>
      <td style="padding:6px 10px;">${days}</td>
      <td style="padding:6px 10px;">${badge}</td>
      <td style="padding:6px 10px;">
        <button class="btn btn-sm btn-success" style="padding:2px 8px;font-size:11px;" onclick="renSendOne(${r.id})">
          <i class="fab fa-whatsapp"></i> Send
        </button>
      </td>
    </tr>`;
  }).join('');

  // Bind checkboxes
  tbody.querySelectorAll('.ren-row-check').forEach(cb => {
    cb.addEventListener('change', (e) => {
      const id = parseInt(e.target.dataset.id);
      if (e.target.checked) renState.selectedIds.add(id);
      else renState.selectedIds.delete(id);
      updateRenSelCount();
    });
  });
}

function renBadge(cat) {
  if (cat === 'expired') return '<span style="background:#dc3545;color:#fff;padding:2px 8px;border-radius:10px;font-size:10px;font-weight:600;">EXPIRED</span>';
  if (cat === 'today') return '<span style="background:#fd7e14;color:#fff;padding:2px 8px;border-radius:10px;font-size:10px;font-weight:600;">TODAY</span>';
  if (cat === 'upcoming') return '<span style="background:#198754;color:#fff;padding:2px 8px;border-radius:10px;font-size:10px;font-weight:600;">UPCOMING</span>';
  return '<span style="background:#6c757d;color:#fff;padding:2px 8px;border-radius:10px;font-size:10px;">--</span>';
}

function renDaysLabel(days) {
  if (days === null || days === undefined) return '--';
  if (days < 0) return `<span style="color:#dc3545;font-weight:700;">${days}d</span>`;
  if (days === 0) return '<span style="color:#fd7e14;font-weight:700;">Today</span>';
  return `<span style="color:#198754;font-weight:600;">${days}d</span>`;
}

// ─── Pagination ──────────────────────────────────────────

function renderRenPagination(pg) {
  const { page, per_page, total, total_pages } = pg;
  const start = total > 0 ? (page - 1) * per_page + 1 : 0;
  const end = Math.min(page * per_page, total);
  document.getElementById('ren-page-info').textContent = total > 0 ? `Showing ${start}-${end} of ${total}` : 'No records';

  const container = document.getElementById('ren-pagination');
  if (total_pages <= 1) { container.innerHTML = ''; return; }

  let html = `<button class="btn btn-outline-secondary btn-sm" ${page <= 1 ? 'disabled' : ''} onclick="renGoPage(${page - 1})">&laquo;</button>`;
  const startP = Math.max(1, page - 2);
  const endP = Math.min(total_pages, page + 2);
  for (let i = startP; i <= endP; i++) {
    html += `<button class="btn btn-${i === page ? 'primary' : 'outline-secondary'} btn-sm" onclick="renGoPage(${i})">${i}</button>`;
  }
  html += `<button class="btn btn-outline-secondary btn-sm" ${page >= total_pages ? 'disabled' : ''} onclick="renGoPage(${page + 1})">&raquo;</button>`;
  container.innerHTML = html;
}

function renGoPage(p) {
  renState.page = p;
  loadRenRecords();
}

// ─── Actions ─────────────────────────────────────────────

async function renSendOne(recordId) {
  const record = renState.records.find(r => r.id === recordId);
  if (!record || !record.mobile) return;

  const template = REN_TEMPLATES[record.category] || 'recharge_reminder';
  const params = [
    record.customer_name || 'Customer',
    record.account_id || '',
    record.expiry_date || '',
  ];

  if (!confirm(`Send "${template}" to ${record.customer_name} (${record.mobile})?`)) return;

  try {
    const res = await fetch(`${REN_API}/send`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        renewal_id: recordId,
        template_name: template,
        params: params,
        operator_name: 'whatsapp_inbox',
      }),
    });
    const data = await res.json();
    if (data.success) {
      alert('✅ Message sent successfully!');
      loadRenRecords();
      loadRenStats();
    } else {
      alert('❌ ' + (data.error || 'Send failed'));
    }
  } catch (e) {
    alert('❌ Network error');
  }
}

async function renBulkSend() {
  const count = renState.selectedIds.size;
  if (count === 0) return;
  if (!confirm(`Send WhatsApp reminders to ${count} selected customers?`)) return;

  try {
    const res = await fetch(`${REN_API}/bulk-send`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        renewal_ids: Array.from(renState.selectedIds),
        operator_name: 'whatsapp_inbox',
      }),
    });
    const data = await res.json();
    if (data.success) {
      const r = data.results;
      alert(`✅ Bulk send: ${r.sent} sent, ${r.failed} failed, ${r.skipped} skipped`);
      renState.selectedIds.clear();
      updateRenSelCount();
      loadRenRecords();
      loadRenStats();
    } else {
      alert('❌ ' + (data.error || 'Bulk send failed'));
    }
  } catch (e) {
    alert('❌ Network error');
  }
}

async function renSync() {
  const btn = document.getElementById('ren-sync-btn');
  btn.disabled = true;
  btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';

  try {
    const res = await fetch(`${REN_API}/sync`, { method: 'POST' });
    const data = await res.json();
    if (data.success) {
      alert(`✅ Sync: ${data.sync.inserted} new, ${data.sync.updated} updated`);
      loadRenStats();
      loadRenRecords();
    } else {
      alert('❌ ' + (data.error || 'Sync failed'));
    }
  } catch (e) {
    alert('❌ Sync failed - check IMS connection');
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<i class="fas fa-rotate"></i> Sync';
  }
}

// ─── Helpers ─────────────────────────────────────────────

function updateRenSelCount() {
  const count = renState.selectedIds.size;
  document.getElementById('ren-sel-count').textContent = count;
  document.getElementById('ren-bulk-btn').disabled = count === 0;
}

function escHtml(text) {
  if (!text) return '';
  const d = document.createElement('div');
  d.textContent = text;
  return d.innerHTML;
}
