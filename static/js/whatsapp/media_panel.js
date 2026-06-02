/**
 * Media Library & Notification Panel
 * Provides media upload with drag-and-drop, grid view with thumbnails/search/type filter,
 * file metadata and usage count display, notification bell indicator with unread count,
 * notification panel with alert list, severity badges, and acknowledge button.
 *
 * APIs:
 *   POST /api/media/upload
 *   GET  /api/media/
 *   GET  /api/notifications/count
 *   GET  /api/notifications/unacknowledged
 *   POST /api/notifications/<id>/acknowledge
 *
 * Requirements: 14.3, 14.4, 25.6, 10.3
 */

const MEDIA_API = '/api/media';
const NOTIF_API = '/api/notifications';

// ─── Media Panel State ──────────────────────────────────────────────────────────

const mediaState = {
  assets: [],
  page: 1,
  perPage: 20,
  total: 0,
  totalPages: 0,
  search: '',
  mediaType: '',
  loaded: false,
  loading: false,
  uploading: false,
  dragOver: false,
};

// ─── Notification State ─────────────────────────────────────────────────────────

const notifState = {
  unreadCount: 0,
  notifications: [],
  panelOpen: false,
  polling: null,
};

// ═══════════════════════════════════════════════════════════════════════════════════
// MEDIA LIBRARY
// ═══════════════════════════════════════════════════════════════════════════════════

function initMediaPanel() {
  if (!mediaState.loaded) {
    mediaState.loaded = true;
  }
  mediaShowGrid();
}

function mediaShowGrid() {
  const container = document.getElementById('mediaPanelContent');
  if (!container) return;

  container.innerHTML = `
    <div class="media-header">
      <h3><i class="fas fa-photo-film"></i> Media Library</h3>
      <div class="media-actions">
        <button class="camp-btn camp-btn-primary" onclick="mediaShowUpload()">
          <i class="fas fa-cloud-upload-alt"></i> Upload Media
        </button>
      </div>
    </div>

    <!-- Filters -->
    <div class="media-filters">
      <div class="media-search-wrap">
        <i class="fas fa-magnifying-glass"></i>
        <input type="text" id="mediaSearchInput" placeholder="Search files..."
               value="${mediaState.search}" oninput="mediaOnSearch(this.value)">
      </div>
      <select id="mediaTypeFilter" class="media-select" onchange="mediaOnTypeFilter(this.value)">
        <option value="">All types</option>
        <option value="image" ${mediaState.mediaType === 'image' ? 'selected' : ''}>Images</option>
        <option value="video" ${mediaState.mediaType === 'video' ? 'selected' : ''}>Videos</option>
        <option value="document" ${mediaState.mediaType === 'document' ? 'selected' : ''}>Documents</option>
      </select>
      <span class="media-count" id="mediaCountLabel">-</span>
    </div>

    <!-- Upload drop zone (hidden by default, shown via upload button) -->
    <div class="media-dropzone" id="mediaDropzone"
         ondragover="mediaDragOver(event)" ondragleave="mediaDragLeave(event)" ondrop="mediaDrop(event)">
      <div class="dropzone-content">
        <i class="fas fa-cloud-upload-alt"></i>
        <p>Drag & drop files here or <label for="mediaFileInput" class="dropzone-browse">browse</label></p>
        <span class="dropzone-hint">Images (≤5MB) • Videos (≤16MB) • Documents (≤100MB)</span>
        <input type="file" id="mediaFileInput" multiple accept="image/*,video/*,.pdf,.doc,.docx,.xls,.xlsx,.ppt,.pptx,.txt,.csv"
               onchange="mediaOnFileSelect(this.files)" style="display:none;">
      </div>
      <div class="dropzone-progress" id="dropzoneProgress" style="display:none;">
        <div class="dropzone-progress-bar" id="dropzoneProgressBar"></div>
        <span class="dropzone-progress-text" id="dropzoneProgressText">Uploading...</span>
      </div>
    </div>

    <!-- Grid -->
    <div class="media-grid" id="mediaGrid">
      <div class="media-grid-loading">Loading media...</div>
    </div>

    <!-- Pagination -->
    <div class="media-pagination" id="mediaPagination"></div>
  `;

  mediaFetchAssets();
}

function mediaShowUpload() {
  const dropzone = document.getElementById('mediaDropzone');
  if (dropzone) {
    dropzone.classList.toggle('visible');
  }
}

// ─── Drag and Drop ──────────────────────────────────────────────────────────────

function mediaDragOver(e) {
  e.preventDefault();
  e.stopPropagation();
  const dropzone = document.getElementById('mediaDropzone');
  if (dropzone) dropzone.classList.add('drag-active');
}

function mediaDragLeave(e) {
  e.preventDefault();
  e.stopPropagation();
  const dropzone = document.getElementById('mediaDropzone');
  if (dropzone) dropzone.classList.remove('drag-active');
}

function mediaDrop(e) {
  e.preventDefault();
  e.stopPropagation();
  const dropzone = document.getElementById('mediaDropzone');
  if (dropzone) dropzone.classList.remove('drag-active');

  const files = e.dataTransfer?.files;
  if (files && files.length > 0) {
    mediaUploadFiles(files);
  }
}

function mediaOnFileSelect(files) {
  if (files && files.length > 0) {
    mediaUploadFiles(files);
  }
}

// ─── Upload ─────────────────────────────────────────────────────────────────────

async function mediaUploadFiles(files) {
  if (mediaState.uploading) return;
  mediaState.uploading = true;

  const progressEl = document.getElementById('dropzoneProgress');
  const progressBar = document.getElementById('dropzoneProgressBar');
  const progressText = document.getElementById('dropzoneProgressText');

  if (progressEl) progressEl.style.display = 'flex';

  let uploaded = 0;
  const total = files.length;
  const errors = [];

  for (let i = 0; i < files.length; i++) {
    const file = files[i];
    if (progressText) progressText.textContent = `Uploading ${i + 1}/${total}: ${file.name}`;
    if (progressBar) progressBar.style.width = `${((i) / total) * 100}%`;

    try {
      const formData = new FormData();
      formData.append('file', file);

      const resp = await fetch(`${MEDIA_API}/upload`, {
        method: 'POST',
        body: formData,
      });

      const data = await resp.json();
      if (!resp.ok) {
        errors.push(`${file.name}: ${data.error || 'Upload failed'}`);
      } else {
        uploaded++;
      }
    } catch (err) {
      errors.push(`${file.name}: Network error`);
    }
  }

  if (progressBar) progressBar.style.width = '100%';
  if (progressText) {
    if (errors.length > 0) {
      progressText.textContent = `Done: ${uploaded}/${total} uploaded. ${errors.length} failed.`;
      progressText.title = errors.join('\n');
    } else {
      progressText.textContent = `All ${uploaded} file(s) uploaded successfully!`;
    }
  }

  mediaState.uploading = false;

  // Refresh grid after short delay
  setTimeout(() => {
    if (progressEl) progressEl.style.display = 'none';
    if (progressBar) progressBar.style.width = '0%';
    mediaState.page = 1;
    mediaFetchAssets();
  }, 2000);
}

// ─── Search & Filter ────────────────────────────────────────────────────────────

let mediaSearchTimeout = null;
function mediaOnSearch(value) {
  clearTimeout(mediaSearchTimeout);
  mediaSearchTimeout = setTimeout(() => {
    mediaState.search = value;
    mediaState.page = 1;
    mediaFetchAssets();
  }, 300);
}

function mediaOnTypeFilter(value) {
  mediaState.mediaType = value;
  mediaState.page = 1;
  mediaFetchAssets();
}

// ─── Fetch Assets ───────────────────────────────────────────────────────────────

async function mediaFetchAssets() {
  if (mediaState.loading) return;
  mediaState.loading = true;

  const params = new URLSearchParams({
    page: mediaState.page,
    per_page: mediaState.perPage,
  });
  if (mediaState.search) params.append('search', mediaState.search);
  if (mediaState.mediaType) params.append('media_type', mediaState.mediaType);

  try {
    const resp = await fetch(`${MEDIA_API}/?${params}`);
    if (!resp.ok) throw new Error('Failed to fetch media');
    const data = await resp.json();

    mediaState.assets = data.assets || [];
    mediaState.total = data.pagination?.total || 0;
    mediaState.totalPages = data.pagination?.total_pages || 0;

    mediaRenderGrid();
    mediaRenderPagination();
  } catch (err) {
    const grid = document.getElementById('mediaGrid');
    if (grid) grid.innerHTML = '<div class="media-grid-empty"><i class="fas fa-exclamation-triangle"></i><p>Failed to load media</p></div>';
  } finally {
    mediaState.loading = false;
  }
}

// ─── Render Grid ────────────────────────────────────────────────────────────────

function mediaRenderGrid() {
  const grid = document.getElementById('mediaGrid');
  const countLabel = document.getElementById('mediaCountLabel');
  if (!grid) return;

  if (countLabel) countLabel.textContent = `${mediaState.total} file${mediaState.total !== 1 ? 's' : ''}`;

  if (mediaState.assets.length === 0) {
    grid.innerHTML = `
      <div class="media-grid-empty">
        <i class="fas fa-photo-film"></i>
        <p>No media files found</p>
        <span>Upload images, videos, or documents to use in campaigns</span>
      </div>
    `;
    return;
  }

  grid.innerHTML = mediaState.assets.map(asset => {
    const thumb = mediaGetThumbnail(asset);
    const size = mediaFormatSize(asset.file_size_bytes);
    const date = mediaFormatDate(asset.created_at);
    const typeIcon = mediaTypeIcon(asset.media_type);

    return `
      <div class="media-card" title="${asset.original_filename}">
        <div class="media-thumb">
          ${thumb}
          <span class="media-type-badge media-type-${asset.media_type}">
            <i class="${typeIcon}"></i> ${asset.media_type}
          </span>
        </div>
        <div class="media-info">
          <div class="media-name">${escapeHtml(asset.original_filename)}</div>
          <div class="media-meta">
            <span>${size}</span>
            <span>•</span>
            <span>${date}</span>
          </div>
          <div class="media-meta">
            <span class="media-usage" title="Used in ${asset.usage_count} template(s)">
              <i class="fas fa-link"></i> ${asset.usage_count} use${asset.usage_count !== 1 ? 's' : ''}
            </span>
            <span>•</span>
            <span>${asset.uploaded_by || 'Unknown'}</span>
          </div>
        </div>
      </div>
    `;
  }).join('');
}

function mediaGetThumbnail(asset) {
  if (asset.media_type === 'image') {
    const src = asset.storage_path ? `/${asset.storage_path}` : '';
    return `<img src="${src}" alt="${escapeHtml(asset.original_filename)}" loading="lazy" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'"><div class="media-thumb-fallback" style="display:none;"><i class="fas fa-image"></i></div>`;
  } else if (asset.media_type === 'video') {
    return `<div class="media-thumb-fallback"><i class="fas fa-video"></i></div>`;
  } else {
    return `<div class="media-thumb-fallback"><i class="fas fa-file-alt"></i></div>`;
  }
}

function mediaTypeIcon(type) {
  switch (type) {
    case 'image': return 'fas fa-image';
    case 'video': return 'fas fa-video';
    case 'document': return 'fas fa-file-alt';
    default: return 'fas fa-file';
  }
}

function mediaFormatSize(bytes) {
  if (!bytes || bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

function mediaFormatDate(dateStr) {
  if (!dateStr) return '-';
  const d = new Date(dateStr);
  if (isNaN(d)) return '-';
  const now = new Date();
  const diff = now - d;
  if (diff < 60000) return 'Just now';
  if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
  if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
  if (diff < 604800000) return `${Math.floor(diff / 86400000)}d ago`;
  return d.toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' });
}

// ─── Pagination ─────────────────────────────────────────────────────────────────

function mediaRenderPagination() {
  const container = document.getElementById('mediaPagination');
  if (!container) return;

  if (mediaState.totalPages <= 1) {
    container.innerHTML = '';
    return;
  }

  let html = '';
  html += `<button class="media-page-btn" ${mediaState.page <= 1 ? 'disabled' : ''} onclick="mediaGoPage(${mediaState.page - 1})"><i class="fas fa-chevron-left"></i></button>`;
  html += `<span class="media-page-info">Page ${mediaState.page} of ${mediaState.totalPages}</span>`;
  html += `<button class="media-page-btn" ${mediaState.page >= mediaState.totalPages ? 'disabled' : ''} onclick="mediaGoPage(${mediaState.page + 1})"><i class="fas fa-chevron-right"></i></button>`;

  container.innerHTML = html;
}

function mediaGoPage(page) {
  if (page < 1 || page > mediaState.totalPages) return;
  mediaState.page = page;
  mediaFetchAssets();
}

// ═══════════════════════════════════════════════════════════════════════════════════
// NOTIFICATION PANEL
// ═══════════════════════════════════════════════════════════════════════════════════

function initNotificationPanel() {
  // Start polling for unread count
  notifFetchCount();
  notifState.polling = setInterval(notifFetchCount, 30000); // Poll every 30s

  // Attach click handler to notification bell
  const bellBtn = document.getElementById('notifBellBtn');
  if (bellBtn) {
    bellBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      notifTogglePanel();
    });
  }

  // Close panel on outside click
  document.addEventListener('click', (e) => {
    if (notifState.panelOpen) {
      const panel = document.getElementById('notifPanel');
      if (panel && !panel.contains(e.target)) {
        notifClosePanel();
      }
    }
  });
}

// ─── Count Badge ────────────────────────────────────────────────────────────────

async function notifFetchCount() {
  try {
    const resp = await fetch(`${NOTIF_API}/count`);
    if (!resp.ok) return;
    const data = await resp.json();
    notifState.unreadCount = data.count || 0;
    notifUpdateBadge();
  } catch (err) {
    // Silent fail — polling will retry
  }
}

function notifUpdateBadge() {
  const badge = document.getElementById('notifBadge');
  if (!badge) return;

  if (notifState.unreadCount > 0) {
    badge.textContent = notifState.unreadCount > 99 ? '99+' : notifState.unreadCount;
    badge.style.display = 'flex';
  } else {
    badge.style.display = 'none';
  }
}

// ─── Panel Toggle ───────────────────────────────────────────────────────────────

function notifTogglePanel() {
  if (notifState.panelOpen) {
    notifClosePanel();
  } else {
    notifOpenPanel();
  }
}

async function notifOpenPanel() {
  notifState.panelOpen = true;
  let panel = document.getElementById('notifPanel');

  if (!panel) {
    panel = document.createElement('div');
    panel.id = 'notifPanel';
    panel.className = 'notif-panel';
    document.body.appendChild(panel);
  }

  panel.classList.add('open');
  panel.innerHTML = `
    <div class="notif-panel-header">
      <h4><i class="fas fa-bell"></i> Notifications</h4>
      <button class="notif-close-btn" onclick="notifClosePanel()"><i class="fas fa-xmark"></i></button>
    </div>
    <div class="notif-panel-body" id="notifPanelBody">
      <div class="notif-loading"><i class="fas fa-spinner fa-spin"></i> Loading...</div>
    </div>
  `;

  // Position panel near the bell button
  const bellBtn = document.getElementById('notifBellBtn');
  if (bellBtn) {
    const rect = bellBtn.getBoundingClientRect();
    panel.style.top = `${rect.bottom + 8}px`;
    panel.style.right = `${window.innerWidth - rect.right}px`;
  }

  await notifFetchAlerts();
}

function notifClosePanel() {
  notifState.panelOpen = false;
  const panel = document.getElementById('notifPanel');
  if (panel) panel.classList.remove('open');
}

// ─── Fetch Alerts ───────────────────────────────────────────────────────────────

async function notifFetchAlerts() {
  try {
    const resp = await fetch(`${NOTIF_API}/unacknowledged`);
    if (!resp.ok) throw new Error('Failed to fetch notifications');
    const data = await resp.json();
    notifState.notifications = data.notifications || [];
    notifRenderAlerts();
  } catch (err) {
    const body = document.getElementById('notifPanelBody');
    if (body) body.innerHTML = '<div class="notif-empty"><i class="fas fa-exclamation-triangle"></i><p>Failed to load notifications</p></div>';
  }
}

// ─── Render Alerts ──────────────────────────────────────────────────────────────

function notifRenderAlerts() {
  const body = document.getElementById('notifPanelBody');
  if (!body) return;

  if (notifState.notifications.length === 0) {
    body.innerHTML = `
      <div class="notif-empty">
        <i class="fas fa-check-circle"></i>
        <p>All caught up!</p>
        <span>No unacknowledged alerts</span>
      </div>
    `;
    return;
  }

  body.innerHTML = notifState.notifications.map(notif => {
    const severity = notif.severity || 'info';
    const severityIcon = notifSeverityIcon(severity);
    const time = mediaFormatDate(notif.created_at);
    const title = escapeHtml(notif.title || 'Alert');
    const details = notif.details ? escapeHtml(JSON.stringify(notif.details).slice(0, 100)) : '';

    return `
      <div class="notif-item notif-severity-${severity}">
        <div class="notif-item-icon">
          <i class="${severityIcon}"></i>
        </div>
        <div class="notif-item-content">
          <div class="notif-item-header">
            <span class="notif-severity-badge notif-badge-${severity}">${severity}</span>
            <span class="notif-item-time">${time}</span>
          </div>
          <div class="notif-item-title">${title}</div>
          ${details ? `<div class="notif-item-details">${details}</div>` : ''}
          <div class="notif-item-type">${escapeHtml(notif.alert_type || '')}</div>
        </div>
        <button class="notif-ack-btn" onclick="notifAcknowledge(${notif.id})" title="Acknowledge">
          <i class="fas fa-check"></i>
        </button>
      </div>
    `;
  }).join('');
}

function notifSeverityIcon(severity) {
  switch (severity) {
    case 'critical': return 'fas fa-circle-exclamation';
    case 'warning': return 'fas fa-triangle-exclamation';
    case 'info': return 'fas fa-circle-info';
    default: return 'fas fa-bell';
  }
}

// ─── Acknowledge ────────────────────────────────────────────────────────────────

async function notifAcknowledge(id) {
  try {
    const resp = await fetch(`${NOTIF_API}/${id}/acknowledge`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    });

    if (resp.ok) {
      // Remove from local state and re-render
      notifState.notifications = notifState.notifications.filter(n => n.id !== id);
      notifState.unreadCount = Math.max(0, notifState.unreadCount - 1);
      notifUpdateBadge();
      notifRenderAlerts();
    }
  } catch (err) {
    // Silent fail
  }
}

// ─── Utility ────────────────────────────────────────────────────────────────────

function escapeHtml(str) {
  if (!str) return '';
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

// ─── Initialize on load ─────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  initNotificationPanel();
});
