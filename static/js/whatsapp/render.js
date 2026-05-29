/**
 * whatsapp/render.js — Render & DOM mutation engine for WhatsApp inbox.
 *
 * PURPOSE:
 *   Centralize ALL DOM rendering and mutation logic into ONE module.
 *   This is the ONLY module responsible for DOM updates.
 *
 *   - polling.js fetches data
 *   - state.js stores state
 *   - render.js updates DOM
 *
 * INITIALIZATION:
 *   Must call window.renderEngine.init() before use:
 *     window.renderEngine.init({
 *       staticBase,          // base URL for static assets (replaces window.STATIC_BASE)
 *       pageUrl,             // base URL for page links (replaces window.PAGE_URL)
 *       dom: {
 *         chatBody,          // DOM element: chat body container
 *         convList,          // DOM element: sidebar conversation list
 *         scrollBtn          // DOM element: scroll-to-bottom button
 *       },
 *       helpers: {
 *         esc,               // HTML escaper (shared with HTML IIFE)
 *         fmtTime,           // timestamp formatter
 *         applyComposerWindowPolicy  // composer enable/disable policy
 *       }
 *     });
 *
 * @see Phase 4.1 of WhatsApp inbox modularization.
 */

(function () {
  'use strict';

  /* ── Guard: prevent double-initialization ──────────────── */
  if (window.__renderEngineInitialized) {
    console.debug('[RENDER] Already initialized — skipping.');
    return;
  }

  /* ── DOM helpers (local copies, not dependent on HTML IIFE) ─ */

  /* ── Message node registry (faster upserts, no duplicate nodes) ── */
  const messageNodeMap = new Map();

  /* ── Render queue protection ──────────────────────────── */
  let isRenderInProgress = false;
  let pendingRenderFrame = false;
  const RENDER_THROTTLE_MS = 50;
  const RENDER_QUEUE_MAX = 3;          // max queued refreshes before coalescing
  let queuedSidebarCount = 0;          // # of queued sidebar refreshes (for coalescing)
  let queuedActiveChatCount = 0;       // # of queued active chat refreshes
  const STALE_RENDER_TIMEOUT_MS = 5000; // max ms a render can run before stuck detection
  let lastSidebarRenderTime = 0;
  let lastActiveChatRenderTime = 0;

  /* ── Render activity tracking (for external stuck detection) ─ */
  let _lastRenderActivityAt = Date.now();
  const _markRenderActive = () => { _lastRenderActivityAt = Date.now(); };
  const _getRenderStaleMs = () => {
    if (!isRenderInProgress) return 0;
    return Date.now() - _lastRenderActivityAt;
  };

  /* ═══════════════════════════════════════════════════════════
     INJECTED DEPENDENCIES (set by init())
     ═══════════════════════════════════════════════════════════ */

  let staticBase = '';
  let pageUrl    = '';

  const dom = {
    chatBody:  null,
    convList:  null,
    scrollBtn: null
  };

  // Helper functions — overridable via init() helpers
  let esc = v => String(v||'').replace(/[&<>'"]/g,
    m => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[m]));

  let fmtTime = ts => (ts && ts.includes(' ')) ? ts.split(' ')[1].slice(0,5) : (ts||'').slice(0,5);

  let applyComposerWindowPolicy = () => {
    const msgInput = document.getElementById('msgInput');
    const status = window.inboxState.chatWindowStatusByMobile.get(window.inboxState.activeMobile) || 'active';
    const isInactive = status === 'inactive';
    if (msgInput) {
      msgInput.disabled = isInactive;
      if (isInactive) msgInput.placeholder = '24h window expired. Send a template to reopen chat.';
      else msgInput.placeholder = 'Write a message…';
    }
    const sendBtn = document.getElementById('sendBtn');
    if (sendBtn) sendBtn.disabled = isInactive;
  };

  /* ═══════════════════════════════════════════════════════════
     INITIALIZATION
     ═══════════════════════════════════════════════════════════ */

  /**
   * One-time initialization with dependency injection.
   *
   * @param {Object} config
   * @param {string} config.staticBase  — Base URL for static assets
   * @param {string} config.pageUrl     — Base URL for page links
   * @param {Object} config.dom
   * @param {Element} config.dom.chatBody   — Chat body container
   * @param {Element} config.dom.convList   — Sidebar conversation list
   * @param {Element} config.dom.scrollBtn  — Scroll-to-bottom button
   * @param {Object}  [config.helpers]
   * @param {Function} config.helpers.esc   — HTML escaper
   * @param {Function} config.helpers.fmtTime — Timestamp formatter
   * @param {Function} config.helpers.applyComposerWindowPolicy — Composer policy
   */
  const init = (config) => {
    if (window.__renderEngineInitialized) return;
    if (!config) { console.warn('[RENDER] init() requires config — skipping'); return; }

    staticBase = config.staticBase || '';
    pageUrl    = config.pageUrl    || '';

    if (config.dom) {
      if (config.dom.chatBody)  dom.chatBody  = config.dom.chatBody;
      if (config.dom.convList)  dom.convList  = config.dom.convList;
      if (config.dom.scrollBtn) dom.scrollBtn = config.dom.scrollBtn;
      console.debug('[RENDER] DOM refs cached:', {
        chatBody: !!dom.chatBody,
        convList: !!dom.convList,
        scrollBtn: !!dom.scrollBtn
      });
    }

    if (config.helpers) {
      if (config.helpers.esc)   esc   = config.helpers.esc;
      if (config.helpers.fmtTime) fmtTime = config.helpers.fmtTime;
      if (config.helpers.applyComposerWindowPolicy) {
        applyComposerWindowPolicy = config.helpers.applyComposerWindowPolicy;
      }
      console.debug('[RENDER] Helper deps injected: esc, fmtTime, applyComposerWindowPolicy');
    }

    window.__renderEngineInitialized = true;
    console.debug('[RENDER] Engine initialized');
    console.debug('[MODULE]', 'render.js loaded');
  };

  /* ═══════════════════════════════════════════════════════════
     UTILITY HELPERS
     ═══════════════════════════════════════════════════════════ */

  const formatUnreadBadge = (count) => (count > 99 ? '99+' : String(count));

  const inferMediaKind = (m = {}) => {
    const type = (m.message_type || '').toLowerCase();
    if (['image', 'sticker'].includes(type)) return 'image';
    if (type === 'video' || type === 'gif') return 'video';
    if (type === 'audio') return 'audio';

    const candidate = `${m.file_name || ''} ${m.media_url || ''}`.toLowerCase();
    if (/\.(webp|png|jpe?g|bmp|svg)(\?|$)/.test(candidate)) return 'image';
    if (/\.(gif|mp4|mov|webm|mkv|3gp)(\?|$)/.test(candidate)) return 'video';
    if (/\.(mp3|ogg|wav|m4a|aac|opus)(\?|$)/.test(candidate)) return 'audio';
    return 'document';
  };

  /* ═══════════════════════════════════════════════════════════
     SCROLL HELPERS
     ═══════════════════════════════════════════════════════════ */

  const nearBottom = () => {
    const el = dom.chatBody;
    return el && (el.scrollHeight - el.scrollTop - el.clientHeight < 160);
  };

  const scrollBottom = () => {
    const el = dom.chatBody;
    console.debug('[SCROLL] scrollBottom triggered');
    return el && requestAnimationFrame(() => { el.scrollTop = el.scrollHeight; });
  };

  /* ═══════════════════════════════════════════════════════════
     CHAT WINDOW HELPERS
     ═══════════════════════════════════════════════════════════ */

  const isActiveChatWindow = createdAt => {
    if (!createdAt) return true;
    try {
      const created = new Date(createdAt);
      const now = new Date();
      const ageMs = now - created;
      const ageHours = ageMs / (1000 * 60 * 60);
      return ageHours < 24;
    } catch {
      return true;
    }
  };

  /* ── Green dot / unread badge update for sidebar ──────────── */
  const updateContactGreenDot = (mobile) => {
    const el = dom.convList;
    if (!el) return;
    const contactEl = el.querySelector(`[data-mobile="${CSS.escape(mobile)}"]`);
    if (!contactEl) return;

    const isActive = mobile === window.inboxState.activeMobile;
    const serverUnreadCount = Number(contactEl.dataset.unreadCount || 0);
    const unreadCount = !isActive ? serverUnreadCount : 0;
    const hasNewMessage = !isActive && unreadCount > 0;

    contactEl.classList.toggle('has-new', hasNewMessage);

    // Update unread badge
    let unreadEl = contactEl.querySelector('.conv-unread');
    if (unreadCount > 0) {
      if (!unreadEl) {
        contactEl.insertAdjacentHTML('beforeend', `<span class="conv-unread">${formatUnreadBadge(unreadCount)}</span>`);
      } else if (unreadEl.textContent !== formatUnreadBadge(unreadCount)) {
        unreadEl.textContent = formatUnreadBadge(unreadCount);
      }
    } else if (unreadEl) {
      unreadEl.remove();
    }

    // Move to top if has new message
    if (hasNewMessage && contactEl.parentElement === el && contactEl !== el.firstChild) {
      console.debug('[SIDEBAR] Moving', contactEl.dataset.name, '(' + mobile + ') to top');
      el.insertBefore(contactEl, el.firstChild);
    }

    console.debug('[RENDER] updateContactGreenDot for', mobile, ': hasNewMessage=' + hasNewMessage + ', unreadCount=' + unreadCount);
  };

  /* ═══════════════════════════════════════════════════════════
     STATE PRUNING HELPER
     ═══════════════════════════════════════════════════════════ */

  const prunePerChatState = contacts => {
    const activeMobile = window.inboxState.activeMobile || '';
    const knownMobiles = new Set((contacts || []).map(c => c?.mobile).filter(Boolean));
    if (activeMobile) knownMobiles.add(activeMobile);

    [window.inboxState.lastMessageIdByMobile, window.inboxState.lastSeenMessageIdByMobile, window.inboxState.unreadByMobile].forEach(stateMap => {
      for (const mobile of stateMap.keys()) {
        if (!knownMobiles.has(mobile)) stateMap.delete(mobile);
      }
    });
  };

  /* ═══════════════════════════════════════════════════════════
     MESSAGE RENDERING
     ═══════════════════════════════════════════════════════════ */

  const renderMsg = m => {
    console.debug('[RENDER MESSAGE] direction:', m.direction, 'id:', m.id);
    const row = document.createElement('div');
    row.className = `msg-row ${m.direction === 'outbound' ? 'outbound' : 'incoming'}`;
    row.dataset.messageId = m.id;
    row.dataset.createdAt = m.created_at || '';

    const status = (m.delivery_status||'').toLowerCase();
    const statusIcon = m.direction === 'outbound'
      ? (status === 'read'
        ? '<i class="fas fa-check-double msg-status-icon read" title="Read"></i>'
        : ['delivered','sent','accepted'].includes(status)
          ? '<i class="fas fa-check-double msg-status-icon" title="Delivered"></i>'
          : '<i class="fas fa-check msg-status-icon" title="Sent"></i>')
      : '';

    let mediaHtml = '';
    const mediaSrc = m.media_public_url || (m.media_url ? `${staticBase}${m.media_url}` : null);

    if (mediaSrc) {
      const mediaKind = inferMediaKind(m);
      if (mediaKind === 'image') {
        const isSticker = (m.message_type || '').toLowerCase() === 'sticker' || /\.webp(\?|$)/i.test(mediaSrc);
        const cls = isSticker ? 'msg-sticker msg-media' : 'msg-media';
        const alt = isSticker ? 'Sticker' : 'Image';
        mediaHtml = `<img class="${cls}" data-type="image" src="${esc(mediaSrc)}" alt="${alt}" loading="lazy" onerror="this.style.display='none';this.nextElementSibling&&this.nextElementSibling.classList.remove('hidden')">
                     <div class="msg-media-placeholder hidden"><i class="fas fa-image"></i> Photo</div>`;
      } else if (mediaKind === 'video') {
        mediaHtml = `<video class="msg-media" data-type="video" src="${esc(mediaSrc)}" controls preload="metadata"></video>`;
      } else if (mediaKind === 'audio') {
        mediaHtml = `<audio class="msg-audio" controls src="${esc(mediaSrc)}" preload="none"></audio>`;
      } else {
        mediaHtml = `<a class="msg-doc-link" href="${esc(mediaSrc)}" target="_blank" rel="noopener" download><i class="fas fa-file-arrow-down"></i><span class="msg-doc-meta"><span class="msg-doc-name">${esc(m.file_name||'Attachment')}</span><span class="msg-doc-type">${esc((m.message_type||'file').replace('_',' '))}</span></span></a>`;
      }
    } else if (['image', 'sticker', 'video', 'audio', 'document', 'gif'].includes((m.message_type || '').toLowerCase())) {
      // No media URL available — show placeholder based on type
      const typeIcons = { image: 'fa-image', sticker: 'fa-note-sticky', video: 'fa-video', audio: 'fa-headphones', document: 'fa-file', gif: 'fa-film' };
      const typeLabels = { image: 'Photo', sticker: 'Sticker', video: 'Video', audio: 'Audio', document: 'Document', gif: 'GIF' };
      const mType = (m.message_type || '').toLowerCase();
      const icon = typeIcons[mType] || 'fa-paperclip';
      const label = typeLabels[mType] || 'Attachment';
      mediaHtml = `<div class="msg-media-placeholder"><i class="fas ${icon}"></i> ${label}</div>`;
    }

    const locHtml = (m.message_type === 'location' && m.latitude != null && m.longitude != null)
      ? `<div class="msg-location-wrap">
           <iframe class="msg-location-frame" src="https://maps.google.com/maps?q=${m.latitude},${m.longitude}&z=15&output=embed" loading="lazy"></iframe>
           <a class="msg-location-link" href="https://www.google.com/maps/search/?api=1&query=${m.latitude},${m.longitude}" target="_blank" rel="noopener">
             <i class="fas fa-map-pin"></i> Open in Google Maps
           </a>
         </div>`
      : '';

    const fallbackText = m.message_type === 'template' ? '📋 Template message' : m.message_type === 'interactive' ? '🔘 Interactive message' : m.message_type === 'sticker' ? '🧩 Sticker' : '';
    const textContent = esc(m.text || fallbackText);
    const showText = Boolean((m.text || fallbackText || '').trim());
    const errorHtml = m.error_reason ? `<div class="msg-error-chip"><i class="fas fa-circle-exclamation"></i> ${esc(m.error_reason)}</div>` : '';
    const senderName = m.direction === 'inbound'
      ? `<div class="msg-sender-name">${esc(m.name || 'Customer')}</div>` : '';

    // AI badge for AI-generated outbound messages
    const isAiMsg = m.sender_type === 'ai' || m.is_ai === true;
    const aiBadgeHtml = isAiMsg
      ? '<span class="msg-ai-badge"><i class="fas fa-robot"></i> AI</span>'
      : '';

    row.innerHTML = `
    ${senderName}
    <div class="msg-bubble">
      ${aiBadgeHtml}
      ${locHtml}${mediaHtml}
      ${showText ? `<p class="msg-text">${textContent}</p>` : ''}
      ${errorHtml}
      <div class="msg-footer"><span class="msg-time">${fmtTime(m.created_at)}</span>${statusIcon}</div>
    </div>`;
    return row;
  };

  const upsertMsg = m => {
    const chatBodyEl = dom.chatBody;
    if (!chatBodyEl) return false;
    const idStr = String(m.id);

    // RECONCILIATION LOGIC: Check if this is a real message that should replace an optimistic one
    const isRealMessage = !String(m.id).startsWith('optimistic_');
    if (isRealMessage && m.direction === 'outbound' && window.inboxState.optimisticMessageIds?.size > 0) {
      // Look for optimistic messages with matching text
      for (const optimisticId of window.inboxState.optimisticMessageIds) {
        const optimisticNode = chatBodyEl.querySelector(`[data-message-id="${CSS.escape(optimisticId)}"]`);
        if (optimisticNode) {
          const optimisticText = optimisticNode.querySelector('.msg-text')?.textContent?.trim() || '';
          const incomingText = m.text?.trim() || '';
          if (optimisticText === incomingText && m.direction === 'outbound') {
            // MATCH FOUND: Replace optimistic message's ID with real message ID
            console.debug('[RECONCILIATION] Found matching optimistic message:', optimisticId, '→', idStr);
            messageNodeMap.delete(optimisticId);
            optimisticNode.dataset.messageId = idStr;
            messageNodeMap.set(idStr, optimisticNode);
            window.inboxState.globalKnownMessageIds.delete(optimisticId);
            window.inboxState.globalKnownMessageIds.add(idStr);
            window.inboxState.optimisticMessageIds.delete(optimisticId);
            
            // Patch status icons on the existing node
            const status = (m.delivery_status || '').toLowerCase();
            const statusIcon = m.direction === 'outbound'
              ? (status === 'read'
                ? '<i class="fas fa-check-double msg-status-icon read" title="Read"></i>'
                : ['delivered', 'sent', 'accepted'].includes(status)
                  ? '<i class="fas fa-check-double msg-status-icon" title="Delivered"></i>'
                  : '<i class="fas fa-check msg-status-icon" title="Sent"></i>')
              : '';
            
            const patchStatus = (selector, html) => {
              let el = optimisticNode.querySelector(selector);
              if (html && !el) {
                el = document.createElement('div');
                const timeEl = optimisticNode.querySelector('.msg-time');
                timeEl ? timeEl.before(el) : optimisticNode.querySelector('.msg-bubble')?.appendChild(el);
              }
              if (el) el.outerHTML = html || '';
            };
            
            patchStatus('.msg-status-icon', statusIcon);
            
            const errorChip = m.error_reason ? `<div class="msg-error-chip"><i class="fas fa-circle-exclamation"></i> ${esc(m.error_reason)}</div>` : '';
            patchStatus('.msg-error-chip', errorChip);
            
            console.debug('[RECONCILIATION] Optimistic message reconciled successfully:', idStr);
            return false;
          }
        }
      }
    }

    // Check message node registry first (faster than querySelector)
    let domNode = messageNodeMap.get(idStr);
    if (domNode && !chatBodyEl.contains(domNode)) {
      // Stale entry — clean up
      messageNodeMap.delete(idStr);
      domNode = null;
    }
    if (!domNode) {
      domNode = chatBodyEl.querySelector(`[data-message-id="${CSS.escape(idStr)}"]`);
      if (domNode) messageNodeMap.set(idStr, domNode);
    }

    // DOM is the source of truth; knownIds is only a best-effort cache.
    if (!domNode) {
      const emptyHint = document.getElementById('chatEmptyHint');
      emptyHint?.remove();
      const newNode = renderMsg(m);
      chatBodyEl.appendChild(newNode);
      messageNodeMap.set(idStr, newNode);
      console.debug('[MESSAGE_NODE] Registered', idStr);
      const isNew = !window.inboxState.globalKnownMessageIds.has(idStr);
      window.inboxState.globalKnownMessageIds.add(idStr);
      if (isNew) {
        window.inboxState.cursors.globalLastMessageId = Math.max(window.inboxState.cursors.globalLastMessageId, +m.id || 0);
        window.inboxState.cursors.globalLastInboxId = Math.max(window.inboxState.cursors.globalLastInboxId, +m.id || 0);
      }
      console.debug('[MESSAGE] upsertMsg added', idStr, '- new:', isNew);
      return true;
    }

    const existing = domNode;

    // Patch status only
    const patchStatus = (selector, html) => {
      let el = existing.querySelector(selector);
      if (html && !el) {
        el = document.createElement('div');
        const timeEl = existing.querySelector('.msg-time');
        timeEl ? timeEl.before(el) : existing.querySelector('.msg-bubble')?.appendChild(el);
      }
      if (el) el.outerHTML = html || '';
    };

    const status = (m.delivery_status||'').toLowerCase();
    const statusIcon = m.direction === 'outbound'
      ? (status === 'read'
        ? '<i class="fas fa-check-double msg-status-icon read" title="Read"></i>'
        : ['delivered','sent','accepted'].includes(status)
          ? '<i class="fas fa-check-double msg-status-icon" title="Delivered"></i>'
          : '<i class="fas fa-check msg-status-icon" title="Sent"></i>')
      : '';
    patchStatus('.msg-status-icon', statusIcon);

    const errorChip = m.error_reason ? `<div class="msg-error-chip"><i class="fas fa-circle-exclamation"></i> ${esc(m.error_reason)}</div>` : '';
    patchStatus('.msg-error-chip', errorChip);

    return false;
  };

  /* ═══════════════════════════════════════════════════════════
     SIDEBAR RENDERING — INTERNAL HELPERS
     ═══════════════════════════════════════════════════════════ */

  const avatarClasses = ['', 'alt-1', 'alt-2'];

  /**
   * Build the HTML string for a brand-new contact sidebar item.
   * @param {Object} c      Contact object
   * @param {number} i      Index (for avatar class cycling)
   * @param {Object} ctx    Context with pre-computed values
   * @returns {string} HTML string
   */
  const buildContactNode = (c, i, ctx) => {
    const { mobile, isActive, unreadCount, effectiveUnread, name, cls, time, preview, isChatWindowActive, windowStatus, aiReplied, needsHuman, showAiUnread, hasNewMessage } = ctx;
    const inactiveClass = !isChatWindowActive ? 'inactive-24h' : '';
    const unreadHtml = effectiveUnread > 0 ? `<span class="conv-unread">${formatUnreadBadge(effectiveUnread)}</span>` : '';
    const aiHtml = showAiUnread ? `<span class="conv-ai-pill">AI replied</span>` : '';
    const needsHumanHtml = needsHuman ? `<span class="conv-needs-human">Needs human</span>` : '';

    return `
      <a class="conv-item ${isActive ? 'active' : ''} ${hasNewMessage ? 'has-new' : ''} ${inactiveClass}"
         href="${pageUrl}?mobile=${encodeURIComponent(mobile)}"
         data-mobile="${esc(mobile)}"
         data-unread-count="${effectiveUnread}"
         data-ai-replied="${aiReplied ? '1' : '0'}"
         data-name="${esc(name.toLowerCase())}"
         title="${isChatWindowActive ? 'Active chat window' : 'Chat window closed (>24h old)'}">
        <div class="conv-avatar ${cls}">${(name[0]||'?').toUpperCase()}</div>
        <div class="conv-meta">
          <div class="conv-row1">
            <div class="conv-name">${esc(name)}</div>
            <div class="conv-time">${esc(time)}</div>
          </div>
          <div class="conv-preview">${esc(preview)}</div>
          <div class="conv-flags">${aiHtml}${needsHumanHtml}</div>
        </div>
        ${unreadHtml}
      </a>`;
  };

  /**
   * Patch an existing contact DOM node with updated data.
   * Avoids full re-render — no flash.
   * @param {Element} existing  Existing DOM node
   * @param {Object}  c         Contact data
   * @param {Object}  ctx       Pre-computed context
   */
  const patchContactNode = (existing, c, ctx) => {
    const { mobile, isActive, unreadCount, effectiveUnread, name, time, preview, isChatWindowActive, aiReplied, needsHuman, showAiUnread, hasNewMessage } = ctx;

    existing.classList.toggle('active', isActive);
    existing.classList.toggle('has-new', hasNewMessage);
    existing.classList.toggle('inactive-24h', !isChatWindowActive);

    const nameEl = existing.querySelector('.conv-name');
    if (nameEl && nameEl.textContent !== name) nameEl.textContent = name;
    const previewEl = existing.querySelector('.conv-preview');
    if (previewEl && previewEl.textContent !== preview) previewEl.textContent = preview;
    const timeEl = existing.querySelector('.conv-time');
    if (timeEl && timeEl.textContent !== time) timeEl.textContent = time;

    existing.dataset.unreadCount = String(unreadCount);
    existing.dataset.aiReplied = aiReplied ? '1' : '0';
    existing.dataset.needsHuman = needsHuman ? '1' : '0';

    let unreadEl = existing.querySelector('.conv-unread');
    if (effectiveUnread > 0) {
      if (!unreadEl) {
        existing.insertAdjacentHTML('beforeend', `<span class="conv-unread">${formatUnreadBadge(effectiveUnread)}</span>`);
      } else if (unreadEl.textContent !== formatUnreadBadge(effectiveUnread)) {
        unreadEl.textContent = formatUnreadBadge(effectiveUnread);
      }
    } else if (unreadEl) {
      unreadEl.remove();
    }

    let flagWrap = existing.querySelector('.conv-flags');
    if (!flagWrap) {
      existing.querySelector('.conv-preview')?.insertAdjacentHTML('afterend', '<div class="conv-flags"></div>');
      flagWrap = existing.querySelector('.conv-flags');
    }
    let aiEl = flagWrap?.querySelector('.conv-ai-pill');
    let nhEl = flagWrap?.querySelector('.conv-needs-human');
    if (showAiUnread) {
      if (!aiEl) flagWrap?.insertAdjacentHTML('beforeend', '<span class="conv-ai-pill">AI replied</span>');
    } else if (aiEl) {
      aiEl.remove();
    }
    if (needsHuman) {
      if (!nhEl) flagWrap?.insertAdjacentHTML('beforeend', '<span class="conv-needs-human">Needs human</span>');
    } else if (nhEl) {
      nhEl.remove();
    }
  };

  /**
   * Sort contacts into section containers by window status priority.
   * Re-appends nodes to enforce correct DOM order.
   */
  const sortContactsByPriority = (orderedBySection) => {
    const el = dom.convList;
    if (!el) return;
    ['active', 'expiring', 'inactive'].forEach(sectionKey => {
      const section = el.querySelector(`[data-section="${sectionKey}"]`);
      if (!section) return;
      orderedBySection[sectionKey].forEach(node => section.appendChild(node));
    });
  };

  /* ═══════════════════════════════════════════════════════════
     SIDEBAR RENDERING — MAIN
     ═══════════════════════════════════════════════════════════ */

  const renderContacts = contacts => {
    const el = dom.convList;
    if (!el) return;
    const renderStart = performance.now();
    prunePerChatState(contacts);
    if (!contacts.length) {
      if (!el.querySelector('.conv-empty')) {
        el.innerHTML = `<div class="conv-empty"><i class="fab fa-whatsapp"></i><h4>No conversations yet</h4><p>New WhatsApp messages will appear here in real time</p></div>`;
      }
      return;
    }
    // Remove empty state if present
    const emptyEl = el.querySelector('.conv-empty');
    if (emptyEl) emptyEl.remove();
    if (!el.querySelector('[data-section="active"]')) {
      el.innerHTML = `
        <div class="conv-section" data-section="active"><div class="conv-section-title">Active</div></div>
        <div class="conv-section" data-section="expiring"><div class="conv-section-title">Expiring Soon</div></div>
        <div class="conv-section" data-section="inactive"><div class="conv-section-title">Inactive</div></div>
      `;
    }
    const orderedBySection = {
      active: [],
      expiring: [],
      inactive: [],
    };

    contacts.forEach((c, i) => {
      const mobile = c.mobile || '';
      if (!mobile) return;
      const isActive = mobile === window.inboxState.activeMobile;
      const name = c.name || mobile || 'Unknown';
      const cls = avatarClasses[i % 3];
      const unreadCount = !isActive ? (window.inboxState.unreadByMobile.get(mobile) || 0) : 0;
      const hasNewMessage = !isActive && unreadCount > 0;
      const time = fmtTime(c.created_at);
      const preview = c.text || c.preview || 'No messages yet';
      // Use last_customer_message_at for 24h window if available, fallback to created_at
      const windowTimestamp = c.last_customer_message_at || c.created_at;
      const isChatWindowActive = isActiveChatWindow(windowTimestamp);
      const windowStatus = c.chat_window_status || (isChatWindowActive ? 'active' : 'inactive');
      const aiReplied = Boolean(c.ai_replied);
      const needsHuman = Boolean(c.needs_human);
      const effectiveUnread = unreadCount;
      const showAiUnread = aiReplied && !isActive && effectiveUnread > 0;

      // Shared context for build/patch helpers
      const ctx = { mobile, isActive, unreadCount, effectiveUnread, name, cls, time, preview, isChatWindowActive: windowStatus !== 'inactive', windowStatus: windowStatus === 'inactive' ? 'inactive' : (windowStatus === 'expiring' ? 'expiring' : 'active'), aiReplied, needsHuman, showAiUnread, hasNewMessage };

      const sectionKey = windowStatus === 'inactive' ? 'inactive' : (windowStatus === 'expiring' ? 'expiring' : 'active');
      const sectionEl = el.querySelector(`[data-section="${sectionKey}"]`) || el;
      const existing = el.querySelector(`[data-mobile="${CSS.escape(mobile)}"]`);

      if (existing) {
        patchContactNode(existing, c, ctx);
        orderedBySection[windowStatus]?.push(existing);
      } else {
        const html = buildContactNode(c, i, ctx);
        sectionEl.insertAdjacentHTML('beforeend', html);
        const inserted = el.querySelector(`[data-mobile="${CSS.escape(mobile)}"]`);
        if (inserted) orderedBySection[windowStatus]?.push(inserted);
        console.debug('[SIDEBAR] Added new contact', name, '(' + mobile + ')' + (hasNewMessage ? ' - with green dot' : ''));
      }
    });

    sortContactsByPriority(orderedBySection);
    applyComposerWindowPolicy();
    const renderDuration = performance.now() - renderStart;
    window.debugMetrics.render.lastSidebarMs = Math.round(renderDuration);
    console.debug('[PERF] renderContacts:', Math.round(renderDuration), 'ms —', contacts.length, 'contacts');
    console.debug('[SIDEBAR] renderContacts complete —', contacts.length, 'contacts');
  };

  /* ── Debounced renderContacts (throttle rapid updates) ──── */
  const debouncedRenderContacts = (contacts) => {
    clearTimeout(window.inboxState.renderContactsTimeout);
    window.inboxState.pendingRenderContacts = contacts;
    window.inboxState.renderContactsTimeout = setTimeout(() => {
      if (window.inboxState.pendingRenderContacts) {
        renderContacts(window.inboxState.pendingRenderContacts);
        window.inboxState.pendingRenderContacts = null;
      }
    }, 300);
  };

  /* ═══════════════════════════════════════════════════════════
     EXPLICIT RENDER TRIGGERS
     ═══════════════════════════════════════════════════════════ */

  /**
   * Refresh the sidebar conversation list.
   * Always reads from window.inboxState.contacts as single source of truth.
   * Protected by render queue guard — if a render is already running,
   * queues one additional render.
   */
  const refreshSidebar = () => {
    if (isRenderInProgress) {
      // ── Queue ceiling: coalesce excessive queuing ─────
      queuedSidebarCount++;
      if (queuedSidebarCount >= RENDER_QUEUE_MAX) {
        console.debug('[RENDER_QUEUE] Sidebar refresh coalesced — dropped', queuedSidebarCount, 'queued calls');
        window.inboxState.stress.queueCoalesceCount++;
        queuedSidebarCount = 0; // reset after coalescing
      }
      pendingRenderFrame = true;
      console.debug('[RENDER_QUEUE] Sidebar refresh queued (count:', queuedSidebarCount, ')');
      return;
    }

    // Render burst throttle: skip if called within RENDER_THROTTLE_MS of last render
    const now = performance.now();
    if (now - lastSidebarRenderTime < RENDER_THROTTLE_MS) {
      console.debug('[PERF] refreshSidebar throttled — burst suppressed');
      window.inboxState.stress.staleRenderDrops++;
      return;
    }
    lastSidebarRenderTime = now;
    queuedSidebarCount = 0; // reset coalesce counter

    isRenderInProgress = true;
    _markRenderActive();
    console.debug('[SIDEBAR] refreshSidebar triggered');

    // ── Stuck render detection ─────────────────────────
    const stuckTimer = setTimeout(() => {
      console.warn('[DOM_WARN] refreshSidebar stuck for', STALE_RENDER_TIMEOUT_MS, 'ms — force-resetting render guard');
      isRenderInProgress = false;
      pendingRenderFrame = false;
    }, STALE_RENDER_TIMEOUT_MS);

    // ── Large sidebar warning ──────────────────────────
    const contacts = window.inboxState.contacts;
    if (contacts && Array.isArray(contacts) && contacts.length > 150) {
      const sidebarContacts = dom.convList?.querySelectorAll('.conv-item').length || 0;
      console.debug('[DOM_WARN] Large sidebar detected —', contacts.length, 'contacts in state,', sidebarContacts, 'in DOM');
      window.debugMetrics.memory.sidebarContactCount = contacts.length;
    }

    const sidebarRefreshStart = performance.now();
    const el = dom.convList;
    if (!el) {
      console.debug('[SIDEBAR] refreshSidebar skipped — no convList DOM ref');
      clearTimeout(stuckTimer);
      isRenderInProgress = false;
      return;
    }

    if (contacts && Array.isArray(contacts) && contacts.length > 0) {
      renderContacts(contacts);
      window.debugMetrics.render.lastSidebarRefreshMs = Math.round(performance.now() - sidebarRefreshStart);
    console.debug('[PERF] refreshSidebar:', Math.round(performance.now() - sidebarRefreshStart), 'ms');
    console.debug('[SYNC] refreshSidebar complete —', contacts.length, 'contacts');
    } else {
      // Green dot fallback for visible contacts when no cached data
      const items = el.querySelectorAll('.conv-item[data-mobile]');
      items.forEach(item => {
        const mobile = item.dataset.mobile;
        if (mobile) updateContactGreenDot(mobile);
      });
      console.debug('[SIDEBAR] refreshSidebar complete — green dots only, no contacts cache');
    }

    clearTimeout(stuckTimer);
    applyComposerWindowPolicy();

    // Track max sidebar render duration
    const sidebarDuration = performance.now() - sidebarRefreshStart;
    window.inboxState.stress.maxSidebarRenderMs = Math.max(
      window.inboxState.stress.maxSidebarRenderMs,
      Math.round(sidebarDuration)
    );

    isRenderInProgress = false;

    // If another refresh was queued during this render, run it once
    if (pendingRenderFrame) {
      pendingRenderFrame = false;
      console.debug('[RENDER_QUEUE] Running queued sidebar refresh');
      refreshSidebar();
    }
  };

  /**
   * Refresh the active conversation (chat body).
   * Re-applies composer window policy, re-enables send button.
   * Does NOT scroll — caller handles scroll when appropriate.
   * Protected by render queue guard.
   */
  const refreshActiveChat = () => {
    if (isRenderInProgress) {
      // ── Queue ceiling: coalesce excessive queuing ─────
      queuedActiveChatCount++;
      if (queuedActiveChatCount >= RENDER_QUEUE_MAX) {
        console.debug('[RENDER_QUEUE] Active chat refresh coalesced — dropped', queuedActiveChatCount, 'queued calls');
        window.inboxState.stress.queueCoalesceCount++;
        queuedActiveChatCount = 0;
      }
      pendingRenderFrame = true;
      console.debug('[RENDER_QUEUE] Active chat refresh queued (count:', queuedActiveChatCount, ')');
      return;
    }

    // Throttle: skip if called within RENDER_THROTTLE_MS of last active chat refresh
    const now = performance.now();
    if (now - lastActiveChatRenderTime < RENDER_THROTTLE_MS) {
      console.debug('[PERF] refreshActiveChat throttled — burst suppressed');
      window.inboxState.stress.staleRenderDrops++;
      return;
    }
    lastActiveChatRenderTime = now;
    queuedActiveChatCount = 0;

    isRenderInProgress = true;
    _markRenderActive();
    console.debug('[ACTIVE_CHAT] refreshActiveChat triggered');
    if (window.inboxState.debugActiveMobile) window.inboxState.debugActiveMobile('refreshActiveChat:start');
    if (!window.inboxState.activeMobile) {
      console.error('[STATE_FATAL] activeMobile missing in refreshActiveChat');
      console.warn('[ACTIVE_CHAT] refreshActiveChat skipped: empty activeMobile');
      isRenderInProgress = false;
      return;
    }

    // ── Stuck render detection ─────────────────────────
    const stuckTimer = setTimeout(() => {
      console.warn('[DOM_WARN] refreshActiveChat stuck for', STALE_RENDER_TIMEOUT_MS, 'ms — force-resetting render guard');
      isRenderInProgress = false;
      pendingRenderFrame = false;
    }, STALE_RENDER_TIMEOUT_MS);

    const activeChatRefreshStart = performance.now();

    applyComposerWindowPolicy();

    // Re-enable send button if activeMobile exists
    const sendBtn = document.getElementById('sendBtn');
    if (sendBtn && window.inboxState.activeMobile) {
      sendBtn.disabled = false;
    }

    window.debugMetrics.render.lastActiveChatRefreshMs = Math.round(performance.now() - activeChatRefreshStart);

    // Track max active chat render duration
    const activeChatDuration = performance.now() - activeChatRefreshStart;
    window.inboxState.stress.maxActiveChatRenderMs = Math.max(
      window.inboxState.stress.maxActiveChatRenderMs,
      Math.round(activeChatDuration)
    );

    clearTimeout(stuckTimer);
    console.debug('[PERF] refreshActiveChat:', Math.round(activeChatDuration), 'ms');
    console.debug('[ACTIVE_CHAT] refreshActiveChat complete');
    isRenderInProgress = false;

    if (pendingRenderFrame) {
      pendingRenderFrame = false;
      console.debug('[RENDER_QUEUE] Running queued active chat refresh');
      refreshActiveChat();
    }
  };

  /* ═══════════════════════════════════════════════════════════
     EXPORTS
     ═══════════════════════════════════════════════════════════ */

  window.renderEngine = {
    init,
    renderMsg,
    upsertMsg,
    renderContacts,
    debouncedRenderContacts,
    nearBottom,
    scrollBottom,
    updateContactGreenDot,
    prunePerChatState,
    applyComposerWindowPolicy,
    isActiveChatWindow,
    inferMediaKind,
    formatUnreadBadge,
    esc,
    fmtTime,
    refreshSidebar,
    refreshActiveChat,

    // Expose for debugging / inspection
    get messageNodeMap()           { return messageNodeMap; },
    get isRenderInProgress()        { return isRenderInProgress; },
    get pendingRenderFrame()        { return pendingRenderFrame; },
    get renderStaleMs()            { return _getRenderStaleMs(); },

    /** Force-reset render guards (for self-healing) */
    forceResetRenderGuard() {
      isRenderInProgress = false;
      pendingRenderFrame = false;
      _lastRenderActivityAt = Date.now();
      console.warn('[RENDER] Render guard force-reset by self-healing');
    },

    /** Clear the message node registry (e.g., on chat switch) */
    clearMessageNodeMap() {
      messageNodeMap.clear();
      console.debug('[MESSAGE_NODE] Registry cleared');
    },
  };

  window.__renderEngineInitialized = true;

})();
