/**
 * whatsapp/events.js — Event & Interaction Layer for WhatsApp inbox.
 *
 * PURPOSE:
 *   Centralize ALL UI event listeners, interaction handlers, and UI lifecycle
 *   (modals, composer, templates, command palette, etc.) into ONE module.
 *
 *   - render.js updates DOM
 *   - polling.js reacts to network updates
 *   - state.js stores state
 *   - events.js reacts to user interaction
 *
 * INITIALIZATION:
 *   Must call window.eventsEngine.init(config) before use, then bindAll():
 *     window.eventsEngine.init({ ... });
 *     window.eventsEngine.bindAll();
 *
 * @see Phase 5 of WhatsApp inbox modularization.
 */

(function () {
  'use strict';

  /* ── Guard: prevent double-initialization ──────────────── */
  if (window.__eventsEngineInitialized) {
    console.debug('[EVENT] Already initialized — skipping.');
    return;
  }

  /* ═══════════════════════════════════════════════════════════
     INJECTED DEPENDENCIES (set by init())
     ═══════════════════════════════════════════════════════════ */

  // API URLs
  let SEND_URL        = '';
  let TEMPLATE_URL    = '';
  let SEND_TPL_URL    = '';
  let SEND_INT_URL    = '';
  let FLOWS_URL       = '';
  let SYNC_FLOWS_URL  = '';

  let PAGE_URL        = '';
  let STATIC_BASE     = '';
  let NOTIFICATION_SOUND_URL = '';
  let SOFT_SOUND_URL  = '';

  // DOM refs
  const dom = {
    chatBody:       null,
    convList:       null,
    chatForm:       null,
    msgInput:       null,
    searchInput:    null,
    scrollBtn:      null,
    connectionDot:  null,
    attachFile:     null,
    attachPreview:  null,
    attachName:     null,
    attachRemove:   null,
    cmdOverlay:     null,
    cmdInput:       null,
    emojiBtn:       null,
    notifBtn:       null,
    notifIcon:      null,
    themeToggle:    null,
    themeIcon:      null,
    newChatBtn:     null,
    templateBtn:    null,
    interactiveBtn: null,
    cmdChip:        null,
    pageLoader:     null,
  };

  // Modal instances
  let templateModal         = null;
  let newConvModal          = null;
  let interactiveModal      = null;
  let templateQuickSendModal = null;
  let mediaModal            = null;

  // Helpers
  let $    = (id) => document.getElementById(id);
  let $q   = (sel, ctx) => (ctx || document).querySelector(sel);
  let $all = (sel, ctx) => [...(ctx || document).querySelectorAll(sel)];
  let esc  = v => String(v||'').replace(/[&<>'\"]/g,
    m => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[m]));

  // Render function deps (from render.js)
  let updateContactGreenDot  = null;
  let scrollBottom           = null;
  let nearBottom             = null;
  let applyComposerWindowPolicy = null;

  /* ── Listener count tracker ────────────────────────────── */
  let _listenerCount = 0;

  /* ── DOM/Memory warning thresholds ─────────────────────── */
  const DOM_WARN_THRESHOLDS = {
    totalNodes: 10000,       // warn if document.querySelectorAll('*').length > 10k
    messageNodes: 2000,      // warn if messageNodeMap.size > 2k
    listeners: 100,          // warn if listener count > 100
    sidebarContacts: 150,    // warn if sidebar contact count > 150
  };

  /* ── Mobile timer drift detection ─────────────────────── */
  const _isMobileAgent = /Mobi|Android|iPhone/i.test(navigator.userAgent);
  let _lastInteractionTs = Date.now();
  const MOBILE_DRIFT_THRESHOLD_MS = 5000; // warn if >5s gap between interactions

  /**
   * Track addEventListener calls for memory leak monitoring.
   * Increments the listener count and returns the standard removeEventListener.
   */
  const trackAddListener = (target, event, handler, options) => {
    if (target && target.addEventListener) {
      target.addEventListener(event, handler, options);
      _listenerCount++;
      if (window.inboxState) window.inboxState.listenerCount = _listenerCount;
      // Log every 10 listeners for healthy awareness
      if (_listenerCount % 10 === 0) {
        console.debug('[MEMORY] Active listeners count:', _listenerCount);
      }
    }
  };

  /**
   * Track removeEventListener calls.
   */
  const trackRemoveListener = (target, event, handler, options) => {
    if (target && target.removeEventListener) {
      target.removeEventListener(event, handler, options);
      _listenerCount = Math.max(0, _listenerCount - 1);
      if (window.inboxState) window.inboxState.listenerCount = _listenerCount;
    }
  };

  /* ═══════════════════════════════════════════════════════════
     INITIALIZATION
     ═══════════════════════════════════════════════════════════ */

  /**
   * One-time initialization with dependency injection.
   */
  const init = (config) => {
    if (window.__eventsEngineInitialized) return;
    if (!config) { console.warn('[EVENT] init() requires config — skipping'); return; }

    // API URLs
    const api = config.apiUrls || {};
    SEND_URL       = api.send        || '';
    TEMPLATE_URL   = api.template    || '';
    SEND_TPL_URL   = api.sendTemplate || '';
    SEND_INT_URL   = api.sendInteractive || '';
    FLOWS_URL      = api.flows       || '';
    SYNC_FLOWS_URL = api.syncFlows   || '';

    PAGE_URL    = config.pageUrl    || '';
    STATIC_BASE = config.staticBase || '';
    NOTIFICATION_SOUND_URL = config.notificationSoundUrl || STATIC_BASE + 'audio/whatsapp_notification.wav';
    SOFT_SOUND_URL  = config.softSoundUrl  || STATIC_BASE + 'audio/soft_sound.mp3';

    // DOM refs
    if (config.dom) {
      const d = config.dom;
      if (d.chatBody)       dom.chatBody       = d.chatBody;
      if (d.convList)       dom.convList       = d.convList;
      if (d.chatForm)       dom.chatForm       = d.chatForm;
      if (d.msgInput)       dom.msgInput       = d.msgInput;
      if (d.searchInput)    dom.searchInput    = d.searchInput;
      if (d.scrollBtn)      dom.scrollBtn      = d.scrollBtn;
      if (d.connectionDot)  dom.connectionDot  = d.connectionDot;
      if (d.attachFile)     dom.attachFile     = d.attachFile;
      if (d.attachPreview)  dom.attachPreview  = d.attachPreview;
      if (d.attachName)     dom.attachName     = d.attachName;
      if (d.attachRemove)   dom.attachRemove   = d.attachRemove;
      if (d.cmdOverlay)     dom.cmdOverlay     = d.cmdOverlay;
      if (d.cmdInput)       dom.cmdInput       = d.cmdInput;
      if (d.emojiBtn)       dom.emojiBtn       = d.emojiBtn;
      if (d.notifBtn)       dom.notifBtn       = d.notifBtn;
      if (d.notifIcon)      dom.notifIcon      = d.notifIcon;
      if (d.themeToggle)    dom.themeToggle    = d.themeToggle;
      if (d.themeIcon)      dom.themeIcon      = d.themeIcon;
      if (d.newChatBtn)     dom.newChatBtn     = d.newChatBtn;
      if (d.templateBtn)    dom.templateBtn    = d.templateBtn;
      if (d.interactiveBtn) dom.interactiveBtn = d.interactiveBtn;
      if (d.cmdChip)        dom.cmdChip        = d.cmdChip;
      if (d.pageLoader)     dom.pageLoader     = d.pageLoader;
    }

    // Modal instances
    if (config.modals) {
      const m = config.modals;
      if (m.templateModal)         templateModal         = m.templateModal;
      if (m.newConvModal)          newConvModal          = m.newConvModal;
      if (m.interactiveModal)      interactiveModal      = m.interactiveModal;
      if (m.templateQuickSendModal) templateQuickSendModal = m.templateQuickSendModal;
      if (m.mediaModal)            mediaModal            = m.mediaModal;
    }

    // Helpers
    if (config.helpers) {
      if (config.helpers.$)    $    = config.helpers.$;
      if (config.helpers.$q)   $q   = config.helpers.$q;
      if (config.helpers.$all) $all = config.helpers.$all;
      if (config.helpers.esc)  esc  = config.helpers.esc;
    }

    // Render function deps
    if (config.renderFns) {
      if (config.renderFns.updateContactGreenDot)  updateContactGreenDot  = config.renderFns.updateContactGreenDot;
      if (config.renderFns.scrollBottom)           scrollBottom           = config.renderFns.scrollBottom;
      if (config.renderFns.nearBottom)             nearBottom             = config.renderFns.nearBottom;
      if (config.renderFns.applyComposerWindowPolicy) applyComposerWindowPolicy = config.renderFns.applyComposerWindowPolicy;
    }

    console.debug('[EVENT] Engine initialized with', Object.keys(config).length, 'config groups');
    console.debug('[EVENT_BIND] Config loaded — API URLs:', Object.keys(api).length, 'DOM refs:', Object.keys(config.dom||{}).length, 'Modals:', Object.keys(config.modals||{}).length);
    console.debug('[MEMORY] Initial listener count:', _listenerCount);

    /* ── TODO(websocket): When migrating to WebSocket transport,
       the events engine will need a reconnect lifecycle hook:
       - eventsEngine.onReconnect() to re-bind modal listeners
         that may have been attached to stale DOM.
       - eventsEngine.onDisconnect() to show connection UI.
       The current polling-based architecture reuses listeners
       across chat switches because DOM nodes are stable. */
  };

  /* ═══════════════════════════════════════════════════════════
     1. THEME MANAGEMENT
     ═══════════════════════════════════════════════════════════ */

  const initTheme = () => {
    const currentTheme = localStorage.getItem('wa_theme') || 'light';
    document.documentElement.setAttribute('data-theme', currentTheme);
    if (dom.themeIcon) {
      dom.themeIcon.className = currentTheme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
    }

    trackAddListener(dom.themeToggle, 'click', () => {
      const current = document.documentElement.getAttribute('data-theme') || 'light';
      const newTheme = current === 'dark' ? 'light' : 'dark';

      document.documentElement.setAttribute('data-theme', newTheme);
      localStorage.setItem('wa_theme', newTheme);

      if (dom.themeIcon) {
        dom.themeIcon.className = newTheme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
      }
    });

    console.debug('[EVENT_BIND] Theme listener registered');
    console.debug('[LIFECYCLE] Theme initialized:', currentTheme);
    console.debug('[EVENT] Theme initialized:', currentTheme);
  };

  /* ── TODO(websocket): Replace initInteractionUnlock with
     a single 'user-activity' custom event that the WS reconnect
     handler can also dispatch. Currently uses 5 separate DOM
     events (pointerdown, keydown, touchstart, click, mousemove). */

  /* ═══════════════════════════════════════════════════════════
     2. AUDIO INITIALIZATION
     ═══════════════════════════════════════════════════════════ */

  const initAudio = () => {
    window.inboxState.notificationAudio = new Audio(NOTIFICATION_SOUND_URL);
    window.inboxState.notificationAudio.preload = 'auto';
    window.inboxState.notificationAudio.volume = 1.0;

    window.inboxState.notificationAudio.addEventListener('error', () => {
      console.warn('⚠️ Notification sound failed to load from:', NOTIFICATION_SOUND_URL);
    });

    window.inboxState.notificationAudio.addEventListener('canplaythrough', () => {
      console.debug('[AUDIO] Notification sound loaded:', NOTIFICATION_SOUND_URL);
    });

    console.debug('[AUDIO] Notification audio created:', NOTIFICATION_SOUND_URL);
    console.debug('[LIFECYCLE] Audio initialized');
    console.debug('[EVENT] Audio initialized');
  };

  /* ═══════════════════════════════════════════════════════════
     3. AUDIO UNLOCK + ACTIVITY TRACKING
     ═══════════════════════════════════════════════════════════ */

  const initInteractionUnlock = () => {
    // Unlock audio after first user interaction
    ['pointerdown', 'keydown', 'touchstart', 'click'].forEach(ev =>
      window.addEventListener(ev, () => {
        window.inboxState.ui.hasInteracted = true;
        console.debug('[EVENT] Audio unlocked on user interaction');
      }, { once: true })
    );

    // Track operator activity for heartbeat adaptation
    ['pointerdown', 'keydown', 'touchstart', 'click', 'mousemove'].forEach(ev => {
      window.addEventListener(ev, () => {
        window.pollingEngine.updateActivity();
      }, { passive: true });
    });

    console.debug('[EVENT_BIND] Interaction unlock + activity tracking listeners registered');
    console.debug('[LIFECYCLE] Interaction unlock + activity tracking initialized');
    console.debug('[EVENT] Interaction unlock + activity tracking initialized');
  };

  /* ═══════════════════════════════════════════════════════════
     4. BEEP (Notification Sound)
     ═══════════════════════════════════════════════════════════ */

  const beep = (inboundMobile) => {
    if (!window.inboxState.ui.soundEnabled || !window.inboxState.ui.hasInteracted) return;

    try {
      // Don't play any sound for messages in the currently active chat
      if (inboundMobile && inboundMobile === window.inboxState.activeMobile) {
        return;
      }

      const audioToPlay = window.inboxState.notificationAudio;
      audioToPlay.currentTime = 0;

      const playPromise = audioToPlay.play();
      if (playPromise !== undefined) {
        playPromise
          .then(() => console.debug('[AUDIO] Sound played successfully'))
          .catch(err => {
            if (err.name === 'NotAllowedError') {
              console.debug('[AUDIO] Browser prevented autoplay - normal');
            } else {
              console.warn('[AUDIO] Sound play rejected:', err.name, err.message);
            }
          });
      }
    } catch (e) {
      console.warn('[AUDIO] Beep error:', e.message);
    }
  };

  /* ═══════════════════════════════════════════════════════════
     5. SOUND TOGGLE
     ═══════════════════════════════════════════════════════════ */

  const updateNotifIcon = () => {
    if (!dom.notifIcon) return;
    if (window.inboxState.ui.soundEnabled) {
      dom.notifIcon.className = 'fas fa-bell';
      dom.notifIcon.style.color = 'var(--text-2)';
      dom.notifIcon.title = '🔔 Sounds ON - Click to test/disable';
    } else {
      dom.notifIcon.className = 'fas fa-bell-slash';
      dom.notifIcon.style.color = 'var(--text-3)';
      dom.notifIcon.title = '🔇 Sounds OFF - Click to enable';
    }
  };

  const initSoundToggle = () => {
    dom.notifBtn?.addEventListener('click', async () => {
      window.inboxState.ui.soundEnabled = !window.inboxState.ui.soundEnabled;
      localStorage.setItem('wa_sound', window.inboxState.ui.soundEnabled ? '1' : '0');
      updateNotifIcon();
      console.debug('[AUDIO] Notifications:', window.inboxState.ui.soundEnabled ? 'ENABLED' : 'DISABLED');

      // If enabling, test the sound
      if (window.inboxState.ui.soundEnabled) {
        try {
          console.debug('[AUDIO] Playing test notification sound...');
          const testAudio = new Audio(NOTIFICATION_SOUND_URL);
          testAudio.volume = 0.8;
          testAudio.preload = 'auto';

          const playPromise = testAudio.play();
          if (playPromise !== undefined) {
            await playPromise;
            console.debug('[AUDIO] Test sound played successfully');
          }
        } catch (err) {
          console.warn('[AUDIO] Test failed:', err.name);
        }
      }
    });

    // Initialize icon on load
    updateNotifIcon();

    // Monitor audio state for debugging
    setInterval(() => {
      if (window.inboxState.ui.soundEnabled && window.inboxState.notificationAudio) {
        if (window.inboxState.notificationAudio.readyState === 0) {
          console.warn('[AUDIO] Notification audio not ready - may have loading issues');
        }
      }
    }, 30000);

    console.debug('[AUDIO] Sound toggle listener registered, monitor interval set');
    console.debug('[LIFECYCLE] Sound toggle initialized');
    console.debug('[EVENT] Sound toggle initialized');
  };

  /* ═══════════════════════════════════════════════════════════
     6. TEXTAREA / COMPOSER
     ═══════════════════════════════════════════════════════════ */

  const resizeInput = () => {
    if (!dom.msgInput) return;
    dom.msgInput.style.height = '22px';
    dom.msgInput.style.height = Math.min(dom.msgInput.scrollHeight, 100) + 'px';
  };

  const initTextarea = () => {
    dom.msgInput?.addEventListener('input', resizeInput);
    dom.msgInput?.addEventListener('keydown', e => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        dom.chatForm?.requestSubmit();
      }
    });
    console.debug('[COMPOSER] Textarea input + enter-to-send listeners registered');
    console.debug('[EVENT] Textarea initialized');
  };

  /* ═══════════════════════════════════════════════════════════
     7. ATTACHMENT HANDLING
     ═══════════════════════════════════════════════════════════ */

  const initAttachment = () => {
    dom.attachFile?.addEventListener('change', () => {
      const file = dom.attachFile.files[0];
      if (file) {
        dom.attachName.textContent = file.name;
        dom.attachPreview.classList.add('show');
      }
    });

    dom.attachRemove?.addEventListener('click', () => {
      dom.attachFile.value = '';
      dom.attachPreview.classList.remove('show');
    });

    console.debug('[COMPOSER] Attachment change + remove listeners registered');
    console.debug('[EVENT] Attachment handling initialized');
  };

  /* ═══════════════════════════════════════════════════════════
     8. QUICK CHIPS + EMOJI
     ═══════════════════════════════════════════════════════════ */

  const initQuickChips = () => {
    $all('.compose-chip[data-insert]').forEach(chip => {
      chip.addEventListener('click', () => {
        if (!dom.msgInput || dom.msgInput.disabled) return;
        const pos = dom.msgInput.selectionStart ?? dom.msgInput.value.length;
        const end = dom.msgInput.selectionEnd ?? dom.msgInput.value.length;
        dom.msgInput.value = dom.msgInput.value.slice(0, pos) + chip.dataset.insert + dom.msgInput.value.slice(end);
        dom.msgInput.focus();
        resizeInput();
      });
    });

    dom.cmdChip?.addEventListener('click', () => openCommandPalette());

    dom.emojiBtn?.addEventListener('click', () => {
      if (!dom.msgInput || dom.msgInput.disabled) return;
      dom.msgInput.value += '😊';
      dom.msgInput.focus();
      resizeInput();
    });

    console.debug('[COMPOSER] Quick chip listeners registered:', $all('.compose-chip[data-insert]').length, 'chips');
    console.debug('[EVENT] Quick chips initialized');
  };

  /* ═══════════════════════════════════════════════════════════
     9. SEARCH (Sidebar)
     ═══════════════════════════════════════════════════════════ */

  const initSearch = () => {
    dom.searchInput?.addEventListener('input', () => {
      const q = dom.searchInput.value.toLowerCase();
      $all('.conv-item', dom.convList).forEach(item => {
        const hay = (item.dataset.name || '') + ' ' + (item.dataset.mobile || '');
        item.style.display = hay.includes(q) ? '' : 'none';
      });
    });

    console.debug('[EVENT_BIND] Search input listener registered');
    console.debug('[LIFECYCLE] Search initialized');
    console.debug('[EVENT] Search initialized');
  };

  /* ═══════════════════════════════════════════════════════════
     10. CONVERSATION CLICK (Chat switching)
     ═══════════════════════════════════════════════════════════ */

  const initConvClick = () => {
    dom.convList?.addEventListener('click', async e => {
      const link = e.target.closest('.conv-item[data-mobile]');
      if (!link) return;
      e.preventDefault();

      const mobile = link.dataset.mobile;
      if (mobile === window.inboxState.activeMobile) return;

      // Remember previous active mobile for abort logic
      const oldMobile = window.inboxState.activeMobile;

      // Swap active highlight immediately — no flicker
      $all('.conv-item').forEach(el => el.classList.toggle('active', el.dataset.mobile === mobile));

      // Update URL without page reload
      history.pushState({}, '', `${PAGE_URL}?mobile=${encodeURIComponent(mobile)}`);

      // Reset chat for new contact
      window.inboxState.setActiveMobile(mobile);
      window.inboxState.cursors.globalLastMessageId = 0;
      window.inboxState.globalKnownMessageIds.clear();
      if (window.renderEngine.clearMessageNodeMap) window.renderEngine.clearMessageNodeMap();
      window.inboxState.ui.conversationOpenedAt = Date.now();

      // Only abort if switching to a different mobile (prevents unnecessary cancellation during heartbeat)
      if (oldMobile && oldMobile !== mobile) {
        window.pollingEngine.abortMessagesPoll();
      }

      // Show lightweight loading overlay — preserve existing messages
      const loadingOverlay = dom.chatBody?.querySelector('.chat-loading-overlay');
      if (loadingOverlay) {
        loadingOverlay.classList.remove('hide');
        loadingOverlay.classList.add('show');
      }

      // Update header name and mobile
      const contactName = link.querySelector('.conv-name')?.textContent || mobile;
      const headerName = document.querySelector('.chat-header-info h3');
      const headerSub  = document.querySelector('.chat-header-info p span[style*="monospace"]');

      if (headerName) headerName.textContent = contactName;
      if (headerSub)  headerSub.textContent  = mobile;

      // Update compose form mobile field
      const mobileInput = dom.chatForm?.querySelector('input[name="mobile"]');
      if (mobileInput) mobileInput.value = mobile;

      // Enable composer
      if (applyComposerWindowPolicy) applyComposerWindowPolicy();

      const sendBtn = document.getElementById('sendBtn');
      if (sendBtn) sendBtn.disabled = false;

      // Restart poll for new contact
      clearTimeout(window.inboxState.pollTimer);
      try {
        await window.pollingEngine.poll();
        // Now that human has opened the chat and poll loaded messages, clear unread badge
        window.inboxState.unreadByMobile.delete(mobile);
        // Refresh the conversation list to update unread badge visually
        const activeLink = dom.convList?.querySelector(`[data-mobile="${CSS.escape(mobile)}"]`);
        if (activeLink) {
          activeLink.dataset.unreadCount = '0';
          activeLink.dataset.aiReplied = '0';
          if (updateContactGreenDot) updateContactGreenDot(mobile);
        }
        await window.pollingEngine.pollSidebar();
      } catch (err) {
        console.error('[EVENT] Error loading chat:', err);
      }

      if (scrollBottom) scrollBottom();
      dom.msgInput?.focus();
    });

    console.debug('[EVENT_BIND] Conversation click delegate listener registered on convList');
    console.debug('[LIFECYCLE] Conversation click initialized');
    console.debug('[EVENT] Conversation click initialized');
  };

  /* ═══════════════════════════════════════════════════════════
     11. FILTER DROPDOWN
     ═══════════════════════════════════════════════════════════ */

  const initFilter = () => {
    $all('.dropdown-item[data-filter]').forEach(item => {
      item.addEventListener('click', () => {
        const textEl = document.getElementById('activeFilterText');
        if (textEl) textEl.textContent = item.textContent;
      });
    });

    console.debug('[EVENT_BIND] Filter dropdown listeners registered:', $all('.dropdown-item[data-filter]').length);
    console.debug('[LIFECYCLE] Filter dropdown initialized');
    console.debug('[EVENT] Filter dropdown initialized');
  };

  /* ═══════════════════════════════════════════════════════════
     12. SCROLL BUTTON
     ═══════════════════════════════════════════════════════════ */

  const initScrollButton = () => {
    dom.chatBody?.addEventListener('scroll', () => {
      if (!dom.scrollBtn) return;
      dom.scrollBtn.classList.toggle('show', !(nearBottom ? nearBottom() : true));
    }, { passive: true });

    dom.scrollBtn?.addEventListener('click', () => {
      if (scrollBottom) scrollBottom();
    });

    console.debug('[EVENT_BIND] Scroll button + chat scroll listeners registered');
    console.debug('[EVENT] Scroll button initialized');
  };

  /* ═══════════════════════════════════════════════════════════
     13. MEDIA VIEWER
     ═══════════════════════════════════════════════════════════ */

  const initMediaViewer = () => {
    dom.chatBody?.addEventListener('click', e => {
      const media = e.target.closest('.msg-media[data-type]');
      if (!media) return;
      const type = media.dataset.type;
      const src  = media.getAttribute('src');
      const content = document.getElementById('mediaViewerContent');
      if (!content || !src) return;
      content.innerHTML = type === 'image'
        ? `<img class="media-viewer-img" src="${esc(src)}" alt="Image">`
        : `<video class="media-viewer-video" controls autoplay src="${esc(src)}"></video>`;
      if (mediaModal) mediaModal.show();
    });

    console.debug('[MODAL] Media viewer listener registered on chatBody delegation');
    console.debug('[LIFECYCLE] Media viewer initialized');
    console.debug('[EVENT] Media viewer initialized');
  };

  /* ═══════════════════════════════════════════════════════════
     14. SEND MESSAGE
     ═══════════════════════════════════════════════════════════ */

  const initSendMessage = () => {
    dom.chatForm?.addEventListener('submit', async e => {
      e.preventDefault();
      const text = dom.msgInput?.value.trim();
      const file = dom.attachFile?.files[0];
      if (!text && !file) return;

      const fd = new FormData(dom.chatForm);
      fd.set('message', text || '');

      // Capture optimistic message data before send
      const optimisticId = `optimistic_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
      const optimisticMsg = {
        id: optimisticId,
        message_id: null,
        name: window.inboxState.activeMobile || 'Me',
        mobile: window.inboxState.activeMobile || '',
        direction: 'outbound',
        from_me: true,
        message_type: 'text',
        text: text || '[media]',
        media_url: null,
        media_public_url: null,
        file_name: null,
        media_mime_type: null,
        latitude: null,
        longitude: null,
        delivery_status: 'sent',
        error_reason: null,
        created_at: new Date().toISOString(),
      };

      // Store optimistic message ID for later cleanup if polling fails
      if (!window.inboxState.optimisticMessageIds) {
        window.inboxState.optimisticMessageIds = new Set();
      }
      window.inboxState.optimisticMessageIds.add(optimisticId);

      // Render optimistic message immediately
      if (upsertMsg) {
        upsertMsg(optimisticMsg);
        console.debug('[SEND] Optimistic message rendered:', optimisticId);
      }

      try {
        const r = await fetch(SEND_URL, { method: 'POST', body: fd });
        if (!r.ok) { 
          const p = await r.json(); 
          alert(p.error || 'Failed to send');
          // Remove optimistic message on send failure
          const optNode = dom.chatBody?.querySelector(`[data-message-id="${CSS.escape(optimisticId)}"]`);
          if (optNode) {
            optNode.remove();
            console.debug('[SEND] Optimistic message removed after send failure:', optimisticId);
          }
          window.inboxState.optimisticMessageIds?.delete(optimisticId);
          return;
        }

        dom.chatForm.reset();
        dom.attachPreview?.classList.remove('show');
        resizeInput();
        dom.msgInput?.focus();

        console.debug('[SEND] Message sent successfully, polling for confirmation...');

        // Poll to reconcile with real message from DB
        await window.pollingEngine.pollMessages();
        await window.pollingEngine.pollSidebar();

        // Clean up optimistic message ID from tracking
        window.inboxState.optimisticMessageIds?.delete(optimisticId);
        console.debug('[SEND] Optimistic message reconciled or replaced by polling:', optimisticId);

        if (scrollBottom) scrollBottom();
      } catch (e) { 
        alert('Network error: ' + e.message);
        // Remove optimistic message on network error
        const optNode = dom.chatBody?.querySelector(`[data-message-id="${CSS.escape(optimisticId)}"]`);
        if (optNode) {
          optNode.remove();
          console.debug('[SEND] Optimistic message removed after network error:', optimisticId);
        }
        window.inboxState.optimisticMessageIds?.delete(optimisticId);
      }
    });

    console.debug('[COMPOSER] Send message submit listener registered');
    console.debug('[LIFECYCLE] Send message initialized');
    console.debug('[EVENT] Send message initialized');
  };

  /* ═══════════════════════════════════════════════════════════
     15. TEMPLATE HELPERS
     ═══════════════════════════════════════════════════════════ */

  const extractVars = text => {
    if (!text) return [];
    const seen = new Set();
    return [...String(text).matchAll(/\{\{\s*([^}]+?)\s*\}\}/g)]
      .map(m => m[1].trim()).filter(v => !seen.has(v) && seen.add(v));
  };

  const getBodyText = t => {
    const body = (t?.components||[]).find(c => String(c?.type||'').toUpperCase() === 'BODY');
    return body?.text || body?.body_text || '';
  };

  const buildParamFields = (vars, wrapId) => {
    const wrap = document.getElementById(wrapId);
    if (!wrap) return;
    wrap.innerHTML = vars.length
      ? vars.map((v, i) => `<div class="col-md-6"><label class="form-label">Param ${i+1} (${esc(v)})</label><input class="form-control tpl-param" data-var="${esc(v)}"></div>`).join('')
      : '<div class="text-muted small">No body parameters required.</div>';
  };

  const onTemplateChange = (selectId, langId, wrapId) => {
    const sel = document.getElementById(selectId);
    const t = window.inboxState.templates.find(x => x.name === sel?.value);
    const vars = extractVars(getBodyText(t));
    buildParamFields(vars, wrapId);
    const langEl = document.getElementById(langId);
    if (langEl) langEl.value = t?.language || 'en_US';
  };

  const loadTemplates = async (force = false) => {
    if (window.inboxState.templates.length && !force) return;
    const r = await fetch(TEMPLATE_URL);
    if (!r.ok) return;
    const data = await r.json();
    window.inboxState.templates = data.data || [];
    const opts = window.inboxState.templates.map(t => `<option value="${esc(t.name)}">${esc(t.name)} (${esc(t.language||'en')})</option>`).join('');

    const templateSelectEl = document.getElementById('templateSelect');
    if (templateSelectEl) templateSelectEl.innerHTML = opts || '<option value="">No templates</option>';

    const newConvTemplateEl = document.getElementById('newConvTemplate');
    if (newConvTemplateEl) newConvTemplateEl.innerHTML = opts || '<option value="">No templates</option>';

    onTemplateChange('templateSelect', 'templateLang', 'templateParamsWrap');
    onTemplateChange('newConvTemplate', 'newConvLang', 'newConvParamsWrap');
    renderApprovedTemplateRows();
  };

  const renderApprovedTemplateRows = () => {
    const wrap = document.getElementById('approvedTemplateRows');
    if (!wrap) return;
    const search = (document.getElementById('templateSearchInput')?.value || '').trim().toLowerCase();
    const statusFilter = document.getElementById('templateStatusFilter')?.value || 'all';
    const categoryFilter = document.getElementById('templateCategoryFilter')?.value || 'all';

    const filtered = window.inboxState.templates.filter(t => {
      const status = String(t.status || '').toUpperCase();
      const category = String(t.category || 'UNCATEGORIZED').toUpperCase();
      const text = `${t.name || ''} ${getBodyText(t) || ''} ${category}`.toLowerCase();
      if (statusFilter !== 'all' && status !== statusFilter) return false;
      if (categoryFilter !== 'all' && category !== categoryFilter) return false;
      if (search && !text.includes(search)) return false;
      return true;
    });

    if (!filtered.length) {
      wrap.innerHTML = `<tr><td colspan="6" style="text-align:center;padding:40px 16px;color:var(--text-3);font-size:13px;">No templates match current filters.</td></tr>`;
      return;
    }

    wrap.innerHTML = filtered.map(t => {
      const status = String(t.status || '').toUpperCase();
      const bodyText = getBodyText(t) || '';
      const vars = extractVars(bodyText);
      const paramCount = vars.length;
      const category = esc(String(t.category || 'UNCATEGORIZED').toUpperCase());
      const preview = esc(bodyText.slice(0, 80)) + (bodyText.length > 80 ? '…' : '');
      const templateId = esc(t.id || t.name || '');
      const shortId = templateId.length > 14 ? templateId.slice(0, 14) + '…' : templateId;

      const badgeCls = status === 'APPROVED'
        ? 'tpl-badge-approved'
        : status === 'REMOVED'
          ? 'tpl-badge-removed'
          : 'tpl-badge-pending';

      const badgeIcon = status === 'APPROVED'
        ? '<i class="fas fa-circle-check" style="font-size:10px;"></i>'
        : status === 'REMOVED'
          ? '<i class="fas fa-circle-xmark" style="font-size:10px;"></i>'
          : '<i class="fas fa-clock" style="font-size:10px;"></i>';

      const statusLabel = status.charAt(0) + status.slice(1).toLowerCase();

      return `<tr style="border-bottom:1px solid var(--border);transition:background .12s;"
               onmouseover="this.style.background='var(--surface-2)'"
               onmouseout="this.style.background=''">
      <td style="padding:10px 16px;">
        <div style="font-weight:600;color:var(--text-1);font-size:13px;">${esc(t.name || '-')}</div>
        <div style="font-size:11px;color:var(--text-3);margin-top:2px;font-family:'DM Mono',monospace;">
          ${paramCount > 0 ? `<span style="margin-right:8px;">${paramCount} param${paramCount !== 1 ? 's' : ''}</span>` : ''}
          <span title="${templateId}">${shortId}</span>
        </div>
      </td>
      <td style="padding:10px 12px;">
        <span class="${badgeCls}" style="display:inline-flex;align-items:center;gap:5px;padding:3px 10px;border-radius:999px;font-size:12px;font-weight:500;">
          ${badgeIcon} ${statusLabel}
        </span>
      </td>
      <td style="padding:10px 12px;color:var(--text-2);font-size:12.5px;">${category}</td>
      <td style="padding:10px 12px;color:var(--text-2);font-size:12.5px;font-family:'DM Mono',monospace;">${esc(t.language || 'en')}</td>
      <td style="padding:10px 12px;color:var(--text-3);font-size:12.5px;max-width:260px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${preview || '<em>No preview</em>'}</td>
      <td style="padding:10px 16px;text-align:right;white-space:nowrap;">
        <button class="quick-send-template-btn" data-template="${esc(t.name)}" title="Send template"
          style="width:32px;height:32px;border-radius:8px;border:1px solid var(--border);background:var(--surface);color:var(--text-2);cursor:pointer;font-size:14px;display:inline-flex;align-items:center;justify-content:center;transition:all .15s;margin-right:4px;"
          onmouseover="this.style.background='var(--green)';this.style.color='#fff';this.style.borderColor='var(--green)'"
          onmouseout="this.style.background='var(--surface)';this.style.color='var(--text-2)';this.style.borderColor='var(--border)'">
          <i class="fas fa-paper-plane" style="pointer-events:none;"></i>
        </button>
        <button class="tpl-more-btn" data-template="${esc(t.name)}" title="More options"
          style="width:32px;height:32px;border-radius:8px;border:1px solid var(--border);background:var(--surface);color:var(--text-2);cursor:pointer;font-size:15px;display:inline-flex;align-items:center;justify-content:center;transition:all .15s;"
          onmouseover="this.style.background='var(--surface-3)';this.style.color='var(--text-1)'"
          onmouseout="this.style.background='var(--surface)';this.style.color='var(--text-2)'">
          <i class="fas fa-ellipsis" style="pointer-events:none;"></i>
        </button>
      </td>
    </tr>`;
    }).join('');
  };

  const refreshTemplateCategories = () => {
    const el = document.getElementById('templateCategoryFilter');
    if (!el) return;
    const unique = [...new Set(window.inboxState.templates.map(t => String(t.category || 'UNCATEGORIZED').toUpperCase()))].sort();
    el.innerHTML = '<option value="all">All categories</option>' + unique.map(c => `<option value="${esc(c)}">${esc(c)}</option>`).join('');
  };

  /* ═══════════════════════════════════════════════════════════
     16. TEMPLATE EVENT BINDING
     ═══════════════════════════════════════════════════════════ */

  const initTemplateSystem = () => {
    // Template select change → update param fields
    document.getElementById('templateSelect')?.addEventListener('change', () => onTemplateChange('templateSelect', 'templateLang', 'templateParamsWrap'));
    document.getElementById('newConvTemplate')?.addEventListener('change', () => onTemplateChange('newConvTemplate', 'newConvLang', 'newConvParamsWrap'));

    // Approved template rows click → quick send or more options
    document.getElementById('approvedTemplateRows')?.addEventListener('click', e => {
      // Quick send
      const btn = e.target.closest('.quick-send-template-btn');
      if (btn) {
        const tName = btn.dataset.template;
        window.inboxState.selectedQuickTemplate = window.inboxState.templates.find(t => t.name === tName);
        if (!window.inboxState.selectedQuickTemplate) return;
        document.getElementById('quickSendTemplateName').textContent = `Send a test message using template "${window.inboxState.selectedQuickTemplate.name}"`;
        document.getElementById('quickTemplateMobile').value = window.inboxState.activeMobile ? `+${window.inboxState.activeMobile}` : '';
        document.getElementById('quickTemplateLang').value = window.inboxState.selectedQuickTemplate.language || 'en_US';
        buildParamFields(extractVars(getBodyText(window.inboxState.selectedQuickTemplate)), 'quickTemplateVars');
        document.getElementById('quickTemplatePreviewText').textContent = getBodyText(window.inboxState.selectedQuickTemplate) || `Template: ${window.inboxState.selectedQuickTemplate.name}`;
        if (templateQuickSendModal) templateQuickSendModal.show();
        return;
      }

      // More options (...)
      const moreBtn = e.target.closest('.tpl-more-btn');
      if (moreBtn) {
        const tName = moreBtn.dataset.template;
        const t = window.inboxState.templates.find(x => x.name === tName);
        if (!t) return;
        const action = confirm(`Template: ${tName}\n\nClick OK to copy template name to clipboard.`);
        if (action) {
          navigator.clipboard?.writeText(tName).then(() => {
            moreBtn.innerHTML = '<i class="fas fa-check" style="pointer-events:none;color:var(--green);"></i>';
            setTimeout(() => { moreBtn.innerHTML = '<i class="fas fa-ellipsis" style="pointer-events:none;"></i>'; }, 1500);
          });
        }
      }
    });

    // Quick template send
    document.getElementById('quickTemplateSendBtn')?.addEventListener('click', async () => {
      const tName = window.inboxState.selectedQuickTemplate?.name;
      const mobile = (document.getElementById('quickTemplateMobile')?.value || window.inboxState.activeMobile || '').replace(/\D/g, '');
      if (!tName || !mobile) return alert('Select template and recipient.');
      const lang = document.getElementById('quickTemplateLang')?.value || 'en_US';
      const params = $all('#quickTemplateVars .tpl-param').map(i => i.value.trim());
      if (params.some(v => !v)) return alert('Fill all variable values.');
      const t = window.inboxState.selectedQuickTemplate || window.inboxState.templates.find(x => x.name === tName);
      const body = { mobile, template_name: tName, language_code: lang, template_preview: getBodyText(t) || `Template: ${tName}`,
        components: params.length ? [{ type:'body', parameters: params.map(text=>({type:'text', text})) }] : [] };
      const r = await fetch(SEND_TPL_URL, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body) });
      const d = await r.json();
      if (!r.ok) return alert(d.error || 'Failed to send.');
      alert('Template sent successfully.');
      if (templateQuickSendModal) templateQuickSendModal.hide();
    });

    // Send template button (modal)
    document.getElementById('sendTemplateBtn')?.addEventListener('click', async () => {
      const tName = document.getElementById('templateSelect')?.value;
      if (!tName) { alert('Select a template'); return; }
      const lang = document.getElementById('templateLang')?.value || 'en_US';
      const params = $all('#templateParamsWrap .tpl-param').map(i => i.value.trim());
      if (params.some(v => !v)) { alert('Fill all parameters'); return; }
      const t = window.inboxState.templates.find(x => x.name === tName);
      const body = { mobile: window.inboxState.activeMobile, template_name: tName, language_code: lang,
        template_preview: getBodyText(t) || `Template: ${tName}`,
        components: params.length ? [{ type:'body', parameters: params.map(text=>({type:'text',text})) }] : [] };
      const r = await fetch(SEND_TPL_URL, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body) });
      const d = await r.json();
      if (!r.ok) { alert(d.error || 'Failed'); return; }
      if (templateModal) templateModal.hide();
      await window.pollingEngine.pollMessages();
      await window.pollingEngine.pollSidebar();
    });

    // Template filter + sync listeners (registered ONCE)
    ['templateSearchInput', 'templateStatusFilter', 'templateCategoryFilter'].forEach(id => {
      const el = document.getElementById(id);
      if (el) {
        el.addEventListener('input', renderApprovedTemplateRows);
        el.addEventListener('change', renderApprovedTemplateRows);
      }
    });

    // Template sync button
    document.getElementById('templateSyncBtn')?.addEventListener('click', async () => {
      const syncBtn = document.getElementById('templateSyncBtn');
      if (syncBtn) { syncBtn.disabled = true; syncBtn.innerHTML = '<i class="fas fa-rotate fa-spin"></i> <span>Syncing…</span>'; }
      await loadTemplates(true);
      refreshTemplateCategories();
      renderApprovedTemplateRows();
      if (syncBtn) { syncBtn.disabled = false; syncBtn.innerHTML = '<i class="fas fa-rotate"></i> <span>Sync</span>'; }
    });

    console.debug('[EVENT_BIND] Template system listeners registered (select, filter, sync, send)');
    console.debug('[LIFECYCLE] Template system initialized');
    console.debug('[EVENT] Template system initialized');
  };

  /* ═══════════════════════════════════════════════════════════
     17. WORKSPACE SWITCHING
     ═══════════════════════════════════════════════════════════ */

  const setWorkspace = async (name = 'inbox') => {
    ['inbox', 'templates', 'campaigns', 'automation'].forEach(key => {
      const panel = document.getElementById(`${key}Panel`);
      if (panel) panel.classList.toggle('active', key === name);
    });
    $all('.workspace-link, .ws-btn').forEach(btn => btn.classList.toggle('active', btn.dataset.workspace === name));
    if (name === 'templates') { await loadTemplates(); refreshTemplateCategories(); renderApprovedTemplateRows(); }
    console.debug('[EVENT] Workspace switched to:', name);
  };

  const initWorkspaceSwitching = () => {
    $all('.workspace-link, .ws-btn').forEach(btn => btn.addEventListener('click', () => setWorkspace(btn.dataset.workspace)));

    // New chat button
    dom.newChatBtn?.addEventListener('click', async () => {
      await loadTemplates();
      if (newConvModal) newConvModal.show();
    });

    // Template button (header dropdown)
    dom.templateBtn?.addEventListener('click', async () => {
      await loadTemplates();
      if (templateModal) templateModal.show();
    });

    console.debug('[EVENT_BIND] Workspace switching listeners registered');
    console.debug('[LIFECYCLE] Workspace switching initialized');
    console.debug('[EVENT] Workspace switching initialized');
  };

  /* ═══════════════════════════════════════════════════════════
     18. NEW CONVERSATION
     ═══════════════════════════════════════════════════════════ */

  const initNewConversation = () => {
    document.getElementById('startConvBtn')?.addEventListener('click', async () => {
      const phone = (document.getElementById('newConvPhone')?.value||'').replace(/\D/g,'');
      const tName = document.getElementById('newConvTemplate')?.value;
      if (!phone) { alert('Enter recipient phone number'); return; }
      if (!tName) { alert('Select a template'); return; }
      const lang = document.getElementById('newConvLang')?.value || 'en_US';
      const params = $all('#newConvParamsWrap .tpl-param').map(i => i.value.trim());
      if (params.some(v => !v)) { alert('Fill all parameters'); return; }
      const t = window.inboxState.templates.find(x => x.name === tName);
      const body = { mobile: phone, template_name: tName, language_code: lang,
        template_preview: getBodyText(t) || `Template: ${tName}`,
        components: params.length ? [{ type:'body', parameters: params.map(text=>({type:'text',text})) }] : [] };
      const r = await fetch(SEND_TPL_URL, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body) });
      const d = await r.json();
      if (!r.ok) { alert(d.error || 'Failed to start conversation'); return; }
      if (newConvModal) newConvModal.hide();
      window.location.href = `${PAGE_URL}?mobile=${encodeURIComponent(phone)}`;
    });

    console.debug('[EVENT_BIND] New conversation start listener registered');
    console.debug('[LIFECYCLE] New conversation initialized');
    console.debug('[EVENT] New conversation initialized');
  };

  /* ═══════════════════════════════════════════════════════════
     19. INTERACTIVE MESSAGE
     ═══════════════════════════════════════════════════════════ */

  const setIType = type => {
    window.inboxState.iType = type;
    $all('#iTypeTabs .nav-link').forEach(b => b.classList.toggle('active', b.dataset.type === type));
    ['paneButton','paneList','paneCta','paneFlow'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.classList.add('d-none');
    });
    const paneMap = { button:'paneButton', list:'paneList', cta_url:'paneCta', flow:'paneFlow' };
    const paneEl = document.getElementById(paneMap[type]);
    if (paneEl) paneEl.classList.remove('d-none');
  };

  const addBtnRow = (title = '') => {
    const wrap = document.getElementById('btnRows');
    if (!wrap || wrap.querySelectorAll('.btn-row').length >= 3) return;
    const row = document.createElement('div');
    row.className = 'btn-row d-flex gap-2 mb-2';
    row.innerHTML = `<span class="input-group-text" style="font-size:12px;width:28px;padding:4px 8px;">${wrap.querySelectorAll('.btn-row').length+1}</span><input class="form-control btn-title-input" maxlength="20" placeholder="Button title" value="${esc(title)}"><button class="btn btn-sm btn-outline-danger" type="button">×</button>`;
    row.querySelector('button').addEventListener('click', () => row.remove());
    wrap.appendChild(row);
  };

  const addListRow = (wrap, title='', desc='') => {
    if (document.querySelectorAll('.list-row').length >= 10) return;
    const row = document.createElement('div');
    row.className = 'list-row border rounded p-2 mb-2';
    row.innerHTML = `<input class="form-control form-control-sm mb-1" maxlength="24" placeholder="Row title" value="${esc(title)}"><input class="form-control form-control-sm" maxlength="72" placeholder="Description (optional)" value="${esc(desc)}"><button class="btn btn-sm btn-outline-danger mt-1" type="button">Remove</button>`;
    row.querySelector('button').addEventListener('click', () => row.remove());
    wrap.appendChild(row);
  };

  const addSection = (sTitle = 'Options') => {
    const wrap = document.getElementById('listSections');
    if (!wrap) return;
    const card = document.createElement('div');
    card.className = 'border rounded p-2 mb-2 list-section';
    card.innerHTML = `<div class="d-flex gap-2 mb-2"><input class="form-control form-control-sm sect-title" maxlength="24" placeholder="Section title" value="${esc(sTitle)}"><button class="btn btn-sm btn-outline-danger" type="button">Remove</button></div><div class="sect-rows"></div><button class="btn btn-sm btn-outline-secondary mt-1" type="button">+ Add row</button>`;
    card.querySelector('.btn-outline-danger').addEventListener('click', () => card.remove());
    const rowsWrap = card.querySelector('.sect-rows');
    card.querySelector('.btn-outline-secondary').addEventListener('click', () => addListRow(rowsWrap));
    wrap.appendChild(card);
    addListRow(rowsWrap, 'Option 1');
  };

  const loadFlows = async (forceSync = false) => {
    const url = forceSync ? `${FLOWS_URL}?sync=1` : FLOWS_URL;
    const r = await fetch(url);
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || 'Failed');
    window.inboxState.flows = data.data || [];
    const sel = document.getElementById('flowSelect');
    if (sel) sel.innerHTML = '<option value="">Select flow</option>' + window.inboxState.flows.map(f =>
      `<option value="${esc(f.id)}">${esc(f.name||f.id)} (${esc((f.status||'').toUpperCase())})</option>`).join('');
  };

  const resetInteractive = () => {
    ['iHeader','iBody','iFooter','listBtnText','ctaText','ctaUrl','flowId'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.value = '';
    });
    const flowBtnText = document.getElementById('flowBtnText');
    if (flowBtnText) flowBtnText.value = 'Open';
    const btnRows = document.getElementById('btnRows');
    if (btnRows) btnRows.innerHTML = '';
    addBtnRow('Yes'); addBtnRow('No');
    const listSections = document.getElementById('listSections');
    if (listSections) listSections.innerHTML = '';
    addSection('Options');
    setIType('button');
  };

  const initInteractive = () => {
    // Type tab click
    $all('#iTypeTabs .nav-link').forEach(b => b.addEventListener('click', () => setIType(b.dataset.type)));

    // Add button row
    document.getElementById('addBtnRow')?.addEventListener('click', () => addBtnRow());

    // Add section
    document.getElementById('addSection')?.addEventListener('click', () => addSection('Section'));

    // Flow select → update flow ID field
    document.getElementById('flowSelect')?.addEventListener('change', () => {
      const flowId = document.getElementById('flowId');
      if (flowId) flowId.value = document.getElementById('flowSelect').value;
    });

    // Sync flows button
    document.getElementById('syncFlowsBtn')?.addEventListener('click', async function() {
      this.disabled = true;
      try { await loadFlows(true); alert(`Synced ${window.inboxState.flows.length} flows.`); }
      catch(e) { alert(e.message); }
      finally { this.disabled = false; }
    });

    // Interactive button (header dropdown) → show modal
    dom.interactiveBtn?.addEventListener('click', async () => {
      resetInteractive();
      try { await loadFlows(); } catch(e) {}
      if (interactiveModal) interactiveModal.show();
    });

    // Send interactive button
    document.getElementById('sendInteractiveBtn')?.addEventListener('click', async () => {
      const body = document.getElementById('iBody')?.value.trim();
      if (!body) { alert('Body text is required'); return; }
      const header = document.getElementById('iHeader')?.value.trim();
      const footer = document.getElementById('iFooter')?.value.trim();
      const payload = { mobile: window.inboxState.activeMobile, interactive_type: window.inboxState.iType, body, header, footer };

      if (window.inboxState.iType === 'button') {
        const titles = $all('#btnRows .btn-title-input').map(i => i.value.trim()).filter(Boolean);
        if (!titles.length) { alert('Add at least one button'); return; }
        payload.buttons = titles.map((title, i) => ({ id:`btn_${i+1}`, title }));
      } else if (window.inboxState.iType === 'list') {
        const lbt = document.getElementById('listBtnText')?.value.trim();
        if (!lbt) { alert('List button text required'); return; }
        const sections = $all('.list-section').map(s => ({
          title: s.querySelector('.sect-title')?.value.trim(),
          rows: $all('.list-row', s).map((r, ri) => ({
            id: `row_${ri+1}_${Date.now()}`,
            title: r.querySelectorAll('input')[0]?.value.trim(),
            description: r.querySelectorAll('input')[1]?.value.trim()
          })).filter(r => r.title)
        })).filter(s => s.rows.length);
        if (!sections.length) { alert('Add at least one row'); return; }
        payload.list_button_text = lbt;
        payload.sections = sections;
      } else if (window.inboxState.iType === 'cta_url') {
        const text = document.getElementById('ctaText')?.value.trim();
        const url  = document.getElementById('ctaUrl')?.value.trim();
        if (!text || !url) { alert('Button text and URL required'); return; }
        payload.button_text = text; payload.button_url = url;
      } else if (window.inboxState.iType === 'flow') {
        const fid = document.getElementById('flowId')?.value.trim();
        const fbt = document.getElementById('flowBtnText')?.value.trim();
        if (!fid || !fbt) { alert('Flow ID and button text required'); return; }
        payload.flow_id = fid; payload.button_text = fbt;
      }

      const r = await fetch(SEND_INT_URL, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) });
      const d = await r.json();
      if (!r.ok) { alert(d.error || 'Failed'); return; }
      if (interactiveModal) interactiveModal.hide();
      await window.pollingEngine.pollMessages();
      await window.pollingEngine.pollSidebar();
    });

    console.debug('[MODAL] Interactive modal listeners registered (tabs, buttons, sections, flows, send)');
    console.debug('[LIFECYCLE] Interactive message initialized');
    console.debug('[EVENT] Interactive message initialized');
  };

  /* ═══════════════════════════════════════════════════════════
     20. COMMAND PALETTE
     ═══════════════════════════════════════════════════════════ */

  function openCommandPalette() {
    dom.cmdOverlay?.classList.add('open');
    setTimeout(() => dom.cmdInput?.focus(), 40);
  }

  function closeCommandPalette() {
    dom.cmdOverlay?.classList.remove('open');
  }

  const initCommandPalette = () => {
    // Click overlay background to close
    dom.cmdOverlay?.addEventListener('click', e => { if (e.target === dom.cmdOverlay) closeCommandPalette(); });

    // Keyboard shortcuts: Cmd+K to open, Escape to close
    document.addEventListener('keydown', e => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); openCommandPalette(); }
      if (e.key === 'Escape') closeCommandPalette();
    });

    // Command items click handler
    $all('.cmd-item').forEach(item => {
      item.addEventListener('click', () => {
        const cmd = item.dataset.cmd;
        closeCommandPalette();
        if (cmd === 'template') dom.templateBtn?.click();
        else if (cmd === 'interactive') dom.interactiveBtn?.click();
        else if (cmd === 'focus') { dom.msgInput?.focus(); }
        else if (cmd === 'dashboard') {
          // This URL is set from the Django template — keep in HTML
          const dashboardUrl = document.querySelector('a[href*="dashboard"]')?.getAttribute('href');
          if (dashboardUrl) window.location.href = dashboardUrl;
        }
        else if (cmd === 'complaints') {
          const complaintsUrl = document.querySelector('a[href*="complaints"]')?.getAttribute('href');
          if (complaintsUrl) window.location.href = complaintsUrl;
        }
      });
    });

    console.debug('[COMMAND] Command palette listeners registered (overlay click, kb shortcut, cmd items)');
    console.debug('[LIFECYCLE] Command palette initialized');
    console.debug('[EVENT] Command palette initialized');
  };

  /* ═══════════════════════════════════════════════════════════
     21. AI TOGGLE
     ═══════════════════════════════════════════════════════════ */

  const initAiToggle = () => {
    const aiCheckbox = document.getElementById('aiToggleBtn');
    const aiLabel    = document.getElementById('aiToggleText');

    if (aiCheckbox) {
      aiCheckbox.addEventListener('change', async () => {
        const phone    = aiCheckbox.dataset.phone;
        const newHuman = aiCheckbox.checked ? 1 : 0;

        // Optimistically update label
        if (aiLabel) aiLabel.textContent = newHuman ? 'Human' : 'AI';

        try {
          const response = await fetch('/api/whatsapp/toggle-ai', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ phone, human_takeover: newHuman })
          });
          const data = await response.json();

          if (!data.success) {
            // Revert on failure
            aiCheckbox.checked = !aiCheckbox.checked;
            if (aiLabel) aiLabel.textContent = aiCheckbox.checked ? 'Human' : 'AI';
          } else {
            aiCheckbox.dataset.human = newHuman;
          }
        } catch (err) {
          console.error('[EVENT] AI toggle failed:', err);
          // Revert on network error
          aiCheckbox.checked = !aiCheckbox.checked;
          if (aiLabel) aiLabel.textContent = aiCheckbox.checked ? 'Human' : 'AI';
        }
      });
    }

    console.debug('[EVENT_BIND] AI toggle change listener registered');
    console.debug('[LIFECYCLE] AI toggle initialized');
    console.debug('[EVENT] AI toggle initialized');
  };

  /* ═══════════════════════════════════════════════════════════
     BIND ALL — Register all event listeners
     ═══════════════════════════════════════════════════════════ */

  const bindAll = () => {
    console.debug('[EVENT] bindAll() — registering all event listeners');

    initTheme();
    initAudio();
    initInteractionUnlock();
    initSoundToggle();
    initTextarea();
    initAttachment();
    initQuickChips();
    initSearch();
    initConvClick();
    initFilter();
    initScrollButton();
    initMediaViewer();
    initSendMessage();
    initTemplateSystem();
    initWorkspaceSwitching();
    initNewConversation();
    initInteractive();
    initCommandPalette();
    initAiToggle();

    // ── Stress hardening: DOM pressure + mobile drift ──
    startDomPressureMonitoring();
    patchMobileTracking();

    console.debug('[EVENT_BIND] All 19 event subsystems + stress protections registered');
    console.debug('[MODULE]', 'events.js loaded');
  };

  /* ═══════════════════════════════════════════════════════════
     22. DOM PRESSURE WARNING INTERVAL
     ═══════════════════════════════════════════════════════════ */

  let _domPressureInterval = null;

  /**
   * Start periodic DOM pressure monitoring.
   * Logs warnings when thresholds are exceeded.
   * Safe to call multiple times — prevents duplicate intervals.
   */
  const startDomPressureMonitoring = () => {
    if (_domPressureInterval) {
      console.debug('[DOM_WARN] Monitoring already started — skipping');
      return;
    }

    console.debug('[DOM_WARN] Starting DOM pressure monitoring (60s interval)');

    _domPressureInterval = setInterval(() => {
      // Track interval in inboxState for cleanup
      if (window.inboxState && window.inboxState.activeTimerIds) {
        if (window.inboxState.activeTimerIds.indexOf(_domPressureInterval) === -1) {
          window.inboxState.activeTimerIds.push(_domPressureInterval);
        }
      }
      const st = window.inboxState;
      if (!st) return;

      const totalNodes = document.querySelectorAll('*').length;
      const rend = window.renderEngine;
      const msgMapSize = rend?.messageNodeMap?.size || 0;
      const activeListeners = _listenerCount;
      const sidebarCount = dom.convList?.querySelectorAll('.conv-item').length || 0;

      // Track in debugMetrics
      window.debugMetrics.memory.domNodeCount = totalNodes;
      window.debugMetrics.memory.messageNodeMapSize = msgMapSize;
      window.debugMetrics.memory.listenerEstimate = activeListeners;
      window.debugMetrics.memory.sidebarContactCount = sidebarCount;

      // ── Warning thresholds ──
      if (totalNodes > DOM_WARN_THRESHOLDS.totalNodes) {
        console.warn('[DOM_WARN] Excessive DOM nodes:', totalNodes, '(threshold:', DOM_WARN_THRESHOLDS.totalNodes, ')');
      }
      if (msgMapSize > DOM_WARN_THRESHOLDS.messageNodes) {
        console.warn('[DOM_WARN] Large messageNodeMap:', msgMapSize, '(threshold:', DOM_WARN_THRESHOLDS.messageNodes, ')');
      }
      if (activeListeners > DOM_WARN_THRESHOLDS.listeners) {
        console.warn('[DOM_WARN] High listener count:', activeListeners, '(threshold:', DOM_WARN_THRESHOLDS.listeners, ')');
      }
      if (sidebarCount > DOM_WARN_THRESHOLDS.sidebarContacts) {
        console.warn('[DOM_WARN] Large sidebar:', sidebarCount, 'contacts (threshold:', DOM_WARN_THRESHOLDS.sidebarContacts, ')');
      }

      // ── Long-session memory protection: periodic pruning ──
      const knownIds = st.globalKnownMessageIds?.size || 0;
      if (knownIds > 5000) {
        if (st.pruneKnownIdsSafe) st.pruneKnownIdsSafe(5000);
      }
      if (msgMapSize > 2000) {
        if (st.pruneMessageNodeMapSafe) st.pruneMessageNodeMapSafe(2000);
      }

      // ── Detect detached / orphaned nodes ──
      if (msgMapSize > 500) {
        const rendEngine = window.renderEngine;
        if (rendEngine?.messageNodeMap && rendEngine.messageNodeMap.size > 0) {
          const chatBodyEl = dom.chatBody;
          // Only prune message nodes if the DOM is non-empty
          // We'll trigger cleanupDetachedNodesSafe from the health snapshot instead
        }
      }

      console.debug('[MEMORY] DOM pressure check:', { totalNodes, msgMapSize, activeListeners, sidebarCount, knownIds });
    }, 60000);
  };

  /**
   * Stop DOM pressure monitoring.
   */
  const stopDomPressureMonitoring = () => {
    if (_domPressureInterval) {
      clearInterval(_domPressureInterval);
      _domPressureInterval = null;
      console.debug('[DOM_WARN] Monitoring stopped');
    }
  };

  /* ═══════════════════════════════════════════════════════════
     23. MOBILE TIMER DRIFT DETECTION
     ═══════════════════════════════════════════════════════════ */

  /**
   * Track interaction timestamps to detect mobile timer drift.
   * Call on each user interaction (keyboard, click, touch).
   * If gap between interactions exceeds threshold on mobile,
   * the browser likely throttled timers (common in hidden tabs).
   */
  const trackMobileInteraction = () => {
    if (!_isMobileAgent) return;
    const now = Date.now();
    const gap = now - _lastInteractionTs;
    _lastInteractionTs = now;

    if (gap > MOBILE_DRIFT_THRESHOLD_MS && gap > 2000) {
      console.debug('[RECOVERY] Mobile interaction gap detected:', Math.round(gap / 1000), 's — possible timer drift');
      // The polling visibilitychange handler will force resync
    }
  };

  /**
   * Patch initInteractionUnlock to also track mobile interactions.
   * The activity tracker already listens for user events — we augment it.
   */
  const patchMobileTracking = () => {
    if (!_isMobileAgent) return;
    ['pointerdown', 'keydown', 'touchstart', 'click'].forEach(ev => {
      window.addEventListener(ev, trackMobileInteraction, { passive: true });
    });
    console.debug('[MEMORY] Mobile timer drift detection registered');
  };

  /* ═══════════════════════════════════════════════════════════
     EXPORTS
     ═══════════════════════════════════════════════════════════ */

  window.eventsEngine = {
    init,
    bindAll,

    // Theme
    initTheme,

    // Audio
    initAudio,
    initInteractionUnlock,
    beep,
    initSoundToggle,
    updateNotifIcon,

    // Composer
    resizeInput,
    initTextarea,
    initAttachment,
    initQuickChips,

    // Search
    initSearch,

    // Conversation
    initConvClick,
    initFilter,

    // Scroll
    initScrollButton,

    // Media
    initMediaViewer,

    // Send
    initSendMessage,

    // Templates
    initTemplateSystem,
    extractVars,
    getBodyText,
    buildParamFields,
    onTemplateChange,
    loadTemplates,
    renderApprovedTemplateRows,
    refreshTemplateCategories,

    // Workspace
    setWorkspace,
    initWorkspaceSwitching,

    // New conversation
    initNewConversation,

    // Interactive
    initInteractive,
    setIType,
    addBtnRow,
    addListRow,
    addSection,
    loadFlows,
    resetInteractive,

    // Command palette
    openCommandPalette,
    closeCommandPalette,
    initCommandPalette,

    // AI
    initAiToggle,
  };

  window.__eventsEngineInitialized = true;

})();
