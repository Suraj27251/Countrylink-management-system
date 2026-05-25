/**
 * whatsapp/polling.js — Polling & synchronization engine for WhatsApp inbox.
 *
 * PURPOSE:
 *   Centralize ALL polling, heartbeat, and sync logic into ONE module.
 *   Eliminate duplicate / scattered polling state from templates/whatsapp.html.
 *
 * RULES:
 *   - Initialize ONLY ONCE via window.pollingEngine.init(config).
 *   - Uses window.inboxState for shared state (never duplicates).
 *   - Polling lifecycle is managed entirely here — no orphaned timers.
 *
 * @see Phase 3 of WhatsApp inbox modularization.
 */

(function () {
  'use strict';

  /* ── Guard: prevent double-initialization ──────────────── */
  if (window.__pollingEngineInitialized) {
    console.debug('[POLLING] Already initialized — skipping.');
    return;
  }

  /* ── Private polling state (not exposed to inboxState) ─── */
  let heartbeatTimer       = null;
  let isHeartbeatTickRunning = false;
  let currentHeartbeatMs   = 1000; // default active ms
  let lastOperatorActivityAt = Date.now();

  /* ── AbortControllers ──────────────────────────────────── */
  let sidebarPollController    = null;
  let messagesPollController   = null;

  /* ── Request tokens (stale response protection) ────────── */
  let activeMessageRequestToken  = 0;
  let activeSidebarRequestToken  = 0;

  /* ── Polling timing constants ──────────────────────────── */
  const POLL_BASE = 2000;
  const POLL_MAX  = 30000;

  /* ── Recovery / cooldown state ────────────────────────── */
  let lastRecoveryTime = 0;
  const MIN_RECOVERY_INTERVAL_MS = 3000;  // min ms between forced recoveries
  let isRecovering = false;

  /* ── Render burst protection ──────────────────────────── */
  let lastRenderTimestamp = 0;
  const RENDER_THROTTLE_MS = 50;

  /* ── Mobile-safe timing ────────────────────────────────── */
  const MOBILE_HEARTBEAT_MULTIPLIER = 1.5; // slower heartbeat on mobile
  const MOBILE_RECOVERY_DELAY_MS = 2000;    // delay recovery on mobile

  /* ── DOM references (set during init) ──────────────────── */
  let connectionDot = null;
  let chatBody      = null;

  /* ── Render function deps (set during init from HTML IIFE) ─ */
  let debouncedRenderContacts = null;
  let nearBottom  = null;
  let scrollBottom = null;
  let upsertMsg   = null;
  let beep        = null;
  let updateContactGreenDot = null;
  let refreshSidebar = null;
  let refreshActiveChat = null;

  /* ── Flag helpers (set during init) ────────────────────── */
  let HEARTBEAT_ACTIVE_MS       = 1000;
  let HEARTBEAT_BACKGROUND_MS   = 5000;
  let HEARTBEAT_IDLE_MS         = 10000;
  let HEARTBEAT_IDLE_AFTER_MS   = 5 * 60 * 1000;
  let API_URL                   = '';
  let POLL_BASE_VALUE           = POLL_BASE;
  let POLL_MAX_VALUE            = POLL_MAX;

  /* ═══════════════════════════════════════════════════════════
     PRIVATE HELPERS
     ═══════════════════════════════════════════════════════════ */

  /* ── Connection indicator ──────────────────────────────── */
  const setPoll = (live, label) => {
    const dot = connectionDot;
    if (!dot) return;

    dot.classList.remove('live', 'degraded', 'offline');
    dot.classList.add(live ? 'live' : 'offline');
    dot.title = live ? 'LIVE' : (label || 'Reconnecting...');
    console.debug('[POLLING] Connection status:', live ? 'LIVE' : (label || 'Offline'));
  };

  /* ── Adaptive heartbeat interval ────────────────────────── */
  const getHeartbeatMs = () => {
    if (document.hidden) return HEARTBEAT_BACKGROUND_MS;
    if (Date.now() - lastOperatorActivityAt > HEARTBEAT_IDLE_AFTER_MS) return HEARTBEAT_IDLE_MS;
    return HEARTBEAT_ACTIVE_MS;
  };


  const parseJsonResponse = (response, contextLabel) => window.whatsappUtils.parseJsonResponse(response, contextLabel, '[POLLING_API]');
  const fetchWithTimeout = (url, options, timeoutMs, contextLabel) => window.whatsappUtils.fetchWithTimeout(url, options, timeoutMs, contextLabel);

  /* ═══════════════════════════════════════════════════════════
     POLLING CORE
     ═══════════════════════════════════════════════════════════ */

  /**
   * Unified poll: fetches sidebar contacts AND messages for the active chat
   * in a single API call. Called by schedulePoll and heartbeatTick.
   *
   * Depends on globals defined in templates/whatsapp.html:
   *   debouncedRenderContacts(), beep(), updateContactGreenDot(),
   *   nearBottom(), upsertMsg(), scrollBottom()
   */
  const poll = async () => {
    console.debug('[POLLING] poll() started — activeMobile:', window.inboxState.activeMobile);
    if (window.inboxState.debugActiveMobile) window.inboxState.debugActiveMobile('poll:begin');

    try {
      const params = new URLSearchParams({
        include_contacts: '1',
        since_inbox_id: String(window.inboxState.cursors.globalLastInboxId)
      });
      if (window.inboxState.activeMobile) {
        params.set('mobile', window.inboxState.activeMobile);
        params.set('since_id', String(window.inboxState.cursors.globalLastMessageId));
      }
      const pollStart = performance.now();
      console.debug('[FETCH_MOBILE]', { endpoint: API_URL, mobile: window.inboxState.activeMobile, source: 'poll' });
      const res = await fetchWithTimeout(`${API_URL}?${params}`, {
        headers: { 'X-Requested-With': 'XMLHttpRequest' }
      }, 20000, 'Polling fetch');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await parseJsonResponse(res, 'Polling fetch');
      window.inboxState.lastPollSuccessAt = Date.now();
      const fetchDuration = performance.now() - pollStart;
      window.debugMetrics.polling.lastFetchMs = Math.round(fetchDuration);
      console.debug('[POLLING] poll() response received — contacts:', data.contacts?.length, 'messages:', data.messages?.length, 'inbox_messages:', data.inbox_messages?.length);
      console.debug('[PERF] poll fetch:', Math.round(fetchDuration), 'ms');

      // ── Contacts / Sidebar ──
      if (data.contacts?.length) {
        // Centralize contact ownership: store then render
        window.inboxState.contacts = data.contacts;
        if (debouncedRenderContacts) debouncedRenderContacts(data.contacts, true);
      }

      // ── Inbox messages (all chats) ──
      if (data.inbox_messages?.length) {
        let newMessagesCount = 0;
        const soundsToPlay = [];
        const contactsToUpdate = new Set();

        data.inbox_messages.forEach(m => {
          const idStr = String(m.id);
          if (window.inboxState.globalKnownMessageIds.has(idStr)) return;
          window.inboxState.globalKnownMessageIds.add(idStr);
          window.inboxState.cursors.globalLastInboxId = Math.max(
            window.inboxState.cursors.globalLastInboxId, +m.id || 0
          );

          const mobile = m.mobile || '';
          const isInbound = m.direction !== 'outbound';
          const isActiveChatMsg = window.inboxState.activeMobile &&
                                  mobile === window.inboxState.activeMobile;

          if (isInbound && mobile) {
            const lastSeenId = window.inboxState.lastSeenMessageIdByMobile.get(mobile) || 0;
            const currentMsgId = +m.id || 0;
            window.inboxState.lastSeenMessageIdByMobile.set(mobile, Math.max(lastSeenId, currentMsgId));

            if (isActiveChatMsg) {
              window.inboxState.unreadByMobile.delete(mobile);
            } else if (currentMsgId > lastSeenId) {
              window.inboxState.unreadByMobile.set(
                mobile,
                (window.inboxState.unreadByMobile.get(mobile) || 0) + 1
              );
              if (beep) beep(mobile);
              contactsToUpdate.add(mobile);
            }
          }
        });

        contactsToUpdate.forEach(mob => { if (updateContactGreenDot) updateContactGreenDot(mob); });
        soundsToPlay.forEach(mob => { if (beep) beep(mob); });

        if (newMessagesCount > 0) {
          console.debug('[POLLING] inbox_messages processed:', newMessagesCount);
        }
      }

      // ── Active chat messages ──
      if (data.messages?.length) {
        const wasNearBottom = nearBottom ? nearBottom() : false;
        let hasNew = false;

        // Seed lastSeenMessageIdByMobile to prevent beeping old messages
        if (window.inboxState.activeMobile && data.last_message_id) {
          const current = window.inboxState.lastSeenMessageIdByMobile.get(window.inboxState.activeMobile) || 0;
          window.inboxState.lastSeenMessageIdByMobile.set(
            window.inboxState.activeMobile,
            Math.max(current, +data.last_message_id || 0)
          );
        }

        data.messages.forEach(m => {
          const isNewMsg = upsertMsg ? upsertMsg(m) : false;

          // Sound for new inbound messages from OTHER chats
          if (isNewMsg && m.direction !== 'outbound' && document.hasFocus() &&
              m.mobile !== window.inboxState.activeMobile) {
            const lastSeen = window.inboxState.lastSeenMessageIdByMobile.get(m.mobile) || 0;
            const curId = +m.id || 0;
            if (curId > lastSeen) {
              window.inboxState.lastSeenMessageIdByMobile.set(m.mobile, Math.max(lastSeen, curId));
              if (beep) beep(m.mobile);
            }
          }
          hasNew = isNewMsg || hasNew;
        });

        if (data.last_message_id) {
          window.inboxState.cursors.globalLastMessageId = Math.max(
            window.inboxState.cursors.globalLastMessageId, +data.last_message_id || 0
          );
        }

        if (data.last_inbox_message_id) {
          window.inboxState.cursors.globalLastInboxId = Math.max(
            window.inboxState.cursors.globalLastInboxId, +data.last_inbox_message_id || 0
          );
        }

        if (hasNew || wasNearBottom) {
          if (scrollBottom) scrollBottom();
        }

        // Refresh active chat UI after processing its messages
        if (window.inboxState.activeMobile && refreshActiveChat) {
          refreshActiveChat();
        }
      }

      // Update cursor from top-level field if present
      if (data.last_inbox_message_id) {
        window.inboxState.cursors.globalLastInboxId = Math.max(
          window.inboxState.cursors.globalLastInboxId, +data.last_inbox_message_id || 0
        );
      }

      window.inboxState.resetPollFails();
      setPoll(true, 'Connected');
      console.debug('[POLLING] poll() completed successfully');

    } catch (err) {
      if (err?.name === 'AbortError') {
        console.debug('[POLLING] poll() aborted');
        return;
      }
      window.inboxState.incrementPollFails();
      throw err; // caller handles backoff
    }
  };

  /**
   * Dedicated message poll for the active chat.
   * Used by explicit refresh (conv-item click) where AbortController + token guard are critical.
   *
   * Depends on globals: nearBottom(), upsertMsg(), scrollBottom(), beep()
   */
  const pollMessages = async () => {
    const requestMobile = (window.inboxState.activeMobile || '').trim();
    if (!requestMobile) { console.error('[STATE_FATAL] activeMobile missing'); debugger; return; }
    const pollMsgStart = performance.now();
    const currentToken = ++activeMessageRequestToken;
    console.debug('[POLLING] pollMessages() started — mobile:', requestMobile, 'token:', currentToken);
    window.inboxState.logToken('messages', currentToken);

    try {
      const sinceId = window.inboxState.lastMessageIdByMobile.get(requestMobile) || 0;
      const params = new URLSearchParams({ mobile: requestMobile, since_id: String(sinceId) });

      window.abortControllerSafe(messagesPollController);
      messagesPollController = new AbortController();
      if (window.inboxState.activeAbortControllers) {
        window.inboxState.activeAbortControllers.push(messagesPollController);
      }

      console.debug('[FETCH_MOBILE]', { endpoint: API_URL, mobile: requestMobile, source: 'pollMessages' });
      const res = await fetchWithTimeout(`${API_URL}?${params}`, {
        headers: { 'X-Requested-With': 'XMLHttpRequest' },
        signal: messagesPollController.signal
      }, 20000, 'Polling messages fetch');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await parseJsonResponse(res, 'Polling fetch');
      window.inboxState.lastPollSuccessAt = Date.now();

      // Stale response guard: verify still on same chat + token is latest
      if (requestMobile !== window.inboxState.activeMobile ||
          currentToken !== activeMessageRequestToken) {
        console.debug('[POLLING] pollMessages() stale response discarded — token mismatch');
        return;
      }

      // Hide loading overlay
      const loadingOverlay = chatBody?.querySelector('.chat-loading-overlay');
      if (loadingOverlay?.classList.contains('show')) {
        loadingOverlay.classList.remove('show');
        loadingOverlay.classList.add('hide');
        setTimeout(() => {
          if (loadingOverlay?.classList.contains('hide')) {
            loadingOverlay.classList.remove('hide');
          }
        }, 200);
      }

      if (data.messages?.length) {
        const wasNearBottom = nearBottom ? nearBottom() : false;
        let hasNew = false;
        let maxRenderedId = sinceId;

        if (data.last_message_id) {
          const newLast = Math.max(
            window.inboxState.lastSeenMessageIdByMobile.get(requestMobile) || 0,
            +data.last_message_id || 0
          );
          window.inboxState.lastSeenMessageIdByMobile.set(requestMobile, newLast);
        }

        data.messages.forEach(m => {
          try {
            const isNewMsg = upsertMsg ? upsertMsg(m) : false;
            hasNew = isNewMsg || hasNew;
            maxRenderedId = Math.max(maxRenderedId, +m.id || 0);
          } catch (err) {
            console.error('[POLLING] Message render failed:', m, err);
          }
        });

        if (data.last_message_id) {
          maxRenderedId = Math.max(maxRenderedId, +data.last_message_id || 0);
        }
        if (maxRenderedId > sinceId) {
          window.inboxState.lastMessageIdByMobile.set(requestMobile, maxRenderedId);
        }

        if (hasNew || wasNearBottom) { if (scrollBottom) scrollBottom(); }

        // Refresh active chat UI after explicit message poll
        if (window.inboxState.activeMobile && refreshActiveChat) {
          refreshActiveChat();
        }
      }

      const pollMsgDuration = performance.now() - pollMsgStart;
      console.debug('[PERF] pollMessages:', Math.round(pollMsgDuration), 'ms — mobile:', requestMobile);

      window.inboxState.resetPollFails();
      setPoll(true, 'Connected');

    } catch (err) {
      if (err?.name !== 'AbortError') {
        const loadingOverlay = chatBody?.querySelector('.chat-loading-overlay');
        if (loadingOverlay?.classList.contains('show')) {
          loadingOverlay.classList.remove('show');
          loadingOverlay.classList.add('hide');
          setTimeout(() => {
            if (loadingOverlay?.classList.contains('hide')) {
              loadingOverlay.classList.remove('hide');
            }
          }, 200);
        }
      }
      throw err;
    }
  };

  /**
   * Schedule the next poll() call with exponential backoff on failure.
   */
  const schedulePoll = (delay) => {
    delay = delay || POLL_BASE_VALUE;
    clearTimeout(window.inboxState.pollTimer);
    console.debug('[POLLING] schedulePoll() — delay:', delay, 'ms');

    window.inboxState.pollTimer = setTimeout(async () => {
      try {
        await poll();
        schedulePoll(POLL_BASE_VALUE);
      } catch (e) {
        const backoff = Math.min(
          POLL_BASE_VALUE * (2 ** window.inboxState.pollFails),
          POLL_MAX_VALUE
        );
        setPoll(false, `Retry in ${Math.round(backoff / 1000)}s`);
        console.debug('[POLLING] schedulePoll() backoff:', backoff, 'ms, fails:', window.inboxState.pollFails);
        schedulePoll(backoff);
      }
    }, delay);
  };

  /* ═══════════════════════════════════════════════════════════
     HEARTBEAT
     ═══════════════════════════════════════════════════════════ */

  /**
   * Heartbeat tick — runs on interval, calls unified poll().
   * Re-entrancy guard prevents overlapping ticks.
   */
  const heartbeatTick = async () => {
    if (isHeartbeatTickRunning) {
      console.debug('[HEARTBEAT] Tick skipped — previous still running');
      return;
    }
    isHeartbeatTickRunning = true;
    console.debug('[HEARTBEAT] Tick started');

    try {
      await poll();
      setPoll(true, 'LIVE');
      console.debug('[HEARTBEAT] Tick completed — LIVE');
    } catch (e) {
      if (e?.name === 'AbortError') return;
      console.warn('[HEARTBEAT] Tick failed:', e);
      setPoll(false, 'Reconnecting...');
    } finally {
      isHeartbeatTickRunning = false;
    }
  };

  /**
   * Start (or restart) the heartbeat interval.
   * Automatically adjusts interval based on visibility + operator activity.
   * Prevents duplicate intervals when config hasn't changed.
   */
  const startHeartbeat = () => {
    const nextMs = getHeartbeatMs();
    if (heartbeatTimer && currentHeartbeatMs === nextMs) {
      console.debug('[HEARTBEAT] startHeartbeat() skipped — already running at', nextMs, 'ms');
      return;
    }
    if (heartbeatTimer) {
      clearInterval(heartbeatTimer);
      console.debug('[HEARTBEAT] Replacing interval: was', currentHeartbeatMs, 'ms, now', nextMs, 'ms');
    }
    currentHeartbeatMs = nextMs;
    heartbeatTimer = setInterval(heartbeatTick, currentHeartbeatMs);
    console.debug('[HEARTBEAT] Started at', currentHeartbeatMs, 'ms (hidden:', document.hidden, ')');
  };

  /**
   * Stop the heartbeat interval.
   * Call before cleanup or full page unload.
   */
  const stopHeartbeat = () => {
    if (heartbeatTimer) {
      clearInterval(heartbeatTimer);
      heartbeatTimer = null;
      console.debug('[HEARTBEAT] Stopped');
    }
  };

  /* ── Visibility change handler ──────────────────────────── */
  /**
   * Handle tab visibility changes to stabilize polling after idle/reconnect.
   * On returning to the tab: force a full poll + sidebar refresh.
   * Prevents stale active chat after long inactivity.
   *
   * Recovery cooldown: prevents overlapping recovery storms.
   * Mobile-safe: adds extra delay on mobile to avoid timer drift.
   */
  const handleVisibilityChange = () => {
    if (!document.hidden) {
      // ── Mobile timer drift check ─────────────────────────
      const driftDetected = detectTimerDrift();

      // ── Recovery cooldown guard ──────────────────────────
      const now = Date.now();
      if (now - lastRecoveryTime < MIN_RECOVERY_INTERVAL_MS) {
        console.debug('[RECOVERY] Suppressed — cooldown active (', Math.round(now - lastRecoveryTime), 'ms since last)',
          driftDetected ? '(timer drift detected)' : '');
        window.inboxState.stress.stalePollDrops++;
        return;
      }
      if (isRecovering) {
        console.debug('[RECOVERY] Suppressed — recovery already in progress',
          driftDetected ? '(timer drift detected)' : '');
        window.inboxState.stress.stalePollDrops++;
        return;
      }
      isRecovering = true;
      lastRecoveryTime = now;
      window.inboxState.stress.recoveryCount++;

      // ── Mobile-safe delay ────────────────────────────────
      const isMobile = /Mobi|Android|iPhone/i.test(navigator.userAgent);
      const recoveryDelay = isMobile ? MOBILE_RECOVERY_DELAY_MS : 0;

      const mobileLog = driftDetected ? ' (timer drift detected + mobile-safe delay: ' + MOBILE_RECOVERY_DELAY_MS + 'ms)'
        : isMobile ? ' (mobile-safe delay: ' + MOBILE_RECOVERY_DELAY_MS + 'ms)' : '';
      console.debug('[RECOVERY] Tab became visible — forcing poll + sidebar refresh', mobileLog);
      window.debugMetrics.sync.lastTabRestore = now;

      setTimeout(() => {
        // Force a full poll cycle to catch up
        poll().then(() => {
          if (refreshSidebar) {
            refreshSidebar();
            console.debug('[RECOVERY] Sidebar refreshed after tab restore');
          }
          if (refreshActiveChat && window.inboxState.activeMobile) {
            refreshActiveChat();
            console.debug('[RECOVERY] Active chat refreshed after tab restore');
          }
        }).catch(err => {
          console.warn('[RECOVERY] Post-restore poll failed:', err);
        }).finally(() => {
          isRecovering = false;
        });

        // Restart heartbeat with current timing
        startHeartbeat();
      }, recoveryDelay);
    } else {
      console.debug('[RECOVERY] Tab hidden — heartbeat will adapt',
        isMobileAgent ? '(mobile timer drift expected)' : '');
      window.inboxState.ui.isWindowFocused = false;
    }
  };

  /**
   * Detection of mobile timer drift.
   * Mobile browsers aggressively throttle timers in background tabs.
   * On return, we skip stale heartbeat ticks and force resync.
   * Called from handleVisibilityChange before initiating recovery.
   */
  const isMobileAgent = /Mobi|Android|iPhone/i.test(navigator.userAgent);
  const detectTimerDrift = () => {
    if (!isMobileAgent) return false;
    const expectedTick = currentHeartbeatMs;
    const actualGap = Date.now() - lastRecoveryTime;
    if (expectedTick > 0 && actualGap > expectedTick * 3) {
      console.debug('[RECOVERY] Mobile timer drift detected — gap:', actualGap, 'ms (expected ~', expectedTick, 'ms)');
      return true;
    }
    return false;
  };

  /* ── Lightweight sidebar-only poll (used after send actions) ─ */
  const pollSidebar = async () => {
    console.debug('[POLLING] pollSidebar() started');
    try {
      const params = new URLSearchParams({ include_contacts: '1' });
      console.debug('[FETCH_MOBILE]', { endpoint: API_URL, mobile: window.inboxState.activeMobile, source: 'pollSidebar' });
      const res = await fetchWithTimeout(`${API_URL}?${params}`, {
        headers: { 'X-Requested-With': 'XMLHttpRequest' }
      }, 20000, 'Polling fetch');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await parseJsonResponse(res, 'Polling fetch');
      window.inboxState.lastPollSuccessAt = Date.now();
      if (data.contacts?.length) {
        window.inboxState.contacts = data.contacts;
        if (debouncedRenderContacts) debouncedRenderContacts(data.contacts, true);
      }
      console.debug('[POLLING] pollSidebar() completed — contacts:', data.contacts?.length);
    } catch (err) {
      console.warn('[POLLING] pollSidebar() failed:', err);
    }
  };

  /* ── Operator activity tracker ──────────────────────────── */
  const updateActivity = () => {
    lastOperatorActivityAt = Date.now();
  };

  /* ═══════════════════════════════════════════════════════════
     INITIALIZATION
     ═══════════════════════════════════════════════════════════ */

  /**
   * One-time initialization.
   *
   * @param {Object} config
   * @param {string} config.apiUrl         — API endpoint for polling
   * @param {number} config.heartbeatActiveMs       — active tab heartbeat interval
   * @param {number} config.heartbeatBackgroundMs   — background tab heartbeat interval
   * @param {number} config.heartbeatIdleMs         — idle heartbeat interval
   * @param {number} config.heartbeatIdleAfterMs    — idle threshold
   * @param {number} config.pollBase        — base poll delay (ms)
   * @param {number} config.pollMax         — max poll delay (ms)
   * @param {Element} config.connectionDotEl — DOM element for live dot
   * @param {Element} [config.chatBodyEl]   — chat body DOM element
   * @param {Object}  [config.renderFns]    — render function deps from HTML IIFE
   * @param {Function} config.renderFns.debouncedRenderContacts
   * @param {Function} config.renderFns.nearBottom
   * @param {Function} config.renderFns.scrollBottom
   * @param {Function} config.renderFns.upsertMsg
   * @param {Function} config.renderFns.beep
   * @param {Function} config.renderFns.updateContactGreenDot
   */
  const init = (config) => {
    if (window.__pollingEngineInitialized) return;

    if (!config || !config.apiUrl) {
      console.warn('[POLLING] init() requires apiUrl — skipping');
      return;
    }

    API_URL                 = config.apiUrl;
    HEARTBEAT_ACTIVE_MS     = config.heartbeatActiveMs     || 1000;
    HEARTBEAT_BACKGROUND_MS = config.heartbeatBackgroundMs || 5000;
    HEARTBEAT_IDLE_MS       = config.heartbeatIdleMs       || 10000;
    HEARTBEAT_IDLE_AFTER_MS = config.heartbeatIdleAfterMs  || 5 * 60 * 1000;
    POLL_BASE_VALUE         = config.pollBase || POLL_BASE;
    POLL_MAX_VALUE          = config.pollMax  || POLL_MAX;
    connectionDot           = config.connectionDotEl || null;
    chatBody                = config.chatBodyEl || null;

    // Wire render burst throttle guard
    lastRenderTimestamp = 0;

    // Register visibility change handler for recovery
    document.addEventListener('visibilitychange', handleVisibilityChange);
    console.debug('[LIFECYCLE] visibilitychange recovery handler registered');

    // Wire up render function deps from the HTML IIFE
    if (config.renderFns) {
      debouncedRenderContacts = config.renderFns.debouncedRenderContacts || null;
      nearBottom  = config.renderFns.nearBottom  || null;
      scrollBottom = config.renderFns.scrollBottom || null;
      upsertMsg   = config.renderFns.upsertMsg   || null;
      beep        = config.renderFns.beep        || null;
      updateContactGreenDot = config.renderFns.updateContactGreenDot || null;
      refreshSidebar = config.renderFns.refreshSidebar || null;
      refreshActiveChat = config.renderFns.refreshActiveChat || null;
    }

    window.__pollingEngineInitialized = true;

    console.debug('[POLLING] Engine initialized — API:', API_URL,
      'heartbeat:', HEARTBEAT_ACTIVE_MS, '/', HEARTBEAT_BACKGROUND_MS, '/', HEARTBEAT_IDLE_MS);
    console.debug('[MODULE]', 'polling.js loaded — engine initialized');

    /* ── TODO(websocket): Replace visibilitychange handler with
       WS reconnect lifecycle event when migrating to WebSocket transport.
       The current polling-based visibilitychange recovery forces a full
       poll cycle. In a WS architecture, reconnect would trigger a
       resync from the last known cursor instead. */
  };

  /* ═══════════════════════════════════════════════════════════
     EXPORTS
     ═══════════════════════════════════════════════════════════ */

  window.pollingEngine = {
    init,
    poll,
    pollMessages,
    pollSidebar,
    schedulePoll,
    heartbeatTick,
    startHeartbeat,
    stopHeartbeat,
    setPoll,
    getHeartbeatMs,
    updateActivity,

    // Expose for debugging / testing
    get activeMessageRequestToken()  { return activeMessageRequestToken; },
    get isHeartbeatTickRunning()      { return isHeartbeatTickRunning; },
    get currentHeartbeatMs()         { return currentHeartbeatMs; },

    // AbortControllers (for external abort on chat switch)
    get sidebarPollController()     { return sidebarPollController; },
    get messagesPollController()    { return messagesPollController; },

    /** Abort the current message poll and create a new controller */
    abortMessagesPoll() {
      messagesPollController?.abort();
      messagesPollController = null;
      ++activeMessageRequestToken;
      console.debug('[POLLING] Messages poll aborted — token bumped to', activeMessageRequestToken);
    },

    /** Abort the current sidebar poll and create a new controller */
    abortSidebarPoll() {
      sidebarPollController?.abort();
      sidebarPollController = null;
      ++activeSidebarRequestToken;
      console.debug('[POLLING] Sidebar poll aborted — token bumped to', activeSidebarRequestToken);
    },

    /** Cancel the scheduled poll timer */
    clearPollTimer() {
      clearTimeout(window.inboxState.pollTimer);
      window.inboxState.pollTimer = null;
      console.debug('[POLLING] Poll timer cleared');
    },

    /** Full cleanup */
    destroy() {
      stopHeartbeat();
      this.abortMessagesPoll();
      this.abortSidebarPoll();
      this.clearPollTimer();
      document.removeEventListener('visibilitychange', handleVisibilityChange);
      window.__pollingEngineInitialized = false;
      console.debug('[POLLING] Engine destroyed');
    }
  };

})();
