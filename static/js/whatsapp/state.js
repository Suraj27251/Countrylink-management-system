/**
 * whatsapp/state.js — Centralized runtime state for WhatsApp inbox.
 *
 * PURPOSE:
 *   Isolate ALL mutable frontend state into ONE object to eliminate
 *   race conditions, stale reads, and shared mutable globals.
 *
 * RULES:
 *   - Initialize ONLY ONCE via initInboxState().
 *   - Read via window.inboxState.<key>.
 *   - Mutate via helpers defined below (never directly except in hot paths).
 *   - All mutations must be guarded against stale async responses.
 *
 * @see Phase 2 of WhatsApp inbox modularization.
 */

(function () {
  'use strict';

  /* ── Guard: prevent double-initialization ──────────────── */
  if (window.__inboxStateInitialized) {
    console.debug('[STATE] Already initialized — skipping.');
    return;
  }

  /* ── Stress-hardening constants ──────────────────────── */
  const MAX_RENDER_QUEUE = 3;          // max queued render frames before coalescing
  const LARGE_SIDEBAR_WARN = 150;       // sidebar contacts threshold for [DOM_WARN]
  const STALE_RENDER_TIMEOUT_MS = 5000; // max ms a render can run before stuck detection

  /* ── State factory ─────────────────────────────────────── */
  function createInitialState() {
    return {
      /* ── Active conversation ──────────────────────────── */
      activeMobile: null,

      /* ── Polling ──────────────────────────────────────── */
      pollTimer: null,
      pollFails: 0,

      /* ── Request tokens (stale response guard) ────────── */
      activeMessageRequestToken: 0,
      activeSidebarRequestToken: 0,

      /* ── Message tracking ─────────────────────────────── */
      /** Global list of rendered message IDs for dedup across chats */
      globalKnownMessageIds: new Set(),
      /** Per-conversation unread counts */
      unreadByMobile: new Map(),
      /** Per-conversation last rendered message ID */
      lastMessageIdByMobile: new Map(),
      /** Per-conversation last SEEN message ID (for notification dedup) */
      lastSeenMessageIdByMobile: new Map(),
      /** Optimistic message IDs pending reconciliation from polling */
      optimisticMessageIds: new Set(),

      /* ── Loading states ───────────────────────────────── */
      loadingStates: {
        sidebar: false,
        messages: false,
      },

      /* ── UI state ─────────────────────────────────────── */
      ui: {
        isWindowFocused: true,
        isLoadingConversation: false,
        lastInteractionAt: Date.now(),
        hasInteracted: false,
        soundEnabled: localStorage.getItem('wa_sound') !== '0',
        conversationOpenedAt: Date.now(),
      },

      /* ── Contacts (centralized source of truth) ──────── */
      contacts: [],

      /* ── Cursors (global offsets for polling) ─────────── */
      cursors: {
        globalLastMessageId: 0,
        globalLastInboxId: 0,
      },

      /* ── Debounced render ─────────────────────────────── */
      renderContactsTimeout: null,
      pendingRenderContacts: null,

      /* ── Template / Flow data ─────────────────────────── */
      templates: [],
      selectedQuickTemplate: null,
      flows: [],
      iType: 'button',

      /* ── Chat window status per mobile (24h window) ──── */
      chatWindowStatusByMobile: new Map(),

      /* ── AbortControllers (for future use) ────────────── */
      messagesAbortController: null,
      sidebarAbortController: null,

      /* ── Notification audio ───────────────────────────── */
      audio: {
        soft: null,
        newConversation: null,
      },
      lastPollSuccessAt: Date.now(),

      /* ── Runtime tracking (for memory/leak observation) ── */
      activeTimerIds: [],        // setTimeout/setInterval IDs for cleanup auditing
      activeAbortControllers: [],  // AbortController instances for leak detection
      listenerCount: 0,          // approximate active event listener count

      /* ── Stress metrics (tracked across sessions) ─────── */
      stress: {
        peakRenderQueueDepth: 0,
        maxSidebarRenderMs: 0,
        maxActiveChatRenderMs: 0,
        recoveryCount: 0,
        staleRenderDrops: 0,
        queueCoalesceCount: 0,
        stalePollDrops: 0,
      },
    };
  }

  /* ── Lifecycle helpers ────────────────────────────────── */
  window.clearTimerSafe = function clearTimerSafe(timerId, label) {
    if (timerId != null) {
      clearTimeout(timerId);
      const idx = (window.inboxState.activeTimerIds || []).indexOf(timerId);
      if (idx !== -1) window.inboxState.activeTimerIds.splice(idx, 1);
      console.debug('[EVENT_CLEANUP] Timer cleared:', label || timerId);
    }
  };

  window.clearIntervalSafe = function clearIntervalSafe(intervalId, label) {
    if (intervalId != null) {
      clearInterval(intervalId);
      const idx = (window.inboxState.activeTimerIds || []).indexOf(intervalId);
      if (idx !== -1) window.inboxState.activeTimerIds.splice(idx, 1);
      console.debug('[EVENT_CLEANUP] Interval cleared:', label || intervalId);
    }
  };

  window.abortControllerSafe = function abortControllerSafe(controller, label) {
    if (controller && typeof controller.abort === 'function') {
      try { controller.abort(); } catch (_) { /* ignore */ }
      const idx = (window.inboxState.activeAbortControllers || []).indexOf(controller);
      if (idx !== -1) window.inboxState.activeAbortControllers.splice(idx, 1);
      console.debug('[EVENT_CLEANUP] AbortController aborted:', label || 'unnamed');
    }
  };

  window.cleanupListenerSafe = function cleanupListenerSafe(target, event, handler, options, label) {
    if (target && typeof target.removeEventListener === 'function') {
      target.removeEventListener(event, handler, options);
      if (typeof handler === 'function') {
        console.debug('[EVENT_CLEANUP] Listener removed:', event, label || '');
      }
    }
  };

  /* ── Initialize ─────────────────────────────────────────── */
  window.inboxState = createInitialState();
  window.__inboxStateInitialized = true;

  /* ── Centralized performance metrics ────────────────── */
  window.debugMetrics = {
    render: {},
    polling: {},
    memory: {
      domNodeCount: 0,
      sidebarContactCount: 0,
      listenerEstimate: 0,
      messageNodeMapSize: 0,
    },
    sync: {},
    queue: {},
  };

  /**
   * Capture a runtime health snapshot — useful for production debugging.
   * @returns {Object} snapshot
   */
  window.debugHealthSnapshot = () => {
    const st = window.inboxState;
    const poll = window.pollingEngine;
    const rend = window.renderEngine;

    return {
      timestamp: Date.now(),
      state: {
        activeMobile: st.activeMobile,
        pollFails: st.pollFails,
        hasInteracted: st.ui.hasInteracted,
        soundEnabled: st.ui.soundEnabled,
        contactsCount: (st.contacts || []).length,
        templatesCount: st.templates.length,
        flowsCount: st.flows.length,
      },
      polling: {
        isHeartbeatRunning: !!st.pollTimer,
        currentHeartbeatMs: poll?.currentHeartbeatMs,
        heartbeatTickRunning: poll?.isHeartbeatTickRunning,
        activeMessageRequestToken: poll?.activeMessageRequestToken,
      },
      render: {
        isRenderInProgress: rend?.isRenderInProgress,
        pendingRenderFrame: rend?.pendingRenderFrame,
        messageNodeMapSize: rend?.messageNodeMap?.size || 0,
      },
      memory: {
        knownIdsSize: st.globalKnownMessageIds?.size || 0,
        unreadByMobileSize: st.unreadByMobile?.size || 0,
        lastSeenSize: st.lastSeenMessageIdByMobile?.size || 0,
        chatWindowStatusSize: st.chatWindowStatusByMobile?.size || 0,
        activeTimers: st.activeTimerIds?.length || 0,
        activeAbortControllers: st.activeAbortControllers?.length || 0,
      },
      metrics: { ...window.debugMetrics },
      stress: st.stress ? { ...st.stress } : {},
    };
  };

  console.debug('[MODULE]', 'state.js loaded');
  console.debug('[STATE] Initialized:', {
    activeMobile: window.inboxState.activeMobile,
    cursors: window.inboxState.cursors,
    ui: window.inboxState.ui,
  });
  console.debug('[LIFECYCLE] debugMetrics + debugHealthSnapshot ready');

  /* ── Safe accessor with fallback ────────────────────────── */
  window.inboxState.safe = function safe(key) {
    const keys = key.split('.');
    let val = window.inboxState;
    for (const k of keys) {
      if (val == null) return undefined;
      val = val[k];
    }
    return val;
  };

  /* ── Log helpers ────────────────────────────────────────── */
  window.inboxState.log = function log(...args) {
    console.debug('[STATE]', ...args);
  };

  window.inboxState.logActiveChat = function logActiveChat(msg) {
    console.debug('[ACTIVE CHAT]', msg, { activeMobile: window.inboxState.activeMobile });
  };

  window.inboxState.logToken = function logToken(area, token) {
    console.debug('[TOKEN]', area, token);
  };

  window.inboxState.logCursor = function logCursor(label) {
    console.debug('[CURSOR]', label, {
      globalLastMessageId: window.inboxState.cursors.globalLastMessageId,
      globalLastInboxId: window.inboxState.cursors.globalLastInboxId,
    });
  };

  window.inboxState.logPollStatus = function logPollStatus(status, detail) {
    console.debug('[POLL STATUS]', status, detail);
  };

  /* ── Stale response guard generator ─────────────────────── */
  /**
   * Returns a function that calls `fn` ONLY if the state's requestToken
   * still matches `expectedToken` — otherwise it's a no-op.
   *
   * Usage:
   *   const token = ++window.inboxState.activeMessageRequestToken;
   *   const guarded = window.inboxState.guard(token, (data) => { ... });
   *   // later: guarded(data);
   */
  window.inboxState.guard = function guard(expectedToken, fn) {
    return function guardedFn(...args) {
      if (window.inboxState.activeMessageRequestToken !== expectedToken ||
          window.inboxState.activeSidebarRequestToken !== expectedToken) {
        console.debug('[TOKEN] Stale response discarded (expected:', expectedToken,
          'msgToken:', window.inboxState.activeMessageRequestToken,
          'sidebarToken:', window.inboxState.activeSidebarRequestToken, ')');
        return;
      }
      return fn(...args);
    };
  };

  /* ── State mutation helpers ─────────────────────────────── */

  /** Set active mobile and log the change */
  window.inboxState.setActiveMobile = function setActiveMobile(mobile) {
    const prev = window.inboxState.activeMobile;
    window.inboxState.activeMobile = mobile;
    window.inboxState.logActiveChat(`Changed: ${prev || '(none)'} → ${mobile || '(none)'}`);

    // Also clear per-conversation tracking for the NEW chat so we start fresh
    if (mobile) {
      // Don't full-clear knownIds — just mark conversationOpenedAt
      window.inboxState.ui.conversationOpenedAt = Date.now();
    }
  };

  /** Reset per-conversation state when switching chats */
  window.inboxState.resetChatState = function resetChatState(mobile) {
    window.inboxState.globalKnownMessageIds.clear();
    window.inboxState.lastMessageIdByMobile.delete(mobile);
    window.inboxState.ui.isLoadingConversation = true;
    window.inboxState.ui.conversationOpenedAt = Date.now();
    window.inboxState.log('[STATE] Chat state reset for:', mobile);
  };

  /** Bump message cursor */
  window.inboxState.updateCursors = function updateCursors(msgId, inboxId) {
    if (msgId) {
      window.inboxState.cursors.globalLastMessageId = Math.max(
        window.inboxState.cursors.globalLastMessageId, +msgId || 0
      );
    }
    if (inboxId) {
      window.inboxState.cursors.globalLastInboxId = Math.max(
        window.inboxState.cursors.globalLastInboxId, +inboxId || 0
      );
    }
  };

  /** Track seen message per mobile for notification dedup */
  window.inboxState.updateLastSeen = function updateLastSeen(mobile, msgId) {
    const current = window.inboxState.lastSeenMessageIdByMobile.get(mobile) || 0;
    const newMax = Math.max(current, +msgId || 0);
    window.inboxState.lastSeenMessageIdByMobile.set(mobile, newMax);
  };

  /** Increment unread count for a mobile (only if not active chat) */
  window.inboxState.incrementUnread = function incrementUnread(mobile, count) {
    count = count || 1;
    const current = window.inboxState.unreadByMobile.get(mobile) || 0;
    window.inboxState.unreadByMobile.set(mobile, current + count);
  };

  /** Clear unread for a mobile */
  window.inboxState.clearUnread = function clearUnread(mobile) {
    window.inboxState.unreadByMobile.delete(mobile);
  };

  /** Safely increment poll failures */
  window.inboxState.incrementPollFails = function incrementPollFails() {
    window.inboxState.pollFails++;
    window.inboxState.logPollStatus('FAIL', window.inboxState.pollFails);
  };

  /** Reset poll failures */
  window.inboxState.resetPollFails = function resetPollFails() {
    window.inboxState.pollFails = 0;
    window.inboxState.logPollStatus('OK', 'Connected');
  };

  /* ── Cleanup — not called normally, available for testing ── */
  /* ── Pruning helpers (long-session memory protection) ── */

  /**
   * Safely prune knownIds Set if it exceeds a threshold.
   * Retains the most recent N entries (FIFO eviction).
   */
  window.inboxState.pruneKnownIdsSafe = function pruneKnownIdsSafe(maxSize) {
    maxSize = maxSize || 5000;
    const ids = window.inboxState.globalKnownMessageIds;
    if (!ids || ids.size <= maxSize) return;
    const entries = Array.from(ids);
    const toDelete = entries.slice(0, entries.length - maxSize);
    toDelete.forEach(function (id) { ids.delete(id); });
    console.debug('[STATE] prunedKnownIdsSafe: removed', toDelete.length, 'entries (kept', maxSize, ')');
  };

  /**
   * Safe prune for messageNodeMap (owned by render.js).
   * Called via window.renderEngine if available.
   */
  window.inboxState.pruneMessageNodeMapSafe = function pruneMessageNodeMapSafe(maxSize) {
    maxSize = maxSize || 2000;
    const rend = window.renderEngine;
    if (!rend || !rend.messageNodeMap || rend.messageNodeMap.size <= maxSize) return;
    const entries = Array.from(rend.messageNodeMap.entries());
    const toDelete = entries.slice(0, entries.length - maxSize);
    toDelete.forEach(function (entry) {
      var key = entry[0];
      if (key != null) rend.messageNodeMap.delete(key);
    });
    const deletedCount = entries.length - maxSize;
    console.debug('[STATE] pruneMessageNodeMapSafe: removed', deletedCount, 'entries (kept', maxSize, ')');
  };

  /**
   * Remove orphaned node references from messageNodeMap.
   */
  window.inboxState.cleanupDetachedNodesSafe = function cleanupDetachedNodesSafe() {
    const rend = window.renderEngine;
    if (!rend || !rend.messageNodeMap) return;
    var removed = 0;
    rend.messageNodeMap.forEach(function (node, key) {
      if (node && node.parentNode == null && !document.contains(node)) {
        rend.messageNodeMap.delete(key);
        removed++;
      }
    });
    if (removed > 0) {
      console.debug('[STATE] cleanupDetachedNodesSafe: removed', removed, 'detached nodes');
    }
  };

  window.inboxState.destroy = function destroy() {
    if (window.inboxState.pollTimer) {
      clearTimeout(window.inboxState.pollTimer);
      window.inboxState.pollTimer = null;
    }
    if (window.inboxState.renderContactsTimeout) {
      clearTimeout(window.inboxState.renderContactsTimeout);
      window.inboxState.renderContactsTimeout = null;
    }
    window.__inboxStateInitialized = false;
    console.debug('[STATE] Destroyed');
  };

})();
