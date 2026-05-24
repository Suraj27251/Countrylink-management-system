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
      notificationAudio: null,
    };
  }

  /* ── Initialize ─────────────────────────────────────────── */
  window.inboxState = createInitialState();
  window.__inboxStateInitialized = true;

  console.debug('[MODULE]', 'state.js loaded');
  console.debug('[STATE] Initialized:', {
    activeMobile: window.inboxState.activeMobile,
    cursors: window.inboxState.cursors,
    ui: window.inboxState.ui,
  });

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
