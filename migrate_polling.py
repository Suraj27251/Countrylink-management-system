#!/usr/bin/env python3
"""
Phase 3 migration: Extract polling code into polling.js, clean up whatsapp.html.

Changes:
1. Add polling.js script tag after state.js
2. Resolve ALL 15 merge conflicts (keep new code)
3. Remove old polling functions (replaced by window.pollingEngine)
4. Fix references: ACTIVE_MOBILE -> window.inboxState.activeMobile
5. Fix poll calls -> pollingEngine calls
6. Add init/config calls
"""

import re

HTML_PATH = 'templates/whatsapp.html'

def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def write_file(path, content):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

def resolve_merge_conflicts(content):
    """
    Resolve ALL git merge conflict markers.
    Keep the NEW code (between ======= and >>>>>>>).
    Remove HEAD code and conflict markers.
    """
    lines = content.split('\n')
    result = []
    i = 0
    conflicts_resolved = 0

    while i < len(lines):
        line = lines[i]

        # Check for <<<<<<< HEAD
        if line.strip().startswith('<<<<<<< HEAD'):
            conflicts_resolved += 1
            i += 1
            # Skip lines until we hit =======
            while i < len(lines) and not lines[i].strip().startswith('======='):
                i += 1
            if i < len(lines):
                i += 1  # Skip the ======= line
            # Now keep lines until we hit >>>>>>>
            while i < len(lines) and not lines[i].strip().startswith('>>>>>>>'):
                result.append(lines[i])
                i += 1
            if i < len(lines):
                i += 1  # Skip the >>>>>>> line
        else:
            result.append(line)
            i += 1

    print(f"  [OK] Resolved {conflicts_resolved} merge conflicts")
    return '\n'.join(result)

def add_polling_script_tag(content):
    """Add polling.js script tag after state.js script tag."""
    target = '<script src="{{ url_for(\'static\', filename=\'js/whatsapp/state.js\') }}"></script>'
    replacement = f'{target}\n<script src="{{{{ url_for(\'static\', filename=\'js/whatsapp/polling.js\') }}}}"></script>'
    
    if target in content:
        content = content.replace(target, replacement)
        print("  [OK] Added polling.js script tag")
    else:
        print("  [WARN] Could not find state.js script tag - might need manual add")
    return content

def remove_old_polling_functions(content):
    """Remove old polling function definitions replaced by polling.js."""
    replacements = [
        # Remove setPoll function (moved to polling.js)
        ('const setPoll = (live, label) => {\n  const dot = connectionDot;\n  if (!dot) return;\n\n  dot.classList.remove(\'live\', \'degraded\', \'offline\');\n  dot.classList.add(live ? \'live\' : \'offline\');\n  dot.title = live ? \'LIVE\' : (label || \'Reconnecting...\');\n};',
         '/* setPoll moved to polling.js */'),

        # Remove getHeartbeatMs function (moved to polling.js)
        ('const getHeartbeatMs = () => {\n  if (document.hidden) return HEARTBEAT_BACKGROUND_MS;\n  if (Date.now() - lastOperatorActivityAt > HEARTBEAT_IDLE_AFTER_MS) return HEARTBEAT_IDLE_MS;\n  return HEARTBEAT_ACTIVE_MS;\n};',
         '/* getHeartbeatMs moved to polling.js */'),

        # Remove heartbeatTick function (moved to polling.js)
        ('const heartbeatTick = async () => {\n  if (isHeartbeatTickRunning) return;\n  isHeartbeatTickRunning = true;\n  try {\n    await pollSidebar();\n    if (ACTIVE_MOBILE) await pollMessages();\n    setPoll(true, \'LIVE\');\n  } catch (e) {\n    if (e?.name === \'AbortError\') return;\n    console.warn(\'Heartbeat poll failed:\', e);\n    setPoll(false, \'Reconnecting...\');\n  } finally {\n    isHeartbeatTickRunning = false;\n  }\n};',
         '/* heartbeatTick moved to polling.js */'),

        # Remove startHeartbeat function (moved to polling.js)
        ('const startHeartbeat = () => {\n  const nextMs = getHeartbeatMs();\n  if (heartbeatTimer && currentHeartbeatMs === nextMs) return;\n  if (heartbeatTimer) clearInterval(heartbeatTimer);\n  currentHeartbeatMs = nextMs;\n  heartbeatTimer = setInterval(heartbeatTick, currentHeartbeatMs);\n};',
         '/* startHeartbeat moved to polling.js */'),

        # Remove lastOperatorActivityAt declaration (moved to polling.js)
        ('let lastOperatorActivityAt = Date.now();',
         '/* lastOperatorActivityAt in polling.js */'),
        
        # Remove isHeartbeatTickRunning declaration (moved to polling.js)
        ('let isHeartbeatTickRunning = false;',
         '/* isHeartbeatTickRunning in polling.js */'),

        # Remove currentHeartbeatMs declaration (moved to polling.js)
        ('let currentHeartbeatMs = HEARTBEAT_ACTIVE_MS;',
         '/* currentHeartbeatMs in polling.js */'),

        # Remove sidebarPollController and messagesPollController declarations (moved to polling.js)
        ('let sidebarPollController = null;\nlet messagesPollController = null;',
         '/* polling controllers moved to polling.js */'),

        # Remove activeMessageRequestToken declaration (moved to polling.js)
        ('let activeMessageRequestToken = 0;',
         '/* activeMessageRequestToken in polling.js */'),

        # Remove heartbeatTimer declaration (moved to polling.js)
        ('let heartbeatTimer = null;',
         '/* heartbeatTimer in polling.js */'),
    ]

    changes = 0
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            changes += 1

    print(f"  [OK] Removed {changes} old polling function declarations")
    return content

def fix_globals_to_inbox_state(content):
    """
    Replace old global references with window.inboxState equivalents.
    Be careful not to replace things already qualified.
    """
    # Use careful replacement with word boundaries
    # Replace ACTIVE_MOBILE (as whole word) with window.inboxState.activeMobile
    content = re.sub(
        r'(?<![.\w])ACTIVE_MOBILE(?![.\w])',
        'window.inboxState.activeMobile',
        content
    )
    
    # Replace seenInboxIds (as whole word) with window.inboxState.globalKnownMessageIds
    # (may not exist after merge conflict resolution)
    content = re.sub(
        r'(?<![.\w])seenInboxIds(?![.\w])',
        'window.inboxState.globalKnownMessageIds',
        content
    )
    
    # Replace lastInboxId (as whole word, standalone)
    content = re.sub(
        r'(?<![.\w])lastInboxId(?![.\w])',
        'window.inboxState.cursors.globalLastInboxId',
        content
    )
    
    print("  [OK] Fixed global references to inboxState")
    return content

def fix_poll_init_calls(content):
    """
    Replace old polling initialization with pollingEngine calls.
    """
    # Replace the main initialization sequence
    old_init = """startHeartbeat();
document.addEventListener('visibilitychange', startHeartbeat);
setInterval(startHeartbeat, 15000);"""
    
    new_init = """// Polling lifecycle managed by pollingEngine
document.addEventListener('visibilitychange', window.pollingEngine.startHeartbeat);
setInterval(window.pollingEngine.startHeartbeat, 15000);
window.pollingEngine.startHeartbeat();"""
    
    if old_init in content:
        content = content.replace(old_init, new_init)
        print("  [OK] Replaced polling init with pollingEngine calls")
    else:
        # Try alternate pattern
        old_init2 = """startHeartbeat();
  document.addEventListener('visibilitychange', startHeartbeat);
  setInterval(startHeartbeat, 15000);"""
        if old_init2 in content:
            content = content.replace(old_init2, new_init)
            print("  [OK] Replaced polling init with pollingEngine calls (alt)")
        else:
            print("  [WARN] Could not find standard polling init pattern")
    
    # Replace standalone poll() calls
    content = content.replace('await poll();', 'await window.pollingEngine.poll();')
    
    # Replace lastOperatorActivityAt = Date.now() with pollingEngine.updateActivity()
    content = content.replace(
        'lastOperatorActivityAt = Date.now();',
        'window.pollingEngine.updateActivity();'
    )
    
    # Replace messagesPollController abort patterns
    content = content.replace(
        'messagesPollController?.abort();\n  messagesPollController = null;',
        'window.pollingEngine.abortMessagesPoll();'
    )
    content = content.replace(
        "messagesPollController?.abort();\n    messagesPollController = null;",
        "window.pollingEngine.abortMessagesPoll();"
    )
    
    # Replace sidebarPollController abort patterns
    content = content.replace(
        'sidebarPollController?.abort();\n  sidebarPollController = null;',
        'window.pollingEngine.abortSidebarPoll();'
    )
    
    # Replace setPoll calls
    content = content.replace(
        "setPoll(true, 'Connected');",
        'window.pollingEngine.setPoll(true, "Connected");'
    )
    content = content.replace(
        'setPoll(true, "Connected");',
        'window.pollingEngine.setPoll(true, "Connected");'
    )
    
    print("  [OK] Fixed poll/init calls to use pollingEngine")
    return content

def add_polling_init_in_script(content):
    """Add pollingEngine.init() call before heartbeat start."""
    init_marker = "startHeartbeat();"
    
    init_code = """
  // -- Initialize polling engine --
  window.pollingEngine.init({
    apiUrl: API_URL,
    heartbeatActiveMs: HEARTBEAT_ACTIVE_MS,
    heartbeatBackgroundMs: HEARTBEAT_BACKGROUND_MS,
    heartbeatIdleMs: HEARTBEAT_IDLE_MS,
    heartbeatIdleAfterMs: HEARTBEAT_IDLE_AFTER_MS,
    pollBase: 2000,
    pollMax: 30000,
    connectionDotEl: connectionDot,
    chatBodyEl: chatBody,
  });
"""
    
    if init_marker in content:
        content = content.replace(init_marker, f"{init_code}\n  {init_marker}")
        print("  [OK] Added pollingEngine.init() call")
    else:
        print("  [WARN] Could not find init insertion point")
    
    return content

def validate_result(content):
    """Check for remaining issues."""
    issues = []
    
    # Check for unresolved merge conflicts
    head_count = content.count('<<<<<<< HEAD')
    if head_count > 0:
        issues.append(f"{head_count} unresolved <<<<<<< HEAD markers")
    
    # Check for remaining standalone ACTIVE_MOBILE
    active_count = len(re.findall(r'(?<![.\w])ACTIVE_MOBILE(?![.\w])', content))
    if active_count > 0:
        issues.append(f"{active_count} standalone ACTIVE_MOBILE references remaining")
    
    # Check for remaining seenInboxIds
    seen_count = len(re.findall(r'(?<![.\w])seenInboxIds(?![.\w])', content))
    if seen_count > 0:
        issues.append(f"{seen_count} standalone seenInboxIds references remaining")
    
    return issues


def main():
    print("=" * 60)
    print("PHASE 3: Polling Modularization")
    print("=" * 60)
    
    content = read_file(HTML_PATH)
    original_len = len(content)
    
    print(f"\n== Read {HTML_PATH} ({original_len} chars)")
    
    # Step 1: Add polling.js script tag
    print("\n[1/6] Adding polling.js script tag...")
    content = add_polling_script_tag(content)
    
    # Step 2: Resolve merge conflicts
    print("\n[2/6] Resolving merge conflicts...")
    content = resolve_merge_conflicts(content)
    
    # Step 3: Remove old polling functions
    print("\n[3/6] Removing old polling function declarations...")
    content = remove_old_polling_functions(content)
    
    # Step 4: Fix global references to inboxState
    print("\n[4/6] Fixing global references...")
    content = fix_globals_to_inbox_state(content)
    
    # Step 5: Fix poll/init calls to use pollingEngine
    print("\n[5/6] Fixing poll/init calls...")
    content = fix_poll_init_calls(content)
    content = add_polling_init_in_script(content)
    
    # Step 6: Validate
    print("\n[6/6] Validating...")
    issues = validate_result(content)
    if issues:
        print(f"\n[WARN] Migration completed with {len(issues)} potential issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("  [OK] No issues found!")
    
    new_len = len(content)
    print(f"\nSize: {original_len} -> {new_len} chars ({new_len - original_len:+d})")
    
    # Write back
    write_file(HTML_PATH, content)
    print(f"\n[DONE] Written back to {HTML_PATH}")
    

if __name__ == '__main__':
    main()
