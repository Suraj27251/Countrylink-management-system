# Implementation Plan: Interactive Tracking Page

## Overview

This plan implements the `track.html` standalone page for CountryLinks — a single HTML file with inline CSS/JS featuring a 3D robot mascot with eye-tracking, GSAP animations, glassmorphism UI, canvas particle system, and a tracking form with validation. Implementation uses vanilla JavaScript with GSAP from CDN and fast-check for property-based testing.

## Tasks

- [x] 1. Set up page structure, styles, and SVG mascot
  - [x] 1.1 Create track.html with HTML structure, meta tags, semantic elements, and inline CSS
    - Create the `track.html` file with DOCTYPE, head (charset, viewport, description meta tags), and semantic body structure (header, main, footer)
    - Implement all inline CSS: CSS custom properties for the color palette (dark blue #0a0e27–#1a1f3a, cyan #00d4ff–#00f5ff, white), gradient background, glassmorphism panel styles (backdrop-filter blur ≥10px, background opacity 0.1–0.3, border opacity 0.1–0.3), responsive breakpoints (320px–1920px), minimum font size 14px, and CSS fallback animations (@keyframes blink, hover transitions, focus glow)
    - Include backdrop-filter fallback class (.no-backdrop-filter with solid dark blue background opacity 0.85–0.95)
    - Add responsive rules: stacked layout below 768px, mascot min-height 120px desktop / 80px mobile, touch targets ≥44px on mobile
    - _Requirements: 4.1, 4.2, 4.3, 4.6, 7.1, 7.2, 7.3, 7.4, 7.6, 8.1, 8.4, 8.5, 8.7_

  - [x] 1.2 Create inline SVG robot mascot with eye sockets and pupils
    - Design and implement the inline SVG mascot with: rounded body, expressive eye sockets with movable pupil elements (using `<circle>` or `<ellipse>` with IDs for JS targeting), antenna with signal wave decorations, 3D-style appearance using gradients/shadows
    - Add `aria-hidden="true"` to the mascot SVG container
    - Ensure vector rendering is crisp at all viewport sizes
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 8.3, 9.4_

  - [x] 1.3 Create tracking form HTML with accessibility attributes
    - Implement the form with: labeled text input (id="tracking-input", maxlength="30", placeholder="Enter Tracking ID"), associated `<label>` element, submit button labeled "Track Status", `aria-live="assertive"` message region (id="form-messages")
    - Add CountryLinks logo above the form (min-height 40px mobile, 60px desktop)
    - Add subtitle text below the form: "Track your broadband installation and service requests in real time"
    - Ensure keyboard navigation (Tab sequencing, Enter submission, visible focus indicators ≥2px outline)
    - _Requirements: 4.4, 4.5, 5.1, 5.2, 5.3, 5.4, 9.1, 9.2, 9.3, 9.6_

- [x] 2. Implement JavaScript modules — Initialization, Eye Tracker, and Animation Engine
  - [x] 2.1 Implement Initialization module and Animation Engine with GSAP detection and CSS fallback
    - Write the Initialization module: feature detection for backdrop-filter (apply fallback class if unsupported), GSAP load check (`window.gsap`), `prefers-reduced-motion` media query detection, module bootstrapping sequence
    - Write the Animation Engine module: `gsapAvailable` flag, `reducedMotion` flag, `animate()` method that uses GSAP when available or CSS transitions as fallback, `createHoverEffect()` for button lift (-2px to -4px translateY + box-shadow), `createFocusEffect()` for input glow (cyan border glow), `onReducedMotionChange()` handler
    - Add GSAP CDN script tag (gsap 3.12.x from cdnjs) with onerror fallback
    - _Requirements: 4.6, 6.1, 6.2, 6.5, 8.2, 8.6, 9.5_

  - [x] 2.2 Implement Eye Tracker module with cursor following, idle detection, and blink
    - Write the EyeTracker module with: `init()` to attach mousemove/mouseleave/focus/keydown listeners, `calculateDisplacement()` constraining to 30% of socket radius, `applyProximityScaling()` (10% at viewport edge to 100% adjacent to mascot), `animateEyes()` with easing (min 200ms), `lookAtInput()` on focus, `resetToCenter()` on mouseleave, `blink()` animation (150–300ms), `startIdleLoop()` after 3s idle (blink every 3–5s), `onKeystroke()` with head movement ≤5px
    - Use GSAP for eye animation when available, direct style manipulation as fallback
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7_

  - [x] 2.3 Write property test for eye displacement bounds (Property 1)
    - **Property 1: Eye displacement is bounded and proportionally scaled**
    - Generate random cursor positions (x, y) within viewport bounds and random viewport/mascot positions
    - Verify displacement magnitude ≤ 0.3 × socketRadius
    - Verify proportional scaling (10% at edge, 100% adjacent)
    - Use fast-check with minimum 100 iterations
    - **Validates: Requirements 1.1, 1.5**

  - [x] 2.4 Write property test for head movement bounds (Property 2)
    - **Property 2: Head movement on keystroke is bounded**
    - Generate random keystroke events
    - Verify resulting head displacement ≤ 5px in both x and y directions
    - Use fast-check with minimum 100 iterations
    - **Validates: Requirements 1.3**

- [x] 3. Implement Particle System and parallax
  - [x] 3.1 Implement Particle System module with canvas rendering and frame rate management
    - Write the ParticleSystem module: `init()` to set up canvas element (full viewport, behind content via z-index), create 30–60 particles on desktop / 15–30 on mobile, `animate()` loop using requestAnimationFrame, `updateParticles()` for position/opacity updates, frame rate monitoring (reduce particles by 50% if <30fps), `pause()`/`resume()`/`destroy()` methods
    - Particles: cyan spectrum colors, varying sizes and speeds, opacity 0.1–0.8, wrap around edges
    - Respect `prefers-reduced-motion` (disable particles entirely)
    - Ensure canvas uses CSS transforms only, no layout recalculations
    - _Requirements: 3.1, 3.2, 3.4, 3.5, 6.4, 6.5, 7.5_

  - [x] 3.2 Implement parallax effect on background elements
    - Add `applyParallax()` to ParticleSystem: calculate offset based on cursor position relative to viewport center, cap displacement at 20px maximum, apply to particle base positions
    - Integrate with mousemove event from EyeTracker (shared cursor position)
    - Disable parallax when `prefers-reduced-motion` is active
    - _Requirements: 3.3, 6.5, 9.5_

  - [x] 3.3 Write property test for parallax displacement bounds (Property 3)
    - **Property 3: Parallax displacement is bounded**
    - Generate random cursor positions across full viewport range
    - Verify parallax offset magnitude ≤ 20px for all particles
    - Use fast-check with minimum 100 iterations
    - **Validates: Requirements 3.3**

- [x] 4. Checkpoint - Verify visual modules
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Implement Form Handler with validation and API submission
  - [x] 5.1 Implement Form Handler module with validation, submission, and error handling
    - Write the FormHandler module: `init()` to attach submit/focus/blur listeners, `validate()` using pattern `/^[a-zA-Z0-9-]{1,30}$/` (return `{valid, message}`), `submit()` using fetch to `/api/track?id={trackingId}` with 15s AbortController timeout, `showLoading()` (disable button, show spinner), `hideLoading()` (re-enable button), `showMessage()` (update aria-live region with error/success/info), `clearMessages()`
    - Handle all error scenarios: empty input, invalid characters, network failure, timeout, server errors (4xx/5xx), invalid JSON
    - Display validation errors inline with ARIA announcements
    - On focus: trigger EyeTracker.lookAtInput() and apply glow effect
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 9.7_

  - [x] 5.2 Write property test for tracking ID validation (Property 4)
    - **Property 4: Tracking ID validation is correct**
    - Generate random valid strings (alphanumeric + hyphens, 1–30 chars) → verify accepted
    - Generate random invalid strings (special chars, empty, >30 chars) → verify rejected
    - Verify: `isValid(s) === /^[a-zA-Z0-9-]{1,30}$/.test(s)` for all strings
    - Use fast-check with minimum 100 iterations
    - **Validates: Requirements 5.6, 5.7**

  - [x] 5.3 Write unit tests for form handler edge cases
    - Test empty input validation message
    - Test boundary lengths (1 char, 30 chars, 31 chars)
    - Test loading state toggle (button disabled/enabled)
    - Test error display in ARIA live region
    - Test timeout handling (15s)
    - _Requirements: 5.5, 5.8, 5.9, 9.7_

- [x] 6. Implement Accessibility module and reduced-motion handling
  - [x] 6.1 Implement Accessibility module with reduced-motion, focus management, and ARIA updates
    - Write the Accessibility module: listen for `prefers-reduced-motion` changes via `matchMedia`, disable/enable non-essential animations (particles, parallax, idle mascot animations, hover lift), preserve essential functionality (form feedback, focus indicators)
    - Ensure all interactive elements have visible focus indicators (≥2px outline)
    - Verify color contrast ≥4.5:1 for all text (enforce via CSS custom properties)
    - Ensure keyboard navigation: Tab through input → button, Enter submits, Escape closes messages
    - _Requirements: 6.5, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7_

- [x] 7. Integration, wiring, and final polish
  - [x] 7.1 Wire all modules together in initialization sequence and verify end-to-end flow
    - Connect Initialization → AnimationEngine → EyeTracker → ParticleSystem → FormHandler → Accessibility in correct boot order
    - Ensure shared cursor position flows from mousemove to both EyeTracker and ParticleSystem parallax
    - Ensure FormHandler focus triggers EyeTracker.lookAtInput()
    - Ensure GSAP fallback path works end-to-end (remove GSAP script to test)
    - Verify page loads and reaches interactive state within 3s on simulated 4G
    - _Requirements: 6.3, 8.1, 8.6_

  - [x] 7.2 Write property test for no horizontal overflow (Property 5)
    - **Property 5: No horizontal overflow at any supported viewport width**
    - Generate random viewport widths between 320 and 1920
    - Verify document scrollWidth ≤ viewport width (requires browser-based test via Playwright)
    - Use fast-check with minimum 100 iterations
    - **Validates: Requirements 7.1**

  - [x] 7.3 Write property test for minimum font size (Property 6)
    - **Property 6: Minimum font size across all viewport widths**
    - Generate random viewport widths between 320 and 1920
    - Verify all text elements have computed font-size ≥ 14px (requires browser-based test via Playwright)
    - Use fast-check with minimum 100 iterations
    - **Validates: Requirements 7.6**

  - [x] 7.4 Write integration tests for accessibility and cross-browser fallbacks
    - Test ARIA live region announces validation errors
    - Test keyboard-only navigation flow (Tab, Enter, Escape)
    - Test backdrop-filter fallback renders solid background
    - Test GSAP CDN failure triggers CSS-only animations
    - Test `prefers-reduced-motion` disables non-essential animations
    - _Requirements: 8.6, 4.6, 9.3, 9.5, 9.7_

- [x] 8. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document using fast-check
- Unit tests validate specific examples and edge cases
- Browser-based property tests (Properties 5 and 6) require Playwright for viewport manipulation
- The single-file constraint means all code goes into `track.html` — test files are separate
- GSAP CDN fallback must be tested by simulating script load failure

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "1.3"] },
    { "id": 2, "tasks": ["2.1", "3.1"] },
    { "id": 3, "tasks": ["2.2", "3.2", "5.1"] },
    { "id": 4, "tasks": ["2.3", "2.4", "3.3", "5.2", "5.3", "6.1"] },
    { "id": 5, "tasks": ["7.1"] },
    { "id": 6, "tasks": ["7.2", "7.3", "7.4"] }
  ]
}
```
