// @ts-check
const { test, expect } = require('@playwright/test');
const path = require('path');

/**
 * Integration tests for accessibility and cross-browser fallbacks.
 * Validates: Requirements 8.6, 4.6, 9.3, 9.5, 9.7
 */

const trackPagePath = path.resolve(__dirname, '../../track.html');
const trackPageURL = `file://${trackPagePath.replace(/\\/g, '/')}`;

// ===== Test a: ARIA live region announces validation errors =====
// Validates: Requirement 9.7
test.describe('ARIA live region announces validation errors', () => {
  test('submitting empty form shows error in aria-live region', async ({ page }) => {
    await page.goto(trackPageURL);
    await page.waitForLoadState('domcontentloaded');

    // Verify #form-messages has aria-live="assertive"
    const formMessages = page.locator('#form-messages');
    await expect(formMessages).toHaveAttribute('aria-live', 'assertive');

    // Submit form with empty input
    await page.locator('.btn-track').click();

    // Verify error message appears in the ARIA live region
    await expect(formMessages).toHaveText('Please enter a Tracking ID');
  });

  test('submitting invalid characters shows format error in aria-live region', async ({ page }) => {
    await page.goto(trackPageURL);
    await page.waitForLoadState('domcontentloaded');

    // Type invalid characters
    await page.locator('#tracking-input').fill('abc@#$');

    // Submit form
    await page.locator('.btn-track').click();

    // Verify format error message appears
    const formMessages = page.locator('#form-messages');
    await expect(formMessages).toHaveText('Tracking ID can only contain letters, numbers, and hyphens');
  });
});

// ===== Test b: Keyboard-only navigation flow =====
// Validates: Requirement 9.3
test.describe('Keyboard-only navigation flow', () => {
  test('Tab, Enter, and Escape keyboard navigation works correctly', async ({ page }) => {
    await page.goto(trackPageURL);
    await page.waitForLoadState('domcontentloaded');

    // Press Tab to focus input
    await page.keyboard.press('Tab');
    // The logo link or first focusable element may get focus first,
    // so we tab until we reach the tracking input
    let activeElementId = await page.evaluate(() => document.activeElement?.id);
    // Keep tabbing until we reach the tracking input
    let maxTabs = 5;
    while (activeElementId !== 'tracking-input' && maxTabs > 0) {
      await page.keyboard.press('Tab');
      activeElementId = await page.evaluate(() => document.activeElement?.id);
      maxTabs--;
    }
    expect(activeElementId).toBe('tracking-input');

    // Type a tracking ID
    await page.keyboard.type('TEST-123');

    // Verify input has the value
    await expect(page.locator('#tracking-input')).toHaveValue('TEST-123');

    // Press Tab to focus button
    await page.keyboard.press('Tab');
    const activeTagAfterTab = await page.evaluate(() => document.activeElement?.tagName.toLowerCase());
    expect(activeTagAfterTab).toBe('button');

    // Press Enter to submit — this will trigger a fetch that will fail (no server),
    // but we verify the form submission occurred by checking loading state or error message
    await page.keyboard.press('Enter');

    // Wait for the form to process (either loading state or error message)
    const formMessages = page.locator('#form-messages');
    // Since there's no server, we expect a network error or timeout message
    await expect(formMessages).not.toBeEmpty({ timeout: 5000 });

    // Press Escape to clear messages
    await page.keyboard.press('Escape');
    await expect(formMessages).toBeEmpty();
  });

  test('input has visible focus indicator when focused via keyboard', async ({ page }) => {
    await page.goto(trackPageURL);
    await page.waitForLoadState('domcontentloaded');

    // Focus the input via keyboard
    await page.locator('#tracking-input').focus();

    // Verify the input has a visible focus style (outline or box-shadow)
    const outlineStyle = await page.locator('#tracking-input').evaluate((el) => {
      const styles = window.getComputedStyle(el);
      return {
        outline: styles.outline,
        outlineWidth: styles.outlineWidth,
        boxShadow: styles.boxShadow
      };
    });

    // Either outline or box-shadow should be present for focus indication
    const hasFocusIndicator =
      (outlineStyle.outlineWidth && outlineStyle.outlineWidth !== '0px') ||
      (outlineStyle.boxShadow && outlineStyle.boxShadow !== 'none');
    expect(hasFocusIndicator).toBeTruthy();
  });
});

// ===== Test c: Backdrop-filter fallback renders solid background =====
// Validates: Requirement 4.6
test.describe('Backdrop-filter fallback renders solid background', () => {
  test('no-backdrop-filter class applies solid background when CSS.supports returns false', async ({ page }) => {
    // Override CSS.supports to return false for backdrop-filter
    await page.addInitScript(() => {
      const originalSupports = CSS.supports.bind(CSS);
      CSS.supports = function(prop, value) {
        if (prop === 'backdrop-filter' || prop === '-webkit-backdrop-filter') {
          return false;
        }
        return originalSupports(prop, value);
      };
    });

    await page.goto(trackPageURL);
    await page.waitForLoadState('domcontentloaded');

    // Verify .glass-panel has class 'no-backdrop-filter'
    const glassPanel = page.locator('#glass-panel');
    await expect(glassPanel).toHaveClass(/no-backdrop-filter/);

    // Verify computed background is solid (opacity 0.85-0.95)
    const bgColor = await glassPanel.evaluate((el) => {
      return window.getComputedStyle(el).backgroundColor;
    });

    // The fallback CSS sets: background: rgba(15, 19, 50, 0.9)
    // Parse the rgba value to check opacity
    const rgbaMatch = bgColor.match(/rgba?\((\d+),\s*(\d+),\s*(\d+),?\s*([\d.]+)?\)/);
    expect(rgbaMatch).not.toBeNull();

    if (rgbaMatch) {
      const alpha = parseFloat(rgbaMatch[4] || '1');
      // Opacity should be between 0.85 and 0.95
      expect(alpha).toBeGreaterThanOrEqual(0.85);
      expect(alpha).toBeLessThanOrEqual(0.95);
    }
  });
});

// ===== Test d: GSAP CDN failure triggers CSS-only animations =====
// Validates: Requirement 8.6
test.describe('GSAP CDN failure triggers CSS-only animations', () => {
  test('when GSAP fails to load, AnimationEngine.gsapAvailable is false and CSS fallbacks work', async ({ page }) => {
    // Block GSAP CDN requests
    await page.route('**/cdnjs.cloudflare.com/**', (route) => {
      route.abort();
    });

    await page.goto(trackPageURL);
    await page.waitForLoadState('domcontentloaded');

    // Verify AnimationEngine.gsapAvailable is false
    const gsapAvailable = await page.evaluate(() => window.AnimationEngine.gsapAvailable);
    expect(gsapAvailable).toBe(false);

    // Verify button still has hover transition (CSS fallback)
    const btnTransition = await page.locator('.btn-track').evaluate((el) => {
      return window.getComputedStyle(el).transition;
    });
    // The CSS defines transition on .btn-track for transform and box-shadow
    expect(btnTransition).toContain('transform');

    // Verify input still has focus glow (CSS fallback)
    // Focus the input and check that box-shadow is applied
    await page.locator('#tracking-input').focus();
    // Wait a moment for the CSS transition to apply
    await page.waitForTimeout(300);

    const inputBoxShadow = await page.locator('#tracking-input').evaluate((el) => {
      return window.getComputedStyle(el).boxShadow;
    });
    // Should have a cyan glow box-shadow when focused
    expect(inputBoxShadow).not.toBe('none');
  });
});

// ===== Test e: prefers-reduced-motion disables non-essential animations =====
// Validates: Requirement 9.5
test.describe('prefers-reduced-motion disables non-essential animations', () => {
  test('ParticleSystem is not running when reduced motion is preferred', async ({ page }) => {
    // Emulate prefers-reduced-motion: reduce
    await page.emulateMedia({ reducedMotion: 'reduce' });

    await page.goto(trackPageURL);
    await page.waitForLoadState('domcontentloaded');

    // Verify ParticleSystem is not running
    const particleState = await page.evaluate(() => {
      return {
        running: window.ParticleSystem.running,
        particleCount: window.ParticleSystem.particles.length
      };
    });
    expect(particleState.running).toBe(false);
    expect(particleState.particleCount).toBe(0);
  });

  test('form still works with reduced motion enabled', async ({ page }) => {
    // Emulate prefers-reduced-motion: reduce
    await page.emulateMedia({ reducedMotion: 'reduce' });

    await page.goto(trackPageURL);
    await page.waitForLoadState('domcontentloaded');

    // Submit form with empty input — validation should still work
    await page.locator('.btn-track').click();

    const formMessages = page.locator('#form-messages');
    await expect(formMessages).toHaveText('Please enter a Tracking ID');

    // Submit form with valid input — form should still submit
    await page.locator('#tracking-input').fill('VALID-ID-123');
    await page.locator('.btn-track').click();

    // Verify loading state was triggered (button gets disabled)
    // Since there's no server, it will eventually show an error, but the form processed
    await expect(formMessages).not.toBeEmpty({ timeout: 5000 });
  });

  test('canvas is empty or not initialized when reduced motion is active', async ({ page }) => {
    // Emulate prefers-reduced-motion: reduce
    await page.emulateMedia({ reducedMotion: 'reduce' });

    await page.goto(trackPageURL);
    await page.waitForLoadState('domcontentloaded');

    // Check that the canvas has no drawn content (all pixels are transparent)
    const canvasIsEmpty = await page.evaluate(() => {
      const canvas = document.getElementById('particle-canvas');
      if (!canvas) return true;
      const ctx = canvas.getContext('2d');
      if (!ctx) return true;
      // Check if canvas has any non-transparent pixels
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const data = imageData.data;
      for (let i = 3; i < data.length; i += 4) {
        if (data[i] > 0) return false; // Found a non-transparent pixel
      }
      return true;
    });
    expect(canvasIsEmpty).toBe(true);
  });
});
