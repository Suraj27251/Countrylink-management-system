/**
 * Property-Based Test: Minimum font size across all viewport widths
 *
 * **Validates: Requirements 7.6**
 *
 * Property 6: For ANY viewport width between 320px and 1920px inclusive,
 * all rendered text elements SHALL have a computed font-size of at least 14px.
 *
 * This test uses Playwright for browser-based viewport manipulation and
 * computed style checking, combined with fast-check for random viewport
 * width generation.
 *
 * Run with: npx playwright test min-font-size.property.test.js
 */

const { test, expect } = require('@playwright/test');
const fc = require('fast-check');
const path = require('path');

const TRACK_HTML_PATH = path.resolve(__dirname, '../../track.html');
const FILE_URL = `file://${TRACK_HTML_PATH.replace(/\\/g, '/')}`;

// Fixed height for viewport (height is not under test)
const VIEWPORT_HEIGHT = 768;

// Minimum font size requirement in pixels
const MIN_FONT_SIZE_PX = 14;

// Selectors for all visible text elements to check
const TEXT_ELEMENT_SELECTORS = [
  'p', 'span', 'label', 'input', 'button', 'a',
  'h1', 'h2', 'h3', 'li', 'td', 'th'
].join(', ');

test.describe('Property 6: Minimum font size across all viewport widths', () => {
  test('all text elements have computed font-size >= 14px for random widths between 320 and 1920', async ({ page }) => {
    // Generate 100 random viewport widths using fast-check
    const widthArbitrary = fc.integer({ min: 320, max: 1920 });
    const widths = fc.sample(widthArbitrary, 100);

    for (const width of widths) {
      // Set viewport to the generated width
      await page.setViewportSize({ width, height: VIEWPORT_HEIGHT });

      // Navigate to the tracking page
      await page.goto(FILE_URL, { waitUntil: 'domcontentloaded' });

      // Check computed font-size of all visible text elements
      const results = await page.evaluate((selectors) => {
        const elements = document.querySelectorAll(selectors);
        const violations = [];

        for (const el of elements) {
          // Skip hidden elements (aria-hidden, display:none, visibility:hidden)
          const style = window.getComputedStyle(el);
          if (style.display === 'none' || style.visibility === 'hidden') {
            continue;
          }
          if (el.closest('[aria-hidden="true"]')) {
            continue;
          }
          // Skip elements with no text content
          if (!el.textContent || el.textContent.trim().length === 0) {
            continue;
          }

          const fontSize = parseFloat(style.fontSize);
          if (fontSize < 14) {
            violations.push({
              tag: el.tagName.toLowerCase(),
              id: el.id || '',
              className: el.className || '',
              text: el.textContent.trim().substring(0, 50),
              fontSize: fontSize
            });
          }
        }

        return violations;
      }, TEXT_ELEMENT_SELECTORS);

      // Property assertion: no text element should have font-size < 14px
      expect(
        results,
        `Font size violation at viewport width ${width}px: ${JSON.stringify(results)}`
      ).toHaveLength(0);
    }
  });

  test('all text elements have computed font-size >= 14px at boundary widths', async ({ page }) => {
    const boundaryWidths = [320, 375, 414, 768, 1024, 1280, 1440, 1920];

    for (const width of boundaryWidths) {
      await page.setViewportSize({ width, height: VIEWPORT_HEIGHT });
      await page.goto(FILE_URL, { waitUntil: 'domcontentloaded' });

      const results = await page.evaluate((selectors) => {
        const elements = document.querySelectorAll(selectors);
        const violations = [];

        for (const el of elements) {
          const style = window.getComputedStyle(el);
          if (style.display === 'none' || style.visibility === 'hidden') {
            continue;
          }
          if (el.closest('[aria-hidden="true"]')) {
            continue;
          }
          if (!el.textContent || el.textContent.trim().length === 0) {
            continue;
          }

          const fontSize = parseFloat(style.fontSize);
          if (fontSize < 14) {
            violations.push({
              tag: el.tagName.toLowerCase(),
              id: el.id || '',
              className: el.className || '',
              text: el.textContent.trim().substring(0, 50),
              fontSize: fontSize
            });
          }
        }

        return violations;
      }, TEXT_ELEMENT_SELECTORS);

      expect(
        results,
        `Font size violation at boundary width ${width}px: ${JSON.stringify(results)}`
      ).toHaveLength(0);
    }
  });
});
