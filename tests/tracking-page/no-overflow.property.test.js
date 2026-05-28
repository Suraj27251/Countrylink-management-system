/**
 * Property-Based Test: No horizontal overflow at any supported viewport width
 *
 * **Validates: Requirements 7.1**
 *
 * Property 5: For ANY viewport width between 320px and 1920px inclusive,
 * the page content SHALL NOT produce horizontal scrolling
 * (document.scrollWidth <= viewport width).
 *
 * This test uses Playwright for browser-based viewport manipulation
 * combined with fast-check for random viewport width generation.
 *
 * Run with: npx playwright test no-overflow.property.test.js
 */

const { test, expect } = require('@playwright/test');
const fc = require('fast-check');
const path = require('path');

const TRACK_HTML_PATH = path.resolve(__dirname, '../../track.html');
const FILE_URL = `file://${TRACK_HTML_PATH.replace(/\\/g, '/')}`;

// Fixed height for viewport (height is not under test)
const VIEWPORT_HEIGHT = 768;

test.describe('Property 5: No horizontal overflow at any supported viewport width', () => {
  test('document scrollWidth <= viewport width for all widths between 320 and 1920', async ({ page }) => {
    // Generate random viewport widths using fast-check
    const widthArbitrary = fc.integer({ min: 320, max: 1920 });
    const widths = fc.sample(widthArbitrary, 100);

    for (const width of widths) {
      // Set viewport to the generated width
      await page.setViewportSize({ width, height: VIEWPORT_HEIGHT });

      // Navigate to the tracking page
      await page.goto(FILE_URL, { waitUntil: 'domcontentloaded' });

      // Measure the document scrollWidth
      const scrollWidth = await page.evaluate(() => {
        return document.documentElement.scrollWidth;
      });

      // Property assertion: no horizontal overflow
      expect(
        scrollWidth,
        `Horizontal overflow detected at viewport width ${width}px: scrollWidth=${scrollWidth}px`
      ).toBeLessThanOrEqual(width);
    }
  });

  test('no horizontal overflow at boundary widths (320, 768, 1024, 1920)', async ({ page }) => {
    const boundaryWidths = [320, 375, 414, 768, 1024, 1280, 1440, 1920];

    for (const width of boundaryWidths) {
      await page.setViewportSize({ width, height: VIEWPORT_HEIGHT });
      await page.goto(FILE_URL, { waitUntil: 'domcontentloaded' });

      const scrollWidth = await page.evaluate(() => {
        return document.documentElement.scrollWidth;
      });

      expect(
        scrollWidth,
        `Horizontal overflow at boundary width ${width}px: scrollWidth=${scrollWidth}px`
      ).toBeLessThanOrEqual(width);
    }
  });
});
