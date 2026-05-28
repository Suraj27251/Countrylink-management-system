/**
 * Property-Based Test: Parallax displacement is bounded
 *
 * **Validates: Requirements 3.3**
 *
 * Property 3: For ANY cursor position (x, y) and ANY viewport dimensions,
 * the calculated parallax offset magnitude is always ≤ 20px (maxParallax).
 */

const fc = require('fast-check');

// ===== Extracted parallax calculation logic from track.html =====
// This mirrors the ParticleSystem.applyParallax() implementation

const MAX_PARALLAX = 20;

/**
 * Calculate parallax offset based on cursor position relative to viewport center.
 * Caps displacement at ±maxParallax (20px) in any direction.
 *
 * @param {number} cursorX - Cursor X position (can be any value, including negative or very large)
 * @param {number} cursorY - Cursor Y position (can be any value, including negative or very large)
 * @param {number} viewportWidth - Viewport width in pixels
 * @param {number} viewportHeight - Viewport height in pixels
 * @returns {{x: number, y: number}} The parallax offset, capped at ±20px
 */
function calculateParallaxOffset(cursorX, cursorY, viewportWidth, viewportHeight) {
    var maxParallax = MAX_PARALLAX;

    // Calculate offset based on cursor position relative to viewport center
    var offsetX = ((cursorX - viewportWidth / 2) / (viewportWidth / 2)) * maxParallax;
    var offsetY = ((cursorY - viewportHeight / 2) / (viewportHeight / 2)) * maxParallax;

    // Cap displacement at maxParallax (20px) in any direction
    offsetX = Math.max(-maxParallax, Math.min(maxParallax, offsetX));
    offsetY = Math.max(-maxParallax, Math.min(maxParallax, offsetY));

    return { x: offsetX, y: offsetY };
}

describe('Property 3: Parallax displacement is bounded', () => {
    it('parallax offset magnitude ≤ 20px for all cursor positions and viewport sizes', () => {
        fc.assert(
            fc.property(
                // Generate random cursor positions (any value, including negative and very large)
                fc.double({ min: -10000, max: 10000, noNaN: true, noDefaultInfinity: true }),
                fc.double({ min: -10000, max: 10000, noNaN: true, noDefaultInfinity: true }),
                // Generate random viewport dimensions (320-1920 width, 480-1080 height)
                fc.integer({ min: 320, max: 1920 }),
                fc.integer({ min: 480, max: 1080 }),
                (cursorX, cursorY, viewportWidth, viewportHeight) => {
                    const offset = calculateParallaxOffset(cursorX, cursorY, viewportWidth, viewportHeight);

                    // Verify: |offsetX| ≤ 20 AND |offsetY| ≤ 20
                    expect(Math.abs(offset.x)).toBeLessThanOrEqual(MAX_PARALLAX);
                    expect(Math.abs(offset.y)).toBeLessThanOrEqual(MAX_PARALLAX);
                }
            ),
            { numRuns: 100 }
        );
    });

    it('parallax offset is zero when cursor is at viewport center', () => {
        fc.assert(
            fc.property(
                // Generate random viewport dimensions
                fc.integer({ min: 320, max: 1920 }),
                fc.integer({ min: 480, max: 1080 }),
                (viewportWidth, viewportHeight) => {
                    const centerX = viewportWidth / 2;
                    const centerY = viewportHeight / 2;
                    const offset = calculateParallaxOffset(centerX, centerY, viewportWidth, viewportHeight);

                    // At center, offset should be 0
                    expect(offset.x).toBeCloseTo(0, 5);
                    expect(offset.y).toBeCloseTo(0, 5);
                }
            ),
            { numRuns: 100 }
        );
    });
});
