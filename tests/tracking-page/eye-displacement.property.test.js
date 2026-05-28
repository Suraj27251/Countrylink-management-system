/**
 * Property-Based Test: Eye Displacement Bounds
 * 
 * Property 1: Eye displacement is bounded and proportionally scaled
 * 
 * **Validates: Requirements 1.1, 1.5**
 * 
 * Tests that for ANY cursor position and mascot center:
 * - |displacement.x| ≤ 5.4 (socketRadiusX * 0.3)
 * - |displacement.y| ≤ 6.0 (socketRadiusY * 0.3)
 * - displacement magnitude ≤ sqrt(5.4² + 6.0²) ≈ 8.07
 * - Proximity scaling: at distance=0, scale=1.0; at distance=maxDistance, scale=0.1
 * - Scaled displacement is always ≤ raw displacement
 */

const fc = require('fast-check');
const { config, calculateDisplacement, applyProximityScaling } = require('./eye-tracker.helpers');

const MAX_X = config.socketRadiusX * config.maxDisplacementRatio; // 18 * 0.3 = 5.4
const MAX_Y = config.socketRadiusY * config.maxDisplacementRatio; // 20 * 0.3 = 6.0
const MAX_MAGNITUDE = Math.sqrt(MAX_X * MAX_X + MAX_Y * MAX_Y);  // ≈ 8.07
const EPSILON = 1e-9; // Floating point tolerance

// Convert to 32-bit floats for fast-check constraints
const MAX_X_F = Math.fround(MAX_X);
const MAX_Y_F = Math.fround(MAX_Y);

describe('Property 1: Eye displacement is bounded and proportionally scaled', () => {

  /**
   * **Validates: Requirements 1.1**
   * 
   * For any cursor position and mascot center, the raw displacement
   * x-component must be within [-5.4, 5.4] and y-component within [-6.0, 6.0].
   */
  it('displacement x and y are bounded within max socket radius ratio', () => {
    fc.assert(
      fc.property(
        fc.float({ min: -5000, max: 5000, noNaN: true }),  // cursorX
        fc.float({ min: -5000, max: 5000, noNaN: true }),  // cursorY
        fc.float({ min: -5000, max: 5000, noNaN: true }),  // mascotCenterX
        fc.float({ min: -5000, max: 5000, noNaN: true }),  // mascotCenterY
        (cursorX, cursorY, mascotCenterX, mascotCenterY) => {
          const mascotCenter = { x: mascotCenterX, y: mascotCenterY };
          const displacement = calculateDisplacement(cursorX, cursorY, mascotCenter);

          expect(Math.abs(displacement.x)).toBeLessThanOrEqual(MAX_X + EPSILON);
          expect(Math.abs(displacement.y)).toBeLessThanOrEqual(MAX_Y + EPSILON);
        }
      ),
      { numRuns: 100 }
    );
  });

  /**
   * **Validates: Requirements 1.1**
   * 
   * For any cursor position and mascot center, the displacement magnitude
   * must not exceed sqrt(5.4² + 6.0²) ≈ 8.07.
   */
  it('displacement magnitude is bounded', () => {
    fc.assert(
      fc.property(
        fc.float({ min: -5000, max: 5000, noNaN: true }),  // cursorX
        fc.float({ min: -5000, max: 5000, noNaN: true }),  // cursorY
        fc.float({ min: -5000, max: 5000, noNaN: true }),  // mascotCenterX
        fc.float({ min: -5000, max: 5000, noNaN: true }),  // mascotCenterY
        (cursorX, cursorY, mascotCenterX, mascotCenterY) => {
          const mascotCenter = { x: mascotCenterX, y: mascotCenterY };
          const displacement = calculateDisplacement(cursorX, cursorY, mascotCenter);

          const magnitude = Math.sqrt(displacement.x * displacement.x + displacement.y * displacement.y);
          expect(magnitude).toBeLessThanOrEqual(MAX_MAGNITUDE + EPSILON);
        }
      ),
      { numRuns: 100 }
    );
  });

  /**
   * **Validates: Requirements 1.5**
   * 
   * Proximity scaling: at distance=0, scale factor is 1.0 (100%).
   * The scaled displacement equals the raw displacement when cursor is at mascot center.
   */
  it('proximity scaling at distance=0 yields full displacement (scale=1.0)', () => {
    fc.assert(
      fc.property(
        fc.float({ min: -MAX_X_F, max: MAX_X_F, noNaN: true }),  // displacement.x
        fc.float({ min: -MAX_Y_F, max: MAX_Y_F, noNaN: true }),  // displacement.y
        fc.integer({ min: 320, max: 1920 }),                       // viewport width
        fc.integer({ min: 320, max: 1080 }),                       // viewport height
        (dx, dy, vw, vh) => {
          const displacement = { x: dx, y: dy };
          const viewport = { width: vw, height: vh };

          const scaled = applyProximityScaling(displacement, 0, viewport);

          expect(scaled.x).toBeCloseTo(displacement.x, 5);
          expect(scaled.y).toBeCloseTo(displacement.y, 5);
        }
      ),
      { numRuns: 100 }
    );
  });

  /**
   * **Validates: Requirements 1.5**
   * 
   * Proximity scaling: at distance=maxDistance (half viewport diagonal),
   * scale factor is 0.1 (10%).
   */
  it('proximity scaling at max distance yields 10% displacement (scale=0.1)', () => {
    fc.assert(
      fc.property(
        fc.float({ min: -MAX_X_F, max: MAX_X_F, noNaN: true }),  // displacement.x
        fc.float({ min: -MAX_Y_F, max: MAX_Y_F, noNaN: true }),  // displacement.y
        fc.integer({ min: 320, max: 1920 }),                       // viewport width
        fc.integer({ min: 320, max: 1080 }),                       // viewport height
        (dx, dy, vw, vh) => {
          const displacement = { x: dx, y: dy };
          const viewport = { width: vw, height: vh };
          const maxDistance = Math.sqrt(vw * vw + vh * vh) / 2;

          const scaled = applyProximityScaling(displacement, maxDistance, viewport);

          expect(scaled.x).toBeCloseTo(displacement.x * 0.1, 5);
          expect(scaled.y).toBeCloseTo(displacement.y * 0.1, 5);
        }
      ),
      { numRuns: 100 }
    );
  });

  /**
   * **Validates: Requirements 1.5**
   * 
   * For any distance, the scaled displacement magnitude is always ≤ the raw displacement magnitude.
   * Proximity scaling only reduces displacement, never amplifies it.
   */
  it('scaled displacement magnitude is always ≤ raw displacement magnitude', () => {
    fc.assert(
      fc.property(
        fc.float({ min: -MAX_X_F, max: MAX_X_F, noNaN: true }),  // displacement.x
        fc.float({ min: -MAX_Y_F, max: MAX_Y_F, noNaN: true }),  // displacement.y
        fc.float({ min: 0, max: Math.fround(5000), noNaN: true }),  // distance
        fc.integer({ min: 320, max: 1920 }),                       // viewport width
        fc.integer({ min: 320, max: 1080 }),                       // viewport height
        (dx, dy, distance, vw, vh) => {
          const displacement = { x: dx, y: dy };
          const viewport = { width: vw, height: vh };

          const scaled = applyProximityScaling(displacement, distance, viewport);

          const rawMagnitude = Math.sqrt(dx * dx + dy * dy);
          const scaledMagnitude = Math.sqrt(scaled.x * scaled.x + scaled.y * scaled.y);

          expect(scaledMagnitude).toBeLessThanOrEqual(rawMagnitude + EPSILON);
        }
      ),
      { numRuns: 100 }
    );
  });

  /**
   * **Validates: Requirements 1.1**
   * 
   * When cursor is at the same position as mascot center (distance=0),
   * displacement should be zero (no direction to move).
   */
  it('displacement is zero when cursor is at mascot center', () => {
    fc.assert(
      fc.property(
        fc.float({ min: -5000, max: 5000, noNaN: true }),  // position x
        fc.float({ min: -5000, max: 5000, noNaN: true }),  // position y
        (px, py) => {
          const mascotCenter = { x: px, y: py };
          const displacement = calculateDisplacement(px, py, mascotCenter);

          expect(displacement.x).toBe(0);
          expect(displacement.y).toBe(0);
        }
      ),
      { numRuns: 100 }
    );
  });
});
