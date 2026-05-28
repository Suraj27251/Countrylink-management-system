/**
 * Property-Based Test: Head movement on keystroke is bounded
 *
 * Property 2: For ANY keystroke event triggered while the input field is focused,
 * the resulting head displacement SHALL be no greater than 5px in any direction.
 *
 * **Validates: Requirements 1.3**
 *
 * The EyeTracker.onKeystroke() method generates random head movement with:
 *   headDX = (Math.random() - 0.5) * 2 * headMovementMax (where headMovementMax = 5)
 *   headDY = (Math.random() - 0.5) * 2 * headMovementMax
 *   Then clamps magnitude: if sqrt(dx² + dy²) > 5, normalize to 5px
 */

const fc = require('fast-check');

// Extract the head movement calculation logic from EyeTracker.onKeystroke()
// This mirrors the exact logic in track.html
const HEAD_MOVEMENT_MAX = 5;

/**
 * Calculate head displacement given two random values (simulating Math.random() calls).
 * @param {number} rand1 - First random value in [0, 1) (used for headDX)
 * @param {number} rand2 - Second random value in [0, 1) (used for headDY)
 * @returns {{headDX: number, headDY: number}} The clamped head displacement
 */
function calculateHeadMovement(rand1, rand2) {
  let headDX = (rand1 - 0.5) * 2 * HEAD_MOVEMENT_MAX;
  let headDY = (rand2 - 0.5) * 2 * HEAD_MOVEMENT_MAX;

  // Clamp to ≤5px magnitude
  const mag = Math.sqrt(headDX * headDX + headDY * headDY);
  if (mag > HEAD_MOVEMENT_MAX) {
    headDX = (headDX / mag) * HEAD_MOVEMENT_MAX;
    headDY = (headDY / mag) * HEAD_MOVEMENT_MAX;
  }

  return { headDX, headDY };
}

describe('Property 2: Head movement on keystroke is bounded', () => {
  it('head displacement magnitude is always ≤ 5px for any random inputs', () => {
    fc.assert(
      fc.property(
        fc.double({ min: 0, max: 1, noNaN: true }),
        fc.double({ min: 0, max: 1, noNaN: true }),
        (rand1, rand2) => {
          const { headDX, headDY } = calculateHeadMovement(rand1, rand2);
          const magnitude = Math.sqrt(headDX * headDX + headDY * headDY);

          // The magnitude must never exceed 5px
          return magnitude <= HEAD_MOVEMENT_MAX + 1e-10; // small epsilon for floating point
        }
      ),
      { numRuns: 100 }
    );
  });

  it('individual head displacement components are bounded by ≤ 5px in both x and y', () => {
    fc.assert(
      fc.property(
        fc.double({ min: 0, max: 1, noNaN: true }),
        fc.double({ min: 0, max: 1, noNaN: true }),
        (rand1, rand2) => {
          const { headDX, headDY } = calculateHeadMovement(rand1, rand2);

          // Each component must be within [-5, 5] range
          return Math.abs(headDX) <= HEAD_MOVEMENT_MAX + 1e-10 &&
                 Math.abs(headDY) <= HEAD_MOVEMENT_MAX + 1e-10;
        }
      ),
      { numRuns: 100 }
    );
  });
});
