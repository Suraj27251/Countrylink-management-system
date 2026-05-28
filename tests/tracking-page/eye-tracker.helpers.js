/**
 * Eye Tracker Helper Module
 * 
 * Extracts the pure calculation functions from the EyeTracker module in track.html
 * for property-based testing. These functions mirror the logic in the inline script.
 */

const config = {
  maxDisplacementRatio: 0.3,   // 30% of eye socket radius
  socketRadiusX: 18,           // rx of eye sockets
  socketRadiusY: 20,           // ry of eye sockets
  headMovementMax: 5           // Max 5px head displacement on keystroke
};

/**
 * Calculate eye displacement from cursor position.
 * Returns displacement constrained to 30% of socket radius.
 *
 * @param {number} cursorX - Cursor X position (viewport coordinates).
 * @param {number} cursorY - Cursor Y position (viewport coordinates).
 * @param {{x: number, y: number}} mascotCenter - Mascot center position.
 * @returns {{x: number, y: number}} Displacement in pixels, constrained to max bounds.
 */
function calculateDisplacement(cursorX, cursorY, mascotCenter) {
  // Direction from mascot center to cursor
  var dx = cursorX - mascotCenter.x;
  var dy = cursorY - mascotCenter.y;

  // Calculate magnitude
  var magnitude = Math.sqrt(dx * dx + dy * dy);

  // Max displacement is 30% of socket radius
  var maxX = config.socketRadiusX * config.maxDisplacementRatio;
  var maxY = config.socketRadiusY * config.maxDisplacementRatio;

  if (magnitude === 0) {
    return { x: 0, y: 0 };
  }

  // Normalize direction and scale to max displacement
  var normalizedX = dx / magnitude;
  var normalizedY = dy / magnitude;

  // Scale displacement proportionally but constrain to max
  var displacementX = normalizedX * maxX;
  var displacementY = normalizedY * maxY;

  return { x: displacementX, y: displacementY };
}

/**
 * Apply proximity-based scaling to displacement.
 * Scales from 10% at viewport edge to 100% when cursor is adjacent to mascot.
 *
 * @param {{x: number, y: number}} displacement - Raw displacement vector.
 * @param {number} distance - Distance from cursor to mascot center in pixels.
 * @param {{width: number, height: number}} viewport - Viewport dimensions.
 * @returns {{x: number, y: number}} Scaled displacement.
 */
function applyProximityScaling(displacement, distance, viewport) {
  // Max possible distance is approximately half the viewport diagonal
  var vw = viewport.width || 1920;
  var vh = viewport.height || 1080;
  var maxDistance = Math.sqrt(vw * vw + vh * vh) / 2;

  // Clamp distance to [0, maxDistance]
  var clampedDistance = Math.max(0, Math.min(distance, maxDistance));

  // Linear interpolation: 100% when distance=0, 10% when distance=maxDistance
  // scale = 1.0 - (0.9 * (distance / maxDistance))
  var scale = 1.0 - (0.9 * (clampedDistance / maxDistance));

  return {
    x: displacement.x * scale,
    y: displacement.y * scale
  };
}

module.exports = {
  config,
  calculateDisplacement,
  applyProximityScaling
};
