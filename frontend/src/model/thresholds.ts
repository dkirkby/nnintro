/**
 * Where a first-hidden-layer unit switches: the line in input space on
 * which its preactivation is zero.
 *
 * A unit of the first hidden layer reads the inputs directly, so its
 * preactivation is affine in them — `z = w₀x₁ + w₁x₂ + b` — and `z = 0` is
 * a straight line across the input domain. That line is the unit's own
 * threshold: for ReLU it is exactly the kink, the boundary between the half
 * of the domain where the unit contributes nothing and the half where it
 * contributes `z`; for tanh and sigmoid it is where the unit is at its most
 * sensitive. Drawn over the prediction map, the lines are what connect the
 * network's parameters to the shape of the surface — a ReLU surface's
 * creases lie on them.
 *
 * Only the first hidden layer has such a line. A later unit reads
 * activations rather than inputs, so its own `z = 0` set is a curve folded
 * by every layer under it, which is a different and much longer story.
 *
 * Pure geometry, like the rest of `model/`: the caller turns a segment in
 * input coordinates into pixels.
 */

import type { LayerParameters } from "./types.js";

/** One unit's threshold, clipped to the domain rectangle. */
export interface ThresholdLine {
  /** The unit's index in the first hidden layer. */
  unit: number;
  /** The segment's ends, in input coordinates. */
  from: [number, number];
  to: [number, number];
}

/**
 * The lines, one per unit of the first hidden layer, in unit order.
 *
 * A unit whose threshold misses the domain is left out rather than
 * reported empty: there is nothing to draw, and the caller should not have
 * to decide what an absent segment means. So are the two degenerate cases
 * — a unit with both input weights zero has no line at all (its
 * preactivation is the constant `b`), and one that is zero everywhere
 * because its weights *and* bias vanish would be the whole plane rather
 * than a line.
 *
 * `parameters` is the whole network: the function reads its first layer and
 * needs the second to know that layer is hidden. A network with no hidden
 * layer has no thresholds, and its output's own `z = 0` is not one of
 * these — it is the prediction, which the map already draws.
 */
export function thresholdLines(
  parameters: LayerParameters[],
  domain: [number, number][],
): ThresholdLine[] {
  if (domain.length !== 2) {
    throw new Error(`a threshold is a line in 2 inputs, not ${domain.length}`);
  }
  if (parameters.length < 2) return [];
  const [first] = parameters;
  const lines: ThresholdLine[] = [];
  for (let unit = 0; unit < first.W.length; unit += 1) {
    const [w1, w2] = first.W[unit];
    const segment = clip(w1, w2, first.b[unit], domain);
    if (segment) lines.push({ unit, from: segment[0], to: segment[1] });
  }
  return lines;
}

/**
 * `w1 x₁ + w2 x₂ + b = 0`, clipped to the rectangle.
 *
 * Every intersection with the four edges is collected and the two furthest
 * apart are kept, which needs no case analysis: a line through a corner
 * yields the same point twice and is deduplicated, a line that misses the
 * rectangle yields fewer than two points, and an edge-parallel line is the
 * ordinary case with one of the two weights zero.
 */
function clip(
  w1: number,
  w2: number,
  b: number,
  [[x1lo, x1hi], [x2lo, x2hi]]: [number, number][],
): [[number, number], [number, number]] | null {
  // Scaled to the rectangle, so the tolerances below are about the domain
  // rather than about how large the weights happen to be.
  const scale = Math.hypot(w1 * (x1hi - x1lo), w2 * (x2hi - x2lo));
  if (!Number.isFinite(scale) || scale === 0) return null;
  const slack = 1e-9 * Math.max(x1hi - x1lo, x2hi - x2lo);

  const points: [number, number][] = [];
  const add = (x1: number, x2: number): void => {
    if (!Number.isFinite(x1) || !Number.isFinite(x2)) return;
    // `+ 0` normalises a negative zero, which an exactly-centred line
    // produces and which would otherwise be printed as one.
    points.push([
      Math.min(x1hi, Math.max(x1lo, x1)) + 0,
      Math.min(x2hi, Math.max(x2lo, x2)) + 0,
    ]);
  };
  if (w2 !== 0) {
    for (const x1 of [x1lo, x1hi]) {
      const x2 = -(w1 * x1 + b) / w2;
      if (x2lo - slack <= x2 && x2 <= x2hi + slack) add(x1, x2);
    }
  }
  if (w1 !== 0) {
    for (const x2 of [x2lo, x2hi]) {
      const x1 = -(w2 * x2 + b) / w1;
      if (x1lo - slack <= x1 && x1 <= x1hi + slack) add(x1, x2);
    }
  }

  let best: [[number, number], [number, number]] | null = null;
  let longest = 0;
  for (let i = 0; i < points.length; i += 1) {
    for (let j = i + 1; j < points.length; j += 1) {
      const length = Math.hypot(points[i][0] - points[j][0], points[i][1] - points[j][1]);
      if (length > longest) {
        longest = length;
        best = [points[i], points[j]];
      }
    }
  }
  // A line that only grazes a corner is a point, not a segment: nothing to
  // draw, and drawing it would put a stray dot on the map.
  return longest > slack ? best : null;
}
