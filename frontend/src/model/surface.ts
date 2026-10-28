/**
 * The prediction surface over a two-dimensional input domain.
 *
 * The input-space view draws this but does not compute it: the numerics stay
 * out of the DOM modules, and keeping the grid here lets the caller reuse it
 * across probe moves, which change nothing about the surface.
 */

import { forward } from "./evaluate.js";
import type { Activation, LayerParameters } from "./types.js";

export interface Surface {
  /** Cells per side. */
  resolution: number;
  /**
   * Row-major, one output per cell centre. Row 0 is the TOP of the plot, so
   * it holds the largest second input: screen order, not value order.
   */
  values: number[];
  /**
   * The largest output magnitude anywhere in the domain, corners included.
   *
   * The shared colour scale is built from this rather than from anything at
   * the probe: a limit that followed the probe would recolour the whole
   * surface while the probe was being dragged. It depends on the parameters
   * alone, so it moves only when an edit moves it.
   */
  extent: number;
}

export const DEFAULT_RESOLUTION = 60;

export function outputSurface(
  parameters: LayerParameters[],
  activation: Activation,
  domain: [number, number][],
  resolution: number = DEFAULT_RESOLUTION,
): Surface {
  if (domain.length !== 2) {
    throw new Error(`the input-space surface needs exactly 2 inputs, got ${domain.length}`);
  }
  const [[x1lo, x1hi], [x2lo, x2hi]] = domain;
  const values: number[] = [];
  let extent = 0;

  function record(x: number[]): number {
    const { output } = forward(parameters, activation, x);
    if (Number.isFinite(output)) extent = Math.max(extent, Math.abs(output));
    return output;
  }

  for (let row = 0; row < resolution; row += 1) {
    const x2 = x2hi - ((row + 0.5) / resolution) * (x2hi - x2lo);
    for (let column = 0; column < resolution; column += 1) {
      const x1 = x1lo + ((column + 0.5) / resolution) * (x1hi - x1lo);
      values.push(record([x1, x2]));
    }
  }
  // The grid samples cell centres, so the domain's own corners -- where a
  // piecewise-linear surface usually peaks -- are never on it.
  for (const x1 of [x1lo, x1hi]) {
    for (const x2 of [x2lo, x2hi]) record([x1, x2]);
  }
  return { resolution, values, extent };
}

/**
 * One line through the input domain: every other input is pinned where the
 * probe left it and one is swept across its whole range.
 *
 * This is the slice the output-space plots draw. Unlike the surface it does
 * depend on the probe, so it is recomputed on every render -- a couple of
 * hundred forward passes, against the surface's few thousand.
 */
export interface Slice {
  /** Which input varies: an index into the domain. */
  axis: number;
  /** Where the other inputs are held. */
  at: number[];
  /** Positions of the varying input, from the low end of its range up. */
  inputs: number[];
  /** The output at each of those positions. */
  values: number[];
}

export const DEFAULT_SLICE_SAMPLES = 161;

export function outputSlice(
  parameters: LayerParameters[],
  activation: Activation,
  domain: [number, number][],
  at: number[],
  axis: number,
  samples: number = DEFAULT_SLICE_SAMPLES,
): Slice {
  if (axis < 0 || axis >= domain.length) {
    throw new Error(`no axis ${axis} in a ${domain.length}-input domain`);
  }
  const [lo, hi] = domain[axis];
  const inputs: number[] = [];
  const values: number[] = [];
  for (let i = 0; i < samples; i += 1) {
    const position = lo + ((hi - lo) * i) / (samples - 1);
    const x = at.slice();
    x[axis] = position;
    inputs.push(position);
    values.push(forward(parameters, activation, x).output);
  }
  return { axis, at: at.slice(), inputs, values };
}
