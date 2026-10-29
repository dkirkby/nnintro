/**
 * What the output would be if one parameter were different.
 *
 * For each parameter in turn, every other parameter and the input are held
 * exactly where they are and that one is swept across the parameter scale.
 * Superimposed, the curves are the network's sensitivity to each of its
 * parameters at this input, read off directly rather than inferred from a
 * gradient: the slope at the current value *is* ∂ŷ/∂θ, and a ReLU kink is
 * the bend where that unit switches on or off.
 *
 * Every curve passes through the current output at the current value, which
 * is why the dots all land on one horizontal line.
 */

import { forward } from "./evaluate.js";
import type { Activation, LayerParameters, ParameterRef } from "./types.js";

export interface ParameterCurve {
  ref: ParameterRef;
  /** The swept values, from −limit up to +limit. */
  values: number[];
  /** The output at each swept value, with everything else held. */
  outputs: number[];
  /** Where this parameter is now … */
  current: number;
  /** … and the output there, which is the same for every curve. */
  output: number;
}

export const DEFAULT_CURVE_SAMPLES = 121;

/** Every parameter, in a stable order: each layer's weights, then its biases. */
export function parameterRefs(parameters: LayerParameters[]): ParameterRef[] {
  const refs: ParameterRef[] = [];
  parameters.forEach((layer, index) => {
    layer.W.forEach((row, out) => {
      row.forEach((_, input) => refs.push({ kind: "weight", layer: index, out, in: input }));
    });
    layer.b.forEach((_, out) => refs.push({ kind: "bias", layer: index, out }));
  });
  return refs;
}

export function parameterCurves(
  parameters: LayerParameters[],
  activation: Activation,
  x: number[],
  limit: number,
  samples: number = DEFAULT_CURVE_SAMPLES,
): ParameterCurve[] {
  // One working copy, mutated and put back: the caller's parameters are the
  // explorer's own state and must come out of this untouched.
  const working = parameters.map((layer) => ({
    W: layer.W.map((row) => row.slice()),
    b: layer.b.slice(),
  }));

  return parameterRefs(parameters).map((ref) => {
    const current = read(working, ref);
    const values: number[] = [];
    const outputs: number[] = [];
    for (let i = 0; i < samples; i += 1) {
      const value = -limit + ((2 * limit) / (samples - 1)) * i;
      write(working, ref, value);
      values.push(value);
      outputs.push(forward(working, activation, x).output);
    }
    write(working, ref, current);
    return { ref, values, outputs, current, output: forward(working, activation, x).output };
  });
}

function read(parameters: LayerParameters[], ref: ParameterRef): number {
  return ref.kind === "weight"
    ? parameters[ref.layer].W[ref.out][ref.in]
    : parameters[ref.layer].b[ref.out];
}

/** In place, on the private working copy only: the shared contract elsewhere
 *  is that a parameter set is never mutated (`model/types.ts`). */
function write(parameters: LayerParameters[], ref: ParameterRef, value: number): void {
  if (ref.kind === "weight") parameters[ref.layer].W[ref.out][ref.in] = value;
  else parameters[ref.layer].b[ref.out] = value;
}
