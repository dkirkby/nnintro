/**
 * The forward pass and its derivatives, in plain TypeScript.
 *
 * This module is the source of mathematical truth for the explorer and has
 * no DOM dependency: views consume the snapshot it returns, and the
 * animation never produces or amends one.
 *
 * Everything here is recomputed after a parameter edit. The gradients
 * shipped with a checkpoint are fixtures for the tests, not a cache.
 */

import type {
  Activation,
  ArchitectureSpec,
  LayerParameters,
  Split,
  Task,
} from "./types.js";

/** The ReLU derivative at exactly zero. A convention, not a derivative:
 *  no unique one exists there, and the UI says so rather than hiding it. */
export const RELU_DERIVATIVE_AT_ZERO = 0;

export function applyActivation(name: Activation, z: number): number {
  switch (name) {
    case "relu":
      return z > 0 ? z : 0;
    case "tanh":
      return Math.tanh(z);
    case "identity":
      return z;
  }
}

export function activationDerivative(name: Activation, z: number): number {
  switch (name) {
    case "relu":
      // z === 0 falls through to the documented convention.
      return z > 0 ? 1 : z < 0 ? 0 : RELU_DERIVATIVE_AT_ZERO;
    case "tanh": {
      const t = Math.tanh(z);
      return 1 - t * t;
    }
    case "identity":
      return 1;
  }
}

export function sigmoid(z: number): number {
  // Branched to stay stable at extreme logits.
  if (z >= 0) return 1 / (1 + Math.exp(-z));
  const e = Math.exp(z);
  return e / (1 + e);
}

/** One forward pass, retaining every intermediate state. */
export interface Trace {
  /** `pre[0]` is the input; `pre[l]` is layer l's preactivation. */
  pre: number[][];
  /** `act[0]` is the input; the output layer's activation is its
   *  preactivation, so a classifier's logit is never squashed in place. */
  act: number[][];
  output: number;
}

export function forward(
  parameters: LayerParameters[],
  activation: Activation,
  x: number[],
): Trace {
  const pre: number[][] = [x.slice()];
  const act: number[][] = [x.slice()];
  const last = parameters.length - 1;
  for (let l = 0; l < parameters.length; l += 1) {
    const { W, b } = parameters[l];
    const input = act[l];
    const z = new Array<number>(b.length);
    for (let i = 0; i < b.length; i += 1) {
      let sum = b[i];
      const row = W[i]; // destination-first: row i holds the inputs to unit i
      for (let j = 0; j < row.length; j += 1) sum += row[j] * input[j];
      z[i] = sum;
    }
    pre.push(z);
    act.push(l === last ? z.slice() : z.map((v) => applyActivation(activation, v)));
  }
  return { pre, act, output: pre[pre.length - 1][0] };
}

/** d(output)/dx at one point, by forward accumulation through the layers.
 *
 *  For a classifier this is the LOGIT sensitivity; convert with
 *  `probabilitySensitivity` and label which one is displayed. */
export function inputSensitivity(
  parameters: LayerParameters[],
  activation: Activation,
  trace: Trace,
): number[] {
  const inputs = trace.act[0].length;
  // jacobian[i][j] = d(unit i of the current layer)/d(x_j)
  let jacobian: number[][] = Array.from({ length: inputs }, (_, i) =>
    Array.from({ length: inputs }, (_, j) => (i === j ? 1 : 0)),
  );
  const last = parameters.length - 1;
  for (let l = 0; l < parameters.length; l += 1) {
    const { W, b } = parameters[l];
    const next: number[][] = [];
    for (let i = 0; i < b.length; i += 1) {
      const row = new Array<number>(inputs).fill(0);
      for (let j = 0; j < W[i].length; j += 1) {
        const w = W[i][j];
        for (let k = 0; k < inputs; k += 1) row[k] += w * jacobian[j][k];
      }
      if (l !== last) {
        const slope = activationDerivative(activation, trace.pre[l + 1][i]);
        for (let k = 0; k < inputs; k += 1) row[k] *= slope;
      }
      next.push(row);
    }
    jacobian = next;
  }
  return jacobian[0];
}

/** Chain rule from logit sensitivity to probability sensitivity. */
export function probabilitySensitivity(
  logit: number,
  logitSensitivity: number[],
): number[] {
  const p = sigmoid(logit);
  const slope = p * (1 - p);
  return logitSensitivity.map((v) => v * slope);
}

export function pointLoss(output: number, target: number, task: Task): number {
  if (task === "regression") {
    const residual = output - target;
    return residual * residual;
  }
  // BCE from logits, in the stable form max(z,0) - z*y + log(1 + exp(-|z|)).
  return (
    Math.max(output, 0) - output * target + Math.log1p(Math.exp(-Math.abs(output)))
  );
}

/** d(point loss)/d(output). Both tasks reduce to a one-line derivative. */
export function lossDerivative(
  output: number,
  target: number,
  task: Task,
): number {
  return task === "regression"
    ? 2 * (output - target)
    : sigmoid(output) - target;
}

/** Zero-filled gradient accumulator shaped like the parameters. */
export function zeroLike(parameters: LayerParameters[]): LayerParameters[] {
  return parameters.map((layer) => ({
    W: layer.W.map((row) => row.map(() => 0)),
    b: layer.b.map(() => 0),
  }));
}

/**
 * d(objective)/d(every parameter), accumulated in reverse.
 *
 * `seed` is d(objective)/d(output) for this example, so the same routine
 * serves a point loss and (accumulated and scaled) a mean loss.
 *
 * This is deliberately *not* surfaced as a lesson: the explorer shows
 * derivatives as properties of the current model, and a separate tool
 * covers how they are computed (PLAN.md section 5).
 */
export function accumulateParameterGradient(
  parameters: LayerParameters[],
  activation: Activation,
  trace: Trace,
  seed: number,
  into: LayerParameters[],
  scale = 1,
): void {
  const last = parameters.length - 1;
  let delta: number[] = [seed];
  for (let l = last; l >= 0; l -= 1) {
    const { W } = parameters[l];
    const input = trace.act[l];
    const target = into[l];
    for (let i = 0; i < delta.length; i += 1) {
      const d = delta[i] * scale;
      target.b[i] += d;
      const row = target.W[i];
      for (let j = 0; j < input.length; j += 1) row[j] += d * input[j];
    }
    if (l === 0) break;
    const upstream = new Array<number>(input.length).fill(0);
    for (let i = 0; i < delta.length; i += 1) {
      const row = W[i];
      for (let j = 0; j < row.length; j += 1) upstream[j] += row[j] * delta[i];
    }
    // Layer l's input is a hidden activation, so its slope applies here.
    for (let j = 0; j < upstream.length; j += 1) {
      upstream[j] *= activationDerivative(activation, trace.pre[l][j]);
    }
    delta = upstream;
  }
}

/** d(point loss)/d(parameters) for one labeled example. */
export function pointGradient(
  parameters: LayerParameters[],
  activation: Activation,
  trace: Trace,
  target: number,
  task: Task,
): LayerParameters[] {
  const into = zeroLike(parameters);
  accumulateParameterGradient(
    parameters,
    activation,
    trace,
    lossDerivative(trace.output, target, task),
    into,
  );
  return into;
}

export interface SplitObjective {
  count: number;
  meanLoss: number;
  gradient: LayerParameters[];
}

/**
 * The mean loss over a split and the gradient OF THAT MEAN.
 *
 * Not a mean of per-example gradient magnitudes: that is a different
 * quantity, and only this one is the slope the explorer claims it is
 * (PLAN.md section 5).
 */
export function splitObjective(
  parameters: LayerParameters[],
  activation: Activation,
  task: Task,
  samples: { x: number[][]; y: number[]; isTest: boolean[] },
  split: Split,
): SplitObjective {
  const gradient = zeroLike(parameters);
  let total = 0;
  let count = 0;
  for (let n = 0; n < samples.y.length; n += 1) {
    if ((split === "test") !== samples.isTest[n]) continue;
    count += 1;
    const trace = forward(parameters, activation, samples.x[n]);
    total += pointLoss(trace.output, samples.y[n], task);
    accumulateParameterGradient(
      parameters,
      activation,
      trace,
      lossDerivative(trace.output, samples.y[n], task),
      gradient,
    );
  }
  if (count === 0) return { count: 0, meanLoss: NaN, gradient };
  const inverse = 1 / count;
  for (const layer of gradient) {
    for (const row of layer.W) for (let j = 0; j < row.length; j += 1) row[j] *= inverse;
    for (let i = 0; i < layer.b.length; i += 1) layer.b[i] *= inverse;
  }
  return { count, meanLoss: total * inverse, gradient };
}

/** The hidden activation to use; a network with no hidden layer has none. */
export function hiddenActivation(architecture: ArchitectureSpec): Activation {
  return architecture.hiddenActivation ?? "identity";
}
