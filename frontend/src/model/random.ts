/**
 * Deterministic seeded initialization for the "randomize" baseline.
 *
 * A random baseline is for exploration, not a claim of training quality,
 * and the explorer records the resulting parameters rather than only the
 * seed: a seed is only reproducible against one implementation of the
 * generator, and the asset outlives this one (PLAN.md section 4).
 */

import type { Activation, LayerParameters } from "./types.js";

/** mulberry32: small, fast, and good enough for an initializer. */
export function createRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Box-Muller, so the draws are normal rather than merely random. */
export function normalDraws(next: () => number): () => number {
  let spare: number | null = null;
  return () => {
    if (spare !== null) {
      const value = spare;
      spare = null;
      return value;
    }
    let u = 0;
    let v = 0;
    while (u === 0) u = next();
    while (v === 0) v = next();
    const radius = Math.sqrt(-2 * Math.log(u));
    const angle = 2 * Math.PI * v;
    spare = radius * Math.sin(angle);
    return radius * Math.cos(angle);
  };
}

/**
 * Activation-aware initialization: He scaling for ReLU hidden layers,
 * Xavier elsewhere, matching `DenseNet._init_parameters` in the pipeline.
 *
 * Unlike training's initialization, biases are randomized (small), so that
 * a randomized network is not born with every unit's kink through the
 * origin.
 */
export function randomParameters(
  layerSizes: number[],
  activation: Activation,
  seed: number,
): LayerParameters[] {
  const normal = normalDraws(createRandom(seed));
  const layers: LayerParameters[] = [];
  for (let l = 0; l < layerSizes.length - 1; l += 1) {
    const fanIn = layerSizes[l];
    const fanOut = layerSizes[l + 1];
    const isOutput = l === layerSizes.length - 2;
    const variance = !isOutput && activation === "relu" ? 2 / fanIn : 1 / fanIn;
    const std = Math.sqrt(variance);
    layers.push({
      W: Array.from({ length: fanOut }, () =>
        Array.from({ length: fanIn }, () => normal() * std),
      ),
      b: Array.from({ length: fanOut }, () => normal() * 0.1),
    });
  }
  return layers;
}
