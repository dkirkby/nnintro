/**
 * What a single edit actually changed, layer by layer.
 *
 * This is the data behind the downstream animation, and it is computed from
 * the before/after snapshots rather than from the animation's progress. The
 * distinction the explorer must not blur: a node can be *structurally*
 * downstream of an edit and still not change value, because a ReLU blocked
 * the change or because contributions cancelled. The ripple shows both, and
 * never implies that everything reachable moved (PLAN.md section 4).
 */

import type { Trace } from "./evaluate.js";
import type { ParameterRef } from "./types.js";

/** Nodes are addressed by layer (0 = input) and index within the layer. */
export interface NodeKey {
  layer: number;
  index: number;
}

export interface RippleNode extends NodeKey {
  before: number;
  after: number;
  delta: number;
  /** True when the value actually moved, not merely could have. */
  changed: boolean;
}

export interface RippleEdge {
  /** Parameter-layer index: the edge from layer `layer` to `layer + 1`. */
  layer: number;
  source: number;
  target: number;
  /** True when the source node's value changed, so something travelled. */
  carried: boolean;
}

export interface RippleStage {
  /** The activation layer this stage lands on (1 = first hidden layer). */
  layer: number;
  edges: RippleEdge[];
  nodes: RippleNode[];
}

export interface Ripple {
  origin: { layer: number; index: number; kind: "parameter" | "input" };
  stages: RippleStage[];
  /** Every node downstream of the origin, whether or not it changed. */
  reachableCount: number;
  /** Those whose value actually moved. */
  changedCount: number;
  /** True when at least one reachable node did not move. */
  hasBlocked: boolean;
}

/**
 * Values this close together are treated as unchanged.
 *
 * Exactly zero would be defensible, but a change of 1e-18 is not a change a
 * student can see or a figure can encode, and calling it one would make the
 * "did this actually move?" reading meaningless.
 */
export const CHANGE_TOLERANCE = 1e-12;

/** Where a ripple starts: an edit's destination neuron, or an input node. */
export function originOf(ref: ParameterRef): { layer: number; index: number } {
  // A weight or bias edit on parameter-layer l first affects activation
  // layer l + 1, at the destination unit.
  return { layer: ref.layer + 1, index: ref.out };
}

export function computeRipple(
  before: Trace,
  after: Trace,
  origin: { layer: number; index: number; kind: "parameter" | "input" },
): Ripple {
  const stages: RippleStage[] = [];
  let reachableCount = 0;
  let changedCount = 0;

  // The nodes that are structurally live at the previous layer: the origin
  // to begin with, then every node of each subsequent layer.
  let liveSources: number[] = [origin.index];

  for (let layer = origin.layer; layer < after.act.length; layer += 1) {
    const nodes: RippleNode[] = after.act[layer].map((value, index) => {
      const previous = before.act[layer][index];
      const delta = value - previous;
      return {
        layer,
        index,
        before: previous,
        after: value,
        delta,
        changed: Math.abs(delta) > CHANGE_TOLERANCE,
      };
    });

    const stageNodes =
      layer === origin.layer
        ? nodes.filter((node) => node.index === origin.index)
        : nodes;

    const edges: RippleEdge[] =
      layer === origin.layer
        ? []
        : liveSources.flatMap((source) =>
            stageNodes.map((node) => ({
              layer: layer - 1,
              source,
              target: node.index,
              carried:
                Math.abs(
                  after.act[layer - 1][source] - before.act[layer - 1][source],
                ) > CHANGE_TOLERANCE,
            })),
          );

    stages.push({ layer, edges, nodes: stageNodes });
    reachableCount += stageNodes.length;
    changedCount += stageNodes.filter((node) => node.changed).length;
    liveSources = stageNodes.map((node) => node.index);
  }

  return {
    origin,
    stages,
    reachableCount,
    changedCount,
    hasBlocked: changedCount < reachableCount,
  };
}

/** A short, honest summary for the status line and the reduced-motion view. */
export function describeRipple(ripple: Ripple): string {
  const { changedCount, reachableCount } = ripple;
  if (changedCount === reachableCount) {
    return `${changedCount} of ${reachableCount} downstream values changed`;
  }
  const blocked = reachableCount - changedCount;
  return (
    `${changedCount} of ${reachableCount} downstream values changed; ` +
    `${blocked} unchanged (blocked or cancelled)`
  );
}
