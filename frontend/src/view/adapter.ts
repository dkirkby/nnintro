/**
 * The single place that translates between model coordinates and the
 * renderer's.
 *
 * **nnet-svg indexes edges source-first** (`sourceLayerIndex`,
 * `sourceNodeIndex` -> `targetNodeIndex`), while canonical model storage is
 * destination-first, `W[out][in]`, matching `torch.nn.Linear`. Every
 * crossing of that boundary goes through this module, and
 * `tests/adapter.test.ts` pins it with an asymmetric matrix where a
 * transpose would otherwise go unnoticed (PLAN.md section 7).
 *
 * Layer indexing also differs by one. The renderer counts activation layers
 * (0 is the input), while parameters are indexed by the gap between them:
 * activation layer `l + 1` is produced by parameter layer `l`.
 */

import type { NodeRef } from "nnet-svg";

import {
  unitOf,
  type LayerParameters,
  type ParameterRef,
  type UnitRef,
} from "../model/types.js";

/** The weight an edge displays: `W[target][source]` of the source layer. */
export function edgeParameter(edge: {
  sourceLayerIndex: number;
  sourceNodeIndex: number;
  targetNodeIndex: number;
}): Extract<ParameterRef, { kind: "weight" }> {
  return {
    kind: "weight",
    layer: edge.sourceLayerIndex,
    out: edge.targetNodeIndex,
    in: edge.sourceNodeIndex,
  };
}

/** The bias a node displays, or null for an input node, which has none. */
export function nodeParameter(node: {
  layerIndex: number;
  nodeIndex: number;
}): ParameterRef | null {
  if (node.layerIndex === 0) return null;
  return { kind: "bias", layer: node.layerIndex - 1, out: node.nodeIndex };
}

/**
 * The unit an edge feeds, which is what selecting that edge selects: the
 * destination neuron, with its bias and every other weight into it.
 */
export function edgeUnit(edge: {
  sourceLayerIndex: number;
  targetNodeIndex: number;
}): UnitRef {
  return { layer: edge.sourceLayerIndex, out: edge.targetNodeIndex };
}

/**
 * The unit a node *is*, or null for an input node, which computes nothing
 * and so owns no parameters.
 */
export function nodeUnit(node: {
  layerIndex: number;
  nodeIndex: number;
}): UnitRef | null {
  const ref = nodeParameter(node);
  return ref === null ? null : unitOf(ref);
}

/** An edge as the renderer and the layout both describe it. */
export type EdgeCoords = {
  sourceLayerIndex: number;
  sourceNodeIndex: number;
  targetNodeIndex: number;
};

/** The value the renderer should show on an edge. */
export function edgeWeight(
  parameters: LayerParameters[],
  edge: EdgeCoords,
): number | null {
  const ref = edgeParameter(edge);
  const layer = parameters[ref.layer];
  if (!layer) return null;
  // Destination-first lookup: row = destination unit, column = source unit.
  return layer.W[ref.out]?.[ref.in] ?? null;
}

/** The value the renderer should show on a node. */
export function nodeBias(
  parameters: LayerParameters[],
  node: NodeRef,
): number | null {
  const ref = nodeParameter(node);
  if (!ref) return null;
  return parameters[ref.layer]?.b[ref.out] ?? null;
}

/** The weighted contribution `w * a` an edge carries for the current input. */
export function edgeContribution(
  parameters: LayerParameters[],
  activations: number[][],
  edge: EdgeCoords,
): number | null {
  const weight = edgeWeight(parameters, edge);
  if (weight === null) return null;
  const source = activations[edge.sourceLayerIndex]?.[edge.sourceNodeIndex];
  return source === undefined ? null : weight * source;
}

/** A stable key for an edge, for selection and animation bookkeeping. */
export function edgeKey(edge: EdgeCoords): string {
  return `e:${edge.sourceLayerIndex}:${edge.sourceNodeIndex}:${edge.targetNodeIndex}`;
}

export function nodeKey(node: { layerIndex: number; nodeIndex: number }): string {
  return `n:${node.layerIndex}:${node.nodeIndex}`;
}

/** Edge key from model coordinates, for the ripple overlay. */
export function edgeKeyFromModel(
  parameterLayer: number,
  source: number,
  target: number,
): string {
  return `e:${parameterLayer}:${source}:${target}`;
}
