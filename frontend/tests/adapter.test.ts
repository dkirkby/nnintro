/**
 * The transpose test.
 *
 * nnet-svg indexes edges source-first; the model stores W[out][in]. A
 * square, symmetric matrix would hide a transpose completely, so every
 * assertion here uses a non-square matrix with distinct entries.
 */
import { describe, expect, it } from "vitest";

import {
  edgeContribution,
  edgeKey,
  edgeKeyFromModel,
  edgeParameter,
  edgeWeight,
  nodeBias,
  nodeParameter,
} from "../src/view/adapter.js";
import type { LayerParameters } from "../src/model/types.js";
import { forward } from "../src/model/evaluate.js";
import { allCheckpoints } from "./fixtures.js";

/** 3 inputs -> 2 units. Every entry distinct; shape is not square. */
const PARAMETERS: LayerParameters[] = [
  {
    W: [
      [1, 2, 3], // unit 0 reads inputs 0,1,2
      [4, 5, 6], // unit 1
    ],
    b: [10, 20],
  },
  { W: [[7, 8]], b: [30] },
];

function edge(sourceLayerIndex: number, sourceNodeIndex: number, targetNodeIndex: number) {
  return {
    kind: "edge" as const,
    sourceLayerIndex,
    sourceNodeIndex,
    targetLayerIndex: sourceLayerIndex + 1,
    targetNodeIndex,
    edgeIndex: 0,
  };
}

describe("edge index translation", () => {
  it("maps a source-first edge onto a destination-first parameter", () => {
    expect(edgeParameter(edge(0, 2, 1))).toEqual({
      kind: "weight",
      layer: 0,
      out: 1, // target
      in: 2, // source
    });
  });

  it("reads the weight the edge actually represents", () => {
    // Input 2 -> unit 1 is W[1][2] = 6. A transpose would read W[2][1],
    // which does not even exist in this non-square matrix.
    expect(edgeWeight(PARAMETERS, edge(0, 2, 1))).toBe(6);
    expect(edgeWeight(PARAMETERS, edge(0, 1, 0))).toBe(2);
    expect(edgeWeight(PARAMETERS, edge(0, 0, 1))).toBe(4);
  });

  it("returns null rather than a wrong number for an out-of-range edge", () => {
    expect(edgeWeight(PARAMETERS, edge(0, 9, 0))).toBeNull();
    expect(edgeWeight(PARAMETERS, edge(5, 0, 0))).toBeNull();
  });

  it("every edge of the layer maps to a distinct weight", () => {
    const seen = new Set<number>();
    for (let source = 0; source < 3; source += 1) {
      for (let target = 0; target < 2; target += 1) {
        const value = edgeWeight(PARAMETERS, edge(0, source, target));
        expect(value).not.toBeNull();
        seen.add(value!);
      }
    }
    expect(seen.size).toBe(6); // a transpose would collide or go out of range
  });

  it("agrees with the key built from model coordinates", () => {
    expect(edgeKey(edge(0, 2, 1))).toBe(edgeKeyFromModel(0, 2, 1));
  });
});

describe("node index translation", () => {
  it("offsets activation layers against parameter layers", () => {
    expect(nodeParameter({ layerIndex: 1, nodeIndex: 1 })).toEqual({
      kind: "bias",
      layer: 0,
      out: 1,
    });
    expect(nodeParameter({ layerIndex: 2, nodeIndex: 0 })).toEqual({
      kind: "bias",
      layer: 1,
      out: 0,
    });
  });

  it("gives the input layer no bias, rather than layer -1", () => {
    expect(nodeParameter({ layerIndex: 0, nodeIndex: 0 })).toBeNull();
    expect(nodeBias(PARAMETERS, { kind: "node", layerIndex: 0, nodeIndex: 0 })).toBeNull();
  });

  it("reads the bias a node represents", () => {
    expect(nodeBias(PARAMETERS, { kind: "node", layerIndex: 1, nodeIndex: 1 })).toBe(20);
    expect(nodeBias(PARAMETERS, { kind: "node", layerIndex: 2, nodeIndex: 0 })).toBe(30);
  });
});

describe("contributions", () => {
  it("multiplies the weight by the SOURCE activation", () => {
    const activations = [[1, 10, 100], [0, 0], [0]];
    // Input 1 (value 10) -> unit 0: W[0][1] = 2, so 20.
    expect(edgeContribution(PARAMETERS, activations, edge(0, 1, 0))).toBe(20);
    // Input 2 (value 100) -> unit 1: W[1][2] = 6, so 600.
    expect(edgeContribution(PARAMETERS, activations, edge(0, 2, 1))).toBe(600);
  });

  it("contributions plus the bias reconstruct the preactivation", () => {
    // The identity the inspector shows students: z = sum(w*a) + b. If the
    // adapter transposed, this would not close.
    for (const checkpoint of allCheckpoints()) {
      const parameters = checkpoint.parameters;
      const probe = checkpoint.goldenProbes[0];
      const trace = forward(
        parameters,
        checkpoint.architecture.hiddenActivation ?? "identity",
        probe.x,
      );
      parameters.forEach((layer, parameterLayer) => {
        layer.b.forEach((bias, unit) => {
          let sum = bias;
          for (let source = 0; source < layer.W[unit].length; source += 1) {
            sum += edgeContribution(parameters, trace.act, edge(parameterLayer, source, unit))!;
          }
          expect(sum).toBeCloseTo(trace.pre[parameterLayer + 1][unit], 12);
        });
      });
    }
  });
});
