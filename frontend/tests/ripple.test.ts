/**
 * The ripple must distinguish "could have changed" from "did change".
 */
import { describe, expect, it } from "vitest";

import { forward } from "../src/model/evaluate.js";
import { computeRipple, originOf } from "../src/model/ripple.js";
import { writeParameter, type LayerParameters } from "../src/model/types.js";

/** A 2-2-1 ReLU net whose second hidden unit is firmly off at the probe. */
const PARAMETERS: LayerParameters[] = [
  { W: [[1, 0], [1, 0]], b: [0, -5] },
  { W: [[1, 1]], b: [0] },
];
const X = [1, 0];

describe("computeRipple", () => {
  it("reports a change that propagates all the way to the output", () => {
    const ref = { kind: "bias", layer: 0, out: 0 } as const;
    const before = forward(PARAMETERS, "relu", X);
    const after = forward(writeParameter(PARAMETERS, ref, 0.5), "relu", X);
    const ripple = computeRipple(before, after, { ...originOf(ref), kind: "parameter" });

    expect(ripple.changedCount).toBe(ripple.reachableCount);
    expect(ripple.hasBlocked).toBe(false);
    expect(ripple.stages.at(-1)!.nodes[0].delta).toBeCloseTo(0.5, 12);
  });

  it("reports a reachable node that a dead ReLU blocked", () => {
    // Unit 1 has preactivation -4: nudging its bias leaves it clamped at 0,
    // so the output is structurally downstream but numerically unmoved.
    const ref = { kind: "bias", layer: 0, out: 1 } as const;
    const before = forward(PARAMETERS, "relu", X);
    const after = forward(writeParameter(PARAMETERS, ref, -4.5), "relu", X);
    const ripple = computeRipple(before, after, { ...originOf(ref), kind: "parameter" });

    expect(ripple.hasBlocked).toBe(true);
    expect(ripple.changedCount).toBe(0);
    // The output node is still reported as reachable, not omitted.
    expect(ripple.stages.at(-1)!.nodes).toHaveLength(1);
    expect(ripple.stages.at(-1)!.nodes[0].changed).toBe(false);
    // And the edge into it carried nothing.
    expect(ripple.stages.at(-1)!.edges.every((edge) => !edge.carried)).toBe(true);
  });

  it("reports cancellation as unchanged, not as unreachable", () => {
    // Two hidden units with opposite output weights: equal nudges cancel.
    const parameters: LayerParameters[] = [
      { W: [[1, 0], [1, 0]], b: [1, 1] },
      { W: [[1, -1]], b: [0] },
    ];
    const before = forward(parameters, "relu", X);
    const edited = writeParameter(parameters, { kind: "weight", layer: 1, out: 0, in: 0 }, 2);
    const after = forward(
      writeParameter(edited, { kind: "weight", layer: 1, out: 0, in: 1 }, -2),
      "relu",
      X,
    );
    const ripple = computeRipple(before, after, {
      layer: 2,
      index: 0,
      kind: "parameter",
    });
    expect(ripple.stages.at(-1)!.nodes[0].changed).toBe(false);
    expect(ripple.hasBlocked).toBe(true);
  });

  it("starts an input edit at the input node", () => {
    const before = forward(PARAMETERS, "relu", X);
    const after = forward(PARAMETERS, "relu", [1.5, 0]);
    const ripple = computeRipple(before, after, { layer: 0, index: 0, kind: "input" });
    expect(ripple.stages[0].layer).toBe(0);
    expect(ripple.stages[0].nodes[0].delta).toBeCloseTo(0.5, 12);
    expect(ripple.stages).toHaveLength(3); // input, hidden, output
  });

  it("an edit's ripple starts at its destination neuron, not the input", () => {
    expect(originOf({ kind: "weight", layer: 1, out: 0, in: 3 })).toEqual({
      layer: 2,
      index: 0,
    });
    expect(originOf({ kind: "bias", layer: 0, out: 2 })).toEqual({
      layer: 1,
      index: 2,
    });
  });
});
