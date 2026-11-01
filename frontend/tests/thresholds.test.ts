/**
 * The thresholds: where each first-hidden-layer unit's preactivation is
 * zero, as a segment of the input domain.
 *
 * The claim the panel makes by drawing them is that the preactivation
 * really is zero along the line, so that is what these check — on hand-made
 * parameters where the answer is obvious, and then on every checkpoint in
 * the published snapshot through the forward pass itself.
 */
import { describe, expect, it } from "vitest";

import { forward } from "../src/model/evaluate.js";
import { thresholdLines } from "../src/model/thresholds.js";
import type { LayerParameters } from "../src/model/types.js";
import { allCheckpoints } from "./fixtures.js";

const SQUARE: [number, number][] = [
  [-1, 1],
  [-1, 1],
];

/** A network of one hidden layer with the given rows, and an output that
 *  reads them: `thresholdLines` needs the second layer to know the first
 *  one is hidden. */
function net(W: number[][], b: number[]): LayerParameters[] {
  return [
    { W, b },
    { W: [W.map(() => 1)], b: [0] },
  ];
}

describe("thresholdLines", () => {
  it("DRAWS A UNIT'S ZERO LINE ACROSS THE DOMAIN", () => {
    // x1 = 0: up the middle, from the bottom edge to the top one.
    expect(thresholdLines(net([[1, 0]], [0]), SQUARE)).toEqual([
      { unit: 0, from: [0, -1], to: [0, 1] },
    ]);
    // 2 x2 - 1 = 0: across at x2 = 0.5.
    const [across] = thresholdLines(net([[0, 2]], [-1]), SQUARE);
    expect(across.from[1]).toBeCloseTo(0.5, 12);
    expect(across.to[1]).toBeCloseTo(0.5, 12);
    expect([across.from[0], across.to[0]].sort()).toEqual([-1, 1]);
  });

  it("clips a diagonal to the square it crosses", () => {
    // x1 + x2 = 0 is the other diagonal of the square, corner to corner.
    const [diagonal] = thresholdLines(net([[1, 1]], [0]), SQUARE);
    const ends = [diagonal.from, diagonal.to].map(([x1, x2]) => `${x1},${x2}`).sort();
    expect(ends).toEqual(["-1,1", "1,-1"]);
    // x1 + x2 = 1 cuts the corner: half a diagonal, inside the square.
    const [cut] = thresholdLines(net([[1, 1]], [-1]), SQUARE);
    for (const [x1, x2] of [cut.from, cut.to]) {
      expect(x1).toBeGreaterThanOrEqual(-1);
      expect(x1).toBeLessThanOrEqual(1);
      expect(x1 + x2).toBeCloseTo(1, 12);
    }
  });

  it("LEAVES OUT A UNIT WITH NOTHING TO DRAW, rather than reporting an empty line", () => {
    // Its threshold is at x1 = 5, nowhere near the domain.
    expect(thresholdLines(net([[1, 0]], [-5]), SQUARE)).toEqual([]);
    // No input weights at all: the preactivation is the constant bias, so
    // there is no line anywhere — not even when that constant is zero.
    expect(thresholdLines(net([[0, 0]], [0.3]), SQUARE)).toEqual([]);
    expect(thresholdLines(net([[0, 0]], [0]), SQUARE)).toEqual([]);
    // Through one corner exactly: a point, not a segment, and a stray dot
    // on the map would say less than nothing.
    expect(thresholdLines(net([[1, 1]], [2]), SQUARE)).toEqual([]);
  });

  it("keeps each line with its own unit, gaps included", () => {
    const lines = thresholdLines(
      net(
        [
          [1, 0],
          [0, 0],
          [0, 1],
        ],
        [0, 0, 0],
      ),
      SQUARE,
    );
    expect(lines.map((line) => line.unit)).toEqual([0, 2]);
  });

  it("HAS NO THRESHOLDS WITHOUT A HIDDEN LAYER", () => {
    // One parameter layer is the output: its own zero line is the
    // prediction's, which the map already draws as a colour.
    expect(thresholdLines([{ W: [[1, 1]], b: [0] }], SQUARE)).toEqual([]);
  });

  it("rejects a domain that is not two inputs", () => {
    expect(() => thresholdLines(net([[1, 0]], [0]), [[-1, 1]])).toThrow();
  });

  it("PUTS THE PREACTIVATION AT ZERO ALONG EVERY LINE, for every checkpoint", () => {
    let drawn = 0;
    for (const checkpoint of allCheckpoints()) {
      const activation = checkpoint.architecture.hiddenActivation;
      const lines = thresholdLines(checkpoint.parameters, checkpoint.dataset.domain);
      if (checkpoint.architecture.layerSizes.length < 3) {
        expect(lines).toEqual([]);
        continue;
      }
      for (const { unit, from, to } of lines) {
        drawn += 1;
        // Both ends and the middle: three points of a straight line is the
        // whole line, and the middle is the one a clipping mistake would
        // put outside the domain.
        for (const t of [0, 0.5, 1]) {
          const x = [from[0] + t * (to[0] - from[0]), from[1] + t * (to[1] - from[1])];
          for (const [axis, [lo, hi]] of checkpoint.dataset.domain.entries()) {
            expect(x[axis]).toBeGreaterThanOrEqual(lo - 1e-9);
            expect(x[axis]).toBeLessThanOrEqual(hi + 1e-9);
          }
          const trace = forward(checkpoint.parameters, activation!, x);
          expect(trace.pre[1][unit]).toBeCloseTo(0, 9);
        }
      }
    }
    // The catalog is 76 checkpoints, nearly all of them with a hidden
    // layer: if this ever drops to nothing the assertion above is vacuous.
    expect(drawn).toBeGreaterThan(100);
  });
});
