/**
 * The prediction surface. Its orientation is the one thing a view cannot
 * check for itself: row 0 is the top of the plot, which is the LARGEST second
 * input, so getting it backwards flips the picture silently.
 */
import { describe, expect, it } from "vitest";

import { forward } from "../src/model/evaluate.js";
import { outputSlice, outputSurface } from "../src/model/surface.js";
import { checkpointNamed, PRIMARY } from "./fixtures.js";

const CHECKPOINT = checkpointNamed(PRIMARY);
const DOMAIN = CHECKPOINT.dataset.domain;
const ACTIVATION = CHECKPOINT.architecture.hiddenActivation!;

describe("outputSurface", () => {
  it("evaluates the network at every cell centre", () => {
    const surface = outputSurface(CHECKPOINT.parameters, ACTIVATION, DOMAIN, 4);
    expect(surface.values).toHaveLength(16);
    const [[x1lo, x1hi], [x2lo, x2hi]] = DOMAIN;
    for (let row = 0; row < 4; row += 1) {
      for (let column = 0; column < 4; column += 1) {
        const x = [
          x1lo + ((column + 0.5) / 4) * (x1hi - x1lo),
          x2hi - ((row + 0.5) / 4) * (x2hi - x2lo),
        ];
        const expected = forward(CHECKPOINT.parameters, ACTIVATION, x).output;
        expect(surface.values[row * 4 + column]).toBeCloseTo(expected, 12);
      }
    }
  });

  it("puts the largest second input in row 0, which is the top of the plot", () => {
    const surface = outputSurface(CHECKPOINT.parameters, ACTIVATION, DOMAIN, 8);
    const last = surface.resolution - 1;
    // The tilted plane falls with x2 (-0.4), so the top row predicts less
    // than the bottom row at the same column.
    expect(surface.values[0]).toBeLessThan(surface.values[last * 8]);
  });

  it("reports an extent that covers the corners the grid misses", () => {
    const surface = outputSurface(CHECKPOINT.parameters, ACTIVATION, DOMAIN, 20);
    const widest = Math.max(...surface.values.map(Math.abs));
    expect(surface.extent).toBeGreaterThanOrEqual(widest);
    const corner = forward(CHECKPOINT.parameters, ACTIVATION, [DOMAIN[0][1], DOMAIN[1][0]]);
    expect(surface.extent).toBeGreaterThanOrEqual(Math.abs(corner.output) - 1e-12);
  });

  it("covers the prediction at every golden probe, so the surface never clips", () => {
    // The colour scale is built from `extent` and never consults the probe;
    // that is only safe if the grid and the corners really do bound it.
    const surface = outputSurface(CHECKPOINT.parameters, ACTIVATION, DOMAIN, 60);
    for (const probe of CHECKPOINT.goldenProbes) {
      expect(surface.extent).toBeGreaterThanOrEqual(Math.abs(probe.output) - 1e-9);
    }
  });

  it("refuses a domain it cannot draw", () => {
    expect(() => outputSurface(CHECKPOINT.parameters, ACTIVATION, [[0, 1]], 4)).toThrow(
      /exactly 2 inputs/,
    );
  });
});

describe("outputSlice", () => {
  const PROBE = [0.649, -0.64];

  it("sweeps one input and holds the other where the probe left it", () => {
    const slice = outputSlice(CHECKPOINT.parameters, ACTIVATION, DOMAIN, PROBE, 0, 9);
    expect(slice.inputs).toHaveLength(9);
    expect(slice.inputs[0]).toBeCloseTo(DOMAIN[0][0], 12);
    expect(slice.inputs[8]).toBeCloseTo(DOMAIN[0][1], 12);
    slice.inputs.forEach((x1, index) => {
      const expected = forward(CHECKPOINT.parameters, ACTIVATION, [x1, PROBE[1]]).output;
      expect(slice.values[index]).toBeCloseTo(expected, 12);
    });
  });

  it("sweeps the second input just as readily", () => {
    const slice = outputSlice(CHECKPOINT.parameters, ACTIVATION, DOMAIN, PROBE, 1, 9);
    expect(slice.at).toEqual(PROBE);
    slice.inputs.forEach((x2, index) => {
      const expected = forward(CHECKPOINT.parameters, ACTIVATION, [PROBE[0], x2]).output;
      expect(slice.values[index]).toBeCloseTo(expected, 12);
    });
  });

  it("CROSSES THE OTHER SLICE AT THE PROBE, which is what makes the pair readable", () => {
    // Both plots mark the probe at the same height; if one slice held the
    // wrong input they would disagree there.
    const at = [0.5, -0.25];
    const horizontal = outputSlice(CHECKPOINT.parameters, ACTIVATION, DOMAIN, at, 0, 201);
    const vertical = outputSlice(CHECKPOINT.parameters, ACTIVATION, DOMAIN, at, 1, 201);
    const output = forward(CHECKPOINT.parameters, ACTIVATION, at).output;
    const nearest = (slice: { inputs: number[]; values: number[] }, target: number) =>
      slice.values[
        slice.inputs.reduce(
          (best, x, i) => (Math.abs(x - target) < Math.abs(slice.inputs[best] - target) ? i : best),
          0,
        )
      ];
    expect(nearest(horizontal, at[0])).toBeCloseTo(output, 2);
    expect(nearest(vertical, at[1])).toBeCloseTo(output, 2);
  });

  it("refuses an axis the domain does not have", () => {
    expect(() => outputSlice(CHECKPOINT.parameters, ACTIVATION, DOMAIN, [0, 0], 2)).toThrow(
      /no axis 2/,
    );
  });
});
