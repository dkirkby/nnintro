/**
 * The parameter sweeps behind "parameters → output".
 *
 * The invariant the panel is drawn around is that every curve passes through
 * the current output at the parameter's current value, which is what puts
 * all the dots on one line.
 */
import { describe, expect, it } from "vitest";

import { parameterCurves, parameterRefs } from "../src/model/curves.js";
import { forward } from "../src/model/evaluate.js";
import {
  cloneParameters,
  parameterSymbol,
  unitParameters,
  writeParameter,
} from "../src/model/types.js";
import { niceLimit } from "../src/view/colormap.js";
import { checkpointNamed, PRIMARY } from "./fixtures.js";

const CHECKPOINT = checkpointNamed(PRIMARY);
const ACTIVATION = CHECKPOINT.architecture.hiddenActivation!;
const PROBE = [0.649, -0.64];
/**
 * The parameter scale `main.ts` builds for this checkpoint, computed rather
 * than written down: "every current value is inside the swept range" is an
 * invariant of that scale, and a hard-coded number only tested it for
 * whichever parameters happened to be published at the time.
 */
const LIMIT = niceLimit(
  CHECKPOINT.parameters.flatMap((layer) => layer.W.flat()),
  CHECKPOINT.parameters.flatMap((layer) => layer.b),
);
/** The caller names the sweeps now; these tests want all of them. */
const ALL = parameterRefs(CHECKPOINT.parameters);

describe("parameterRefs", () => {
  it("names every weight and bias exactly once", () => {
    const refs = parameterRefs(CHECKPOINT.parameters);
    const expected = CHECKPOINT.parameters.reduce(
      (total, layer) => total + layer.W.flat().length + layer.b.length,
      0,
    );
    expect(refs).toHaveLength(expected);
    expect(new Set(refs.map((ref) => JSON.stringify(ref))).size).toBe(expected);
  });
});

describe("a unit's parameters, in the order the panel lists them", () => {
  it("PUTS THE BIAS FIRST, then the weights top to bottom as drawn", () => {
    // nnet-svg lays a layer out in index order downward, so source index 0
    // is the top node: ascending `in` is top to bottom in the diagram.
    const refs = unitParameters(CHECKPOINT.parameters, { layer: 0, out: 2 });
    expect(refs[0]).toEqual({ kind: "bias", layer: 0, out: 2 });
    expect(refs.slice(1)).toEqual(
      CHECKPOINT.parameters[0].W[2].map((_, input) => ({
        kind: "weight",
        layer: 0,
        out: 2,
        in: input,
      })),
    );
    expect(refs.map(parameterSymbol)).toEqual(["b", "w\u2080", "w\u2081"]);
  });

  it("is exactly the parameters of that one neuron, and no others", () => {
    const all = parameterRefs(CHECKPOINT.parameters).map((ref) => JSON.stringify(ref));
    const units = CHECKPOINT.parameters.flatMap((layer, index) =>
      layer.b.map((_, out) => ({ layer: index, out })),
    );
    const covered = units.flatMap((unit) =>
      unitParameters(CHECKPOINT.parameters, unit).map((ref) => JSON.stringify(ref)),
    );
    // Every parameter belongs to exactly one unit: a partition, so no
    // selection can ever leave a parameter unreachable or offer it twice.
    expect(covered.slice().sort()).toEqual(all.slice().sort());
    expect(new Set(covered).size).toBe(all.length);
  });
});

describe("parameterCurves", () => {
  it("SWEEPS ONLY THE PARAMETERS IT WAS GIVEN, in that order", () => {
    const refs = unitParameters(CHECKPOINT.parameters, { layer: 1, out: 0 });
    const curves = parameterCurves(CHECKPOINT.parameters, ACTIVATION, PROBE, LIMIT, refs, 5);
    expect(curves.map((curve) => curve.ref)).toEqual(refs);
  });

  it("sweeps each parameter across the scale with everything else held", () => {
    const curves = parameterCurves(CHECKPOINT.parameters, ACTIVATION, PROBE, LIMIT, ALL, 5);
    for (const curve of curves) {
      expect(curve.values[0]).toBeCloseTo(-LIMIT, 12);
      expect(curve.values[4]).toBeCloseTo(LIMIT, 12);
      curve.values.forEach((value, index) => {
        const altered = writeParameter(CHECKPOINT.parameters, curve.ref, value);
        expect(curve.outputs[index]).toBeCloseTo(
          forward(altered, ACTIVATION, PROBE).output,
          12,
        );
      });
    }
  });

  it("PASSES THROUGH THE CURRENT OUTPUT at every parameter's current value", () => {
    const output = forward(CHECKPOINT.parameters, ACTIVATION, PROBE).output;
    const curves = parameterCurves(CHECKPOINT.parameters, ACTIVATION, PROBE, LIMIT, ALL, 41);
    for (const curve of curves) {
      expect(curve.output).toBeCloseTo(output, 12);
      // The sweep really does reach that value where the dot is drawn:
      // interpolating the curve at `current` reproduces it.
      const index = curve.values.findIndex((value) => value >= curve.current);
      expect(index).toBeGreaterThan(0);
      const [lo, hi] = [curve.values[index - 1], curve.values[index]];
      const t = (curve.current - lo) / (hi - lo);
      const interpolated =
        curve.outputs[index - 1] + t * (curve.outputs[index] - curve.outputs[index - 1]);
      // Linear between samples except across a ReLU kink, so this is close
      // rather than exact.
      expect(interpolated).toBeCloseTo(output, 2);
    }
  });

  it("keeps every current value inside the swept range", () => {
    // The scale is built from the parameters themselves, so a dot can never
    // fall off the end of the axis.
    const curves = parameterCurves(CHECKPOINT.parameters, ACTIVATION, PROBE, LIMIT, ALL, 5);
    for (const curve of curves) {
      expect(Math.abs(curve.current)).toBeLessThanOrEqual(LIMIT);
    }
  });

  it("LEAVES THE CALLER's PARAMETERS UNTOUCHED", () => {
    // It sweeps by mutating a private copy; the explorer's own state is the
    // array handed in.
    const before = cloneParameters(CHECKPOINT.parameters);
    parameterCurves(CHECKPOINT.parameters, ACTIVATION, PROBE, LIMIT, ALL, 7);
    expect(CHECKPOINT.parameters).toEqual(before);
  });

  it("bends where a ReLU switches, instead of running straight", () => {
    // A first-layer bias drives its unit through zero somewhere in the
    // sweep, so its curve is not a single straight line.
    const curves = parameterCurves(CHECKPOINT.parameters, ACTIVATION, PROBE, LIMIT, ALL, 101);
    const bent = curves.filter((curve) => {
      const slopes = curve.outputs
        .slice(1)
        .map((value, index) => value - curve.outputs[index]);
      const first = slopes.find((slope) => Math.abs(slope) > 1e-12) ?? 0;
      return slopes.some((slope) => Math.abs(slope - first) > 1e-9);
    });
    expect(bent.length).toBeGreaterThan(0);
  });
});
