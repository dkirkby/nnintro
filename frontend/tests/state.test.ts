/**
 * The interaction contract: an edit preserves the selection, the baseline
 * stays immutable, provenance is honest, and a restore is exact.
 */
import { beforeEach, describe, expect, it } from "vitest";

import {
  createExplorer,
  describeProvenance,
  OPENING_PROBE,
  type Explorer,
} from "../src/state.js";
import {
  readParameter,
  unitOf,
  type ParameterRef,
  type UnitRef,
} from "../src/model/types.js";
import { allCheckpoints, checkpointNamed, PRIMARY } from "./fixtures.js";

const CHECKPOINT = checkpointNamed(PRIMARY);
const WEIGHT: ParameterRef = { kind: "weight", layer: 0, out: 1, in: 0 };
const BIAS: ParameterRef = { kind: "bias", layer: 0, out: 2 };
/** The unit each of those belongs to: what a click on either now selects. */
const WEIGHT_UNIT: UnitRef = unitOf(WEIGHT);
const BIAS_UNIT: UnitRef = unitOf(BIAS);
const OUTPUT_UNIT: UnitRef = { layer: CHECKPOINT.parameters.length - 1, out: 0 };

describe("createExplorer", () => {
  let explorer: Explorer;
  beforeEach(() => {
    explorer = createExplorer(CHECKPOINT);
  });

  it("starts from the pre-trained checkpoint and says so", () => {
    expect(describeProvenance(explorer.snapshot().provenance)).toContain("Pre-trained");
    expect(explorer.snapshot().parameters).toEqual(CHECKPOINT.parameters);
  });

  it("reproduces the checkpoint's published losses", () => {
    expect(explorer.snapshot().train.meanLoss).toBeCloseTo(CHECKPOINT.metrics.trainLoss, 12);
    expect(explorer.snapshot().test.meanLoss).toBeCloseTo(CHECKPOINT.metrics.testLoss, 12);
  });

  it("OPENS ON THE OUTPUT UNIT, never on nothing", () => {
    // There is no empty selection and no way to deselect: the explorer
    // starts on the parameters nearest the prediction.
    expect(explorer.snapshot().selection).toEqual(OUTPUT_UNIT);
  });

  it("OPENS WITH THE PROBE OFF THE ORIGIN, at (0.5, 0.5) here", () => {
    // The map draws its axes through the origin; a probe there would sit on
    // them, and would zero every input term of the first layer's sums.
    expect(explorer.snapshot().x).toEqual([OPENING_PROBE, OPENING_PROBE]);
    expect(explorer.snapshot().probe.kind).toBe("free");
  });

  // The Stage 0 exit condition.
  it("PRESERVES THE SELECTION across a parameter edit", () => {
    explorer.selectUnit(WEIGHT_UNIT);
    explorer.setParameter(WEIGHT, 0.5);
    expect(explorer.snapshot().selection).toEqual(WEIGHT_UNIT);
    explorer.setParameter(BIAS, -0.25);
    expect(explorer.snapshot().selection).toEqual(WEIGHT_UNIT);
  });

  it("preserves the selection across a restore", () => {
    explorer.selectUnit(BIAS_UNIT);
    explorer.setParameter(BIAS, 1.5);
    explorer.restoreParameter(BIAS);
    expect(explorer.snapshot().selection).toEqual(BIAS_UNIT);
    explorer.restoreBaseline();
    expect(explorer.snapshot().selection).toEqual(BIAS_UNIT);
  });

  it("preserves the selection when the probe moves", () => {
    explorer.selectUnit(WEIGHT_UNIT);
    explorer.setProbe({ kind: "free", x: [0.2, -0.4] });
    expect(explorer.snapshot().selection).toEqual(WEIGHT_UNIT);
  });

  it("selects one unit whichever of its parameters was clicked", () => {
    // A weight and the bias of the same neuron are the same selection: an
    // edge selects the unit it feeds, with every other edge into it.
    explorer.selectUnit(unitOf({ kind: "weight", layer: 0, out: 2, in: 1 }));
    expect(explorer.snapshot().selection).toEqual(BIAS_UNIT);
  });

  it("never mutates the checkpoint it was built from", () => {
    const before = JSON.stringify(CHECKPOINT.parameters);
    explorer.setParameter(WEIGHT, 99);
    explorer.setParameter(BIAS, -99);
    expect(JSON.stringify(CHECKPOINT.parameters)).toBe(before);
  });

  it("recomputes derivatives after an edit rather than reusing the asset's", () => {
    const before = explorer.snapshot().train.gradient[0].W[1][0];
    explorer.setParameter(WEIGHT, readParameter(CHECKPOINT.parameters, WEIGHT) + 0.5);
    expect(explorer.snapshot().train.gradient[0].W[1][0]).not.toBe(before);
  });

  it("bumps the version on every change, so stale work is detectable", () => {
    const start = explorer.snapshot().version;
    explorer.selectUnit(WEIGHT_UNIT);
    explorer.setParameter(WEIGHT, 0.1);
    expect(explorer.snapshot().version).toBeGreaterThan(start);
  });

  it("reports provenance as modified, counting the edits", () => {
    explorer.setParameter(WEIGHT, 0.1);
    explorer.setParameter(BIAS, 0.2);
    const text = describeProvenance(explorer.snapshot().provenance);
    expect(text).toContain("Modified from");
    expect(text).toContain("Pre-trained");
    expect(text).toContain("2 edits");
  });

  it("rejects a non-finite value rather than storing it", () => {
    expect(() => explorer.setParameter(WEIGHT, Number.NaN)).toThrow(RangeError);
    expect(() => explorer.setParameter(WEIGHT, Number.POSITIVE_INFINITY)).toThrow(RangeError);
    expect(explorer.snapshot().parameters).toEqual(CHECKPOINT.parameters);
  });

  it("restores one parameter without disturbing the others", () => {
    explorer.setParameter(WEIGHT, 1.25);
    explorer.setParameter(BIAS, -1.25);
    explorer.restoreParameter(WEIGHT);
    expect(readParameter(explorer.snapshot().parameters, WEIGHT)).toBe(
      readParameter(CHECKPOINT.parameters, WEIGHT),
    );
    expect(readParameter(explorer.snapshot().parameters, BIAS)).toBe(-1.25);
  });

  it("restore-baseline returns to the checkpoint exactly", () => {
    explorer.setParameter(WEIGHT, 1.25);
    explorer.restoreBaseline();
    expect(explorer.snapshot().parameters).toEqual(CHECKPOINT.parameters);
  });

  it("randomize makes a new baseline that restore returns to, not the checkpoint", () => {
    explorer.randomize(42);
    const randomized = JSON.stringify(explorer.snapshot().parameters);
    expect(describeProvenance(explorer.snapshot().provenance)).toBe("Random (seed 42)");
    explorer.setParameter(WEIGHT, 3);
    explorer.restoreBaseline();
    expect(JSON.stringify(explorer.snapshot().parameters)).toBe(randomized);
  });

  it("randomize is deterministic for a given seed", () => {
    explorer.randomize(7);
    const first = JSON.stringify(explorer.snapshot().parameters);
    const other = createExplorer(CHECKPOINT);
    other.randomize(7);
    expect(JSON.stringify(other.snapshot().parameters)).toBe(first);
  });

  it("keeps the previous state for before/after comparison", () => {
    explorer.setProbe({ kind: "sample", index: 0 });
    const before = explorer.snapshot().pointLoss;
    explorer.setParameter(BIAS, readParameter(CHECKPOINT.parameters, BIAS) + 0.3);
    const edit = explorer.lastEdit();
    expect(edit).not.toBeNull();
    expect(edit!.previous.pointLoss).toBe(before);
    expect(edit!.after - edit!.before).toBeCloseTo(0.3, 12);
  });

  it("a free probe in an analytic dataset has a target; the loss is real", () => {
    explorer.setProbe({ kind: "free", x: [0.5, -0.5] });
    const snapshot = explorer.snapshot();
    expect(snapshot.target).toBeCloseTo(0.5 * 0.5 - 0.3 * -0.5 + 0.2, 12);
    expect(snapshot.pointLoss).not.toBeNull();
  });

  it("notifies subscribers and can be unsubscribed", () => {
    let count = 0;
    const stop = explorer.subscribe(() => {
      count += 1;
    });
    explorer.setParameter(WEIGHT, 0.75);
    expect(count).toBe(1);
    stop();
    explorer.setParameter(WEIGHT, 0.8);
    expect(count).toBe(1);
  });
});

describe("edit records as subscribers see them", () => {
  it("lastEdit() is already populated when subscribers run", () => {
    // The views build the ripple and the predicted-vs-actual comparison
    // inside the subscription, so an edit recorded after the notification
    // would leave both silently missing.
    const explorer = createExplorer(CHECKPOINT);
    const seen: (ReturnType<Explorer["lastEdit"]>)[] = [];
    explorer.subscribe(() => seen.push(explorer.lastEdit()));
    explorer.setParameter(BIAS, 0.75);
    expect(seen).toHaveLength(1);
    expect(seen[0]).not.toBeNull();
    expect(seen[0]!.ref).toEqual(BIAS);
    expect(seen[0]!.after).toBe(0.75);
    expect(seen[0]!.previous.pointLoss).not.toBeNull();
  });

  it("clears the edit when the change was not an edit", () => {
    const explorer = createExplorer(CHECKPOINT);
    explorer.setParameter(BIAS, 0.75);
    const seen: (ReturnType<Explorer["lastEdit"]>)[] = [];
    explorer.subscribe(() => seen.push(explorer.lastEdit()));
    explorer.selectUnit(WEIGHT_UNIT);
    explorer.setProbe({ kind: "free", x: [0.2, -0.4] });
    expect(seen).toEqual([null, null]);
  });

  it("restoreParameter reports itself as an edit too", () => {
    const explorer = createExplorer(CHECKPOINT);
    explorer.setParameter(BIAS, 0.75);
    const seen: (ReturnType<Explorer["lastEdit"]>)[] = [];
    explorer.subscribe(() => seen.push(explorer.lastEdit()));
    explorer.restoreParameter(BIAS);
    expect(seen[0]).not.toBeNull();
    expect(seen[0]!.before).toBe(0.75);
  });
});

/**
 * The browser's copy of the generators' formulas.
 *
 * `state.ts` has to evaluate a target at a point the pipeline never
 * sampled, so it duplicates the formula. That duplication is only safe if
 * it is checked: every published dataset that declares an analytic target
 * must have one here, agreeing with the pipeline's own samples.
 */
describe("analytic targets", () => {
  it("EXIST FOR EVERY DATASET THAT CLAIMS ONE, and agree with its samples", () => {
    const seen = new Set<string>();
    for (const checkpoint of allCheckpoints()) {
      if (seen.has(checkpoint.dataset.id)) continue;
      seen.add(checkpoint.dataset.id);
      // Construction throws if the formula is missing, which is the point.
      const explorer = createExplorer(checkpoint);
      const { samples, hasAnalyticTarget } = checkpoint.dataset;
      if (!hasAnalyticTarget) continue;
      // Probing a sample's own coordinates as a *free* probe must give the
      // target the pipeline recorded for it.
      for (const index of [0, 7, samples.y.length - 1]) {
        explorer.setProbe({ kind: "free", x: samples.x[index] });
        expect(explorer.snapshot().target).toBeCloseTo(samples.y[index], 12);
      }
    }
    expect(seen.size).toBe(4);
  });

  it("GIVE A STOCHASTIC CLASSIFIER NO TARGET AND NO LOSS at a free probe", () => {
    const classifier = allCheckpoints().find(
      (entry) => entry.dataset.task === "classification",
    )!;
    const explorer = createExplorer(classifier);
    explorer.setProbe({ kind: "free", x: [0.21, -0.37] });
    const snapshot = explorer.snapshot();
    expect(snapshot.target).toBeNull();
    expect(snapshot.pointLoss).toBeNull();
    expect(snapshot.pointGradient).toBeNull();
    // A sample, by contrast, has the label it was given.
    explorer.setProbe({ kind: "sample", index: 3 });
    expect(explorer.snapshot().target).toBe(classifier.dataset.samples.y[3]);
  });
});
