/**
 * The interaction contract: an edit preserves the selection, the baseline
 * stays immutable, provenance is honest, and undo is exact.
 */
import { beforeEach, describe, expect, it } from "vitest";

import { createExplorer, describeProvenance, type Explorer } from "../src/state.js";
import { readParameter, type ParameterRef } from "../src/model/types.js";
import { loadCheckpoint, loadManifest } from "./fixtures.js";

const CHECKPOINT = loadCheckpoint(loadManifest().checkpoints[0].url);
const WEIGHT: ParameterRef = { kind: "weight", layer: 0, out: 1, in: 0 };
const BIAS: ParameterRef = { kind: "bias", layer: 0, out: 2 };

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

  // The Stage 0 exit condition.
  it("PRESERVES THE SELECTION across a parameter edit", () => {
    explorer.selectParameter(WEIGHT);
    explorer.setParameter(WEIGHT, 0.5);
    expect(explorer.snapshot().selection).toEqual(WEIGHT);
    explorer.setParameter(BIAS, -0.25);
    expect(explorer.snapshot().selection).toEqual(WEIGHT);
  });

  it("preserves the selection across undo, redo and restore", () => {
    explorer.selectParameter(BIAS);
    explorer.setParameter(BIAS, 1.5);
    explorer.undo();
    expect(explorer.snapshot().selection).toEqual(BIAS);
    explorer.redo();
    expect(explorer.snapshot().selection).toEqual(BIAS);
    explorer.restoreBaseline();
    expect(explorer.snapshot().selection).toEqual(BIAS);
  });

  it("preserves the selection when the probe moves", () => {
    explorer.selectParameter(WEIGHT);
    explorer.setProbe({ kind: "free", x: [0.2, -0.4] });
    expect(explorer.snapshot().selection).toEqual(WEIGHT);
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
    explorer.selectParameter(WEIGHT);
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

  it("undo restores the parameters exactly", () => {
    const before = JSON.stringify(explorer.snapshot().parameters);
    explorer.setParameter(WEIGHT, 1.25);
    expect(JSON.stringify(explorer.snapshot().parameters)).not.toBe(before);
    explorer.undo();
    expect(JSON.stringify(explorer.snapshot().parameters)).toBe(before);
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
    expect(snapshot.target).toBeCloseTo(0.6 * 0.5 - 0.4 * -0.5 + 0.2, 12);
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
    explorer.selectParameter(WEIGHT);
    explorer.undo();
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
