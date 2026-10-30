/**
 * An independent check on both kinds of derivative: centered finite
 * differences, which share no code with the analytic path.
 *
 * ReLU kinks are excluded by construction rather than by a loose tolerance:
 * a difference quotient straddling a kink is not approximating anything.
 */
import { describe, expect, it } from "vitest";

import {
  activationDerivative,
  forward,
  hiddenActivation,
  inputSensitivity,
  pointGradient,
  pointLoss,
  splitObjective,
} from "../src/model/evaluate.js";
import { writeParameter, type ParameterRef } from "../src/model/types.js";
import { allCheckpoints } from "./fixtures.js";

/** True when no preactivation sits near a kink, so a step of h is safe. */
function awayFromKinks(pre: number[][], h: number): boolean {
  return pre.slice(1, -1).every((layer) => layer.every((z) => Math.abs(z) > 100 * h));
}

describe.each(allCheckpoints())("$checkpointId", (checkpoint) => {
  const activation = hiddenActivation(checkpoint.architecture);
  const { parameters, dataset } = checkpoint;
  const task = dataset.task;

  it("input sensitivity matches centered finite differences", () => {
    const h = 1e-6;
    let checked = 0;
    for (const probe of checkpoint.goldenProbes) {
      const trace = forward(parameters, activation, probe.x);
      if (!awayFromKinks(trace.pre, h)) continue;
      checked += 1;
      const analytic = inputSensitivity(parameters, activation, trace);
      probe.x.forEach((_, axis) => {
        const hi = probe.x.slice();
        const lo = probe.x.slice();
        hi[axis] += h;
        lo[axis] -= h;
        const numeric =
          (forward(parameters, activation, hi).output -
            forward(parameters, activation, lo).output) /
          (2 * h);
        // Scale-aware, like the gradient checks below. A confident
        // classifier's logit sensitivity runs into the hundreds, and a
        // centered difference of a value that size cannot be held to 5e-7
        // absolute — the catalog's steepest case misses it by 6e-7 while
        // agreeing to 2 parts in 1e9.
        expect(Math.abs(numeric - analytic[axis])).toBeLessThan(
          1e-5 * Math.max(1, Math.abs(analytic[axis])),
        );
      });
    }
    expect(checked).toBeGreaterThan(0);
  });

  it("point-loss parameter gradients match centered finite differences", () => {
    const h = 1e-7;
    const probe = checkpoint.goldenProbes.find(
      (p) => p.target !== null && awayFromKinks(forward(parameters, activation, p.x).pre, h),
    );
    expect(probe).toBeDefined();
    const trace = forward(parameters, activation, probe!.x);
    const analytic = pointGradient(parameters, activation, trace, probe!.target!, task);

    const refs: ParameterRef[] = [];
    parameters.forEach((layer, index) => {
      layer.W.forEach((row, out) =>
        row.forEach((_, inIndex) =>
          refs.push({ kind: "weight", layer: index, out, in: inIndex }),
        ),
      );
      layer.b.forEach((_, out) => refs.push({ kind: "bias", layer: index, out }));
    });

    // Every parameter, not a sample: the networks here are small enough.
    for (const ref of refs) {
      const base =
        ref.kind === "weight"
          ? parameters[ref.layer].W[ref.out][ref.in]
          : parameters[ref.layer].b[ref.out];
      const evaluate = (value: number) => {
        const edited = writeParameter(parameters, ref, value);
        return pointLoss(
          forward(edited, activation, probe!.x).output,
          probe!.target!,
          task,
        );
      };
      const numeric = (evaluate(base + h) - evaluate(base - h)) / (2 * h);
      const want =
        ref.kind === "weight"
          ? analytic[ref.layer].W[ref.out][ref.in]
          : analytic[ref.layer].b[ref.out];
      // Scale-aware: a gradient of 1e3 cannot be checked to 1e-9 absolute.
      expect(Math.abs(numeric - want)).toBeLessThan(1e-5 * Math.max(1, Math.abs(want)));
    }
  });

  it("mean-loss gradients match centered finite differences", () => {
    const h = 1e-7;
    const base = splitObjective(parameters, activation, task, dataset.samples, "train");
    // Derived from the shape rather than written out: the catalog has a
    // 2 -> 1 network whose first layer has a single destination unit, so a
    // hard-coded `out: 1` indexes past the end of it.
    const last = parameters.length - 1;
    const firstOut = parameters[0].b.length - 1;
    const firstIn = parameters[0].W[0].length - 1;
    const refs: ParameterRef[] = [
      { kind: "weight", layer: 0, out: 0, in: 0 },
      // The far corner of the first layer, so the check is asymmetric
      // wherever the shape allows it to be.
      { kind: "weight", layer: 0, out: firstOut, in: firstIn },
      { kind: "bias", layer: 0, out: 0 },
      { kind: "weight", layer: last, out: 0, in: 0 },
      { kind: "bias", layer: last, out: 0 },
    ];
    for (const ref of refs) {
      const current =
        ref.kind === "weight"
          ? parameters[ref.layer].W[ref.out][ref.in]
          : parameters[ref.layer].b[ref.out];
      const evaluate = (value: number) =>
        splitObjective(
          writeParameter(parameters, ref, value),
          activation,
          task,
          dataset.samples,
          "train",
        ).meanLoss;
      const numeric = (evaluate(current + h) - evaluate(current - h)) / (2 * h);
      const want =
        ref.kind === "weight"
          ? base.gradient[ref.layer].W[ref.out][ref.in]
          : base.gradient[ref.layer].b[ref.out];
      expect(Math.abs(numeric - want)).toBeLessThan(1e-5 * Math.max(1, Math.abs(want)));
    }
  });

  it("a first-order step predicts a small actual change", () => {
    // The claim the UI makes to students, checked numerically: the local
    // slope predicts a small change, and only a small one.
    const probe = checkpoint.goldenProbes.find((p) => p.target !== null)!;
    const trace = forward(parameters, activation, probe.x);
    const analytic = pointGradient(parameters, activation, trace, probe.target!, task);
    const ref: ParameterRef = { kind: "bias", layer: parameters.length - 1, out: 0 };
    const slope = analytic[ref.layer].b[ref.out];
    const before = pointLoss(trace.output, probe.target!, task);
    const base = parameters[ref.layer].b[ref.out];

    const delta = 1e-4;
    const actual =
      pointLoss(
        forward(writeParameter(parameters, ref, base + delta), activation, probe.x).output,
        probe.target!,
        task,
      ) - before;
    expect(Math.abs(actual - slope * delta)).toBeLessThan(1e-6);
  });
});

describe("the ReLU kink convention", () => {
  it("is zero at exactly zero, and is a convention rather than a limit", () => {
    expect(activationDerivative("relu", 0)).toBe(0);
    // The one-sided limits disagree, which is why a convention is needed.
    expect(activationDerivative("relu", 1e-300)).toBe(1);
    expect(activationDerivative("relu", -1e-300)).toBe(0);
  });

  it("matches the convention the asset declares", () => {
    for (const checkpoint of allCheckpoints()) {
      if (checkpoint.architecture.hiddenActivation !== "relu") continue;
      expect(checkpoint.architecture.reluDerivativeAtZero).toBe(
        activationDerivative("relu", 0),
      );
    }
  });

  it("is visible in the kink probe: the exported slope uses it", () => {
    for (const checkpoint of allCheckpoints()) {
      const probe = checkpoint.goldenProbes.find((p) => p.id === "relu-kink");
      if (!probe) continue;
      // The kinked unit is off and contributes nothing to the slope.
      const activation = hiddenActivation(checkpoint.architecture);
      const trace = forward(checkpoint.parameters, activation, probe.x);
      expect(trace.pre[1].some((z) => z === 0)).toBe(true);
      const got = inputSensitivity(checkpoint.parameters, activation, trace);
      got.forEach((value, axis) =>
        expect(value).toBeCloseTo(probe.inputSensitivity[axis], 9),
      );
    }
  });
});
