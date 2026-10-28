/**
 * The Stage 0 exit condition: the browser's forward pass and derivatives
 * reproduce PyTorch's, on every golden probe of every published checkpoint.
 *
 * Both sides work in float64, so the agreement is held to 1e-9 rather than
 * to a loose visual tolerance.
 */
import { describe, expect, it } from "vitest";

import {
  forward,
  hiddenActivation,
  inputSensitivity,
  pointGradient,
  pointLoss,
  splitObjective,
} from "../src/model/evaluate.js";
import { allCheckpoints } from "./fixtures.js";

const TOL = 1e-9;

describe.each(allCheckpoints())("$checkpointId", (checkpoint) => {
  const activation = hiddenActivation(checkpoint.architecture);
  const { parameters, dataset } = checkpoint;

  it("publishes probes at all", () => {
    expect(checkpoint.goldenProbes.length).toBeGreaterThan(0);
  });

  describe.each(checkpoint.goldenProbes)("probe $id", (probe) => {
    const trace = forward(parameters, activation, probe.x);

    it("reproduces every preactivation and activation", () => {
      expect(trace.pre.length).toBe(probe.pre.length);
      for (let layer = 0; layer < probe.pre.length; layer += 1) {
        for (let unit = 0; unit < probe.pre[layer].length; unit += 1) {
          expect(trace.pre[layer][unit]).toBeCloseTo(probe.pre[layer][unit], 9);
          expect(trace.act[layer][unit]).toBeCloseTo(probe.act[layer][unit], 9);
        }
      }
    });

    it("reproduces the output", () => {
      expect(trace.output).toBeCloseTo(probe.output, 9);
    });

    it("reproduces the input sensitivity", () => {
      const got = inputSensitivity(parameters, activation, trace);
      expect(got.length).toBe(probe.inputSensitivity.length);
      got.forEach((value, axis) => {
        expect(value).toBeCloseTo(probe.inputSensitivity[axis], 9);
      });
    });

    it("reproduces the point loss and its parameter gradient", () => {
      if (probe.target === null) {
        // An unlabeled probe has no loss to reproduce, by design.
        expect(probe.pointLoss).toBeNull();
        expect(probe.pointGradient).toBeNull();
        return;
      }
      expect(pointLoss(trace.output, probe.target, dataset.task)).toBeCloseTo(
        probe.pointLoss!,
        9,
      );
      const got = pointGradient(
        parameters,
        activation,
        trace,
        probe.target,
        dataset.task,
      );
      got.forEach((layer, index) => {
        const want = probe.pointGradient![index];
        layer.W.forEach((row, out) =>
          row.forEach((value, inIndex) => {
            expect(Math.abs(value - want.W[out][inIndex])).toBeLessThan(TOL);
          }),
        );
        layer.b.forEach((value, out) => {
          expect(Math.abs(value - want.b[out])).toBeLessThan(TOL);
        });
      });
    });
  });

  describe.each(["train", "test"] as const)("the %s split", (split) => {
    it("reproduces the mean loss and the gradient of that mean", () => {
      const want = checkpoint.splitGradients[split];
      const got = splitObjective(
        parameters,
        activation,
        dataset.task,
        dataset.samples,
        split,
      );
      expect(got.count).toBe(want.count);
      expect(got.meanLoss).toBeCloseTo(want.meanLoss, 12);
      got.gradient.forEach((layer, index) => {
        layer.W.forEach((row, out) =>
          row.forEach((value, inIndex) => {
            expect(Math.abs(value - want.gradient[index].W[out][inIndex])).toBeLessThan(TOL);
          }),
        );
        layer.b.forEach((value, out) => {
          expect(Math.abs(value - want.gradient[index].b[out])).toBeLessThan(TOL);
        });
      });
    });
  });
});
