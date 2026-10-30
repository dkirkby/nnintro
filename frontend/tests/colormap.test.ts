/**
 * The colour maps: one diverging scale per family of quantity, with zero at
 * the neutral surface colour so sign is readable and a near-zero value looks
 * like one.
 */
import { describe, expect, it } from "vitest";

import type { DatasetSpec, Task } from "../src/model/types.js";
import {
  createColorScale,
  gradientBar,
  niceLimit,
  outputLimit,
  snugLimit,
} from "../src/view/colormap.js";

describe("niceLimit", () => {
  it("rounds up to the next round number so colours do not shift underfoot", () => {
    expect(niceLimit([0.45])).toBeCloseTo(0.5, 12);
    expect(niceLimit([0.6])).toBeCloseTo(0.75, 12);
    expect(niceLimit([2.3])).toBeCloseTo(3, 12);
    expect(niceLimit([8.1])).toBeCloseTo(10, 12);
  });

  it("is already round for a value that is itself a stop", () => {
    expect(niceLimit([2])).toBeCloseTo(2, 12);
    expect(niceLimit([-5])).toBeCloseTo(5, 12);
  });

  it("takes the largest magnitude across every group, sign-blind", () => {
    expect(niceLimit([0.1, 0.2], [-1.4], [0.3])).toBeCloseTo(1.5, 12);
  });

  it("IS A NUMBER THAT PRINTS AS ROUND, since it is an axis tick", () => {
    // 3 * 0.1 is 0.30000000000000004 in binary, which is what the
    // parameters-to-output axis would then be labelled with.
    expect(String(niceLimit([0.28]))).toBe("0.3");
    expect(String(niceLimit([0.07]))).toBe("0.075");
    expect(String(niceLimit([2.9e-4]))).toBe("0.0003");
    expect(String(niceLimit([1.1]))).toBe("1.5");
  });

  it("ignores non-finite values and never returns zero", () => {
    expect(niceLimit([Number.NaN, Number.POSITIVE_INFINITY])).toBe(1);
    expect(niceLimit([0, 0])).toBe(1);
    expect(niceLimit([])).toBe(1);
  });
});

describe("a colour scale", () => {
  const scale = createColorScale(2, "output");

  it("puts zero at the neutral surface colour", () => {
    expect(scale.css(0)).toBe("color-mix(in oklab, var(--output-pos) 0.0%, var(--surface))");
  });

  it("is linear in the value and separates the two signs", () => {
    expect(scale.css(1)).toContain("var(--output-pos) 50.0%");
    expect(scale.css(-1)).toContain("var(--output-neg) 50.0%");
    expect(scale.css(2)).toContain("var(--output-pos) 100.0%");
  });

  it("clamps rather than wrapping beyond the limit", () => {
    expect(scale.css(40)).toBe(scale.css(2));
    expect(scale.paint(-40)).toBe(scale.paint(-2));
  });

  it("marks a non-finite value instead of painting it as a number", () => {
    expect(scale.css(Number.NaN)).toBe("var(--nonfinite)");
    expect(scale.paint(Number.POSITIVE_INFINITY)).not.toBe(scale.paint(2));
  });

  it("paints a concrete colour for the canvas, which has no custom properties", () => {
    // Zero is the surface token itself; full saturation is not.
    expect(scale.paint(0)).toMatch(/^oklab\([\d.]+% -?[\d.]+ -?[\d.]+\)$/);
    expect(scale.paint(0)).not.toBe(scale.paint(2));
    expect(scale.paint(1)).not.toBe(scale.paint(-1));
  });

  it("agrees with the DOM rendering about which values are neutral", () => {
    // Same map, two renderings: the midpoint of the mix is the midpoint of
    // the range in both.
    expect(scale.css(0)).toContain("0.0%");
    expect(scale.paint(0)).toBe(createColorScale(7, "output").paint(0));
  });

  it("spans only the range asked for in a gradient", () => {
    const stops = scale.stops(-1, 1, 3);
    expect(stops).toHaveLength(3);
    expect(stops[0]).toBe(scale.css(-1));
    expect(stops[1]).toBe(scale.css(0));
    expect(stops[2]).toBe(scale.css(1));
  });
});

describe("the families", () => {
  it("mix their own inks, so no two families share a colour", () => {
    const value = 0.7;
    const colours = (["param", "output"] as const).map((family) =>
      createColorScale(1, family).css(value),
    );
    expect(colours[0]).toContain("var(--param-pos)");
    expect(colours[1]).toContain("var(--output-pos)");
    expect(new Set(colours).size).toBe(2);
  });

  it("paints distinct canvas colours for the same number", () => {
    const painted = (["param", "output"] as const).map((family) =>
      createColorScale(1, family).paint(0.7),
    );
    expect(new Set(painted).size).toBe(2);
  });

  it("agrees with itself about zero whatever the family", () => {
    // Zero is the surface colour in every family: the maps differ in their
    // inks, not in where neutral sits.
    const zeros = (["param", "output"] as const).map((family) =>
      createColorScale(1, family).paint(0),
    );
    expect(new Set(zeros).size).toBe(1);
  });

  it("builds a gradient across a range for the bars", () => {
    const scale = createColorScale(1.5, "param");
    expect(gradientBar(scale, -1, 1, "to right")).toBe(
      `linear-gradient(to right, ${scale.stops(-1, 1).join(", ")})`,
    );
  });
});

/**
 * The limit for a scale that cannot shift underfoot.
 *
 * `niceLimit`'s ladder is there so a limit computed from the parameters
 * stays put as they are edited. The output scale is computed from a
 * dataset's fixed targets, which cannot move, so the ladder would only
 * spend the scale: the tilted plane's targets used to reach 1.07 and the
 * ladder rounded that to 1.5, which is why nothing on its map was ever
 * fully coloured.
 */
describe("snugLimit", () => {
  it("SATURATES AT THE LARGEST VALUE, within two significant digits", () => {
    expect(snugLimit([1.072467603122828, -0.75])).toBe(1.1);
    expect(snugLimit([0.9982551740776434])).toBe(1);
    // The largest MAGNITUDE, either sign, as the scale is diverging.
    expect(snugLimit([-0.0405, 0.0302])).toBe(0.041);
    expect(snugLimit([37])).toBe(37);
    expect(snugLimit([36.2])).toBe(37);
  });

  it("is never looser than the ladder, and usually tighter", () => {
    for (const largest of [0.37, 1.072, 1.5, 2.9, 4.01, 12.4]) {
      expect(snugLimit([largest])).toBeLessThanOrEqual(niceLimit([largest]));
      expect(snugLimit([largest])).toBeGreaterThanOrEqual(largest);
    }
  });

  it("leaves a value that already has two digits alone", () => {
    // The rounding in the decade itself must not push the limit up a step.
    for (const exact of [1.1, 0.031, 3.7, 150]) {
      expect(snugLimit([exact])).toBe(exact);
    }
  });

  it("falls back to 1 where every value is zero, as niceLimit does", () => {
    expect(snugLimit([0, -0])).toBe(1);
    expect(snugLimit([])).toBe(1);
    expect(snugLimit([Number.NaN, Number.POSITIVE_INFINITY])).toBe(1);
  });
});

/**
 * Which number is full red: the generator's range of targets, not the
 * sample's.
 *
 * The distinction is the whole reason `dataset.targetRange` is published.
 * The tilted plane's targets reach 1 at the corner (+1, -1) and 400 uniform
 * draws do not land on a corner, so a scale fitted to the draw would stop
 * short of the range the generator actually covers -- and would move if the
 * sampling seed ever did.
 */
describe("outputLimit", () => {
  /** A dataset spec with only the fields the limit reads filled in. */
  function datasetOf(
    task: Task,
    targetRange: [number, number] | null,
    y: number[],
  ): DatasetSpec {
    return {
      id: "test-set",
      label: "Test set",
      task,
      description: "a dataset built for this test",
      generatorVersion: 1,
      domain: [
        [-1, 1],
        [-1, 1],
      ],
      inputNames: ["x1", "x2"],
      targetName: "y",
      hasAnalyticTarget: targetRange !== null,
      targetRange,
      preprocessing: null,
      samples: { x: y.map(() => [0, 0]), y, isTest: y.map(() => false) },
    };
  }

  it("SPANS THE GENERATOR'S RANGE, not the range the samples happened to draw", () => {
    // The tilted plane's own numbers: the range is [-0.6, 1] and the most
    // extreme of its 400 samples is 0.902.
    const dataset = datasetOf("regression", [-0.6, 1], [0.9021074585014741, -0.55]);
    expect(outputLimit(dataset, 3.7)).toBe(1);
    // Fitted to the draw instead, the scale would end here -- and the
    // extreme sample would be the extreme colour of a scale that means
    // nothing in particular.
    expect(snugLimit(dataset.samples.y)).toBe(0.91);
  });

  it("ignores the surface, so an edit cannot move a regression's scale", () => {
    const dataset = datasetOf("regression", [-1, 1], [0.4]);
    for (const extent of [0.01, 1, 12, 400]) {
      expect(outputLimit(dataset, extent)).toBe(1);
    }
  });

  it("falls back to the drawn targets where the generator declares no range", () => {
    // Not every generator has a range to declare: one sampled from measured
    // data has only the targets it drew, and those are then the only range
    // there is.
    const dataset = datasetOf("regression", null, [0.9021074585014741, -0.55]);
    expect(outputLimit(dataset, 3.7)).toBe(0.91);
  });

  it("TAKES A CLASSIFIER'S LIMIT FROM THE SURFACE, since 0 and 1 are labels", () => {
    // Everything a classifier's panels draw is a logit, and its targets are
    // not on that axis at all, so they must not reach the limit.
    const dataset = datasetOf("classification", null, [0, 1, 1, 0]);
    expect(outputLimit(dataset, 4.2)).toBe(5);
    expect(outputLimit(dataset, 0.4)).toBe(0.5);
  });
});
