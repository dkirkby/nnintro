/**
 * The shared colour map: one scale for every value the explorer paints, with
 * zero at the neutral surface colour so sign is readable and a near-zero
 * value looks like one.
 */
import { describe, expect, it } from "vitest";

import { createColorScale, niceLimit } from "../src/view/colormap.js";

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

  it("ignores non-finite values and never returns zero", () => {
    expect(niceLimit([Number.NaN, Number.POSITIVE_INFINITY])).toBe(1);
    expect(niceLimit([0, 0])).toBe(1);
    expect(niceLimit([])).toBe(1);
  });
});

describe("a colour scale", () => {
  const scale = createColorScale(2);

  it("puts zero at the neutral surface colour", () => {
    expect(scale.css(0)).toBe("color-mix(in oklab, var(--pos) 0.0%, var(--surface))");
  });

  it("is linear in the value and separates the two signs", () => {
    expect(scale.css(1)).toContain("var(--pos) 50.0%");
    expect(scale.css(-1)).toContain("var(--neg) 50.0%");
    expect(scale.css(2)).toContain("var(--pos) 100.0%");
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
    expect(scale.paint(0)).toBe(createColorScale(7).paint(0));
  });

  it("spans only the range asked for in a gradient", () => {
    const stops = scale.stops(-1, 1, 3);
    expect(stops).toHaveLength(3);
    expect(stops[0]).toBe(scale.css(-1));
    expect(stops[1]).toBe(scale.css(0));
    expect(stops[2]).toBe(scale.css(1));
  });
});
