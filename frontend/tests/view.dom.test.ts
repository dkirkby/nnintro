/**
 * @vitest-environment jsdom
 *
 * The view half of the Stage 0 exit condition: an edit preserves the
 * selection in the rendered SVG, the three colour families stay apart, and
 * the three plots of the input-to-output panel share their axes exactly.
 */
import { beforeEach, describe, expect, it } from "vitest";

import { parameterCurves } from "../src/model/curves.js";
import { outputSlice, outputSurface } from "../src/model/surface.js";
import { createExplorer, type Explorer } from "../src/state.js";
import type { ParameterRef } from "../src/model/types.js";
import { edgeParameter } from "../src/view/adapter.js";
import { createColorScale, niceLimit, type Palette } from "../src/view/colormap.js";
import { createInputOutputView, type InputOutputView } from "../src/view/inputoutput.js";
import { createNetworkView, format, type NetworkView } from "../src/view/network.js";
import { createParamSpaceView, type ParamSpaceView } from "../src/view/paramspace.js";
import { loadCheckpoint, loadManifest } from "./fixtures.js";

const CHECKPOINT = loadCheckpoint(loadManifest().checkpoints[0].url);
const WEIGHT: ParameterRef = { kind: "weight", layer: 0, out: 1, in: 0 };
const ACTIVATION = CHECKPOINT.architecture.hiddenActivation!;
const SURFACE = outputSurface(CHECKPOINT.parameters, ACTIVATION, CHECKPOINT.dataset.domain, 12);
const PALETTE: Palette = {
  input: createColorScale(niceLimit(CHECKPOINT.dataset.domain.flat()), "input"),
  param: createColorScale(
    niceLimit(
      CHECKPOINT.parameters.flatMap((layer) => layer.W.flat()),
      CHECKPOINT.parameters.flatMap((layer) => layer.b),
    ),
    "param",
  ),
  output: createColorScale(
    niceLimit([SURFACE.extent], CHECKPOINT.dataset.samples.y),
    "output",
  ),
};

describe("the network view", () => {
  let container: HTMLElement;
  let explorer: Explorer;
  let view: NetworkView;
  let selected: ParameterRef | null = null;

  beforeEach(() => {
    container = document.createElement("div");
    document.body.replaceChildren(container);
    explorer = createExplorer(CHECKPOINT);
    selected = null;
    view = createNetworkView({
      container,
      layerSizes: CHECKPOINT.architecture.layerSizes,
      idPrefix: "test",
      onSelect: (ref) => {
        selected = ref;
        explorer.selectParameter(ref);
      },
    });
    view.render(explorer.snapshot(), "weights", PALETTE);
  });

  it("renders one edge per weight and one node per unit", () => {
    const sizes = CHECKPOINT.architecture.layerSizes;
    const expectedEdges = sizes.slice(0, -1).reduce((sum, n, i) => sum + n * sizes[i + 1], 0);
    expect(container.querySelectorAll('[data-key^="e:"]')).toHaveLength(expectedEdges);
    expect(container.querySelectorAll('[data-key^="n:"]')).toHaveLength(
      sizes.reduce((a, b) => a + b, 0),
    );
  });

  it("clicking an edge selects the weight that edge represents", () => {
    const element = container.querySelector<SVGElement>('[data-key="e:0:0:1"]');
    expect(element).not.toBeNull();
    element!.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
    // source 0 -> target 1 is W[1][0], destination-first.
    expect(selected).toEqual({ kind: "weight", layer: 0, out: 1, in: 0 });
    expect(selected).toEqual(edgeParameter({
      sourceLayerIndex: 0,
      sourceNodeIndex: 0,
      targetNodeIndex: 1,
    }));
  });

  it("keyboard activation selects the same parameter as a click", () => {
    const element = container.querySelector<SVGElement>('[data-key="n:1:2"]');
    element!.dispatchEvent(
      new window.KeyboardEvent("keydown", { key: "Enter", bubbles: true }),
    );
    expect(selected).toEqual({ kind: "bias", layer: 0, out: 2 });
  });

  it("the input layer's nodes are not selectable, having no bias", () => {
    const element = container.querySelector<SVGElement>('[data-key="n:0:0"]');
    element!.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
    expect(selected).toBeNull();
  });

  it("KEEPS THE SELECTION VISIBLE after an edit re-renders the view", () => {
    explorer.selectParameter(WEIGHT);
    view.render(explorer.snapshot(), "weights", PALETTE);
    expect(container.querySelector('[data-key="e:0:0:1"]')!.classList).toContain(
      "is-selected",
    );

    explorer.setParameter(WEIGHT, 0.5);
    view.render(explorer.snapshot(), "weights", PALETTE);
    const stillSelected = container.querySelector('[data-key="e:0:0:1"]');
    expect(stillSelected!.classList).toContain("is-selected");
    expect(container.querySelectorAll(".is-selected")).toHaveLength(1);
  });

  it("shows the edited value, not the checkpoint's", () => {
    explorer.setParameter(WEIGHT, -2.5);
    view.render(explorer.snapshot(), "weights", PALETTE);
    const label = container
      .querySelector('[data-key="e:0:0:1"]')!
      .getAttribute("aria-label");
    expect(label).toContain("-2.5");
  });
});

describe("the input \u2192 output panel", () => {
  const PROBE = [0.649, -0.64];
  let container: HTMLElement;
  let explorer: Explorer;
  let view: InputOutputView;

  function frame(showSamples = false) {
    const snapshot = explorer.snapshot();
    return {
      surface: SURFACE,
      slices: CHECKPOINT.dataset.domain.map((_, axis) =>
        outputSlice(snapshot.parameters, ACTIVATION, CHECKPOINT.dataset.domain, snapshot.x, axis),
      ),
      palette: PALETTE,
      showSamples,
    };
  }

  /** The map is 300 units square; x1 runs right and x2 runs up. */
  const toPixelX = (x: number) => ((x - CHECKPOINT.dataset.domain[0][0]) / 2) * 300;
  const toPixelY = (y: number) => 300 - ((y - CHECKPOINT.dataset.domain[1][0]) / 2) * 300;

  beforeEach(() => {
    container = document.createElement("div");
    document.body.replaceChildren(container);
    explorer = createExplorer(CHECKPOINT);
    explorer.setProbe({ kind: "free", x: PROBE });
    view = createInputOutputView({
      container,
      dataset: CHECKPOINT.dataset,
      onProbe: (probe) => explorer.setProbe(probe),
    });
  });

  it("SHOWS NO SAMPLES until they are asked for", () => {
    view.render(explorer.snapshot(), frame());
    expect(container.querySelectorAll(".sample")).toHaveLength(0);
    view.render(explorer.snapshot(), frame(true));
    expect(container.querySelectorAll(".sample")).toHaveLength(
      CHECKPOINT.dataset.samples.y.length,
    );
  });

  it("marks the probe with crosshairs and nothing else", () => {
    view.render(explorer.snapshot(), frame());
    // Two in the map, one through each slice; no probe dot, no arrow.
    expect(container.querySelectorAll(".io-marks .crosshair")).toHaveLength(2);
    expect(container.querySelectorAll(".io-slice-x1 .crosshair")).toHaveLength(1);
    expect(container.querySelectorAll(".io-slice-x2 .crosshair")).toHaveLength(1);
    expect(container.querySelector(".probe")).toBeNull();
    expect(container.querySelector(".sensitivity-arrow")).toBeNull();
  });

  it("LINES THE SLICES UP WITH THE MAP's OWN AXES", () => {
    view.render(explorer.snapshot(), frame());
    const inMap = Array.from(container.querySelectorAll(".io-marks .crosshair"));
    const horizontal = inMap.find((node) => node.getAttribute("y1") === node.getAttribute("y2"))!;
    const vertical = inMap.find((node) => node.getAttribute("x1") === node.getAttribute("x2"))!;
    expect(Number(vertical.getAttribute("x1"))).toBeCloseTo(toPixelX(PROBE[0]), 2);
    expect(Number(horizontal.getAttribute("y1"))).toBeCloseTo(toPixelY(PROBE[1]), 2);

    // The slice below shares the map's x1 pixel, and the one to its left
    // shares the map's x2 pixel: that is what makes them read as one figure.
    const below = container.querySelector(".io-slice-x1 .crosshair")!;
    expect(below.getAttribute("x1")).toBe(vertical.getAttribute("x1"));
    const left = container.querySelector(".io-slice-x2 .crosshair")!;
    expect(left.getAttribute("y1")).toBe(horizontal.getAttribute("y1"));
  });

  it("puts each slice's curve on the crosshair at the prediction", () => {
    const snapshot = explorer.snapshot();
    view.render(snapshot, frame());
    // The slices are 75 units deep, +limit at the top of the one below and
    // at the LEFT of the rotated one.
    const expected = 75 / 2 - (snapshot.trace.output / PALETTE.output.limit) * (75 / 2);
    const below = container.querySelector(".io-slice-x1 .slice-point")!;
    expect(Number(below.getAttribute("cy"))).toBeCloseTo(expected, 2);
    expect(Number(below.getAttribute("cx"))).toBeCloseTo(toPixelX(PROBE[0]), 2);
    const left = container.querySelector(".io-slice-x2 .slice-point")!;
    expect(Number(left.getAttribute("cx"))).toBeCloseTo(expected, 2);
    expect(Number(left.getAttribute("cy"))).toBeCloseTo(toPixelY(PROBE[1]), 2);
  });

  it("runs the rotated slice's output axis leftward, away from the map", () => {
    view.render(explorer.snapshot(), frame());
    const left = container.querySelector<HTMLElement>(".io-slice-x2 .slice-gradient")!;
    expect(left.style.backgroundImage).toContain("linear-gradient(to left");
    const below = container.querySelector<HTMLElement>(".io-slice-x1 .slice-gradient")!;
    expect(below.style.backgroundImage).toContain("linear-gradient(to top");
    for (const gradient of [left, below]) {
      expect(gradient.style.backgroundImage).toContain(
        PALETTE.output.css(PALETTE.output.limit),
      );
    }
  });

  it("KEEPS THE TWO FAMILIES APART: axis bars read inputs, slices read outputs", () => {
    view.render(explorer.snapshot(), frame(true));
    const bar = container.querySelector<HTMLElement>(".axis-bar-x")!;
    expect(bar.style.backgroundImage).toContain("var(--input-");
    expect(bar.style.backgroundImage).not.toContain("var(--output-");
    expect(bar.style.backgroundImage).toContain(
      PALETTE.input.css(CHECKPOINT.dataset.domain[0][0]),
    );

    const gradient = container.querySelector<HTMLElement>(".io-slice-x1 .slice-gradient")!;
    expect(gradient.style.backgroundImage).toContain("var(--output-");
    const sample = container.querySelector(".sample")!;
    expect(sample.getAttribute("fill")).toBe(
      PALETTE.output.css(CHECKPOINT.dataset.samples.y[0]),
    );
  });

  it("draws the whole sweep, one point per sample", () => {
    const current = frame();
    view.render(explorer.snapshot(), current);
    const curve = container.querySelector(".io-slice-x1 .slice-curve")!;
    expect(curve.getAttribute("points")!.split(" ")).toHaveLength(
      current.slices[0].inputs.length,
    );
  });
});

describe("the network's node colours", () => {
  let container: HTMLElement;
  let explorer: Explorer;
  let view: NetworkView;

  beforeEach(() => {
    container = document.createElement("div");
    document.body.replaceChildren(container);
    explorer = createExplorer(CHECKPOINT);
    explorer.setProbe({ kind: "free", x: [0.649, -0.64] });
    view = createNetworkView({
      container,
      layerSizes: CHECKPOINT.architecture.layerSizes,
      idPrefix: "families",
      onSelect: () => {},
    });
  });

  it("FILLS EACH NODE FROM ITS OWN FAMILY's MAP", () => {
    const snapshot = explorer.snapshot();
    view.render(snapshot, "weights", PALETTE);
    const last = CHECKPOINT.architecture.layerSizes.length - 1;

    const input = container.querySelector('[data-key="n:0:0"]')!;
    expect(input.getAttribute("fill")).toBe(PALETTE.input.css(snapshot.x[0]));

    // A hidden unit wears its bias, not its activation.
    const hidden = container.querySelector('[data-key="n:1:0"]')!;
    expect(hidden.getAttribute("fill")).toBe(
      PALETTE.param.css(snapshot.parameters[0].b[0]),
    );
    expect(hidden.getAttribute("fill")).not.toBe(
      PALETTE.param.css(snapshot.trace.act[1][0]),
    );

    const output = container.querySelector(`[data-key="n:${last}:0"]`)!;
    expect(output.getAttribute("fill")).toBe(PALETTE.output.css(snapshot.trace.output));
  });

  it("still shows the activation as the number on a hidden node", () => {
    const snapshot = explorer.snapshot();
    view.render(snapshot, "weights", PALETTE);
    const hidden = container.querySelector('[data-key="n:1:0"]')!;
    expect(hidden.getAttribute("aria-label")).toContain(
      `activation ${format(snapshot.trace.act[1][0])}`,
    );
    expect(hidden.getAttribute("aria-label")).toContain(
      `bias ${format(snapshot.parameters[0].b[0])}`,
    );
  });

  it("does not change a node's colour when the edge mode changes", () => {
    const snapshot = explorer.snapshot();
    view.render(snapshot, "weights", PALETTE);
    const before = container.querySelector('[data-key="n:1:0"]')!.getAttribute("fill");
    view.render(snapshot, "contributions", PALETTE);
    expect(container.querySelector('[data-key="n:1:0"]')!.getAttribute("fill")).toBe(before);
  });
});

describe("the parameters → output panel", () => {
  let container: HTMLElement;
  let explorer: Explorer;
  let view: ParamSpaceView;
  let edited: [ParameterRef, number] | null = null;

  const WEIGHT_REF: ParameterRef = { kind: "weight", layer: 0, out: 1, in: 0 };

  function frame() {
    const snapshot = explorer.snapshot();
    const curves = parameterCurves(
      snapshot.parameters,
      ACTIVATION,
      snapshot.x,
      PALETTE.param.limit,
    );
    return {
      curves,
      scale: PALETTE.param,
      outputLimit: niceLimit(curves.flatMap((curve) => curve.outputs)),
    };
  }

  beforeEach(() => {
    container = document.createElement("div");
    document.body.replaceChildren(container);
    explorer = createExplorer(CHECKPOINT);
    explorer.setProbe({ kind: "free", x: [0.649, -0.64] });
    edited = null;
    view = createParamSpaceView({
      container,
      onEdit: (ref, value) => {
        edited = [ref, value];
        explorer.setParameter(ref, value);
      },
    });
  });

  it("draws one curve and one dot per parameter", () => {
    const current = frame();
    view.render(explorer.snapshot(), current);
    const count = CHECKPOINT.parameters.reduce(
      (total, layer) => total + layer.W.flat().length + layer.b.length,
      0,
    );
    expect(current.curves).toHaveLength(count);
    expect(container.querySelectorAll(".param-curve")).toHaveLength(count);
    expect(container.querySelectorAll(".param-dot")).toHaveLength(count);
  });

  it("PUTS EVERY DOT ON THE OUTPUT LINE, which is the panel's whole claim", () => {
    view.render(explorer.snapshot(), frame());
    const level = Number(container.querySelector(".output-level")!.getAttribute("y1"));
    for (const dot of Array.from(container.querySelectorAll(".param-dot"))) {
      expect(Number(dot.getAttribute("cy"))).toBeCloseTo(level, 2);
    }
  });

  it("highlights the selected parameter's curve and nothing else", () => {
    view.render(explorer.snapshot(), frame());
    expect(container.querySelectorAll(".param-curve.is-selected")).toHaveLength(0);

    explorer.selectParameter(WEIGHT_REF);
    view.render(explorer.snapshot(), frame());
    expect(container.querySelectorAll(".param-curve.is-selected")).toHaveLength(1);
    expect(container.querySelectorAll(".param-dot.is-selected")).toHaveLength(1);
  });

  it("puts the slider on the same axis as the plot, disabled until something is selected", () => {
    view.render(explorer.snapshot(), frame());
    const slider = container.querySelector<HTMLInputElement>(".param-slider")!;
    expect(slider.disabled).toBe(true);
    expect(Number(slider.min)).toBeCloseTo(-PALETTE.param.limit, 12);
    expect(Number(slider.max)).toBeCloseTo(PALETTE.param.limit, 12);

    explorer.selectParameter(WEIGHT_REF);
    view.render(explorer.snapshot(), frame());
    expect(slider.disabled).toBe(false);
    expect(Number(slider.value)).toBeCloseTo(
      CHECKPOINT.parameters[0].W[1][0],
      6,
    );
  });

  it("EDITS THE SELECTED PARAMETER as the slider is dragged", () => {
    explorer.selectParameter(WEIGHT_REF);
    view.render(explorer.snapshot(), frame());
    const slider = container.querySelector<HTMLInputElement>(".param-slider")!;
    slider.value = "0.25";
    slider.dispatchEvent(new window.Event("input", { bubbles: true }));
    expect(edited).toEqual([WEIGHT_REF, 0.25]);
    expect(explorer.snapshot().parameters[0].W[1][0]).toBe(0.25);

    // And the redrawn plot follows the new state: the dots are still on the
    // line, which has moved with the output.
    view.render(explorer.snapshot(), frame());
    const level = Number(container.querySelector(".output-level")!.getAttribute("y1"));
    for (const dot of Array.from(container.querySelectorAll(".param-dot"))) {
      expect(Number(dot.getAttribute("cy"))).toBeCloseTo(level, 2);
    }
  });

  it("carries the parameter colour scale on its x axis", () => {
    view.render(explorer.snapshot(), frame());
    const bar = container.querySelector<HTMLElement>(".param-axis")!;
    expect(bar.style.backgroundImage).toContain("var(--param-");
    expect(bar.style.backgroundImage).toContain(PALETTE.param.css(-PALETTE.param.limit));
  });
});
