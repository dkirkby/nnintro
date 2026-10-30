/**
 * @vitest-environment jsdom
 *
 * The view half of the Stage 0 exit condition: an edit preserves the
 * selection in the rendered SVG, the three colour families stay apart, and
 * the three plots of the input-to-output panel share their axes exactly.
 *
 * Selection is per *unit*: a click on a node or on any edge into it selects
 * that neuron and every weight feeding it, and the parameters-to-output
 * panel shows exactly those parameters, one slider each.
 */
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { parameterCurves } from "../src/model/curves.js";
import { outputSlice, outputSurface } from "../src/model/surface.js";
import { createExplorer, type Explorer } from "../src/state.js";
import {
  describeParameter,
  unitOf,
  unitParameters,
  type ParameterRef,
  type UnitRef,
} from "../src/model/types.js";
import { edgeParameter, edgeUnit, nodeUnit } from "../src/view/adapter.js";
import { createColorScale, niceLimit, type Palette } from "../src/view/colormap.js";
import { createInputOutputView, type InputOutputView } from "../src/view/inputoutput.js";
import { createHelp, type Help } from "../src/view/help.js";
import { createInspector, type Inspector } from "../src/view/inspector.js";
import { createNetworkView, format, type NetworkView } from "../src/view/network.js";
import { createParamSpaceView, type ParamSpaceView } from "../src/view/paramspace.js";
import { createTensorsView, type TensorsView } from "../src/view/tensors.js";
import { loadCheckpoint, loadManifest } from "./fixtures.js";

const CHECKPOINT = loadCheckpoint(loadManifest().checkpoints[0].url);
const WEIGHT: ParameterRef = { kind: "weight", layer: 0, out: 1, in: 0 };
/** Hidden unit 1: what a click on that weight's edge now selects. */
const WEIGHT_UNIT: UnitRef = unitOf(WEIGHT);
const ACTIVATION = CHECKPOINT.architecture.hiddenActivation!;
const SURFACE = outputSurface(CHECKPOINT.parameters, ACTIVATION, CHECKPOINT.dataset.domain, 12);
const PALETTE: Palette = {
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
  let selected: UnitRef | null = null;

  beforeEach(() => {
    container = document.createElement("div");
    document.body.replaceChildren(container);
    explorer = createExplorer(CHECKPOINT);
    selected = null;
    view = createNetworkView({
      container,
      layerSizes: CHECKPOINT.architecture.layerSizes,
      idPrefix: "test",
      onSelect: (unit) => {
        selected = unit;
        if (unit) explorer.selectUnit(unit);
      },
    });
    view.render(explorer.snapshot(), "weights", PALETTE);
  });

  it("renders one edge per weight and one node per unit", () => {
    const sizes = CHECKPOINT.architecture.layerSizes;
    const expectedEdges = sizes.slice(0, -1).reduce((sum, n, i) => sum + n * sizes[i + 1], 0);
    expect(container.querySelectorAll('.dn-edge[data-key^="e:"]')).toHaveLength(expectedEdges);
    expect(container.querySelectorAll('[data-key^="n:"]')).toHaveLength(
      sizes.reduce((a, b) => a + b, 0),
    );
    // Plus a wide transparent twin for each, so a hairline is clickable.
    expect(container.querySelectorAll(".edge-target")).toHaveLength(expectedEdges);
  });

  it("GIVES A ZERO-WEIGHT EDGE A VISIBLE, SELECTABLE FORM", () => {
    explorer.setParameter(WEIGHT, 0);
    view.render(explorer.snapshot(), "weights", PALETTE);
    const edge = container.querySelector('.dn-edge[data-key="e:0:0:1"]')!;
    // Dotted grey from CSS, since the colour scale has nothing to show.
    expect(edge.classList).toContain("is-faint");

    // And the wide transparent twin carries the same key, so a click on it
    // selects the unit even where the line itself is a hairline.
    const target = container.querySelector('.edge-target[data-key="e:0:0:1"]')!;
    target.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
    expect(selected).toEqual(WEIGHT_UNIT);
  });

  it("RAILS EVERY EDGE INTO THE SELECTED UNIT, leaving each its colour", () => {
    explorer.selectUnit(WEIGHT_UNIT);
    view.render(explorer.snapshot(), "weights", PALETTE);
    // One unit, every edge feeding it: two rails each.
    const feeding = CHECKPOINT.architecture.layerSizes[0];
    const rails = Array.from(container.querySelectorAll<SVGLineElement>(".edge-outline"));
    expect(rails).toHaveLength(2 * feeding);

    const edge = container.querySelector<SVGLineElement>('.dn-edge[data-key="e:0:0:1"]')!;
    const at = (node: SVGLineElement) => ({
      x: (Number(node.getAttribute("x1")) + Number(node.getAttribute("x2"))) / 2,
      y: (Number(node.getAttribute("y1")) + Number(node.getAttribute("y2"))) / 2,
      dx: Number(node.getAttribute("x2")) - Number(node.getAttribute("x1")),
      dy: Number(node.getAttribute("y2")) - Number(node.getAttribute("y1")),
    });
    const line = at(edge);
    // This edge's own two rails, picked out of the unit's set by direction.
    const [one, two] = rails
      .map(at)
      .filter((rail) => Math.abs(rail.dx - line.dx) < 0.05 && Math.abs(rail.dy - line.dy) < 0.05);
    // Parallel to the edge, and offset to opposite sides of it by the same
    // distance, so the edge itself is untouched between them.
    // Coordinates are written to two decimals of SVG user units.
    for (const rail of [one, two]) {
      expect(rail.dx).toBeCloseTo(line.dx, 1);
      expect(rail.dy).toBeCloseTo(line.dy, 1);
    }
    expect(one.x - line.x).toBeCloseTo(line.x - two.x, 1);
    expect(one.y - line.y).toBeCloseTo(line.y - two.y, 1);
    expect(Math.hypot(one.x - line.x, one.y - line.y)).toBeGreaterThan(0.5);
    // The edge keeps the colour the scale gave it.
    expect(edge.getAttribute("stroke")).toContain("var(--param-");

    // The unit's own node wears the outline itself, and it is the only
    // selected node: a sibling unit's edges and node are untouched.
    expect(container.querySelectorAll(".dn-node.is-selected")).toHaveLength(1);
    expect(container.querySelector('[data-key="n:1:1"]')!.classList).toContain(
      "is-selected",
    );
    expect(container.querySelector('[data-key="n:1:2"]')!.classList).not.toContain(
      "is-selected",
    );
    expect(container.querySelector('[data-key="e:0:0:2"]')!.classList).not.toContain(
      "is-selected",
    );
  });

  it("SELECTS A NODE'S WHOLE UNIT: the node and every edge feeding it", () => {
    const node = container.querySelector<SVGElement>('[data-key="n:1:2"]')!;
    node.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
    expect(selected).toEqual({ layer: 0, out: 2 });

    view.render(explorer.snapshot(), "weights", PALETTE);
    expect(container.querySelector('[data-key="n:1:2"]')!.classList).toContain(
      "is-selected",
    );
    const inputs = CHECKPOINT.architecture.layerSizes[0];
    for (let source = 0; source < inputs; source += 1) {
      expect(
        container.querySelector(`.dn-edge[data-key="e:0:${source}:2"]`)!.classList,
      ).toContain("is-selected");
    }
    // Nothing else: one node and its own incoming edges.
    expect(container.querySelectorAll(".dn-edge.is-selected")).toHaveLength(inputs);
    expect(container.querySelectorAll(".dn-node.is-selected")).toHaveLength(1);
  });

  it("an edge selects the unit it FEEDS, with its siblings into that unit", () => {
    // e:0:0:1 is input 0 -> hidden unit 1, so the selection is hidden unit
    // 1: the edge from input 1 into the same unit comes with it, and the
    // edge from input 0 into a different unit does not.
    const element = container.querySelector<SVGElement>('[data-key="e:0:0:1"]')!;
    element.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
    expect(selected).toEqual(edgeUnit({ sourceLayerIndex: 0, targetNodeIndex: 1 }));

    view.render(explorer.snapshot(), "weights", PALETTE);
    expect(container.querySelector('.dn-edge[data-key="e:0:1:1"]')!.classList).toContain(
      "is-selected",
    );
    expect(
      container.querySelector('.dn-edge[data-key="e:0:0:2"]')!.classList,
    ).not.toContain("is-selected");
  });

  it("opens on the output unit and its final-layer edges", () => {
    const last = CHECKPOINT.architecture.layerSizes.length - 1;
    const width = CHECKPOINT.architecture.layerSizes[last - 1];
    expect(explorer.snapshot().selection).toEqual(
      nodeUnit({ layerIndex: last, nodeIndex: 0 }),
    );
    expect(container.querySelector(`[data-key="n:${last}:0"]`)!.classList).toContain(
      "is-selected",
    );
    expect(container.querySelectorAll(".dn-edge.is-selected")).toHaveLength(width);
    for (let source = 0; source < width; source += 1) {
      expect(
        container.querySelector(`.dn-edge[data-key="e:${last - 1}:${source}:0"]`)!
          .classList,
      ).toContain("is-selected");
    }
  });

  it("still reads an edge destination-first on the way to its unit", () => {
    const element = container.querySelector<SVGElement>('[data-key="e:0:0:1"]');
    expect(element).not.toBeNull();
    element!.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
    // source 0 -> target 1 is W[1][0], destination-first, and that weight's
    // unit is the one selected.
    expect(selected).toEqual(
      unitOf(
        edgeParameter({ sourceLayerIndex: 0, sourceNodeIndex: 0, targetNodeIndex: 1 }),
      ),
    );
    expect(selected).toEqual({ layer: 0, out: 1 });
  });

  it("keyboard activation selects the same unit as a click", () => {
    const element = container.querySelector<SVGElement>('[data-key="n:1:2"]');
    element!.dispatchEvent(
      new window.KeyboardEvent("keydown", { key: "Enter", bubbles: true }),
    );
    expect(selected).toEqual({ layer: 0, out: 2 });
  });

  it("the input layer's nodes own nothing, so selecting one selects nothing", () => {
    // The one click that changes nothing: there is no way to deselect, so
    // the unit that was selected stays selected.
    const before = explorer.snapshot().selection;
    const element = container.querySelector<SVGElement>('[data-key="n:0:0"]');
    element!.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
    expect(selected).toBeNull();
    expect(explorer.snapshot().selection).toEqual(before);
  });

  it("KEEPS THE SELECTION VISIBLE after an edit re-renders the view", () => {
    explorer.selectUnit(WEIGHT_UNIT);
    view.render(explorer.snapshot(), "weights", PALETTE);
    const inputs = CHECKPOINT.architecture.layerSizes[0];
    expect(container.querySelector('[data-key="e:0:0:1"]')!.classList).toContain(
      "is-selected",
    );

    explorer.setParameter(WEIGHT, 0.5);
    view.render(explorer.snapshot(), "weights", PALETTE);
    const stillSelected = container.querySelector('[data-key="e:0:0:1"]');
    expect(stillSelected!.classList).toContain("is-selected");
    // The whole unit, and only it: its node plus one edge per input.
    expect(container.querySelectorAll(".dn-edge.is-selected")).toHaveLength(inputs);
    expect(container.querySelectorAll(".dn-node.is-selected")).toHaveLength(1);
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

  it("DRAWS THE TWO INPUT AXES through the origin, under everything else", () => {
    view.render(explorer.snapshot(), frame(true));
    const axes = Array.from(container.querySelectorAll<SVGElement>(".io-marks .map-axis"));
    expect(axes).toHaveLength(2);
    const vertical = axes.find((node) => node.getAttribute("x1") === node.getAttribute("x2"))!;
    const horizontal = axes.find((node) => node.getAttribute("y1") === node.getAttribute("y2"))!;
    expect(Number(vertical.getAttribute("x1"))).toBeCloseTo(toPixelX(0), 2);
    expect(Number(horizontal.getAttribute("y1"))).toBeCloseTo(toPixelY(0), 2);

    // First in document order: a reference line is drawn under the samples
    // and the crosshairs, never over them.
    const marks = Array.from(container.querySelector(".io-marks")!.children);
    expect(marks.slice(0, 2)).toEqual(axes);
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

it("labels the two axes in plain text, with no colour of their own", () => {
    view.render(explorer.snapshot(), frame(true));
    const strip = container.querySelector<HTMLElement>(".axis-strip-x")!;
    expect(strip.style.backgroundImage).toBe("");
    expect(Array.from(strip.children).map((node) => node.textContent)).toEqual([
      String(CHECKPOINT.dataset.domain[0][0]),
      CHECKPOINT.dataset.inputNames[0],
      String(CHECKPOINT.dataset.domain[0][1]),
    ]);

    // The output family is still the one that colours this panel.
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

    // An input carries no colour: its value is a coordinate, and the
    // input-to-output panel's axes are where coordinates are read.
    const input = container.querySelector('[data-key="n:0:0"]')!;
    expect(input.getAttribute("fill")).toBe("var(--surface)");

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

  /** Exactly what `main.ts` hands the panel: the selected unit's sweeps. */
  function frame() {
    const snapshot = explorer.snapshot();
    const curves = parameterCurves(
      snapshot.parameters,
      ACTIVATION,
      snapshot.x,
      PALETTE.param.limit,
      unitParameters(snapshot.parameters, snapshot.selection),
    );
    return {
      curves,
      paramScale: PALETTE.param,
      outputScale: PALETTE.output,
      // The same 5% headroom `main.ts` gives the axis.
      outputLimit: niceLimit(
        curves.flatMap((curve) => curve.outputs.map((value) => value * 1.05)),
      ),
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

  it("DRAWS ONLY THE SELECTED UNIT's PARAMETERS, one curve and dot each", () => {
    explorer.selectUnit(WEIGHT_UNIT);
    const current = frame();
    view.render(explorer.snapshot(), current);
    // Hidden unit 1: its bias plus one weight per input, and nothing from
    // any other unit.
    const count = 1 + CHECKPOINT.architecture.layerSizes[0];
    expect(current.curves).toHaveLength(count);
    expect(container.querySelectorAll(".param-curve")).toHaveLength(count);
    expect(container.querySelectorAll(".param-dot")).toHaveLength(count);
    expect(
      Array.from(container.querySelectorAll(".param-curve")).map((node) =>
        node.getAttribute("data-param"),
      ),
    ).toEqual(["b[0][1]", "W[0][1][0]", "W[0][1][1]"]);
  });

  it("PUTS EVERY DOT ON THE OUTPUT LINE, which is the panel's whole claim", () => {
    view.render(explorer.snapshot(), frame());
    const level = Number(container.querySelector(".output-level")!.getAttribute("y1"));
    for (const dot of Array.from(container.querySelectorAll(".param-dot"))) {
      expect(Number(dot.getAttribute("cy"))).toBeCloseTo(level, 2);
    }
  });

  it("LISTS ONE SLIDER PER PARAMETER, the bias first then the weights", () => {
    explorer.selectUnit(WEIGHT_UNIT);
    view.render(explorer.snapshot(), frame());
    const sliders = Array.from(
      container.querySelectorAll<HTMLInputElement>(".param-slider"),
    );
    // Bias at the top, then each weight in the order its edge appears from
    // top to bottom in the network diagram (ascending source index).
    expect(sliders.map((slider) => slider.dataset.param)).toEqual([
      "b[0][1]",
      "W[0][1][0]",
      "W[0][1][1]",
    ]);
    expect(
      Array.from(container.querySelectorAll(".param-row-name")).map(
        (node) => node.textContent,
      ),
    ).toEqual(["b", "w\u2080", "w\u2081"]);

    // Every slider is on the colour bar's own axis, and none is disabled:
    // there is no empty selection to disable them for.
    for (const slider of sliders) {
      expect(slider.disabled).toBe(false);
      expect(Number(slider.min)).toBeCloseTo(-PALETTE.param.limit, 12);
      expect(Number(slider.max)).toBeCloseTo(PALETTE.param.limit, 12);
    }
    expect(Number(sliders[0].value)).toBeCloseTo(CHECKPOINT.parameters[0].b[1], 6);
    expect(Number(sliders[1].value)).toBeCloseTo(CHECKPOINT.parameters[0].W[1][0], 6);
  });

  it("names each curve at its dot, so the sliders and the lines match up", () => {
    explorer.selectUnit(WEIGHT_UNIT);
    view.render(explorer.snapshot(), frame());
    expect(
      Array.from(container.querySelectorAll(".param-symbol")).map(
        (node) => node.textContent,
      ),
    ).toEqual(["b", "w\u2080", "w\u2081"]);
  });

  it("REBUILDS THE ROWS when the selection moves to another unit", () => {
    explorer.selectUnit(WEIGHT_UNIT);
    view.render(explorer.snapshot(), frame());
    const output = { layer: CHECKPOINT.parameters.length - 1, out: 0 };
    explorer.selectUnit(output);
    view.render(explorer.snapshot(), frame());
    expect(
      Array.from(container.querySelectorAll<HTMLInputElement>(".param-slider")).map(
        (slider) => slider.dataset.param,
      ),
    ).toEqual(
      unitParameters(explorer.snapshot().parameters, output).map(describeParameter),
    );
  });

  it("EDITS THE PARAMETER WHOSE SLIDER WAS DRAGGED, not some other one", () => {
    explorer.selectUnit(WEIGHT_UNIT);
    view.render(explorer.snapshot(), frame());
    // The third row is W[0][1][1], so a drag there must not move W[0][1][0].
    const sliders = Array.from(
      container.querySelectorAll<HTMLInputElement>(".param-slider"),
    );
    const untouched = CHECKPOINT.parameters[0].W[1][0];
    sliders[2].value = "0.25";
    sliders[2].dispatchEvent(new window.Event("input", { bubbles: true }));
    expect(edited).toEqual([{ kind: "weight", layer: 0, out: 1, in: 1 }, 0.25]);
    expect(explorer.snapshot().parameters[0].W[1][1]).toBe(0.25);
    expect(explorer.snapshot().parameters[0].W[1][0]).toBe(untouched);

    // And the row for the parameter that was edited follows the new state.
    view.render(explorer.snapshot(), frame());
    expect(Number(sliders[2].value)).toBeCloseTo(0.25, 6);

    // The redrawn plot follows too: the dots are still on the line, which
    // has moved with the output.
    const level = Number(container.querySelector(".output-level")!.getAttribute("y1"));
    for (const dot of Array.from(container.querySelectorAll(".param-dot"))) {
      expect(Number(dot.getAttribute("cy"))).toBeCloseTo(level, 2);
    }
  });

  it("edits the bias from the top row", () => {
    explorer.selectUnit(WEIGHT_UNIT);
    view.render(explorer.snapshot(), frame());
    const slider = container.querySelector<HTMLInputElement>(".param-slider")!;
    slider.value = "-0.5";
    slider.dispatchEvent(new window.Event("input", { bubbles: true }));
    expect(edited).toEqual([{ kind: "bias", layer: 0, out: 1 }, -0.5]);
    expect(explorer.snapshot().parameters[0].b[1]).toBe(-0.5);
  });

  it("carries the parameter scale on its x axis and the output scale up its back", () => {
    const current = frame();
    view.render(explorer.snapshot(), current);
    const bar = container.querySelector<HTMLElement>(".param-axis")!;
    expect(bar.style.backgroundImage).toContain("var(--param-");
    expect(bar.style.backgroundImage).toContain(PALETTE.param.css(-PALETTE.param.limit));

    const plot = container.querySelector<SVGElement>(".param-plot")!;
    expect(plot.style.backgroundImage).toContain("linear-gradient(to top");
    expect(plot.style.backgroundImage).toContain("var(--output-");
    // Over this plot's own axis, which is taller than the output scale's
    // limit, so its ends are the saturated colour.
    expect(plot.style.backgroundImage).toContain(
      PALETTE.output.css(current.outputLimit),
    );
  });
});

describe("the inspector", () => {
  let container: HTMLElement;
  let explorer: Explorer;
  let inspector: Inspector;

  beforeEach(() => {
    container = document.createElement("div");
    document.body.replaceChildren(container);
    explorer = createExplorer(CHECKPOINT);
    explorer.setProbe({ kind: "free", x: [0.649, -0.64] });
    inspector = createInspector({
      container,
      activation: ACTIVATION,
      targetName: CHECKPOINT.dataset.targetName,
      inputNames: CHECKPOINT.dataset.inputNames,
    });
  });

  /** The activation plot's polyline, as (x, y) pairs in its own user units. */
  function curve(): [number, number][] {
    const points = container
      .querySelector<SVGElement>(".phi-plot .phi-curve")!
      .getAttribute("points")!;
    return points.split(" ").map((pair) => {
      const [x, y] = pair.split(",").map(Number);
      return [x, y] as [number, number];
    });
  }

  function dot(): { x: number; y: number } {
    const circle = container.querySelector<SVGElement>(".phi-plot .phi-dot")!;
    return { x: Number(circle.getAttribute("cx")), y: Number(circle.getAttribute("cy")) };
  }

  /**
   * The drawn curve's height at an x, between the samples either side.
   * Coordinates are written to two decimals, so comparisons against this
   * carry a tolerance of a hundredth of a user unit rather than being exact.
   */
  function curveAt(x: number): number {
    const points = curve();
    const index = points.findIndex(([px]) => px >= x);
    if (index <= 0) return points[Math.max(index, 0)][1];
    const [x0, y0] = points[index - 1];
    const [x1, y1] = points[index];
    return y0 + ((x - x0) / (x1 - x0)) * (y1 - y0);
  }

  /** The second panel the inspector renders: "Neuron: …". */
  function neuron(): HTMLElement {
    return container.querySelectorAll<HTMLElement>("section.panel")[1];
  }

  it("TABULATES ONE ROW PER INPUT TERM, and no bias row", () => {
    explorer.selectUnit(WEIGHT_UNIT);
    inspector.render(explorer.snapshot(), null);
    const heads = Array.from(neuron().querySelectorAll("thead th"));
    expect(heads.map((head) => head.textContent)).toEqual([
      "term",
      "w\u1d62",
      "a\u1d62",
      "w\u1d62\u00b7a\u1d62",
    ]);
    const rows = Array.from(neuron().querySelectorAll("tbody tr"));
    // The bias is not a term of the sum over i; it enters in the line below.
    expect(rows.map((row) => row.children[0].textContent)).toEqual(
      CHECKPOINT.parameters[0].W[WEIGHT_UNIT.out].map((_, i) => `i=${i}`),
    );
  });

  it("WRITES OUT THE TWO STEPS under the table, with their values", () => {
    explorer.selectUnit(WEIGHT_UNIT);
    const snapshot = explorer.snapshot();
    inspector.render(snapshot, null);
    const lines = Array.from(neuron().querySelectorAll(".formula")).map(
      (line) => line.textContent,
    );
    expect(lines[0]).toBe(
      `Preactivation z = \u03a3 w\u1d62 a\u1d62 + bias = ` +
        `${format(snapshot.trace.pre[1][WEIGHT_UNIT.out])}`,
    );
    expect(lines[1]).toBe(
      `Activation a = ReLU(z) = ${format(snapshot.trace.act[1][WEIGHT_UNIT.out])}`,
    );
  });

  it("ENDS WITH THE PLOT, beside the arithmetic and with nothing after it", () => {
    explorer.selectUnit(WEIGHT_UNIT);
    inspector.render(explorer.snapshot(), null);
    // The plot is the second half of the neuron panel's two-column body,
    // which is itself the last thing in the panel.
    const body = neuron().lastElementChild!;
    expect(body.classList.contains("neuron-body")).toBe(true);
    expect(body.children).toHaveLength(2);
    expect(body.children[0].classList.contains("neuron-math")).toBe(true);
    expect(body.lastElementChild!.classList.contains("phi-plot")).toBe(true);
  });

  it("names the output unit's activation as the identity it is", () => {
    explorer.selectUnit({ layer: CHECKPOINT.parameters.length - 1, out: 0 });
    inspector.render(explorer.snapshot(), null);
    const lines = Array.from(neuron().querySelectorAll(".formula")).map(
      (line) => line.textContent,
    );
    expect(lines[1]).toContain("Activation a = identity(z) = ");
    // No slope line: the identity has nothing to disclose.
    expect(lines).toHaveLength(2);
    expect(neuron().textContent).toContain("the output layer applies no nonlinearity");
  });

  it("PLOTS THE ACTIVATION FUNCTION WITH THE UNIT's OWN (z, a) ON IT", () => {
    explorer.selectUnit(WEIGHT_UNIT);
    inspector.render(explorer.snapshot(), null);
    const plot = container.querySelector<SVGElement>(".phi-plot")!;
    expect(plot.dataset.activation).toBe(ACTIVATION);

    // The panel's one structural claim: the dot is a point *of* the drawn
    // function, not a mark placed near it. Both activations are piecewise
    // linear or sampled finely enough for the straight segments to hold.
    const here = dot();
    expect(here.y).toBeCloseTo(curveAt(here.x), 1);
    // And inside the frame, never on it: the axis carries headroom so an
    // edit cannot push the dot out of view.
    const [first] = curve();
    const last = curve().at(-1)!;
    expect(here.x).toBeGreaterThan(first[0]);
    expect(here.x).toBeLessThan(last[0]);
  });

  it("MOVES THE DOT ALONG THE CURVE when the bias changes", () => {
    explorer.selectUnit(WEIGHT_UNIT);
    inspector.render(explorer.snapshot(), null);
    const before = dot();

    const bias = explorer.snapshot().parameters[0].b[WEIGHT_UNIT.out];
    explorer.setParameter({ kind: "bias", layer: 0, out: WEIGHT_UNIT.out }, bias + 0.4);
    inspector.render(explorer.snapshot(), explorer.lastEdit());
    const after = dot();

    // A larger bias is a larger z, so the dot slides to the right — and it
    // is still a point of the function, which the bias does not change.
    expect(after.x).toBeGreaterThan(before.x);
    expect(after.y).toBeCloseTo(curveAt(after.x), 1);
  });

  it("draws the output unit's identity, since that layer has no nonlinearity", () => {
    const output: UnitRef = { layer: CHECKPOINT.parameters.length - 1, out: 0 };
    explorer.selectUnit(output);
    inspector.render(explorer.snapshot(), null);
    expect(container.querySelector<SVGElement>(".phi-plot")!.dataset.activation).toBe(
      "identity",
    );
    // a = z over the whole range: one straight line, corner to corner of a
    // square frame on one scale.
    const points = curve();
    const [x0, y0] = points[0];
    const [x1, y1] = points.at(-1)!;
    expect(y0 - y1).toBeCloseTo(x1 - x0, 1);
    for (const [x, y] of points) {
      expect(y).toBeCloseTo(y0 + ((x - x0) / (x1 - x0)) * (y1 - y0), 1);
    }
  });

  it("has no empty state, since the selection is never empty", () => {
    inspector.render(explorer.snapshot(), null);
    expect(container.textContent).not.toContain("Nothing selected");
    expect(container.textContent).toContain("Neuron:");
    expect(container.querySelector(".phi-plot")).not.toBeNull();
  });

  it("no longer carries the parameter table or the downstream-effect table", () => {
    explorer.selectUnit(WEIGHT_UNIT);
    explorer.setParameter(WEIGHT, 1.75);
    inspector.render(explorer.snapshot(), explorer.lastEdit());
    expect(container.querySelector(".param-table")).toBeNull();
    expect(container.textContent).not.toContain("Downstream effect");
    // The edit still reports itself, which is the panel that stayed.
    expect(container.textContent).toContain("actual \u0394\u2113");
  });
});

describe("a panel's help popup", () => {
  let panel: HTMLElement;
  let legend: HTMLElement;
  let help: Help;

  function button(): HTMLButtonElement {
    return panel.querySelector<HTMLButtonElement>(".help-button")!;
  }
  function popup(): HTMLElement {
    return document.querySelector<HTMLElement>(".help-popup")!;
  }

  beforeEach(() => {
    document.body.replaceChildren();
    panel = document.createElement("section");
    panel.className = "panel";
    const heading = document.createElement("h2");
    heading.textContent = "Network";
    legend = document.createElement("p");
    legend.className = "legend";
    legend.textContent = "Edge colour and width: weight w.";
    panel.append(heading, legend);
    document.body.append(panel);
    help = createHelp(panel, legend, "Network");
  });

  afterEach(() => help.destroy());

  it("TAKES THE EXPLANATION OUT OF THE PANEL, so it cannot cover it", () => {
    // The popup is a child of the body, not of the panel, and the placement
    // in help.ts puts it beside the panel; nothing nests it over the plot.
    expect(panel.contains(legend)).toBe(false);
    expect(popup().parentElement).toBe(document.body);
    expect(popup().contains(legend)).toBe(true);
    expect(help.isOpen()).toBe(false);
    expect(popup().hidden).toBe(true);
    expect(button().getAttribute("aria-expanded")).toBe("false");
  });

  it("opens and closes from the question mark, and closes on Escape", () => {
    button().dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
    expect(help.isOpen()).toBe(true);
    expect(button().getAttribute("aria-expanded")).toBe("true");

    button().dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
    expect(help.isOpen()).toBe(false);

    button().dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
    document.dispatchEvent(new window.KeyboardEvent("keydown", { key: "Escape" }));
    expect(help.isOpen()).toBe(false);
  });

  it("closes on a click outside itself, but not on one inside", () => {
    button().dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
    popup().dispatchEvent(new window.MouseEvent("pointerdown", { bubbles: true }));
    expect(help.isOpen()).toBe(true);
    panel.dispatchEvent(new window.MouseEvent("pointerdown", { bubbles: true }));
    expect(help.isOpen()).toBe(false);
  });

  it("SHOWS WHATEVER THE PANEL LAST WROTE, since it holds the element itself", () => {
    // The views keep writing their legend as the state changes; moving the
    // element rather than copying its text is what keeps the popup current.
    legend.textContent = "Edge colour and width: contribution w\u00b7a.";
    button().dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
    expect(popup().textContent).toContain("contribution w\u00b7a");
  });
});

describe("the parameter tensors panel", () => {
  let container: HTMLElement;
  let explorer: Explorer;
  let view: TensorsView;

  /** Every tensor drawn, in document order: bias then weights, per layer. */
  function blocks(): SVGElement[] {
    return Array.from(container.querySelectorAll<SVGElement>(".tensor-block"));
  }
  function dotsOf(block: SVGElement): SVGElement[] {
    return Array.from(block.querySelectorAll<SVGElement>(".tensor-dot"));
  }
  function bands(): SVGElement[] {
    return Array.from(container.querySelectorAll<SVGElement>(".tensor-band"));
  }
  /** The slots a band covers: how the selection is read off the diagram. */
  function covered(band: SVGElement, block: SVGElement): SVGElement[] {
    const x = Number(band.getAttribute("x"));
    const width = Number(band.getAttribute("width"));
    return dotsOf(block).filter((dot) => {
      const cx = Number(dot.getAttribute("cx"));
      return cx > x && cx < x + width;
    });
  }

  beforeEach(() => {
    container = document.createElement("div");
    document.body.replaceChildren(container);
    explorer = createExplorer(CHECKPOINT);
    view = createTensorsView({ container });
  });

  it("DRAWS ONE SLOT PER PARAMETER, as a bias vector over a weight matrix", () => {
    view.render(explorer.snapshot());
    expect(blocks()).toHaveLength(2 * CHECKPOINT.parameters.length);
    CHECKPOINT.parameters.forEach((layer, index) => {
      const [bias, weights] = [blocks()[2 * index], blocks()[2 * index + 1]];
      expect(dotsOf(bias)).toHaveLength(layer.b.length);
      expect(dotsOf(weights)).toHaveLength(layer.W.flat().length);
    });
  });

  it("names each tensor by its shape and the layers it joins", () => {
    view.render(explorer.snapshot());
    const titles = blocks().map((block) => block.querySelector("title")!.textContent);
    expect(titles[0]).toBe(`1×${CHECKPOINT.parameters[0].b.length} biases for hidden layer 1`);
    expect(titles[1]).toBe(
      `${CHECKPOINT.parameters[0].W[0].length}×${CHECKPOINT.parameters[0].b.length} ` +
        `weights connecting the input layer to hidden layer 1`,
    );
    expect(titles.at(-1)).toContain("to the output layer");
  });

  it("SHADES THE SELECTED UNIT's OWN BAND: its bias slot and its weights", () => {
    explorer.selectUnit({ layer: 0, out: 2 });
    view.render(explorer.snapshot());
    const [biasBand, weightBand] = bands();
    expect(bands()).toHaveLength(2);
    // One slot of the bias vector, and the whole column under it: every
    // weight feeding that unit, and nothing else.
    expect(covered(biasBand, blocks()[0])).toHaveLength(1);
    expect(covered(weightBand, blocks()[1])).toHaveLength(
      CHECKPOINT.parameters[0].W[0].length,
    );
    expect(biasBand.getAttribute("x")).toBe(weightBand.getAttribute("x"));
    // In this layer's group, not another's.
    expect(covered(weightBand, blocks()[3])).toHaveLength(0);
  });

  it("moves the band one column along when the next unit is selected", () => {
    explorer.selectUnit({ layer: 0, out: 1 });
    view.render(explorer.snapshot());
    const before = Number(bands()[0].getAttribute("x"));
    const width = Number(bands()[0].getAttribute("width"));
    explorer.selectUnit({ layer: 0, out: 2 });
    view.render(explorer.snapshot());
    expect(Number(bands()[0].getAttribute("x"))).toBeCloseTo(before + width, 6);
  });
});
