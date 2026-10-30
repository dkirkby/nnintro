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
import { createColorScale, niceLimit, outputLimit, type Palette } from "../src/view/colormap.js";
import { createInputOutputView, type InputOutputView } from "../src/view/inputoutput.js";
import { createHelp, type Help } from "../src/view/help.js";
import { createInspector, type Inspector } from "../src/view/inspector.js";
import {
  createNetworkView,
  format,
  type NetworkFrame,
  type NetworkView,
} from "../src/view/network.js";
import { createParamSpaceView, type ParamSpaceView } from "../src/view/paramspace.js";
import { createTensorsView, type TensorsView } from "../src/view/tensors.js";
import { checkpointNamed, CLASSIFIER, PRIMARY } from "./fixtures.js";

const CHECKPOINT = checkpointNamed(PRIMARY);
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
  // The generator's own range of targets, which is what `main.ts` fits a
  // regression's output scale to: a property of the dataset alone, so an
  // edit cannot move it and the colour under the probe means the same
  // number either side of one.
  output: createColorScale(outputLimit(CHECKPOINT.dataset, SURFACE.extent), "output"),
};

/**
 * The frame `main.ts` hands the network panel, with everything switched
 * on: the display preferences are the reader's and live in the shell, so
 * the view is always told all of them.
 */
function netFrame(over: Partial<NetworkFrame> = {}): NetworkFrame {
  return {
    mode: "weights",
    palette: PALETTE,
    showSamples: false,
    showValues: true,
    showParameters: true,
    ...over,
  };
}

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
    view.render(explorer.snapshot(), netFrame());
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

  /**
   * The panel's own two switches, which build the figure up rather than
   * showing all of it at once: the shape of the network, then what the
   * parameters do to it, then the numbers themselves.
   */
  describe("putting the numbers and the parameters away", () => {
    const edges = () =>
      Array.from(container.querySelectorAll<SVGElement>('.dn-edge[data-key^="e:"]'));
    const nodes = () =>
      Array.from(container.querySelectorAll<SVGElement>('.dn-node[data-key^="n:"]'));

    it("TAKES THE NUMBER OFF EVERY NODE, and touches nothing else", () => {
      const coloured = nodes().map((node) => node.getAttribute("fill"));
      expect(container.querySelectorAll(".dn-node-label").length).toBeGreaterThan(0);

      view.render(explorer.snapshot(), netFrame({ showValues: false }));
      expect(container.querySelectorAll(".dn-node-label")).toHaveLength(0);
      // The parameters are a separate switch: the colours stay.
      expect(nodes().map((node) => node.getAttribute("fill"))).toEqual(coloured);
    });

    it("DRAWS THE SHAPE ALONE with the parameters off", () => {
      // A zero weight would be dotted grey on the parameter scale; with no
      // parameter to show there is nothing faint about it.
      explorer.setParameter(WEIGHT, 0);
      view.render(explorer.snapshot(), netFrame({ showParameters: false }));

      const widths = new Set(edges().map((edge) => edge.getAttribute("stroke-width")));
      const strokes = new Set(edges().map((edge) => edge.getAttribute("stroke")));
      expect(widths.size).toBe(1);
      expect(strokes).toEqual(new Set(["var(--ink-dim)"]));
      expect(container.querySelectorAll(".dn-edge.is-faint")).toHaveLength(0);
      // Every node the same ink, inputs included: nothing here stands for
      // a number any more.
      expect(new Set(nodes().map((node) => node.getAttribute("fill")))).toEqual(
        new Set(["var(--ink)"]),
      );
      // The numbers are the other switch, and it is still on.
      expect(container.querySelectorAll(".dn-node-label").length).toBeGreaterThan(0);
    });

    it("keeps the selected unit legible once every node is the same ink", () => {
      explorer.selectUnit(WEIGHT_UNIT);
      view.render(explorer.snapshot(), netFrame({ showParameters: false }));
      const node = container.querySelector(`[data-key="n:1:${WEIGHT_UNIT.out}"]`)!;
      // `is-plain` is what the stylesheet hangs the inside-out selected
      // node on; the rails along its edges are drawn either way.
      expect(node.classList.contains("is-plain")).toBe(true);
      expect(node.classList.contains("is-selected")).toBe(true);
      expect(container.querySelectorAll(".edge-outline")).toHaveLength(
        2 * CHECKPOINT.architecture.layerSizes[0],
      );
    });

    it("can put both away at once", () => {
      view.render(
        explorer.snapshot(),
        netFrame({ showValues: false, showParameters: false }),
      );
      expect(container.querySelectorAll(".dn-node-label")).toHaveLength(0);
      expect(new Set(edges().map((edge) => edge.getAttribute("stroke")))).toEqual(
        new Set(["var(--ink-dim)"]),
      );
      // Still a network: every node and every edge is there to be clicked.
      expect(nodes().length).toBeGreaterThan(0);
      expect(container.querySelectorAll(".edge-target").length).toBe(edges().length);
    });
  });

  it("GIVES A ZERO-WEIGHT EDGE A VISIBLE, SELECTABLE FORM", () => {
    explorer.setParameter(WEIGHT, 0);
    view.render(explorer.snapshot(), netFrame());
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
    view.render(explorer.snapshot(), netFrame());
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

    view.render(explorer.snapshot(), netFrame());
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

    view.render(explorer.snapshot(), netFrame());
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
    view.render(explorer.snapshot(), netFrame());
    const inputs = CHECKPOINT.architecture.layerSizes[0];
    expect(container.querySelector('[data-key="e:0:0:1"]')!.classList).toContain(
      "is-selected",
    );

    explorer.setParameter(WEIGHT, 0.5);
    view.render(explorer.snapshot(), netFrame());
    const stillSelected = container.querySelector('[data-key="e:0:0:1"]');
    expect(stillSelected!.classList).toContain("is-selected");
    // The whole unit, and only it: its node plus one edge per input.
    expect(container.querySelectorAll(".dn-edge.is-selected")).toHaveLength(inputs);
    expect(container.querySelectorAll(".dn-node.is-selected")).toHaveLength(1);
  });

  it("shows the edited value, not the checkpoint's", () => {
    explorer.setParameter(WEIGHT, -2.5);
    view.render(explorer.snapshot(), netFrame());
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
    view.render(snapshot, netFrame());
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
    view.render(snapshot, netFrame());
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
    view.render(snapshot, netFrame());
    const before = container.querySelector('[data-key="n:1:0"]')!.getAttribute("fill");
    view.render(snapshot, netFrame({ mode: "contributions" }));
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
      legend: document.createElement("p"),
      outputName: "ŷ",
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
  /**
   * The inspector renders into two panel bodies the shell owns, rather
   * than building its own panels: they are the workspace's, so that
   * folding or moving one is the reader's arrangement and not something a
   * checkpoint switch can take away.
   */
  let prediction: HTMLElement;
  let neuronBody: HTMLElement;
  /** The neuron panel's name lives in its title bar, so it is handed out. */
  let neuronTitle = "";

  beforeEach(() => {
    container = document.createElement("div");
    prediction = document.createElement("div");
    neuronBody = document.createElement("div");
    container.append(prediction, neuronBody);
    document.body.replaceChildren(container);
    explorer = createExplorer(CHECKPOINT);
    explorer.setProbe({ kind: "free", x: [0.649, -0.64] });
    neuronTitle = "";
    inspector = createInspector({
      prediction,
      neuron: neuronBody,
      setNeuronTitle: (title) => {
        neuronTitle = title;
      },
      activation: ACTIVATION,
      task: CHECKPOINT.dataset.task,
      targetName: CHECKPOINT.dataset.targetName,
      inputNames: CHECKPOINT.dataset.inputNames,
    });
  });

  /**
   * The plot's half-range, read off the `+N` tick it draws. The axis is
   * fitted to the unit's own z, so this is not a constant.
   */
  function axisLimit(): number {
    const ticks = Array.from(container.querySelectorAll(".phi-plot .slice-tick"));
    const positive = ticks.find((tick) => tick.textContent?.startsWith("+"))!;
    return Number(positive.textContent!.slice(1));
  }

  /** Where the dot stands on the z axis, inverting the plot's own mapping. */
  function dotZ(): number {
    // PHI = { size: 150, pad: 12 } in view/inspector.ts.
    const span = 150 - 2 * 12;
    const limit = axisLimit();
    return ((dot().x - 12) / span) * 2 * limit - limit;
  }

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

  /** What the inspector rendered into the neuron panel's body. */
  function neuron(): HTMLElement {
    return neuronBody.querySelector<HTMLElement>(".panel-content")!;
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
    const before = { z: dotZ() };

    const bias = explorer.snapshot().parameters[0].b[WEIGHT_UNIT.out];
    explorer.setParameter({ kind: "bias", layer: 0, out: WEIGHT_UNIT.out }, bias + 0.4);
    inspector.render(explorer.snapshot(), explorer.lastEdit());
    const after = { z: dotZ() };

    // z = Σ wᵢaᵢ + b, so a bias edit moves the dot along the z axis by
    // exactly the edit: this is the arithmetic and the plot agreeing.
    //
    // Deliberately read in z rather than in pixels. The axis is fitted to
    // the magnitude of this unit's own z, so an edit can widen the frame by
    // outgrowing a ladder stop *or* narrow it by moving z toward zero —
    // which is what the published checkpoint does here, 1.5 down to 1. The
    // dot then sits at a smaller pixel x while standing at a larger z, so a
    // pixel assertion would only be a statement about which checkpoint
    // happens to be published.
    // To two decimals: the dot's cx is written with toFixed(2), so reading
    // z back out of it cannot be sharper than that rounding.
    expect(after.z - before.z).toBeCloseTo(0.4, 2);
    // And it is still a point of the function, which the bias does not move.
    const here = dot();
    expect(here.y).toBeCloseTo(curveAt(here.x), 1);
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
    // The name is the panel's title now, not a heading inside its body.
    // The explorer opens on the output unit, which here is layer 2.
    expect(neuronTitle).toBe("Neuron: layer 2, unit 0");
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

  it("NAMES THE TWO ROWS ONCE, rotated in the gutter on the left", () => {
    view.render(explorer.snapshot());
    const labels = Array.from(container.querySelectorAll<SVGElement>(".tensor-axis"));
    // Every group shares the two bands, so the names are the diagram's own
    // rows rather than something repeated per layer.
    expect(labels.map((node) => node.textContent)).toEqual(["bias", "weight"]);
    for (const node of labels) {
      expect(node.getAttribute("transform")).toMatch(/^rotate\(-90 /);
    }
    // "bias" on the bias strip, "weight" below it on the weight block, and
    // both left of every slot.
    const [bias, weight] = labels.map((node) => Number(node.getAttribute("y")));
    expect(bias).toBeLessThan(weight);
    const leftmost = Math.min(
      ...dotsOf(blocks()[0]).map((dot) => Number(dot.getAttribute("cx"))),
    );
    for (const node of labels) {
      expect(Number(node.getAttribute("x"))).toBeLessThan(leftmost);
    }
  });

  it("NAMES A TENSOR FROM ANYWHERE INSIDE IT, not only over its slots", () => {
    view.render(explorer.snapshot());
    for (const block of blocks()) {
      // The <title> is the group's first child, so the frame and the gaps
      // between the slots resolve their tooltip to it too. `.tensor-frame`
      // carries `pointer-events: all` for the same reason: an unfilled
      // shape is not hit-tested.
      expect(block.firstElementChild!.tagName).toBe("title");
      expect(block.querySelector(".tensor-frame")).not.toBeNull();
    }
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

/**
 * A classifier's output is a logit, and the panels have to say so.
 *
 * The network's scalar output is never a probability: sigmoid is applied for
 * display and labeled, and the logit stays visible beside it because the
 * logit is what the parameters produce and what every plot draws (PLAN.md
 * section 3). The one thing that cannot be left on the logit axis is the
 * target, which is a class label.
 */
describe("a classifier's panels", () => {
  const CLASSIFIER_CHECKPOINT = checkpointNamed(CLASSIFIER);
  const CLASSIFIER_ACTIVATION = CLASSIFIER_CHECKPOINT.architecture.hiddenActivation!;
  const CLASSIFIER_SURFACE = outputSurface(
    CLASSIFIER_CHECKPOINT.parameters,
    CLASSIFIER_ACTIVATION,
    CLASSIFIER_CHECKPOINT.dataset.domain,
    12,
  );
  // The targets are deliberately NOT in this limit: 0 and 1 are class
  // labels, not points on the logit axis the surface spans.
  const CLASSIFIER_PALETTE: Palette = {
    param: createColorScale(
      niceLimit(
        CLASSIFIER_CHECKPOINT.parameters.flatMap((layer) => layer.W.flat()),
        CLASSIFIER_CHECKPOINT.parameters.flatMap((layer) => layer.b),
      ),
      "param",
    ),
    output: createColorScale(
      outputLimit(CLASSIFIER_CHECKPOINT.dataset, CLASSIFIER_SURFACE.extent),
      "output",
    ),
  };

  let container: HTMLElement;
  let explorer: Explorer;

  beforeEach(() => {
    container = document.createElement("div");
    document.body.replaceChildren(container);
    explorer = createExplorer(CLASSIFIER_CHECKPOINT);
    explorer.setProbe({ kind: "sample", index: 0 });
  });

  it("IS A CLASSIFIER AT ALL, with a logit and no analytic target", () => {
    expect(CLASSIFIER_CHECKPOINT.dataset.task).toBe("classification");
    expect(CLASSIFIER_CHECKPOINT.architecture.outputTransform).toBe("sigmoid");
    expect(CLASSIFIER_CHECKPOINT.dataset.hasAnalyticTarget).toBe(false);
  });

  it("READS THE OUTPUT AS A LOGIT, with the probability beside it", () => {
    const inspector = createInspector({
      prediction: container,
      neuron: document.createElement("div"),
      setNeuronTitle: () => {},
      activation: CLASSIFIER_ACTIVATION,
      task: "classification",
      targetName: CLASSIFIER_CHECKPOINT.dataset.targetName,
      inputNames: CLASSIFIER_CHECKPOINT.dataset.inputNames,
    });
    inspector.render(explorer.snapshot(), null);
    const labels = Array.from(container.querySelectorAll(".row-label")).map(
      (element) => element.textContent,
    );
    expect(labels).toContain("logit z");
    expect(labels).toContain("probability p");
    expect(labels).not.toContain("prediction ŷ");

    // The two numbers are one sigmoid apart, and the probability is the one
    // the sensitivity defaults to.
    const snapshot = explorer.snapshot();
    expect(snapshot.probability).toBeCloseTo(
      1 / (1 + Math.exp(-snapshot.trace.output)),
      12,
    );
    expect(labels).toContain("∂p/∂x");
    expect(labels).toContain("∂z/∂x");
  });

  it("DRAWS A CLASS LABEL AT THE END OF THE SCALE, not at its own value", () => {
    const view = createInputOutputView({
      container,
      dataset: CLASSIFIER_CHECKPOINT.dataset,
      onProbe: () => {},
    });
    const snapshot = explorer.snapshot();
    view.render(snapshot, {
      surface: CLASSIFIER_SURFACE,
      slices: CLASSIFIER_CHECKPOINT.dataset.domain.map((_, axis) =>
        outputSlice(
          snapshot.parameters,
          CLASSIFIER_ACTIVATION,
          CLASSIFIER_CHECKPOINT.dataset.domain,
          snapshot.x,
          axis,
        ),
      ),
      palette: CLASSIFIER_PALETTE,
      showSamples: true,
    });

    const limit = CLASSIFIER_PALETTE.output.limit;
    const ends = [
      CLASSIFIER_PALETTE.output.css(-limit),
      CLASSIFIER_PALETTE.output.css(limit),
    ];
    const fills = new Set(
      Array.from(container.querySelectorAll(".sample")).map((mark) =>
        mark.getAttribute("fill"),
      ),
    );
    // Exactly two colours, one per class, and both are scale ends.
    expect(fills.size).toBe(2);
    for (const fill of fills) expect(ends).toContain(fill);
  });

  it("SAYS THE NEUTRAL BAND IS THE DECISION BOUNDARY, since a zero logit is p = 0.5", () => {
    expect(CLASSIFIER_PALETTE.output.css(0)).toBe(CLASSIFIER_PALETTE.output.css(-0));
    const snapshot = explorer.snapshot();
    // The claim the legend makes: the colour scale's neutral point and the
    // p = 0.5 contour are the same locus.
    expect(1 / (1 + Math.exp(-0))).toBe(0.5);
    expect(snapshot.probability).not.toBeNull();
  });
});

describe("the samples on the map", () => {
  it("DRAW TRAINING AND HELD-OUT POINTS ALIKE, with no split glyph", () => {
    const container = document.createElement("div");
    document.body.replaceChildren(container);
    const explorer = createExplorer(CHECKPOINT);
    const view = createInputOutputView({
      container,
      dataset: CHECKPOINT.dataset,
      onProbe: () => {},
    });
    const snapshot = explorer.snapshot();
    view.render(snapshot, {
      surface: SURFACE,
      slices: CHECKPOINT.dataset.domain.map((_, axis) =>
        outputSlice(snapshot.parameters, ACTIVATION, CHECKPOINT.dataset.domain, snapshot.x, axis),
      ),
      palette: PALETTE,
      showSamples: true,
    });

    const samples = Array.from(container.querySelectorAll(".sample"));
    expect(samples).toHaveLength(CHECKPOINT.dataset.samples.y.length);
    expect(CHECKPOINT.dataset.samples.isTest.some(Boolean)).toBe(true);
    // The split is a number in "Prediction and loss", not a glyph: no
    // checkpoint here can show a gap the map would explain.
    for (const sample of samples) {
      expect(sample.getAttribute("class")).toBe("sample");
    }
  });
});

/**
 * Snapping the probe onto a sample.
 *
 * A free point next to a dot is not the dot: it has whatever target the
 * generator defines there, or none at all on a classifier. Landing the
 * probe *on* a sample is the only way to read that sample's own target and
 * its own point loss, so a drag inside a drawn dot takes it rather than the
 * pixel under the pointer.
 */
describe("snapping the probe onto a sample", () => {
  /** The map's mark layer, with the geometry jsdom does not provide. */
  function mapOf(container: HTMLElement): SVGElement {
    const map = container.querySelector<SVGElement>(".io-marks")!;
    // jsdom gives every element a zero-sized box, and the panel converts a
    // client position through it. Reporting the view box's own 300 square
    // makes a client coordinate one user unit, which is what the
    // expectations below are written in.
    Object.defineProperty(map, "getBoundingClientRect", {
      value: () => ({ left: 0, top: 0, width: 300, height: 300 }),
    });
    Object.defineProperty(map, "setPointerCapture", { value: () => {} });
    return map;
  }

  function drag(map: SVGElement, px: number, py: number): void {
    map.dispatchEvent(
      new window.MouseEvent("pointerdown", { bubbles: true, clientX: px, clientY: py }),
    );
  }

  const { domain, samples } = CHECKPOINT.dataset;
  /** Somewhere that is not a sample, for showing the enlarged dot go away. */
  const PROBE_AWAY = [0.649, -0.64];
  const toPixelX = (x: number) => ((x - domain[0][0]) / (domain[0][1] - domain[0][0])) * 300;
  const toPixelY = (y: number) => 300 - ((y - domain[1][0]) / (domain[1][1] - domain[1][0])) * 300;
  const pixelOf = (n: number) => [toPixelX(samples.x[n][0]), toPixelY(samples.x[n][1])];

  /**
   * A sample no other dot is near, found rather than guessed.
   *
   * Four hundred dots in a 300-unit square overlap, and the nearest one
   * wins a snap — correctly — so a test that names a sample has to name one
   * where "nearest" is not in doubt.
   */
  function isolatedSample(): number {
    for (let n = 0; n < samples.x.length; n += 1) {
      const [px, py] = pixelOf(n);
      const alone = samples.x.every((x, other) => {
        if (other === n) return true;
        return Math.hypot(toPixelX(x[0]) - px, toPixelY(x[1]) - py) > 10;
      });
      if (alone) return n;
    }
    throw new Error("every sample has a neighbour within a dot's width");
  }

  /** A pixel in the map that is in no dot at all, found rather than guessed. */
  function emptyPixel(): [number, number] {
    for (let px = 4; px < 300; px += 3) {
      for (let py = 4; py < 300; py += 3) {
        const clear = samples.x.every((x) => {
          const [sx, sy] = [toPixelX(x[0]), toPixelY(x[1])];
          return Math.hypot(sx - px, sy - py) > 8;
        });
        if (clear) return [px, py];
      }
    }
    throw new Error("the map has no empty pixel, so this test cannot run");
  }

  let container: HTMLElement;
  let explorer: Explorer;
  let view: InputOutputView;

  function frame(showSamples: boolean) {
    const snapshot = explorer.snapshot();
    return {
      surface: SURFACE,
      slices: domain.map((_, axis) =>
        outputSlice(snapshot.parameters, ACTIVATION, domain, snapshot.x, axis),
      ),
      palette: PALETTE,
      showSamples,
    };
  }

  beforeEach(() => {
    container = document.createElement("div");
    document.body.replaceChildren(container);
    explorer = createExplorer(CHECKPOINT);
    view = createInputOutputView({
      container,
      dataset: CHECKPOINT.dataset,
      onProbe: (probe) => explorer.setProbe(probe),
    });
  });

  it("TAKES THE SAMPLE a drag lands inside, with its own target", () => {
    view.render(explorer.snapshot(), frame(true));
    const n = isolatedSample();
    const [px, py] = pixelOf(n);
    // Inside the dot but not at its centre: the probe is the sample
    // exactly, not the pixel the pointer was on.
    drag(mapOf(container), px + 2, py - 1);
    const snapshot = explorer.snapshot();
    expect(snapshot.probe).toEqual({ kind: "sample", index: n });
    expect(snapshot.x).toEqual(samples.x[n]);
    expect(snapshot.target).toBe(samples.y[n]);
  });

  it("leaves the probe free where the drag lands in no dot", () => {
    view.render(explorer.snapshot(), frame(true));
    const [px, py] = emptyPixel();
    drag(mapOf(container), px, py);
    expect(explorer.snapshot().probe.kind).toBe("free");
  });

  it("DOES NOT SNAP ONTO A DOT NOBODY CAN SEE, while the samples are hidden", () => {
    view.render(explorer.snapshot(), frame(false));
    const [px, py] = pixelOf(isolatedSample());
    drag(mapOf(container), px, py);
    // Silently becoming a sample probe -- which has a target where a free
    // one may not -- with nothing on screen to explain it.
    expect(explorer.snapshot().probe.kind).toBe("free");
  });

  it("DRAWS THE SNAPPED DOT LARGER AND OVER THE CROSSHAIRS", () => {
    explorer.setProbe({ kind: "sample", index: 7 });
    view.render(explorer.snapshot(), frame(true));

    const probed = container.querySelector<SVGElement>(".sample.is-probe");
    expect(probed).not.toBeNull();
    expect(container.querySelectorAll(".sample.is-probe")).toHaveLength(1);
    const others = Array.from(
      container.querySelectorAll<SVGElement>(".sample:not(.is-probe)"),
    );
    expect(others).toHaveLength(samples.y.length - 1);
    for (const other of others) {
      expect(Number(probed!.getAttribute("r"))).toBeGreaterThan(
        Number(other.getAttribute("r")),
      );
    }

    // Over the crosshairs, which is the whole point: the dot being probed
    // is the one thing in the map they must not hide.
    const order = Array.from(probed!.parentNode!.children);
    const crosshairs = order.filter((node) =>
      (node.getAttribute("class") ?? "").startsWith("crosshair"),
    );
    expect(crosshairs.length).toBeGreaterThan(0);
    for (const line of crosshairs) {
      expect(order.indexOf(probed!)).toBeGreaterThan(order.indexOf(line));
    }
    // And the crosshairs meet at that sample, so they point at it.
    expect(probed!.getAttribute("cx")).toBe(
      toPixelX(samples.x[7][0]).toFixed(2),
    );
  });

  it("drops the enlarged dot again when the probe goes back to a free point", () => {
    explorer.setProbe({ kind: "sample", index: 7 });
    view.render(explorer.snapshot(), frame(true));
    explorer.setProbe({ kind: "free", x: PROBE_AWAY });
    view.render(explorer.snapshot(), frame(true));
    expect(container.querySelectorAll(".sample.is-probe")).toHaveLength(0);
    expect(container.querySelectorAll(".sample")).toHaveLength(samples.y.length);
  });
});

/**
 * The output bar in the network panel's right-hand gutter.
 *
 * The output node is the prediction -- its fill and its badge -- and the bar
 * is the axis that fill came off, so the prediction and the target it is
 * being compared against are two marks on one scale rather than two printed
 * numbers to relate. It appears only with the samples shown and the probe on
 * one of them: both conditions the reader can see on screen.
 */
describe("the output bar beside the output node", () => {
  interface Marker {
    side: string;
    tipX: number;
    tipY: number;
    baseX: number;
    title: string;
  }

  function markers(container: HTMLElement): Marker[] {
    return Array.from(container.querySelectorAll<SVGElement>(".output-marker")).map(
      (node) => {
        const points = (node.getAttribute("points") ?? "")
          .split(" ")
          .map((pair) => pair.split(",").map(Number));
        return {
          side: node.classList.contains("is-left") ? "left" : "right",
          baseX: points[0][0],
          tipX: points[2][0],
          tipY: points[2][1],
          title: node.querySelector("title")?.textContent ?? "",
        };
      },
    );
  }

  /** The bar's own frame, which is where the geometry is read from. */
  function frameOf(container: HTMLElement) {
    const rect = container.querySelector<SVGElement>(".output-bar-frame")!;
    const read = (name: string) => Number(rect.getAttribute(name));
    return { x: read("x"), y: read("y"), width: read("width"), height: read("height") };
  }

  /** Where a value belongs on the drawn bar, pinned to its ends. */
  function rowOf(container: HTMLElement, value: number, limit: number): number {
    const bar = frameOf(container);
    const fraction = (limit - value) / (2 * limit);
    return bar.y + bar.height * Math.min(1, Math.max(0, fraction));
  }

  function mount(checkpoint: ReturnType<typeof checkpointNamed>) {
    const container = document.createElement("div");
    document.body.replaceChildren(container);
    const explorer = createExplorer(checkpoint);
    const view = createNetworkView({
      container,
      layerSizes: checkpoint.architecture.layerSizes,
      idPrefix: "bar-test",
      onSelect: () => {},
    });
    return { container, explorer, view };
  }

  it("APPEARS ONLY WITH THE SAMPLES SHOWN AND THE PROBE ON ONE of them", () => {
    const { container, explorer, view } = mount(CHECKPOINT);
    // A free probe on this generator HAS a target -- the tilted plane is
    // analytic everywhere -- so this is the display rule, not the absence
    // of a number: a mark with no dot on screen to explain where it came
    // from is worse than no mark.
    expect(explorer.snapshot().target).not.toBeNull();
    view.render(explorer.snapshot(), netFrame({ showSamples: true }));
    expect(container.querySelector(".output-bar")).toBeNull();

    explorer.setProbe({ kind: "sample", index: 3 });
    view.render(explorer.snapshot(), netFrame());
    expect(container.querySelector(".output-bar")).toBeNull();

    view.render(explorer.snapshot(), netFrame({ showSamples: true }));
    expect(container.querySelector(".output-bar")).not.toBeNull();
    expect(markers(container)).toHaveLength(2);
  });

  it("MARKS THE PREDICTION LEFT AND THE TARGET RIGHT, on one scale", () => {
    const { container, explorer, view } = mount(CHECKPOINT);
    explorer.setProbe({ kind: "sample", index: 3 });
    const snapshot = explorer.snapshot();
    view.render(snapshot, netFrame({ showSamples: true }));

    const bar = frameOf(container);
    const limit = PALETTE.output.limit;
    const output = snapshot.trace.act[snapshot.trace.act.length - 1][0];
    const [left, right] = markers(container).sort((a, b) => a.baseX - b.baseX);

    expect(left.side).toBe("left");
    expect(right.side).toBe("right");
    // Each tip touches its own edge of the bar, at its own value's row.
    expect(left.tipX).toBeCloseTo(bar.x, 6);
    expect(right.tipX).toBeCloseTo(bar.x + bar.width, 6);
    expect(left.tipY).toBeCloseTo(rowOf(container, output, limit), 2);
    expect(right.tipY).toBeCloseTo(
      rowOf(container, CHECKPOINT.dataset.samples.y[3], limit),
      2,
    );
    // The bases point away from the bar, so the triangles face it.
    expect(left.baseX).toBeLessThan(left.tipX);
    expect(right.baseX).toBeGreaterThan(right.tipX);
    expect(left.title).toContain(format(output));
    expect(right.title).toBe(`target ${format(CHECKPOINT.dataset.samples.y[3])}`);
  });

  it("names the target marker in rotated text, on the marker's own row", () => {
    const { container, explorer, view } = mount(CHECKPOINT);
    explorer.setProbe({ kind: "sample", index: 3 });
    view.render(explorer.snapshot(), netFrame({ showSamples: true }));
    const label = container.querySelector<SVGElement>(".output-bar-label")!;
    expect(label.textContent).toBe("target");
    const x = Number(label.getAttribute("x"));
    const y = Number(label.getAttribute("y"));
    expect(label.getAttribute("transform")).toBe(`rotate(-90 ${x.toFixed(2)} ${y.toFixed(2)})`);
    const right = markers(container).find((mark) => mark.side === "right")!;
    expect(y).toBeCloseTo(right.tipY, 2);
    // Beyond the marker it labels, so it never sits over the bar.
    expect(x).toBeGreaterThan(right.baseX);
  });

  it("PINS A PREDICTION THAT RUNS OFF THE SCALE to the end it ran past", () => {
    const { container, explorer, view } = mount(CHECKPOINT);
    explorer.setProbe({ kind: "sample", index: 3 });
    const last = CHECKPOINT.parameters.length - 1;
    // The scale is the dataset's targets, and a prediction is free to leave
    // them: the marker can then only say which end it left by.
    explorer.setParameter({ kind: "bias", layer: last, out: 0 }, 20 * PALETTE.output.limit);
    const snapshot = explorer.snapshot();
    view.render(snapshot, netFrame({ showSamples: true }));
    const bar = frameOf(container);
    const left = markers(container).find((mark) => mark.side === "left")!;
    expect(left.tipY).toBeCloseTo(bar.y, 2);
    // Position is pinned; the number is not silently changed with it.
    expect(left.title).toContain(
      format(snapshot.trace.act[snapshot.trace.act.length - 1][0]),
    );
  });

  it("RESERVES THE GUTTER whether the bar is drawn or not", () => {
    // A frame that grew when a sample was selected would resize the whole
    // diagram at the moment the reader was looking at one node of it.
    const { container, explorer, view } = mount(checkpointNamed("tilted-plane__2-4-4-1__relu"));
    const viewBoxes: string[] = [];
    const rightmost = () => {
      const nodes = Array.from(container.querySelectorAll<SVGElement>(".dn-node"));
      return Math.max(
        ...nodes.map((node) => Number(node.getAttribute("cx")) + Number(node.getAttribute("r"))),
      );
    };

    view.render(explorer.snapshot(), netFrame({ showSamples: true }));
    viewBoxes.push(container.querySelector("svg")!.getAttribute("viewBox")!);
    const diagram = rightmost();

    explorer.setProbe({ kind: "sample", index: 3 });
    view.render(explorer.snapshot(), netFrame({ showSamples: true }));
    viewBoxes.push(container.querySelector("svg")!.getAttribute("viewBox")!);
    expect(viewBoxes[0]).toBe(viewBoxes[1]);
    expect(rightmost()).toBeCloseTo(diagram, 6);

    // And the bar is in the gutter the network was kept out of, clear of
    // the badge written across the output node as well as of the node.
    const bar = frameOf(container);
    const marks = markers(container);
    const width = Number(viewBoxes[1].split(" ")[2]);
    const badge = Array.from(container.querySelectorAll<SVGElement>(".dn-node-label"));
    const widest = Math.max(
      ...badge.map(
        (text) => Number(text.getAttribute("x")) + ((text.textContent ?? "").length * 18) / 4,
      ),
    );
    expect(Math.min(...marks.map((mark) => mark.baseX))).toBeGreaterThan(widest);
    expect(bar.x).toBeGreaterThan(diagram);
    const label = container.querySelector<SVGElement>(".output-bar-label")!;
    expect(Number(label.getAttribute("x"))).toBeLessThan(width);
  });

  it("PUTS A CLASS LABEL AT THE END OF THE SCALE, not on the logit axis", () => {
    const classifier = checkpointNamed(CLASSIFIER);
    const { container, explorer, view } = mount(classifier);
    const palette = PALETTE_FOR(classifier, classifier.architecture.hiddenActivation!);
    // A free point in a stochastic classification dataset has no label at
    // all, so there is nothing to mark until the probe is on a sample.
    expect(explorer.snapshot().target).toBeNull();
    view.render(explorer.snapshot(), netFrame({ palette, showSamples: true }));
    expect(container.querySelector(".output-bar")).toBeNull();

    const index = classifier.dataset.samples.y.findIndex((y) => y === 1);
    explorer.setProbe({ kind: "sample", index });
    view.render(explorer.snapshot(), netFrame({ palette, showSamples: true }));
    const bar = frameOf(container);
    const right = markers(container).find((mark) => mark.side === "right")!;
    // 0 and 1 are not points on the logit axis the bar spans, so the mark
    // goes to the end of the scale its class wants -- the same convention
    // the map uses to colour that very sample.
    expect(right.tipY).toBeCloseTo(bar.y, 2);
    // Formatted as the inspector formats the same number, where it reads
    // `class y`, rather than prettified here alone.
    expect(right.title).toBe(`target class ${format(1)}`);

    const zero = classifier.dataset.samples.y.findIndex((y) => y === 0);
    explorer.setProbe({ kind: "sample", index: zero });
    view.render(explorer.snapshot(), netFrame({ palette, showSamples: true }));
    const bottom = markers(container).find((mark) => mark.side === "right")!;
    expect(bottom.tipY).toBeCloseTo(bar.y + bar.height, 2);

    // The label is pulled back inside the bar rather than centred on a
    // marker at its very end, where half the word would hang past the
    // scale it names. A classifier's target is always at an end, so this
    // is the usual case here rather than an edge of one.
    const label = Number(
      container.querySelector<SVGElement>(".output-bar-label")!.getAttribute("y"),
    );
    expect(label).toBeLessThan(bottom.tipY);
    expect(label).toBeGreaterThan(bar.y + bar.height - 30);
  });
});

/**
 * Every panel against every shape the controls can now select.
 *
 * The explorer used to have one architecture, so depth 0 (no hidden layer
 * at all, one parameter layer) and depth 2 were unreachable and untested.
 * These are the shapes themselves rather than the numbers: that each panel
 * draws the right number of things for the architecture it was handed, and
 * that nothing throws on the edges of the family.
 */
describe("the panels at every selectable shape", () => {
  const SHAPES = [
    "tilted-plane__2-1__none",
    "tilted-plane__2-2-1__identity",
    "curved-surface__2-2-4-1__tanh",
    "xor-quadrants__2-4-4-1__relu",
  ];

  describe.each(SHAPES)("%s", (id) => {
    const checkpoint = checkpointNamed(id);
    const sizes = checkpoint.architecture.layerSizes;
    const activation = checkpoint.architecture.hiddenActivation ?? "identity";
    let container: HTMLElement;
    let explorer: Explorer;

    beforeEach(() => {
      container = document.createElement("div");
      document.body.replaceChildren(container);
      explorer = createExplorer(checkpoint);
    });

    it("OPENS ON THE OUTPUT UNIT, whatever the depth", () => {
      expect(explorer.snapshot().selection).toEqual({ layer: sizes.length - 2, out: 0 });
    });

    it("DRAWS ONE DOT PER PARAMETER in the tensor panel", () => {
      const view = createTensorsView({ container });
      view.render(explorer.snapshot());
      const expected = checkpoint.parameters.reduce(
        (total, layer) => total + layer.b.length + layer.W.length * layer.W[0].length,
        0,
      );
      expect(container.querySelectorAll(".tensor-dot")).toHaveLength(expected);
      // Two rectangles per parameter layer: the biases and their matrix.
      expect(container.querySelectorAll(".tensor-block")).toHaveLength(
        2 * checkpoint.parameters.length,
      );
    });

    it("DRAWS EVERY NODE AND EDGE of the network", () => {
      const view = createNetworkView({
        container,
        layerSizes: sizes,
        idPrefix: `t-${id}`,
        onSelect: () => {},
      });
      view.render(explorer.snapshot(), netFrame({ palette: PALETTE_FOR(checkpoint, activation) }));
      const edges = sizes.slice(1).reduce((total, out, index) => total + out * sizes[index], 0);
      expect(container.querySelectorAll('[data-key^="n:"]')).toHaveLength(
        sizes.reduce((a, b) => a + b, 0),
      );
      expect(container.querySelectorAll('.dn-edge[data-key^="e:"]')).toHaveLength(edges);
    });

    it("SWEEPS THE OUTPUT UNIT's OWN PARAMETERS, bias plus its fan-in", () => {
      const view = createParamSpaceView({
        container,
        legend: document.createElement("p"),
        outputName: checkpoint.dataset.task === "classification" ? "z" : "ŷ",
        onEdit: () => {},
      });
      const snapshot = explorer.snapshot();
      const palette = PALETTE_FOR(checkpoint, activation);
      const curves = parameterCurves(
        snapshot.parameters,
        activation,
        snapshot.x,
        palette.param.limit,
        unitParameters(snapshot.parameters, snapshot.selection),
      );
      view.render(snapshot, {
        curves,
        paramScale: palette.param,
        outputScale: palette.output,
        outputLimit: niceLimit(curves.flatMap((curve) => curve.outputs)),
      });
      // The fan-in of the output unit is the last hidden width, or the
      // input width when there is no hidden layer at all.
      expect(curves).toHaveLength(1 + sizes[sizes.length - 2]);
      expect(container.querySelectorAll(".param-curve")).toHaveLength(curves.length);
      expect(container.querySelectorAll(".param-slider")).toHaveLength(curves.length);
    });

    it("WRITES ONE TABLE ROW PER INPUT TERM of the selected neuron", () => {
      const inspector = createInspector({
        prediction: document.createElement("div"),
        neuron: container,
        setNeuronTitle: () => {},
        activation,
        task: checkpoint.dataset.task,
        targetName: checkpoint.dataset.targetName,
        inputNames: checkpoint.dataset.inputNames,
      });
      inspector.render(explorer.snapshot(), null);
      expect(container.querySelectorAll("table.contributions tbody tr")).toHaveLength(
        sizes[sizes.length - 2],
      );
      // The output layer applies no nonlinearity, so its unit plots the
      // identity whatever the hidden activation is.
      expect(
        container.querySelector<SVGElement>(".phi-plot")!.dataset.activation,
      ).toBe("identity");
    });
  });
});

/** The palette `main.ts` would build for this checkpoint. */
function PALETTE_FOR(
  checkpoint: ReturnType<typeof checkpointNamed>,
  activation: Parameters<typeof outputSurface>[1],
): Palette {
  const surface = outputSurface(checkpoint.parameters, activation, checkpoint.dataset.domain, 8);
  return {
    param: createColorScale(
      niceLimit(
        checkpoint.parameters.flatMap((layer) => layer.W.flat()),
        checkpoint.parameters.flatMap((layer) => layer.b),
      ),
      "param",
    ),
    output: createColorScale(outputLimit(checkpoint.dataset, surface.extent), "output"),
  };
}
