/**
 * @vitest-environment jsdom
 *
 * The view half of the Stage 0 exit condition: an edit preserves the
 * selection in the rendered SVG, and a new edit cancels a stale animation
 * instead of letting it finish over the new state.
 */
import { beforeEach, describe, expect, it } from "vitest";

import { forward } from "../src/model/evaluate.js";
import { computeRipple, originOf } from "../src/model/ripple.js";
import { outputSlice, outputSurface } from "../src/model/surface.js";
import { createExplorer, type Explorer } from "../src/state.js";
import type { ParameterRef } from "../src/model/types.js";
import { edgeParameter } from "../src/view/adapter.js";
import { createColorScale, niceLimit } from "../src/view/colormap.js";
import { createInputSpaceView, type InputSpaceView } from "../src/view/inputspace.js";
import { createNetworkView, type NetworkView } from "../src/view/network.js";
import { createOutputSpaceView, type OutputSpaceView } from "../src/view/outputspace.js";
import { createRippleAnimator } from "../src/view/ripple.js";
import { loadCheckpoint, loadManifest } from "./fixtures.js";

const CHECKPOINT = loadCheckpoint(loadManifest().checkpoints[0].url);
const WEIGHT: ParameterRef = { kind: "weight", layer: 0, out: 1, in: 0 };
const ACTIVATION = CHECKPOINT.architecture.hiddenActivation!;
const SURFACE = outputSurface(CHECKPOINT.parameters, ACTIVATION, CHECKPOINT.dataset.domain, 12);
const SCALE = createColorScale(
  niceLimit([SURFACE.extent], CHECKPOINT.dataset.samples.y, CHECKPOINT.dataset.domain.flat()),
);

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
    view.render(explorer.snapshot(), "activations", SCALE);
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
    view.render(explorer.snapshot(), "activations", SCALE);
    expect(container.querySelector('[data-key="e:0:0:1"]')!.classList).toContain(
      "is-selected",
    );

    explorer.setParameter(WEIGHT, 0.5);
    view.render(explorer.snapshot(), "activations", SCALE);
    const stillSelected = container.querySelector('[data-key="e:0:0:1"]');
    expect(stillSelected!.classList).toContain("is-selected");
    expect(container.querySelectorAll(".is-selected")).toHaveLength(1);
  });

  it("shows the edited value, not the checkpoint's", () => {
    explorer.setParameter(WEIGHT, -2.5);
    view.render(explorer.snapshot(), "weights", SCALE);
    const label = container
      .querySelector('[data-key="e:0:0:1"]')!
      .getAttribute("aria-label");
    expect(label).toContain("-2.5");
  });
});

describe("the ripple animator", () => {
  let container: HTMLElement;
  let view: NetworkView;

  function makeRipple(ref: ParameterRef, value: number) {
    const explorer = createExplorer(CHECKPOINT);
    // Off the origin, where the explorer now opens: an input of exactly zero
    // carries no change out of a first-layer weight, so nothing would
    // animate and the test would pass vacuously.
    explorer.setProbe({ kind: "free", x: [0.73, -0.41] });
    const before = explorer.snapshot().trace;
    explorer.setParameter(ref, value);
    const after = explorer.snapshot().trace;
    return computeRipple(before, after, { ...originOf(ref), kind: "parameter" });
  }

  beforeEach(() => {
    container = document.createElement("div");
    document.body.replaceChildren(container);
    view = createNetworkView({
      container,
      layerSizes: CHECKPOINT.architecture.layerSizes,
      idPrefix: "ripple-test",
      onSelect: () => {},
    });
    view.render(createExplorer(CHECKPOINT).snapshot(), "activations", SCALE);
  });

  it("marks changed and blocked nodes without any motion when asked", async () => {
    const animator = createRippleAnimator({
      overlay: () => view.overlay(),
      layout: () => view.layout(),
      prefersReducedMotion: () => true,
    });
    await animator.play(makeRipple(WEIGHT, 2));
    expect(animator.reducedMotion).toBe(true);
    // Static deltas instead of a pulse, and no zipline group was created.
    expect(view.overlay().querySelectorAll("circle").length).toBeGreaterThan(0);
    expect(view.overlay().querySelectorAll("g").length).toBe(0);
  });

  it("CANCELS A STALE RUN when a new edit arrives", async () => {
    const animator = createRippleAnimator({
      overlay: () => view.overlay(),
      layout: () => view.layout(),
      prefersReducedMotion: () => false,
    });
    const first = animator.play(makeRipple(WEIGHT, 2));
    const generationDuringFirst = animator.generation;
    // Guard against a vacuous pass: the first run really did start, and
    // really did mount a zipline group to animate.
    expect(animator.running).toBe(true);
    expect(view.overlay().querySelectorAll("g").length).toBeGreaterThan(0);

    // A second edit lands before the first run can finish.
    const second = animator.play(makeRipple(WEIGHT, -2));
    expect(animator.generation).toBeGreaterThan(generationDuringFirst);

    // The stale run resolves rather than hanging or continuing.
    await expect(first).resolves.toBeUndefined();
    animator.cancel();
    await expect(second).resolves.toBeUndefined();
    expect(animator.running).toBe(false);
  });

  it("lets overlapping runs finish without destroying each other's groups", async () => {
    // Each run owns the controller it mounted. Sharing one slot let a stale
    // run tear down the group of the run that had replaced it, which then
    // threw on its own cleanup.
    const animator = createRippleAnimator({
      overlay: () => view.overlay(),
      layout: () => view.layout(),
      prefersReducedMotion: () => false,
    });
    const runs = [
      animator.play(makeRipple(WEIGHT, 2)),
      animator.play(makeRipple(WEIGHT, -2)),
      animator.play(makeRipple(WEIGHT, 1.5)),
    ];
    animator.cancel();
    await expect(Promise.all(runs)).resolves.toEqual([undefined, undefined, undefined]);
    expect(animator.running).toBe(false);
    expect(view.overlay().querySelectorAll("g")).toHaveLength(0);
  });

  it("cancel() stops a run and leaves nothing animating", async () => {
    const animator = createRippleAnimator({
      overlay: () => view.overlay(),
      layout: () => view.layout(),
      prefersReducedMotion: () => false,
    });
    const run = animator.play(makeRipple(WEIGHT, 3));
    animator.cancel();
    await expect(run).resolves.toBeUndefined();
    expect(animator.running).toBe(false);
  });

  it("destroy() clears the overlay and refuses further runs", async () => {
    const animator = createRippleAnimator({
      overlay: () => view.overlay(),
      layout: () => view.layout(),
      prefersReducedMotion: () => true,
    });
    await animator.play(makeRipple(WEIGHT, 2));
    animator.destroy();
    expect(view.overlay().childElementCount).toBe(0);
    await animator.play(makeRipple(WEIGHT, 1));
    expect(view.overlay().childElementCount).toBe(0);
  });

  it("animates only the edges that actually carried a change", async () => {
    // A bias edit on a unit that is off at this probe changes nothing
    // downstream, so there is nothing to animate even though the output is
    // structurally reachable.
    const explorer = createExplorer(CHECKPOINT);
    const probe = CHECKPOINT.goldenProbes.find((p) => p.id === "most-inactive")!;
    explorer.setProbe({ kind: "free", x: probe.x });
    const before = explorer.snapshot().trace;
    const offUnit = before.act[1].findIndex((a) => a === 0);
    expect(offUnit).toBeGreaterThanOrEqual(0);

    const ref: ParameterRef = { kind: "bias", layer: 0, out: offUnit };
    const current = explorer.snapshot().parameters[0].b[offUnit];
    // Still negative after the nudge: the unit stays off.
    explorer.setParameter(ref, current - 0.1);
    const ripple = computeRipple(before, explorer.snapshot().trace, {
      ...originOf(ref),
      kind: "parameter",
    });
    expect(ripple.hasBlocked).toBe(true);
    expect(ripple.stages.flatMap((s) => s.edges).every((e) => !e.carried)).toBe(true);
  });

  it("the output really did change for an edit that is not blocked", () => {
    const explorer = createExplorer(CHECKPOINT);
    const before = forward(explorer.snapshot().parameters, "relu", explorer.snapshot().x);
    explorer.setParameter({ kind: "bias", layer: 1, out: 0 }, 5);
    const after = explorer.snapshot().trace;
    expect(after.output).not.toBeCloseTo(before.output, 6);
  });
});

describe("the input space", () => {
  let container: HTMLElement;
  let explorer: Explorer;
  let view: InputSpaceView;

  beforeEach(() => {
    container = document.createElement("div");
    document.body.replaceChildren(container);
    explorer = createExplorer(CHECKPOINT);
    view = createInputSpaceView({
      container,
      dataset: CHECKPOINT.dataset,
      onProbe: (probe) => explorer.setProbe(probe),
    });
  });

  it("SHOWS NO SAMPLES until they are asked for", () => {
    view.render(explorer.snapshot(), { surface: SURFACE, scale: SCALE, showSamples: false });
    expect(container.querySelectorAll(".sample")).toHaveLength(0);
    // The probe and its sensitivity arrow are still there to be dragged.
    expect(container.querySelectorAll(".probe")).toHaveLength(1);

    view.render(explorer.snapshot(), { surface: SURFACE, scale: SCALE, showSamples: true });
    expect(container.querySelectorAll(".sample")).toHaveLength(
      CHECKPOINT.dataset.samples.y.length,
    );
  });

  it("colours the samples with the shared scale, held-out ones included", () => {
    view.render(explorer.snapshot(), { surface: SURFACE, scale: SCALE, showSamples: true });
    const marks = Array.from(container.querySelectorAll<SVGElement>(".sample"));
    const first = CHECKPOINT.dataset.samples.y[0];
    expect(marks[0].getAttribute("fill")).toBe(SCALE.css(first));
    // A held-out point keeps its fill: it is dashed, not hollow, so the
    // "does this dot vanish into the surface" reading still works.
    const test = marks.find((mark) => mark.classList.contains("is-test"));
    expect(test!.getAttribute("fill")).toMatch(/^color-mix/);
  });

  it("draws a colour bar along each input axis, over the domain", () => {
    view.render(explorer.snapshot(), { surface: SURFACE, scale: SCALE, showSamples: false });
    const [[x1lo, x1hi]] = CHECKPOINT.dataset.domain;
    const bar = container.querySelector<HTMLElement>(".axis-bar-x")!;
    expect(bar.style.backgroundImage).toContain("linear-gradient(to right");
    expect(bar.style.backgroundImage).toContain(SCALE.css(x1lo));
    expect(bar.style.backgroundImage).toContain(SCALE.css(x1hi));
    expect(bar.querySelector(".axis-bar-label")!.textContent).toBe(
      CHECKPOINT.dataset.inputNames[0],
    );

    // The second axis runs upwards, so its gradient must not be the first's.
    const other = container.querySelector<HTMLElement>(".axis-bar-y")!;
    expect(other.style.backgroundImage).toContain("linear-gradient(to top");
  });

  it("reads the same colour for a value as the network's nodes do", () => {
    const network = document.createElement("div");
    document.body.append(network);
    const nodes = createNetworkView({
      container: network,
      layerSizes: CHECKPOINT.architecture.layerSizes,
      idPrefix: "shared-scale",
      onSelect: () => {},
    });
    // Away from the centre, so neither input is the colour map's zero.
    explorer.setProbe({ kind: "free", x: [0.73, -0.41] });
    const snapshot = explorer.snapshot();
    nodes.render(snapshot, "activations", SCALE);
    view.render(snapshot, { surface: SURFACE, scale: SCALE, showSamples: false });

    // The first input's node, the bar along that axis and the surface all
    // take their colour from one scale: the node fill is exactly the colour
    // that scale gives the probe's coordinate.
    const input = network.querySelector('[data-key="n:0:0"]')!;
    expect(input.getAttribute("fill")).toBe(SCALE.css(snapshot.x[0]));
    const bar = container.querySelector<HTMLElement>(".axis-bar-x")!;
    expect(bar.style.backgroundImage).toContain(SCALE.css(CHECKPOINT.dataset.domain[0][0]));

    // And the output node is the colour of the surface under the probe.
    const last = CHECKPOINT.architecture.layerSizes.length - 1;
    const output = network.querySelector(`[data-key="n:${last}:0"]`)!;
    expect(output.getAttribute("fill")).toBe(SCALE.css(snapshot.trace.output));
  });
});


describe("the output space", () => {
  let container: HTMLElement;
  let explorer: Explorer;
  let view: OutputSpaceView;

  function slicesFor(): ReturnType<typeof outputSlice>[] {
    const snapshot = explorer.snapshot();
    return CHECKPOINT.dataset.domain.map((_, axis) =>
      outputSlice(CHECKPOINT.parameters, ACTIVATION, CHECKPOINT.dataset.domain, snapshot.x, axis),
    );
  }

  beforeEach(() => {
    container = document.createElement("div");
    document.body.replaceChildren(container);
    explorer = createExplorer(CHECKPOINT);
    explorer.setProbe({ kind: "free", x: [0.649, -0.64] });
    view = createOutputSpaceView({ container, dataset: CHECKPOINT.dataset });
  });

  it("draws one plot per input, each over the shared colour scale", () => {
    view.render(explorer.snapshot(), { slices: slicesFor(), scale: SCALE });
    const plots = container.querySelectorAll(".slice");
    expect(plots).toHaveLength(CHECKPOINT.dataset.inputNames.length);
    for (const gradient of Array.from(container.querySelectorAll<HTMLElement>(".slice-gradient"))) {
      // Upright: the top of the plot is +limit, so the gradient runs to top.
      expect(gradient.style.backgroundImage).toContain("linear-gradient(to top");
      expect(gradient.style.backgroundImage).toContain(SCALE.css(SCALE.limit));
      expect(gradient.style.backgroundImage).toContain(SCALE.css(-SCALE.limit));
    }
  });

  it("captions each plot with the input it is holding, and where", () => {
    view.render(explorer.snapshot(), { slices: slicesFor(), scale: SCALE });
    const captions = Array.from(container.querySelectorAll(".slice-caption")).map(
      (node) => node.textContent,
    );
    expect(captions[0]).toBe("ŷ(x1) at x2 = -0.640");
    expect(captions[1]).toBe("ŷ(x2) at x1 = 0.649");
  });

  it("MARKS THE PROBE AT THE PREDICTION's HEIGHT in both plots", () => {
    const snapshot = explorer.snapshot();
    view.render(snapshot, { slices: slicesFor(), scale: SCALE });
    const svg = container.querySelector(".slice-marks")!;
    const height = Number(svg.getAttribute("viewBox")!.split(" ")[3]);
    const expected = height / 2 - (snapshot.trace.output / SCALE.limit) * (height / 2);
    for (const dot of Array.from(container.querySelectorAll(".slice-point"))) {
      // Both slices pass through the probe, so both dots sit at one height.
      // The attribute is written to two decimals of SVG user units.
      expect(Number(dot.getAttribute("cy"))).toBeCloseTo(expected, 2);
    }
  });

  it("draws the whole sweep, one point per sample", () => {
    const slices = slicesFor();
    view.render(explorer.snapshot(), { slices, scale: SCALE });
    const curve = container.querySelector(".slice-curve")!;
    expect(curve.getAttribute("points")!.split(" ")).toHaveLength(slices[0].inputs.length);
  });

  it("says so when a curve runs past the scale instead of hiding the clipping", () => {
    const slices = slicesFor();
    view.render(explorer.snapshot(), { slices, scale: SCALE });
    expect(container.querySelector(".legend")!.textContent).not.toContain("cut off");

    // A scale far too small for this model: the curve leaves the box.
    view.render(explorer.snapshot(), { slices, scale: createColorScale(0.05) });
    expect(container.querySelector(".legend")!.textContent).toContain("cut off");
  });
});
