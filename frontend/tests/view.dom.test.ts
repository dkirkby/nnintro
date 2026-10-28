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
import { createExplorer, type Explorer } from "../src/state.js";
import type { ParameterRef } from "../src/model/types.js";
import { edgeParameter } from "../src/view/adapter.js";
import { createNetworkView, type NetworkView } from "../src/view/network.js";
import { createRippleAnimator } from "../src/view/ripple.js";
import { loadCheckpoint, loadManifest } from "./fixtures.js";

const CHECKPOINT = loadCheckpoint(loadManifest().checkpoints[0].url);
const WEIGHT: ParameterRef = { kind: "weight", layer: 0, out: 1, in: 0 };

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
    view.render(explorer.snapshot(), "activations");
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
    view.render(explorer.snapshot(), "activations");
    expect(container.querySelector('[data-key="e:0:0:1"]')!.classList).toContain(
      "is-selected",
    );

    explorer.setParameter(WEIGHT, 0.5);
    view.render(explorer.snapshot(), "activations");
    const stillSelected = container.querySelector('[data-key="e:0:0:1"]');
    expect(stillSelected!.classList).toContain("is-selected");
    expect(container.querySelectorAll(".is-selected")).toHaveLength(1);
  });

  it("shows the edited value, not the checkpoint's", () => {
    explorer.setParameter(WEIGHT, -2.5);
    view.render(explorer.snapshot(), "weights");
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
    view.render(createExplorer(CHECKPOINT).snapshot(), "activations");
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
