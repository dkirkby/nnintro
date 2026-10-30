/**
 * @vitest-environment jsdom
 *
 * The catalog controls and the index behind them.
 *
 * The load-bearing claim is PLAN.md section 3's: a control must never be
 * able to ask for a configuration the published catalog does not have. So
 * these tests cover both halves — that the index reports the published
 * product honestly, and that a hole in it disables the stop that would
 * reach it rather than producing a failed fetch.
 */
import { beforeEach, describe, expect, it } from "vitest";

import { createCatalogIndex, type Selection } from "../src/model/catalog.js";
import type { Activation, Manifest, ManifestEntry } from "../src/model/types.js";
import { createSelector, describeArchitecture, type Selector } from "../src/view/selector.js";
import { allCheckpoints, loadManifest } from "./fixtures.js";

const MANIFEST = loadManifest();

/** A manifest built to order, so a deliberate hole can be tested. */
function manifestOf(entries: [string, number[], Activation | null][]): Manifest {
  const datasets = [...new Set(entries.map(([dataset]) => dataset))].map((id) => ({
    id,
    label: id === "a-set" ? "A set" : "B set",
    task: "regression" as const,
    description: `the ${id}`,
    generatorVersion: 1,
    hasAnalyticTarget: true,
    targetRange: [-1, 1] as [number, number],
    sampleCount: 4,
  }));
  const checkpoints: ManifestEntry[] = entries.map(([dataset, layerSizes, activation]) => ({
    checkpointId: `${dataset}__${layerSizes.join("-")}__${activation ?? "none"}`,
    dataset,
    task: "regression",
    layerSizes,
    hiddenActivation: activation,
    url: `${dataset}-${layerSizes.join("-")}-${activation ?? "none"}.json`,
    contentHash: "sha256:none",
    metrics: { trainLoss: 0, testLoss: 0 },
  }));
  return { schemaVersion: "test", datasets, checkpoints };
}

describe("the catalog index", () => {
  const index = createCatalogIndex(MANIFEST);

  it("ENUMERATES THE PUBLISHED PRODUCT, rather than the family it expects", () => {
    expect(index.datasets.map((entry) => entry.id)).toEqual([
      "tilted-plane",
      "curved-surface",
      "gaussian-clusters",
      "xor-quadrants",
    ]);
    expect(index.depths).toEqual([0, 1, 2]);
    expect(index.widths).toEqual([2, 4]);
    expect(index.activations).toEqual(["relu", "tanh", "identity"]);
  });

  it("COVERS EVERY COMBINATION A CONTROL CAN REACH: 4 datasets x 19 variants", () => {
    const missing: string[] = [];
    for (const dataset of index.datasets) {
      for (const widths of [[], ...index.widths.map((w) => [w]), ...pairs(index.widths)]) {
        // With no hidden layer there is no activation to vary, so that
        // architecture contributes one entry rather than three.
        const activations = widths.length > 0 ? index.activations : [index.activations[0]];
        for (const activation of activations) {
          const selection = { dataset: dataset.id, widths, activation };
          if (!index.has(selection)) missing.push(`${dataset.id} ${describeArchitecture(selection)} ${activation}`);
        }
      }
    }
    expect(missing).toEqual([]);
    expect(MANIFEST.checkpoints).toHaveLength(76);
  });

  it("KEYS A DEPTH-0 ENTRY WITHOUT AN ACTIVATION, whichever one is remembered", () => {
    // The control carries an activation even at depth 0 so that toggling
    // depth off and back on returns the previous choice; it must not make
    // the linear model unreachable.
    for (const activation of index.activations) {
      const found = index.find({ dataset: "tilted-plane", widths: [], activation });
      expect(found?.checkpointId).toBe("tilted-plane__2-1__none");
    }
  });

  it("RETURNS NULL for a combination the snapshot does not have", () => {
    expect(index.find({ dataset: "tilted-plane", widths: [3], activation: "relu" })).toBeNull();
    expect(index.find({ dataset: "no-such-set", widths: [4], activation: "relu" })).toBeNull();
  });
});

function pairs(widths: number[]): number[][] {
  return widths.flatMap((a) => widths.map((b) => [a, b]));
}

describe("the catalog controls", () => {
  let container: HTMLElement;
  let selector: Selector;
  let asked: Selection[];
  let refused: string[];

  /** Two datasets, depths 0–2, widths 2 and 4 — with one hole in it. */
  const HOLE: Selection = { dataset: "b-set", widths: [4], activation: "tanh" };
  const index = createCatalogIndex(
    manifestOf(
      ["a-set", "b-set"].flatMap((dataset) =>
        [
          [[2, 1], null] as [number[], Activation | null],
          ...([2, 4].flatMap((w) =>
            (["relu", "tanh"] as Activation[]).map(
              (a) => [[2, w, 1], a] as [number[], Activation | null],
            ),
          ) as [number[], Activation | null][]),
          ...([2, 4].flatMap((w) =>
            (["relu", "tanh"] as Activation[]).map(
              (a) => [[2, w, w, 1], a] as [number[], Activation | null],
            ),
          ) as [number[], Activation | null][]),
        ]
          .filter(
            ([sizes, activation]) =>
              !(
                dataset === HOLE.dataset &&
                sizes.length === 3 &&
                sizes[1] === HOLE.widths[0] &&
                activation === HOLE.activation
              ),
          )
          .map(
            ([sizes, activation]) =>
              [dataset, sizes, activation] as [string, number[], Activation | null],
          ),
      ),
    ),
  );

  function stops(label: string): { text: string; current: boolean; missing: boolean }[] {
    const group = Array.from(container.querySelectorAll(".selector-group")).find(
      (element) => element.querySelector(".selector-label")?.textContent === label,
    )!;
    return Array.from(group.querySelectorAll(".stop-mark")).map((mark) => ({
      text: mark.textContent ?? "",
      current: mark.classList.contains("is-current"),
      missing: mark.classList.contains("stop-missing"),
    }));
  }

  function slider(label: string): HTMLInputElement {
    const group = Array.from(container.querySelectorAll(".selector-group")).find(
      (element) => element.querySelector(".selector-label")?.textContent === label,
    )!;
    return group.querySelector<HTMLInputElement>(".stop-slider")!;
  }

  function drag(label: string, value: number): void {
    const input = slider(label);
    input.value = String(value);
    input.dispatchEvent(new Event("input"));
  }

  beforeEach(() => {
    container = document.createElement("div");
    document.body.replaceChildren(container);
    asked = [];
    refused = [];
    selector = createSelector({
      container,
      index,
      onSelect: (selection) => asked.push(selection),
      onUnavailable: (message) => refused.push(message),
    });
    selector.sync({ dataset: "a-set", widths: [4], activation: "relu" });
  });

  it("WRITES EVERY STOP OUT, and marks the current one", () => {
    expect(stops("hidden layers").map((stop) => stop.text)).toEqual(["0", "1", "2"]);
    expect(stops("hidden layers").find((stop) => stop.current)?.text).toBe("1");
    expect(stops("width of hidden layer 1").map((stop) => stop.text)).toEqual(["2", "4"]);
    expect(stops("width of hidden layer 1").find((stop) => stop.current)?.text).toBe("4");
  });

  it("DISABLES THE WIDTH OF A LAYER THAT DOES NOT EXIST, keeping its place", () => {
    // Every control stays in the row at every depth. The depth slider is
    // directly above these, so a row that reflowed on a depth change would
    // move a control out from under the pointer that had just used it.
    const group = slider("width of hidden layer 2").closest<HTMLElement>(".selector-group")!;
    expect(group.hidden).toBe(false);
    expect(slider("width of hidden layer 2").disabled).toBe(true);
    expect(group.classList.contains("is-inert")).toBe(true);
    // Greyed, but not struck through: the catalog has those widths, there
    // is just no layer to apply them to yet.
    expect(stops("width of hidden layer 2").some((stop) => stop.missing)).toBe(false);

    selector.sync({ dataset: "a-set", widths: [4, 2], activation: "relu" });
    expect(slider("width of hidden layer 2").disabled).toBe(false);
    expect(group.classList.contains("is-inert")).toBe(false);
  });

  it("previews what a new layer would get, rather than showing a blank", () => {
    // Deepening inherits the last existing layer's width, so the disabled
    // control reads 4 at 2–4–1: it is the next stop up, already shown.
    expect(stops("width of hidden layer 2").find((stop) => stop.current)?.text).toBe("4");
    selector.sync({ dataset: "a-set", widths: [2], activation: "relu" });
    expect(stops("width of hidden layer 2").find((stop) => stop.current)?.text).toBe("2");
  });

  it("A NEW LAYER INHERITS THE LAST ONE's WIDTH: 2–4–1 deepens to 2–4–4–1", () => {
    drag("hidden layers", 2);
    expect(asked).toHaveLength(1);
    expect(describeArchitecture(asked[0])).toBe("2–4–4–1");
  });

  it("TURNS THE ACTIVATION OFF with no hidden layer to apply it to", () => {
    drag("hidden layers", 0);
    selector.sync(asked[0]);
    const buttons = Array.from(container.querySelectorAll<HTMLButtonElement>("[data-activation]"));
    expect(buttons.every((button) => button.disabled)).toBe(true);
    expect(buttons.every((button) => button.getAttribute("aria-pressed") === "false")).toBe(true);
  });

  it("DISABLES A BUTTON WHOSE COMBINATION HAS NO CHECKPOINT", () => {
    // b-set has no 2–4–1 tanh, so from a-set/2–4–1/tanh the dataset button
    // for b-set cannot be pressed.
    selector.sync({ dataset: "a-set", widths: [4], activation: "tanh" });
    const button = container.querySelector<HTMLButtonElement>('[data-dataset="b-set"]')!;
    expect(button.disabled).toBe(true);
    // And from relu, where b-set does have an entry, it can.
    selector.sync({ dataset: "a-set", widths: [4], activation: "relu" });
    expect(button.disabled).toBe(false);
  });

  it("STRIKES THROUGH A SLIDER STOP WITH NO CHECKPOINT, and snaps back off it", () => {
    selector.sync({ dataset: "b-set", widths: [2], activation: "tanh" });
    const marks = stops("width of hidden layer 1");
    expect(marks.map((stop) => [stop.text, stop.missing])).toEqual([
      ["2", false],
      ["4", true],
    ]);

    drag("width of hidden layer 1", 1);
    expect(asked).toEqual([]);
    expect(refused[0]).toContain("no checkpoint");
    // Snapped back, so the control still shows what is actually loaded.
    expect(slider("width of hidden layer 1").value).toBe("0");
  });

  it("KEEPS THE SLIDERS LIVE WHILE A CHECKPOINT LOADS", () => {
    // A discrete slider fires an event per stop it crosses, so disabling it
    // mid-load would break the drag. `main.ts` discards superseded loads
    // instead, which is why nothing here has a busy state.
    drag("hidden layers", 2);
    drag("hidden layers", 0);
    expect(asked.map(describeArchitecture)).toEqual(["2–4–4–1", "2–1"]);
    expect(slider("hidden layers").disabled).toBe(false);
  });
});

/**
 * The colour scale's ends travel with the asset.
 *
 * `view/colormap.ts` paints a regression's output scale over
 * `dataset.targetRange`, which the browser cannot work out for itself: it
 * has each generator's formula but not a search over the domain. So the
 * snapshot has to carry it, and a regression dataset that arrived without
 * one would quietly fall back to the range its own samples happened to
 * draw.
 */
describe("the published target ranges", () => {
  it("DECLARES ONE PER REGRESSION DATASET, and none for a classifier", () => {
    const declared = Object.fromEntries(
      MANIFEST.datasets.map((entry) => [entry.id, entry.targetRange]),
    );
    // Both regression generators top out at |y| = 1 by construction, which
    // is what puts full red and full blue at a round number.
    expect(declared).toEqual({
      "tilted-plane": [-0.6, 1],
      "curved-surface": [-1, 1],
      "gaussian-clusters": null,
      "xor-quadrants": null,
    });
  });

  it("agrees with every checkpoint, and holds every sample", () => {
    for (const { dataset } of allCheckpoints()) {
      const entry = MANIFEST.datasets.find((row) => row.id === dataset.id)!;
      expect(dataset.targetRange).toEqual(entry.targetRange);
      if (dataset.targetRange === null) continue;
      const [lo, hi] = dataset.targetRange;
      // A sample outside the range would saturate the very scale its own
      // dataset defines.
      expect(Math.min(...dataset.samples.y)).toBeGreaterThanOrEqual(lo);
      expect(Math.max(...dataset.samples.y)).toBeLessThanOrEqual(hi);
    }
  });
});
