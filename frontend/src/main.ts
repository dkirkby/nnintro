/**
 * Wiring for the Stage 0 spike.
 *
 * The order of operations on every change is fixed: state recomputes the
 * snapshot synchronously, the views render it, and only then does the
 * animation start. The animation is an explanation of a change that has
 * already happened.
 *
 * This module also owns the two things both panels have to agree on: the
 * prediction surface (recomputed only when the parameters change, not when
 * the probe moves) and the shared colour scale derived from it.
 */

import { fetchCheckpoint, fetchManifest } from "./model/catalog.js";
import { parameterCurves } from "./model/curves.js";
import { hiddenActivation } from "./model/evaluate.js";
import { computeRipple, originOf, type Ripple } from "./model/ripple.js";
import { outputSlice, outputSurface, type Surface } from "./model/surface.js";
import { describeParameter, type LayerParameters } from "./model/types.js";
import {
  createExplorer,
  describeProvenance,
  type EditRecord,
  type Explorer,
  type Snapshot,
} from "./state.js";
import { createColorScale, niceLimit, type Palette } from "./view/colormap.js";
import { createInputOutputView } from "./view/inputoutput.js";
import { createInspector } from "./view/inspector.js";
import {
  createNetworkView,
  format,
  DISPLAY_MODES,
  NODE_LEGEND,
  type DisplayMode,
} from "./view/network.js";
import { createParamSpaceView } from "./view/paramspace.js";

async function start(): Promise<void> {
  const found = document.querySelector<HTMLElement>("#app");
  if (!found) throw new Error("no #app");
  const app: HTMLElement = found;

  const manifest = await fetchManifest();
  const checkpoint = await fetchCheckpoint(manifest.checkpoints[0].url);
  const explorer = createExplorer(checkpoint);
  const activation = hiddenActivation(checkpoint.architecture);
  const { dataset } = checkpoint;

  app.innerHTML = `
    <header>
      <h1>Dense network explorer</h1>
      <p class="subtitle">
        ${dataset.description} ·
        ${checkpoint.architecture.layerSizes.join("–")} ·
        ${checkpoint.architecture.hiddenActivation ?? "no hidden layer"}
      </p>
      <p class="provenance" id="provenance"></p>
    </header>
    <div class="controls" id="controls"></div>
    <div class="workspace">
      <section class="panel io-panel">
        <h2>Input → output</h2>
        <div id="input-output"></div>
        <p class="legend" id="input-legend"></p>
      </section>
      <div class="stack">
        <section class="panel network-panel">
          <h2>Network</h2>
          <div id="network"></div>
          <p class="legend" id="legend"></p>
        </section>
        <section class="panel">
          <h2>Parameters → output</h2>
          <div id="param-space"></div>
        </section>
      </div>
      <div class="inspector" id="inspector"></div>
    </div>
    <p class="status" id="status" role="status" aria-live="polite"></p>
  `;

  const networkView = createNetworkView({
    container: app.querySelector("#network")!,
    layerSizes: checkpoint.architecture.layerSizes,
    idPrefix: `nn-${checkpoint.checkpointId}`,
    onSelect: (ref) => explorer.selectParameter(ref),
  });
  const inputOutput = createInputOutputView({
    container: app.querySelector("#input-output")!,
    dataset,
    onProbe: (probe) => explorer.setProbe(probe),
  });
  const inspector = createInspector({
    container: app.querySelector("#inspector")!,
    activation,
    targetName: dataset.targetName,
    inputNames: dataset.inputNames,
    onEdit: (ref, value) => explorer.setParameter(ref, value),
    onRestore: (ref) => explorer.restoreParameter(ref),
  });
  const paramSpace = createParamSpaceView({
    container: app.querySelector("#param-space")!,
    onEdit: (ref, value) => explorer.setParameter(ref, value),
  });

  let mode: DisplayMode = "weights";
  // Off to begin with: the panel is there to be probed, and four hundred
  // dots answer the question before the student has asked it.
  let showSamples = false;
  let ripple: Ripple | null = null;
  let cached: { parameters: LayerParameters[]; surface: Surface } | null = null;

  const controls = buildControls(app.querySelector("#controls")!, explorer, {
    setMode: (next) => {
      mode = next;
      render(explorer.snapshot());
    },
    toggleSamples: () => {
      showSamples = !showSamples;
      render(explorer.snapshot());
    },
  });

  /** The surface only depends on the parameters, which the probe never moves. */
  function surfaceFor(parameters: LayerParameters[]): Surface {
    if (!cached || cached.parameters !== parameters) {
      cached = { parameters, surface: outputSurface(parameters, activation, dataset.domain) };
    }
    return cached.surface;
  }

  /**
   * The three scales, one per family of quantity (see `view/colormap.ts`).
   *
   * Each limit is a function of the parameters and the dataset alone, never
   * of where the probe is, so dragging the probe cannot recolour anything
   * underneath it:
   *
   * - **input**: the domain the two axis bars span.
   * - **param**: every weight and bias, so an edge and a hidden node's fill
   *   are the same quantity on the same scale.
   * - **output**: the whole prediction surface and the targets, which count
   *   even while the samples are hidden, so showing them recolours nothing.
   */
  function paletteFor(snapshot: Snapshot, surface: Surface): Palette {
    const weights = snapshot.parameters.flatMap((layer) => layer.W.flat());
    const biases = snapshot.parameters.flatMap((layer) => layer.b);
    return {
      input: createColorScale(niceLimit(dataset.domain.flat()), "input"),
      param: createColorScale(niceLimit(weights, biases), "param"),
      output: createColorScale(
        niceLimit([surface.extent], dataset.samples.y),
        "output",
      ),
    };
  }

  function render(snapshot: Snapshot): void {
    const surface = surfaceFor(snapshot.parameters);
    const palette = paletteFor(snapshot, surface);
    // The slices follow the probe, so unlike the surface they are recomputed
    // on every render.
    const slices = dataset.domain.map((_, axis) =>
      outputSlice(snapshot.parameters, activation, dataset.domain, snapshot.x, axis),
    );

    // One sweep per parameter, across the scale the x axis of that panel
    // shows, at wherever the probe is now.
    const curves = parameterCurves(
      snapshot.parameters,
      activation,
      snapshot.x,
      palette.param.limit,
    );

    networkView.render(snapshot, mode, palette);
    inputOutput.render(snapshot, { surface, slices, palette, showSamples });
    paramSpace.render(snapshot, {
      curves,
      scale: palette.param,
      // Tall enough for every curve, so no sweep is cut off at an edge the
      // reader cannot see.
      outputLimit: niceLimit(curves.flatMap((curve) => curve.outputs)),
    });
    inspector.render(snapshot, explorer.lastEdit(), ripple);
    controls.sync(mode, showSamples);
    app.querySelector("#provenance")!.textContent = describeProvenance(snapshot.provenance);

    app.querySelector("#legend")!.textContent =
      `${DISPLAY_MODES.find((entry) => entry.id === mode)!.legend} ${NODE_LEGEND}`;
    app.querySelector("#input-legend")!.textContent =
      `Drag anywhere in the map to move the probe. The slices hold the other ` +
      `input where it is: left, ${dataset.inputNames[1]} varies at ` +
      `${dataset.inputNames[0]} = ${format(snapshot.x[0])}; below, ` +
      `${dataset.inputNames[0]} varies at ${dataset.inputNames[1]} = ` +
      `${format(snapshot.x[1])}. Slice height and background are the same ` +
      `±${palette.output.limit} output scale; the two bars are the input ` +
      `scale, which the input nodes also use.` +
      (showSamples ? " Dashed marks are held-out samples." : "");
  }

  explorer.subscribe((snapshot) => {
    const edit = explorer.lastEdit();
    ripple = edit ? rippleFor(edit, snapshot) : null;
    render(snapshot);
    if (edit) {
      setStatus(
        app,
        `${describeParameter(edit.ref)}: ${format(edit.before)} → ${format(edit.after)}.`,
      );
    }
  });

  // The canvas cannot resolve the theme's custom properties, so it is the one
  // surface a light/dark switch does not repaint by itself.
  globalThis
    .matchMedia?.("(prefers-color-scheme: dark)")
    .addEventListener("change", () => render(explorer.snapshot()));

  render(explorer.snapshot());
  setStatus(app, "Drag in the map to probe, then select an edge or a node.");
}

function rippleFor(edit: EditRecord, snapshot: Snapshot): Ripple {
  return computeRipple(edit.previous.trace, snapshot.trace, {
    ...originOf(edit.ref),
    kind: "parameter",
  });
}

function setStatus(app: HTMLElement, message: string): void {
  app.querySelector("#status")!.textContent = message;
}

interface ControlHandlers {
  setMode: (mode: DisplayMode) => void;
  toggleSamples: () => void;
}

/**
 * The buttons, plus a `sync` that re-reads the pressed states from the
 * caller's own variables: the controls hold no copy of the state they set.
 */
function buildControls(
  container: HTMLElement,
  explorer: Explorer,
  handlers: ControlHandlers,
): { sync: (mode: DisplayMode, showSamples: boolean) => void } {
  const modes = document.createElement("div");
  modes.className = "control-group";
  modes.setAttribute("role", "group");
  modes.setAttribute("aria-label", "display mode");
  const modeButtons = DISPLAY_MODES.map((entry) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = entry.label;
    button.addEventListener("click", () => handlers.setMode(entry.id));
    modes.append(button);
    return { id: entry.id, button };
  });

  const samples = document.createElement("button");
  samples.type = "button";
  samples.textContent = "Show samples";
  samples.addEventListener("click", () => handlers.toggleSamples());
  const samplesGroup = document.createElement("div");
  samplesGroup.className = "control-group";
  samplesGroup.append(samples);

  const actions = document.createElement("div");
  actions.className = "control-group";
  const buttons: [string, () => void][] = [
    ["Load pre-trained", () => explorer.loadPretrained()],
    ["Randomize all", () => explorer.randomize()],
    ["Restore baseline", () => explorer.restoreBaseline()],
    ["Undo", () => explorer.undo()],
    ["Redo", () => explorer.redo()],
  ];
  for (const [label, action] of buttons) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = label;
    button.addEventListener("click", action);
    actions.append(button);
  }
  container.replaceChildren(modes, samplesGroup, actions);

  return {
    sync(mode, showSamples) {
      for (const entry of modeButtons) {
        entry.button.setAttribute("aria-pressed", String(entry.id === mode));
      }
      samples.setAttribute("aria-pressed", String(showSamples));
    },
  };
}

void start().catch((error: unknown) => {
  const fallback = document.querySelector("#app");
  if (fallback instanceof HTMLElement) {
    fallback.textContent = `Could not start the explorer: ${String(error)}`;
  }
  throw error;
});
