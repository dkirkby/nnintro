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
import { outputSlice, outputSurface, type Surface } from "./model/surface.js";
import {
  unitParameters,
  type LayerParameters,
} from "./model/types.js";
import {
  createExplorer,
  describeProvenance,
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
import { createHelp } from "./view/help.js";
import { createParamSpaceView } from "./view/paramspace.js";
import { createTensorsView } from "./view/tensors.js";

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
      <div class="column">
        <section class="panel io-panel">
          <h2>Inputs → output</h2>
          <div id="input-output"></div>
          <p class="legend" id="input-legend"></p>
        </section>
        <section class="panel tensor-panel">
          <h2>Parameter Tensors</h2>
          <div id="tensors"></div>
          <p class="legend" id="tensor-legend">
            Every parameter the network has, as the shapes they are stored in:
            one group per layer, the bias vector above its weight matrix, and
            one light dot per value. A column of a group is one unit of the
            layer it feeds — its bias and the weights into it — so selecting a
            unit in the network panel shades exactly that band. A row of a
            weight matrix is one source the layer reads from. Hover a
            rectangle to see what it is.
          </p>
        </section>
      </div>
      <div class="stack">
        <section class="panel network-panel">
          <h2>Network</h2>
          <div id="network"></div>
          <p class="legend" id="legend"></p>
        </section>
        <section class="panel param-panel">
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
    // An input node owns no parameters, so a click on one selects nothing
    // and leaves the current unit alone: there is no way to deselect.
    onSelect: (unit) => {
      if (unit) explorer.selectUnit(unit);
    },
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
  });
  const paramSpace = createParamSpaceView({
    container: app.querySelector("#param-space")!,
    onEdit: (ref, value) => explorer.setParameter(ref, value),
  });
  const tensors = createTensorsView({ container: app.querySelector("#tensors")! });

  // Each panel's own explanation, in a popup beside it rather than printed
  // under it: it is a paragraph read once, and it was crowding out the plot
  // it explains. `createHelp` moves the legend element itself, so whatever
  // writes its text keeps the popup current (see view/help.ts).
  // Held by reference, because `createHelp` moves each element out of its
  // panel: after this they are no longer reachable through `app`.
  const legends = {
    inputs: app.querySelector<HTMLElement>("#input-legend")!,
    network: app.querySelector<HTMLElement>("#legend")!,
  };
  const explained: [string, HTMLElement, string][] = [
    [".io-panel", legends.inputs, "Inputs → output"],
    [".network-panel", legends.network, "Network"],
    [".param-panel", app.querySelector<HTMLElement>("#param-space .legend")!, "Parameters → output"],
    [".tensor-panel", app.querySelector<HTMLElement>("#tensor-legend")!, "Parameter Tensors"],
  ];
  for (const [panel, legend, name] of explained) {
    createHelp(app.querySelector(panel)!, legend, name);
  }

  let mode: DisplayMode = "weights";
  // Off to begin with: the panel is there to be probed, and four hundred
  // dots answer the question before the student has asked it.
  let showSamples = false;
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
   * - **param**: every weight and bias, so an edge and a hidden node's fill
   *   are the same quantity on the same scale.
   * - **output**: the whole prediction surface and the targets, which count
   *   even while the samples are hidden, so showing them recolours nothing.
   */
  function paletteFor(snapshot: Snapshot, surface: Surface): Palette {
    const weights = snapshot.parameters.flatMap((layer) => layer.W.flat());
    const biases = snapshot.parameters.flatMap((layer) => layer.b);
    return {
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

    // One sweep per parameter of the selected unit — its bias, then its
    // weights top to bottom — across the scale the x axis of that panel
    // shows, at wherever the probe is now.
    const curves = parameterCurves(
      snapshot.parameters,
      activation,
      snapshot.x,
      palette.param.limit,
      unitParameters(snapshot.parameters, snapshot.selection),
    );

    networkView.render(snapshot, mode, palette);
    tensors.render(snapshot);
    inputOutput.render(snapshot, { surface, slices, palette, showSamples });
    paramSpace.render(snapshot, {
      curves,
      paramScale: palette.param,
      outputScale: palette.output,
      // Tall enough for every curve, so no sweep is cut off at an edge the
      // reader cannot see. The 5% headroom is for the case where the sweep
      // exactly reaches a round number — common when the selected unit is
      // off and the current output *is* the extreme — which would otherwise
      // draw the output line along the frame itself.
      outputLimit: niceLimit(
        curves.flatMap((curve) => curve.outputs.map((value) => value * 1.05)),
      ),
    });
    inspector.render(snapshot, explorer.lastEdit());
    controls.sync(mode, showSamples);
    app.querySelector("#provenance")!.textContent = describeProvenance(snapshot.provenance);

    legends.network.textContent =
      `${DISPLAY_MODES.find((entry) => entry.id === mode)!.legend} ${NODE_LEGEND}`;
    legends.inputs.textContent =
      `Drag anywhere in the map to move the probe. The slices hold the other ` +
      `input where it is: left, ${dataset.inputNames[1]} varies at ` +
      `${dataset.inputNames[0]} = ${format(snapshot.x[0])}; below, ` +
      `${dataset.inputNames[0]} varies at ${dataset.inputNames[1]} = ` +
      `${format(snapshot.x[1])}. Slice height and background are the same ` +
      `±${palette.output.limit} output scale.` +
      (showSamples ? " Dashed marks are held-out samples." : "");
  }

  // No running commentary on edits: an edit's before and after are in the
  // panels themselves — the slider's own readout, and the comparison in
  // "Prediction and loss" — and a line repeating them below the fold was
  // one more place to look.
  explorer.subscribe(render);

  // The canvas cannot resolve the theme's custom properties, so it is the one
  // surface a light/dark switch does not repaint by itself.
  globalThis
    .matchMedia?.("(prefers-color-scheme: dark)")
    .addEventListener("change", () => render(explorer.snapshot()));

  render(explorer.snapshot());
  // Standing instructions, not a log: nothing below the panels changes as
  // the explorer is used, so nothing down here can go stale.
  setStatus(
    app,
    "Drag in the map to probe. Click a node, or any edge into it, to select " +
      "that unit.",
  );
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
