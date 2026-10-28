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
import { createColorScale, niceLimit, type ColorScale } from "./view/colormap.js";
import { createInputSpaceView } from "./view/inputspace.js";
import { createInspector } from "./view/inspector.js";
import { createOutputSpaceView } from "./view/outputspace.js";
import {
  createNetworkView,
  format,
  DISPLAY_MODES,
  type DisplayMode,
} from "./view/network.js";
import { createRippleAnimator } from "./view/ripple.js";

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
      <section class="panel">
        <h2>Input space</h2>
        <div id="input-space"></div>
        <p class="legend" id="input-legend"></p>
      </section>
      <section class="panel network-panel">
        <h2>Network</h2>
        <div id="network"></div>
        <p class="legend" id="legend"></p>
      </section>
      <section class="panel output-panel">
        <h2>Output space</h2>
        <div id="output-space"></div>
      </section>
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
  const inputSpace = createInputSpaceView({
    container: app.querySelector("#input-space")!,
    dataset,
    onProbe: (probe) => explorer.setProbe(probe),
  });
  const outputSpace = createOutputSpaceView({
    container: app.querySelector("#output-space")!,
    dataset,
  });
  const inspector = createInspector({
    container: app.querySelector("#inspector")!,
    activation,
    targetName: dataset.targetName,
    inputNames: dataset.inputNames,
    onEdit: (ref, value) => explorer.setParameter(ref, value),
    onRestore: (ref) => explorer.restoreParameter(ref),
  });
  const animator = createRippleAnimator({
    overlay: () => networkView.overlay(),
    layout: () => networkView.layout(),
  });

  let mode: DisplayMode = "activations";
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
   * One scale for everything drawn as a colour: the surface, the axis bars,
   * the samples and the network's nodes.
   *
   * It covers what the input space itself draws: the whole prediction
   * surface, the targets (which count even while the samples are hidden, so
   * showing them never recolours the panel) and the input domain the axis
   * bars span. All three depend on the parameters alone, so dragging the
   * probe never recolours the panel underneath it.
   *
   * A hidden unit can run past that limit, and then its node saturates
   * rather than widening the scale: stretching the whole surface to fit one
   * large activation would flatten the thing the panel is for. The node
   * keeps its number, and the legend states the limit.
   */
  function scaleFor(surface: Surface): ColorScale {
    return createColorScale(
      niceLimit([surface.extent], dataset.samples.y, dataset.domain.flat()),
    );
  }

  function render(snapshot: Snapshot): void {
    const surface = surfaceFor(snapshot.parameters);
    const scale = scaleFor(surface);

    networkView.render(snapshot, mode, scale);
    inputSpace.render(snapshot, { surface, scale, showSamples });
    // The slices follow the probe, so unlike the surface they are recomputed
    // on every render.
    outputSpace.render(snapshot, {
      scale,
      slices: dataset.domain.map((_, axis) =>
        outputSlice(snapshot.parameters, activation, dataset.domain, snapshot.x, axis),
      ),
    });
    inspector.render(snapshot, explorer.lastEdit(), ripple);
    controls.sync(mode, showSamples);
    app.querySelector("#provenance")!.textContent = describeProvenance(snapshot.provenance);
    app.querySelector("#legend")!.textContent =
      DISPLAY_MODES.find((entry) => entry.id === mode)!.legend;
    app.querySelector("#input-legend")!.textContent =
      `Drag to move the probe. Colour is ${dataset.targetName} on a shared ` +
      `±${scale.limit} scale, read by the surface, the two axis bars and the ` +
      `network's nodes, which saturate beyond it; the arrow is the local ` +
      `∂ŷ/∂x direction.` +
      (showSamples ? " Dashed marks are held-out samples." : "");
  }

  explorer.subscribe((snapshot) => {
    const edit = explorer.lastEdit();
    // A new change makes any run in flight obsolete before anything is
    // drawn, so a stale pulse can never outlive the state that caused it.
    animator.cancel();
    ripple = edit ? rippleFor(edit, snapshot) : null;
    render(snapshot);
    if (ripple) {
      const description = describeParameter(edit!.ref);
      setStatus(
        app,
        `${description}: ${format(edit!.before)} → ${format(edit!.after)}.` +
          (animator.reducedMotion ? " Showing static deltas." : ""),
      );
      void animator.play(ripple);
    }
  });

  // The canvas cannot resolve the theme's custom properties, so it is the one
  // surface a light/dark switch does not repaint by itself.
  globalThis
    .matchMedia?.("(prefers-color-scheme: dark)")
    .addEventListener("change", () => render(explorer.snapshot()));

  render(explorer.snapshot());
  setStatus(app, "Probe the input space, then select an edge or a node.");
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
