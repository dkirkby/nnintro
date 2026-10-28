/**
 * Wiring for the Stage 0 spike.
 *
 * The order of operations on every change is fixed: state recomputes the
 * snapshot synchronously, the views render it, and only then does the
 * animation start. The animation is an explanation of a change that has
 * already happened.
 */

import { fetchCheckpoint, fetchManifest } from "./model/catalog.js";
import { hiddenActivation } from "./model/evaluate.js";
import { computeRipple, originOf, type Ripple } from "./model/ripple.js";
import { describeParameter } from "./model/types.js";
import {
  createExplorer,
  describeProvenance,
  type EditRecord,
  type Explorer,
  type Snapshot,
} from "./state.js";
import { createInputSpaceView } from "./view/inputspace.js";
import { createInspector } from "./view/inspector.js";
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

  app.innerHTML = `
    <header>
      <h1>Dense network explorer</h1>
      <p class="subtitle">
        ${checkpoint.dataset.description} ·
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
        <p class="legend">
          Drag to move the probe. Filled marks are samples (outlined: held out);
          the arrow is the local ∂ŷ/∂x direction.
        </p>
      </section>
      <section class="panel network-panel">
        <h2>Network</h2>
        <div id="network"></div>
        <p class="legend" id="legend"></p>
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
    dataset: checkpoint.dataset,
    activation,
    onProbe: (probe) => explorer.setProbe(probe),
  });
  const inspector = createInspector({
    container: app.querySelector("#inspector")!,
    activation,
    targetName: checkpoint.dataset.targetName,
    inputNames: checkpoint.dataset.inputNames,
    onEdit: (ref, value) => explorer.setParameter(ref, value),
    onRestore: (ref) => explorer.restoreParameter(ref),
  });
  const animator = createRippleAnimator({
    overlay: () => networkView.overlay(),
    layout: () => networkView.layout(),
  });

  let mode: DisplayMode = "activations";
  let ripple: Ripple | null = null;

  buildControls(app.querySelector("#controls")!, explorer, () => mode, (next) => {
    mode = next;
    render(explorer.snapshot());
  });

  function render(snapshot: Snapshot): void {
    networkView.render(snapshot, mode);
    inputSpace.render(snapshot);
    inspector.render(snapshot, explorer.lastEdit(), ripple);
    app.querySelector("#provenance")!.textContent = describeProvenance(snapshot.provenance);
    app.querySelector("#legend")!.textContent =
      DISPLAY_MODES.find((entry) => entry.id === mode)!.legend;
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

  render(explorer.snapshot());
  setStatus(app, "Select an edge or a node to begin.");
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

function buildControls(
  container: HTMLElement,
  explorer: Explorer,
  mode: () => DisplayMode,
  setMode: (mode: DisplayMode) => void,
): void {
  const group = document.createElement("div");
  group.className = "control-group";
  group.setAttribute("role", "group");
  group.setAttribute("aria-label", "display mode");
  for (const entry of DISPLAY_MODES) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = entry.label;
    button.addEventListener("click", () => {
      setMode(entry.id);
      for (const sibling of group.children) {
        sibling.setAttribute(
          "aria-pressed",
          String((sibling as HTMLElement).textContent === DISPLAY_MODES.find((m) => m.id === mode())!.label),
        );
      }
    });
    button.setAttribute("aria-pressed", String(entry.id === mode()));
    group.append(button);
  }

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
  container.replaceChildren(group, actions);
}

void start().catch((error: unknown) => {
  const fallback = document.querySelector("#app");
  if (fallback instanceof HTMLElement) {
    fallback.textContent = `Could not start the explorer: ${String(error)}`;
  }
  throw error;
});
