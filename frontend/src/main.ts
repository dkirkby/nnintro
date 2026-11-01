/**
 * Wiring.
 *
 * The module is in two halves. The **shell** is built once: the panels, the
 * `?` popups, the catalog controls and the display controls. A **session**
 * is built per checkpoint: the explorer and the five views that read its
 * snapshot. Switching dataset, depth, width or activation tears down the
 * session and mounts a new one, which is what makes the reset honest —
 * parameters are never carried across a change of problem or of shape
 * (PLAN.md section 4), because the object that held them is gone.
 *
 * Within a session the order of operations on every change is fixed: state
 * recomputes the snapshot synchronously, then the views render it.
 *
 * This module also owns the two things every panel has to agree on: the
 * prediction surface (recomputed only when the parameters change, not when
 * the probe moves) and the colour scales derived from it.
 */

import {
  createCatalogIndex,
  createCheckpointCache,
  fetchManifest,
  type CatalogIndex,
  type Selection,
} from "./model/catalog.js";
import { parameterCurves } from "./model/curves.js";
import { hiddenActivation } from "./model/evaluate.js";
import { outputSlice, outputSurface, type Surface } from "./model/surface.js";
import { thresholdLines } from "./model/thresholds.js";
import { unitParameters, type Checkpoint, type LayerParameters } from "./model/types.js";
import { createExplorer, describeProvenance, type Explorer, type Snapshot } from "./state.js";
import { createColorScale, niceLimit, outputLimit, type Palette } from "./view/colormap.js";
import { createHelp, type Help } from "./view/help.js";
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
import { createSelector, describeArchitecture } from "./view/selector.js";
import { createTensorsView } from "./view/tensors.js";
import { createWorkspace, type PanelSpec } from "./view/workspace.js";

const SHELL = `
  <header>
    <h1>Dense network explorer</h1>
    <p class="subtitle" id="subtitle"></p>
    <p class="provenance" id="provenance"></p>
  </header>
  <div class="controls" id="catalog"></div>
  <div class="controls" id="controls"></div>
  <div id="workspace"></div>
  <p class="status" id="status" role="status" aria-live="polite"></p>
`;

/**
 * The panels, and where they sit before anyone moves them.
 *
 * The workspace is a canvas the reader arranges (`view/workspace.ts`), so
 * this is a default rather than a layout: six cards on a twelve-column
 * grid, laid out as the fixed three columns used to be — the input space
 * and the tensors down the left, the network and its parameter sweeps on
 * the right, the inspector along the bottom.
 *
 * `limits` is what each panel's content can live with. They are generous
 * rather than exact: a plot that scales with its box can be read at a wide
 * range of sizes, and the floor is the width below which its own labels
 * start colliding.
 */
const PANELS: PanelSpec[] = [
  {
    id: "inputs",
    title: "Inputs \u2192 output",
    rect: { x: 0, y: 0, w: 5, h: 36 },
    limits: { minW: 3, maxW: 12, minH: 14, maxH: 80 },
    body: `
      <div class="panel-toggles" id="input-toggles"></div>
      <div id="input-output"></div>
      <p class="legend" id="input-legend"></p>
    `,
  },
  {
    id: "network",
    title: "Network",
    rect: { x: 5, y: 0, w: 7, h: 30 },
    limits: { minW: 3, maxW: 12, minH: 10, maxH: 80 },
    body: `
      <div class="panel-toggles" id="network-toggles"></div>
      <div class="network-figure" id="network"></div>
      <p class="legend" id="legend"></p>
    `,
  },
  {
    id: "params",
    title: "Parameters \u2192 output",
    rect: { x: 5, y: 30, w: 7, h: 30 },
    limits: { minW: 3, maxW: 12, minH: 12, maxH: 80 },
    body: `
      <div id="param-space"></div>
      <p class="legend" id="param-legend"></p>
    `,
  },
  {
    id: "tensors",
    title: "Parameter Tensors",
    rect: { x: 0, y: 36, w: 5, h: 13 },
    limits: { minW: 2, maxW: 12, minH: 6, maxH: 50 },
    body: `
      <div id="tensors"></div>
      <p class="legend" id="tensor-legend">
        Every parameter the network has, as the shapes they are stored in:
        one group per layer, the bias vector above its weight matrix \u2014
        the two rows are named in the gutter on the left \u2014 and one light
        dot per value. A column of a group is one unit of the layer it
        feeds \u2014 its bias and the weights into it \u2014 so selecting a unit
        in the network panel shades exactly that band. A row of a weight
        matrix is one source the layer reads from. Hover anywhere inside a
        rectangle to see what it is.
      </p>
    `,
  },
  {
    id: "prediction",
    title: "Prediction and loss",
    rect: { x: 0, y: 48, w: 5, h: 16 },
    limits: { minW: 2, maxW: 12, minH: 6, maxH: 50 },
    body: `<div id="prediction"></div>`,
  },
  {
    id: "neuron",
    // Replaced on every render with the unit that is actually selected.
    title: "Neuron",
    rect: { x: 5, y: 56, w: 7, h: 20 },
    limits: { minW: 3, maxW: 12, minH: 8, maxH: 50 },
    body: `<div id="neuron"></div>`,
  },
];

/**
 * The input panel's one switch, named once: the shell builds the box and
 * the session has to find it again to grey it out where the network has no
 * hidden layer to have thresholds.
 */
const THRESHOLDS = "Thresholds";

/** One checkpoint's explorer and views: everything a switch replaces. */
interface Session {
  explorer: Explorer;
  render(): void;
  destroy(): void;
}

async function start(): Promise<void> {
  const found = document.querySelector<HTMLElement>("#app");
  if (!found) throw new Error("no #app");
  const app: HTMLElement = found;

  const manifest = await fetchManifest();
  const index = createCatalogIndex(manifest);
  const loadCheckpoint = createCheckpointCache();

  app.innerHTML = SHELL;

  // The panels are a canvas the reader arranges rather than a fixed grid:
  // each one can be folded down to its title bar, dragged anywhere, and
  // resized, so the explorer can be built up a piece at a time instead of
  // arriving as a wall of five views. The arrangement is the reader's and
  // is kept between sessions; `Reset layout` puts it back.
  //
  // Shell-owned, like the help popups and for the same reason: a checkpoint
  // switch rebuilds the views, and a panel that went with them would take
  // the reader's arrangement with it.
  const workspace = createWorkspace({
    container: app.querySelector("#workspace")!,
    panels: PANELS,
  });

  // Each panel's own explanation, in a popup beside it rather than printed
  // under it: it is a paragraph read once, and it was crowding out the plot
  // it explains. `createHelp` moves the legend element itself, so whatever
  // writes its text keeps the popup current (see view/help.ts).
  //
  // Held by reference, because `createHelp` moves each element out of its
  // panel: after this they are no longer reachable through `app`. They are
  // also shell-owned rather than session-owned, so a checkpoint switch does
  // not leave a second `?` button behind.
  const legends = {
    inputs: app.querySelector<HTMLElement>("#input-legend")!,
    network: app.querySelector<HTMLElement>("#legend")!,
    params: app.querySelector<HTMLElement>("#param-legend")!,
    tensors: app.querySelector<HTMLElement>("#tensor-legend")!,
  };
  const helps: Help[] = (
    [
      ["inputs", legends.inputs, "Inputs → output"],
      ["network", legends.network, "Network"],
      ["params", legends.params, "Parameters → output"],
      ["tensors", legends.tensors, "Parameter Tensors"],
    ] as [string, HTMLElement, string][]
  ).map(([panel, legend, name]) => createHelp(workspace.panel(panel), legend, name));

  // Display preferences, as opposed to model state: they belong to the
  // reader rather than to the checkpoint, so they survive a switch.
  let mode: DisplayMode = "weights";
  // Off to begin with: the panel is there to be probed, and four hundred
  // dots answer the question before the student has asked it.
  let showSamples = false;
  // Both on: the explorer is about the parameters, so it opens showing
  // them. They are here to be put away by a reader who wants the shape of
  // the network on its own, or wants the colours without the numbers over
  // them — which is why these are the network panel's own controls rather
  // than part of the row at the top.
  let showValues = true;
  let showParameters = true;
  // On as well, and for the same reason: the lines are where the first
  // hidden layer's units switch, which is the most direct thing the map can
  // say about the parameters. A reader who wants the surface alone turns
  // them off.
  let showThresholds = true;

  let session: Session | null = null;
  let selection: Selection = openingSelection(index);
  /** Which load is the current one: a superseded fetch must not mount. */
  let request = 0;

  buildToggles(app.querySelector("#network-toggles")!, [
    [
      "Values",
      (next) => {
        showValues = next;
        session?.render();
      },
    ],
    [
      "Parameters",
      (next) => {
        showParameters = next;
        session?.render();
        // The display modes only switch what an edge's colour and width
        // show, so with the parameters off there is nothing for them to
        // switch: they are disabled in place rather than left live and inert.
        controls.sync(mode, showSamples, showParameters);
      },
    ],
  ]);

  const inputToggles = buildToggles(app.querySelector("#input-toggles")!, [
    [
      THRESHOLDS,
      (next) => {
        showThresholds = next;
        session?.render();
      },
    ],
  ]);

  const controls = buildControls(app.querySelector("#controls")!, {
    current: () => session?.explorer ?? null,
    setMode: (next) => {
      mode = next;
      session?.render();
    },
    toggleSamples: () => {
      showSamples = !showSamples;
      session?.render();
    },
    resetLayout: () => {
      workspace.reset();
      setStatus(app, `Layout reset: every panel expanded and back where it started. ${RESET_HINT}`);
    },
  });

  const selector = createSelector({
    container: app.querySelector("#catalog")!,
    index,
    // Reported in the status line rather than left as an unhandled
    // rejection: a switch is a user action, so its failure belongs on
    // screen.
    onSelect: (next) => {
      void choose(next).catch((error: unknown) => {
        setStatus(app, `Could not switch: ${String(error)}`);
      });
    },
    onUnavailable: (message) => setStatus(app, message),
  });

  /**
   * Switch to another configuration.
   *
   * The new session is mounted only once its checkpoint has arrived, so a
   * slow fetch leaves the current figure on screen rather than blanking it.
   * A discrete slider fires an event for every stop it crosses, so several
   * loads can be in flight at once: the last one asked for is the one that
   * mounts, and the others are discarded rather than mounted in whatever
   * order they happen to resolve.
   */
  async function choose(next: Selection): Promise<void> {
    const entry = index.find(next);
    if (!entry) throw new Error(`no checkpoint for ${JSON.stringify(next)}`);
    const token = ++request;
    selector.sync(next);
    setStatus(app, `Loading ${entry.checkpointId}…`);
    try {
      const checkpoint = await loadCheckpoint(entry.url);
      if (token !== request) return;
      const previous = selection;
      // What the reader was looking at, so the comparison a switch is for
      // survives it. Parameters never carry — that is the reset — but where
      // the probe is, and which unit is open, are the reader's place in the
      // figure rather than anything the new asset disagrees with.
      const was = session?.explorer.snapshot() ?? null;
      const sameDataset = previous.dataset === next.dataset;
      const sameShape =
        session !== null &&
        describeArchitecture(previous) === describeArchitecture(next);
      selection = next;
      session?.destroy();
      session = mount(checkpoint);
      // The probe is a point of the dataset's own domain, so it is only
      // meaningful while the dataset is the same one.
      if (was && sameDataset) session.explorer.setProbe(was.probe);
      // A unit index only exists in a network of the same shape, which is
      // exactly the case when only the activation changed.
      if (was && sameShape) session.explorer.selectUnit(was.selection);
      session.render();
      announceReset(previous, next);
    } catch (error) {
      if (token !== request) return;
      // `selection` is only advanced after the fetch resolves, so the
      // controls resync to the configuration that is actually on screen.
      setStatus(app, `Could not load ${entry.checkpointId}: ${String(error)}`);
      selector.sync(selection);
    }
  }

  /** What changed, and the fact that the parameters went back to the asset. */
  function announceReset(previous: Selection, next: Selection): void {
    const changed: string[] = [];
    if (previous.dataset !== next.dataset) {
      changed.push(`dataset → ${index.dataset(next.dataset)?.label ?? next.dataset}`);
    }
    if (describeArchitecture(previous) !== describeArchitecture(next)) {
      changed.push(`architecture → ${describeArchitecture(next)}`);
    }
    if (previous.activation !== next.activation && next.widths.length > 0) {
      changed.push(`activation → ${next.activation}`);
    }
    setStatus(
      app,
      changed.length === 0
        ? RESET_HINT
        : `${changed.join(", ")}. Fresh baseline: any edits were discarded, ` +
            `since parameters are never carried across a change of problem ` +
            `or of shape. ${RESET_HINT}`,
    );
  }

  /** Build one checkpoint's explorer and views. */
  function mount(checkpoint: Checkpoint): Session {
    const explorer = createExplorer(checkpoint);
    const activation = hiddenActivation(checkpoint.architecture);
    const { dataset } = checkpoint;
    const isClassifier = dataset.task === "classification";

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
      prediction: app.querySelector("#prediction")!,
      neuron: app.querySelector("#neuron")!,
      setNeuronTitle: (title) => workspace.setTitle("neuron", title),
      activation,
      task: dataset.task,
      targetName: dataset.targetName,
      inputNames: dataset.inputNames,
    });
    const paramSpace = createParamSpaceView({
      container: app.querySelector("#param-space")!,
      legend: legends.params,
      outputName: isClassifier ? "z" : "ŷ",
      onEdit: (ref, value) => explorer.setParameter(ref, value),
    });
    const tensors = createTensorsView({ container: app.querySelector("#tensors")! });

    // A threshold belongs to a unit of the first hidden layer, so a network
    // without one has none to draw: the box is greyed in place rather than
    // left live over a map it cannot change.
    inputToggles.setEnabled(THRESHOLDS, checkpoint.architecture.layerSizes.length > 2);

    let cached: { parameters: LayerParameters[]; surface: Surface } | null = null;

    /** The surface only depends on the parameters, which the probe never moves. */
    function surfaceFor(parameters: LayerParameters[]): Surface {
      if (!cached || cached.parameters !== parameters) {
        cached = {
          parameters,
          surface: outputSurface(parameters, activation, dataset.domain),
        };
      }
      return cached.surface;
    }

    /**
     * The two scales, one per family of quantity (see `view/colormap.ts`).
     *
     * Each limit is a function of the parameters and the dataset alone,
     * never of where the probe is, so dragging the probe cannot recolour
     * anything underneath it:
     *
     * - **param**: every weight and bias, so an edge and a hidden node's
     *   fill are the same quantity on the same scale.
     * - **output**: the range of the dataset's own targets, as its
     *   generator defines it over the whole domain — so full red and full
     *   blue are the most extreme targets there are, rather than the most
     *   extreme the sampling happened to draw. `outputLimit`
     *   (`view/colormap.ts`) owns that rule and the classifier's, where
     *   there is no target range to span.
     */
    function paletteFor(snapshot: Snapshot, surface: Surface): Palette {
      const weights = snapshot.parameters.flatMap((layer) => layer.W.flat());
      const biases = snapshot.parameters.flatMap((layer) => layer.b);
      return {
        param: createColorScale(niceLimit(weights, biases), "param"),
        output: createColorScale(outputLimit(dataset, surface.extent), "output"),
      };
    }

    function render(snapshot: Snapshot): void {
      const surface = surfaceFor(snapshot.parameters);
      const palette = paletteFor(snapshot, surface);
      // The slices follow the probe, so unlike the surface they are
      // recomputed on every render.
      const slices = dataset.domain.map((_, axis) =>
        outputSlice(snapshot.parameters, activation, dataset.domain, snapshot.x, axis),
      );
      // Where each first-hidden-layer unit switches. Computed here rather
      // than in the panel, like everything else it draws, and cheap enough
      // to redo per render: it is one line per hidden unit of the first
      // layer, and the widest this catalog goes is eight.
      const thresholds = thresholdLines(snapshot.parameters, dataset.domain);

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

      networkView.render(snapshot, {
        mode,
        palette,
        showSamples,
        showValues,
        showParameters,
      });
      tensors.render(snapshot);
      inputOutput.render(snapshot, {
        surface,
        slices,
        palette,
        showSamples,
        thresholds,
        showThresholds,
      });
      paramSpace.render(snapshot, {
        curves,
        paramScale: palette.param,
        outputScale: palette.output,
        // Tall enough for every curve, so no sweep is cut off at an edge
        // the reader cannot see. The 5% headroom is for the case where the
        // sweep exactly reaches a round number — common when the selected
        // unit is off and the current output *is* the extreme — which would
        // otherwise draw the output line along the frame itself.
        outputLimit: niceLimit(
          curves.flatMap((curve) => curve.outputs.map((value) => value * 1.05)),
        ),
      });
      inspector.render(snapshot, explorer.lastEdit());
      controls.sync(mode, showSamples, showParameters);
      app.querySelector("#provenance")!.textContent = describeProvenance(snapshot.provenance);
      app.querySelector("#subtitle")!.textContent =
        `${dataset.label} · ${dataset.description} · ` +
        `${checkpoint.architecture.layerSizes.join("–")} · ` +
        `${checkpoint.architecture.hiddenActivation ?? "no hidden layer"}`;

      legends.network.textContent =
        `${DISPLAY_MODES.find((entry) => entry.id === mode)!.legend} ${NODE_LEGEND}`;
      legends.inputs.textContent =
        `Drag anywhere in the map to move the probe. The slices hold the ` +
        `other input where it is: left, ${dataset.inputNames[1]} varies at ` +
        `${dataset.inputNames[0]} = ${format(snapshot.x[0])}; below, ` +
        `${dataset.inputNames[0]} varies at ${dataset.inputNames[1]} = ` +
        `${format(snapshot.x[1])}. Slice height and background are the same ` +
        `±${palette.output.limit} ${isClassifier ? "logit" : "output"} scale` +
        (isClassifier
          ? `, fitted to this network's own range.`
          : `, which is the whole range of ${dataset.label}'s targets — ` +
            `the range its formula covers over the whole square, not just ` +
            `the range these ${dataset.samples.y.length} samples happened ` +
            `to draw — and so fixed while you edit: full red and full blue ` +
            `are the most extreme targets there are, and a prediction ` +
            `beyond them saturates.`) +
        ` With the samples shown, dragging inside a dot snaps the probe onto ` +
        `that sample, which is where its target and its point loss come from.` +
        (thresholds.length === 0
          ? ` This network has no hidden layer, so there are no thresholds ` +
            `to show.`
          : ` The dotted lines are the thresholds: one per unit of the ` +
            `first hidden layer, where that unit's own preactivation is ` +
            `zero — for ${activation} that is ` +
            (activation === "relu"
              ? `its kink, with the unit contributing nothing on one side ` +
                `of the line and z on the other, so the creases in the ` +
                `surface lie along them`
              : `where the unit is at its most sensitive`) +
            `. Hover one to see which unit it belongs to, and watch it move ` +
            `as you edit that unit's weights or bias.`) +
        (isClassifier
          ? ` The neutral band is the decision boundary: the logit is zero ` +
            `there, so p = 0.5. Samples are drawn at the end of the scale ` +
            `their class wants, so a sample over a neutral background is one ` +
            `the model is unsure about.`
          : "");
    }

    // No running commentary on edits: an edit's before and after are in the
    // panels themselves — the slider's own readout, and the comparison in
    // "Prediction and loss" — and a line repeating them below the fold was
    // one more place to look.
    const unsubscribe = explorer.subscribe(render);

    return {
      explorer,
      render: () => render(explorer.snapshot()),
      destroy() {
        unsubscribe();
        networkView.destroy();
        inputOutput.destroy();
        paramSpace.destroy();
        tensors.destroy();
        inspector.destroy();
      },
    };
  }

  // The canvas cannot resolve the theme's custom properties, so it is the
  // one surface a light/dark switch does not repaint by itself.
  globalThis
    .matchMedia?.("(prefers-color-scheme: dark)")
    .addEventListener("change", () => session?.render());

  // Closing over `helps` so the popups are not collected while the page
  // lives; nothing else holds them.
  globalThis.addEventListener("pagehide", () => {
    for (const help of helps) help.destroy();
  });

  await choose(selection);
  // A default height is a guess about a window nobody has seen, and these
  // plots scale with their width: the same panel is taller on a wide
  // screen. Now that the views have drawn, each panel takes the height its
  // own content asks for. An arrangement restored from a previous session
  // is the reader's own and is left exactly as they left it.
  if (!workspace.isRestored()) workspace.fit();
}

/** Standing instructions, printed under the panels and never updated. */
const RESET_HINT =
  "Drag in the map to probe. Click a node, or any edge into it, to select that unit.";

/**
 * Where the explorer opens: the first dataset with one hidden layer at the
 * wider stop and ReLU, which is the configuration Stage 0 shipped and the
 * one with the most to look at. Falls back to whatever the manifest lists
 * first, so a catalog missing that entry still starts.
 */
function openingSelection(index: CatalogIndex): Selection {
  const preferred: Selection = {
    dataset: index.datasets[0].id,
    widths: [index.widths[index.widths.length - 1]],
    activation: "relu",
  };
  if (index.has(preferred)) return preferred;
  for (const dataset of index.datasets) {
    for (const widths of [[index.widths[0]], []]) {
      for (const activation of index.activations) {
        const candidate: Selection = { dataset: dataset.id, widths, activation };
        if (index.has(candidate)) return candidate;
      }
    }
  }
  throw new Error("the manifest offers no selectable configuration");
}

function setStatus(app: HTMLElement, message: string): void {
  app.querySelector("#status")!.textContent = message;
}

interface ControlHandlers {
  /** The live explorer, which a checkpoint switch replaces. */
  current: () => Explorer | null;
  setMode: (mode: DisplayMode) => void;
  toggleSamples: () => void;
  /** Back to every panel expanded and where it shipped. */
  resetLayout: () => void;
}

/**
 * The display controls, built once over whichever session is current: the
 * mode and sample toggles are reader preferences, and the parameter actions
 * are dispatched to the live explorer rather than bound to one.
 */
function buildControls(
  container: HTMLElement,
  handlers: ControlHandlers,
): {
  sync: (mode: DisplayMode, showSamples: boolean, showParameters: boolean) => void;
} {
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
  const buttons: [string, (explorer: Explorer) => void][] = [
    ["Load pre-trained", (explorer) => explorer.loadPretrained()],
    ["Randomize all", (explorer) => explorer.randomize()],
    ["Restore baseline", (explorer) => explorer.restoreBaseline()],
  ];
  for (const [label, action] of buttons) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = label;
    button.addEventListener("click", () => {
      const explorer = handlers.current();
      if (explorer) action(explorer);
    });
    actions.append(button);
  }
  // The layout is the reader's, not the model's, so its control sits apart
  // from the ones that act on parameters.
  const layout = document.createElement("div");
  layout.className = "control-group";
  const reset = document.createElement("button");
  reset.type = "button";
  reset.textContent = "Reset layout";
  reset.title = "Expand every panel and put it back where it started";
  reset.addEventListener("click", () => handlers.resetLayout());
  layout.append(reset);

  container.replaceChildren(modes, samplesGroup, actions, layout);

  return {
    sync(mode, showSamples, showParameters) {
      for (const entry of modeButtons) {
        entry.button.setAttribute("aria-pressed", String(entry.id === mode));
        // Nothing to switch while the edges carry no parameter: the group
        // stays where it is, greyed, rather than disappearing from a row
        // the reader has just learned the shape of.
        entry.button.disabled = !showParameters;
      }
      modes.classList.toggle("is-inert", !showParameters);
      samples.setAttribute("aria-pressed", String(showSamples));
    },
  };
}

/** A panel's own switches, and the ones that can go inert. */
interface Toggles {
  /**
   * Grey one out where it has nothing to act on, leaving it in place and
   * still checked: a control that vanished would move the others out from
   * under the pointer, and one left live and inert would lie.
   */
  setEnabled(name: string, enabled: boolean): void;
}

/**
 * A panel's own boxes, over the figure they act on rather than in the row
 * at the top: they are about that one figure.
 *
 * The network's two — the numbers on the nodes, and the parameters the
 * colours stand for — are there to build the figure up: the shape of the
 * network first, then what the parameters do to it, then the numbers
 * themselves. The input panel's one puts the first hidden layer's
 * thresholds over the map. Like the other display preferences they belong
 * to the reader and are held by the shell, so a checkpoint switch does not
 * quietly turn them back on.
 *
 * Every box starts checked, which is what makes these switches for putting
 * something away rather than for finding it.
 */
function buildToggles(
  container: HTMLElement,
  boxes: [string, (next: boolean) => void][],
): Toggles {
  const inputs = new Map<string, HTMLInputElement>();
  container.replaceChildren(
    ...boxes.map(([name, set]) => {
      const label = document.createElement("label");
      label.className = "toggle";
      const box = document.createElement("input");
      box.type = "checkbox";
      box.checked = true;
      box.addEventListener("change", () => set(box.checked));
      label.append(box, document.createTextNode(` ${name}`));
      inputs.set(name, box);
      return label;
    }),
  );
  return {
    setEnabled(name, enabled) {
      const box = inputs.get(name);
      if (!box) return;
      box.disabled = !enabled;
      box.closest(".toggle")?.classList.toggle("is-inert", !enabled);
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
