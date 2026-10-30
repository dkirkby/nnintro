/**
 * The catalog controls: which dataset, how deep, how wide, which hidden
 * activation.
 *
 * Depth and width are **discrete sliders with labeled stops** (PLAN.md
 * section 3): the quantities are ordered, and a slider says so where a row
 * of buttons would not. Dataset and activation are named rather than
 * ordered, so they are button groups.
 *
 * Every stop is checked against the published catalog before it is offered.
 * A combination with no checkpoint is not reachable: its button is
 * disabled, and a slider dragged onto it snaps back and says why. The
 * controls never ask for an asset that does not exist (PLAN.md section 3),
 * which is why they enumerate the manifest instead of the architecture
 * family they happen to know about.
 */

import type { CatalogIndex, Selection } from "../model/catalog.js";
import type { Activation } from "../model/types.js";

export interface SelectorOptions {
  container: HTMLElement;
  index: CatalogIndex;
  /** Called with a selection that is known to exist in the catalog. */
  onSelect: (selection: Selection) => void;
  /** Called when a control was dragged onto a combination with no checkpoint. */
  onUnavailable: (message: string) => void;
}

export interface Selector {
  /**
   * Re-reads every control's state from the live selection: which stop is
   * current, and which of the others the catalog actually covers.
   *
   * Nothing here is disabled while a checkpoint is loading. A discrete
   * slider fires an event per stop it crosses, so disabling it mid-drag
   * would break the gesture; the caller discards superseded loads instead.
   */
  sync(selection: Selection): void;
  destroy(): void;
}

const ACTIVATION_LABELS: Record<Activation, string> = {
  relu: "ReLU",
  tanh: "tanh",
  identity: "identity",
};

/** The architecture a selection describes, as the header writes it. */
export function describeArchitecture(selection: Selection): string {
  return [2, ...selection.widths, 1].join("–");
}

export function createSelector(options: SelectorOptions): Selector {
  const { container, index, onSelect, onUnavailable } = options;
  const cleanup: (() => void)[] = [];
  /** The selection every control reads when it builds its own candidates. */
  let current: Selection = {
    dataset: index.datasets[0].id,
    widths: [],
    activation: index.activations[0] ?? "relu",
  };

  /** Ask for a selection, or explain why it cannot be had. */
  function request(next: Selection, what: string): void {
    if (index.has(next)) {
      onSelect(next);
      return;
    }
    onUnavailable(
      `The catalog has no checkpoint for ${what}, so that stop is not ` +
        `available. Nothing changed.`,
    );
    sync(current);
  }

  // --- dataset: named, so a button group ---
  const datasetGroup = group("dataset");
  const datasetButtons = index.datasets.map((dataset) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = dataset.label;
    button.title = dataset.description;
    button.dataset.dataset = dataset.id;
    const onClick = () =>
      request({ ...current, dataset: dataset.id }, `${dataset.label} at this architecture`);
    button.addEventListener("click", onClick);
    cleanup.push(() => button.removeEventListener("click", onClick));
    datasetGroup.body.append(button);
    return { id: dataset.id, button };
  });

  // --- depth and width: ordered, so discrete sliders ---
  const depths = index.depths;
  const depthSlider = stepper("hidden layers", depths.map(String));
  const onDepth = () => {
    const depth = depths[Number(depthSlider.input.value)];
    // A layer appearing has to be given a width. It inherits the width of
    // the last layer that already existed, so 2–4–1 deepens to 2–4–4–1
    // rather than jumping to some unrelated default.
    const widths = current.widths.slice(0, depth);
    while (widths.length < depth) {
      widths.push(widths[widths.length - 1] ?? index.widths[0]);
    }
    request(
      { ...current, widths },
      depth === 1 ? "one hidden layer here" : `${depth} hidden layers here`,
    );
  };
  depthSlider.input.addEventListener("input", onDepth);
  cleanup.push(() => depthSlider.input.removeEventListener("input", onDepth));

  const maxDepth = depths[depths.length - 1];
  const widthSliders = Array.from({ length: maxDepth }, (_, layer) => {
    const slider = stepper(`width of hidden layer ${layer + 1}`, index.widths.map(String));
    const onInput = () => {
      const widths = current.widths.slice();
      widths[layer] = index.widths[Number(slider.input.value)];
      request({ ...current, widths }, `width ${widths[layer]} in hidden layer ${layer + 1}`);
    };
    slider.input.addEventListener("input", onInput);
    cleanup.push(() => slider.input.removeEventListener("input", onInput));
    return slider;
  });

  // --- activation: named, so a button group ---
  const activationGroup = group("hidden activation");
  const activationButtons = index.activations.map((name) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = ACTIVATION_LABELS[name];
    button.dataset.activation = name;
    const onClick = () => request({ ...current, activation: name }, `${ACTIVATION_LABELS[name]} here`);
    button.addEventListener("click", onClick);
    cleanup.push(() => button.removeEventListener("click", onClick));
    activationGroup.body.append(button);
    return { name, button };
  });

  container.replaceChildren(
    datasetGroup.root,
    depthSlider.root,
    ...widthSliders.map((slider) => slider.root),
    activationGroup.root,
  );

  function sync(selection: Selection): void {
    current = selection;
    const depth = selection.widths.length;

    for (const entry of datasetButtons) {
      entry.button.setAttribute("aria-pressed", String(entry.id === selection.dataset));
      entry.button.disabled = !index.has({ ...selection, dataset: entry.id });
    }

    depthSlider.set(
      depths.indexOf(depth),
      depths.map((candidate) => {
        const widths = selection.widths.slice(0, candidate);
        while (widths.length < candidate) {
          widths.push(widths[widths.length - 1] ?? index.widths[0]);
        }
        return index.has({ ...selection, widths });
      }),
    );

    widthSliders.forEach((slider, layer) => {
      // A width control for a layer that does not exist is not disabled but
      // absent: at depth 0 there is no such thing as the width of hidden
      // layer 1, and a greyed-out slider would imply there is.
      slider.root.hidden = layer >= depth;
      if (layer >= depth) return;
      slider.set(
        index.widths.indexOf(selection.widths[layer]),
        index.widths.map((candidate) => {
          const widths = selection.widths.slice();
          widths[layer] = candidate;
          return index.has({ ...selection, widths });
        }),
      );
    });

    // With no hidden layer there is no hidden activation to choose, and the
    // catalog has one entry rather than three. The buttons stay visible and
    // disabled, because the choice returns the moment a layer does.
    activationGroup.root.classList.toggle("is-inert", depth === 0);
    for (const entry of activationButtons) {
      entry.button.setAttribute(
        "aria-pressed",
        String(depth > 0 && entry.name === selection.activation),
      );
      entry.button.disabled =
        depth === 0 || !index.has({ ...selection, activation: entry.name });
    }
  }

  return {
    sync,
    destroy() {
      for (const undo of cleanup) undo();
      container.replaceChildren();
    },
  };
}

/** A labeled group of buttons. */
function group(label: string): { root: HTMLElement; body: HTMLElement } {
  const root = document.createElement("div");
  root.className = "selector-group";
  const caption = document.createElement("span");
  caption.className = "selector-label";
  caption.textContent = label;
  const body = document.createElement("div");
  body.className = "control-group";
  body.setAttribute("role", "group");
  body.setAttribute("aria-label", label);
  root.append(caption, body);
  return { root, body };
}

/**
 * A discrete slider over `stops`, with every stop written out beneath it.
 *
 * The slider's value is a stop *index*, not the quantity: the stops are
 * whatever the catalog covers, and need be neither consecutive nor evenly
 * spaced for the control to work.
 */
function stepper(
  label: string,
  stops: string[],
): {
  root: HTMLElement;
  input: HTMLInputElement;
  set: (index: number, available: boolean[]) => void;
} {
  const root = document.createElement("div");
  root.className = "selector-group";
  const caption = document.createElement("span");
  caption.className = "selector-label";
  caption.textContent = label;

  const input = document.createElement("input");
  input.type = "range";
  input.className = "stop-slider";
  input.min = "0";
  input.max = String(stops.length - 1);
  input.step = "1";
  input.setAttribute("aria-label", label);

  const scale = document.createElement("div");
  scale.className = "stop-scale";
  // One column per stop, so each label sits under its own notch.
  scale.style.gridTemplateColumns = `repeat(${stops.length}, 1fr)`;
  const marks = stops.map((text) => {
    const mark = document.createElement("span");
    mark.className = "stop-mark";
    mark.textContent = text;
    scale.append(mark);
    return mark;
  });

  const track = document.createElement("div");
  track.className = "stop-track";
  track.append(input, scale);
  root.append(caption, track);

  return {
    root,
    input,
    set(index, available) {
      input.value = String(index);
      input.setAttribute("aria-valuetext", stops[index]);
      marks.forEach((mark, stop) => {
        mark.classList.toggle("is-current", stop === index);
        // Said before it is dragged onto, not only after: a stop with no
        // checkpoint is struck through and titled.
        mark.classList.toggle("stop-missing", !available[stop]);
        if (available[stop]) mark.removeAttribute("title");
        else mark.title = "no checkpoint in the catalog for this combination";
      });
    },
  };
}
