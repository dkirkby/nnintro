/**
 * Parameters → output: every parameter's effect on the prediction, at once.
 *
 * One line per parameter, each the output as that parameter alone is swept
 * across the parameter scale with everything else — the other parameters and
 * the probe — held. A dot marks where each parameter actually is, and since
 * every curve passes through the current output at the current value, the
 * dots all land on the horizontal line. A line's slope under its dot is
 * ∂ŷ/∂θ, and a bend is a ReLU switching on or off.
 *
 * The parameter colour scale is the x axis, and the slider under it edits
 * whichever parameter is selected, along that same axis.
 */

import type { ParameterCurve } from "../model/curves.js";
import { describeParameter, sameParameter, type ParameterRef } from "../model/types.js";
import type { Snapshot } from "../state.js";
import { gradientBar, type ColorScale } from "./colormap.js";
import { format } from "./network.js";

const SVG = "http://www.w3.org/2000/svg";
const PLOT = { width: 300, height: 140 };

export interface ParamSpaceView {
  render(snapshot: Snapshot, frame: ParamSpaceFrame): void;
  destroy(): void;
}

export interface ParamSpaceFrame {
  curves: ParameterCurve[];
  /** The parameter scale: the x axis, and the colour bar under it. */
  scale: ColorScale;
  /** Half-height of the output axis, covering every curve. */
  outputLimit: number;
}

export interface ParamSpaceOptions {
  container: HTMLElement;
  onEdit: (ref: ParameterRef, value: number) => void;
}

export function createParamSpaceView(options: ParamSpaceOptions): ParamSpaceView {
  const { container, onEdit } = options;

  const svg = document.createElementNS(SVG, "svg");
  svg.setAttribute("viewBox", `0 0 ${PLOT.width} ${PLOT.height}`);
  svg.setAttribute("class", "param-plot");
  svg.setAttribute("role", "img");

  const bar = document.createElement("div");
  bar.className = "axis-bar param-axis";
  bar.setAttribute("role", "img");
  const low = document.createElement("span");
  const high = document.createElement("span");
  const name = document.createElement("span");
  name.className = "axis-bar-label";
  bar.append(low, name, high);

  const slider = document.createElement("input");
  slider.type = "range";
  slider.className = "param-slider";

  const hint = document.createElement("p");
  hint.className = "legend";

  container.replaceChildren(svg, bar, slider, hint);

  /** Set on every render so the listener always edits the live selection. */
  let selected: ParameterRef | null = null;
  function onInput(): void {
    if (!selected) return;
    const next = Number(slider.value);
    if (Number.isFinite(next)) onEdit(selected, next);
  }
  slider.addEventListener("input", onInput);

  function render(snapshot: Snapshot, frame: ParamSpaceFrame): void {
    const { curves, scale, outputLimit } = frame;
    selected = snapshot.selection;

    const toX = (value: number) => ((value + scale.limit) / (2 * scale.limit)) * PLOT.width;
    const toY = (value: number) =>
      PLOT.height / 2 -
      (Math.max(-1, Math.min(1, value / outputLimit)) * PLOT.height) / 2;

    const marks: SVGElement[] = [];
    const zero = toX(0);
    marks.push(line(zero, 0, zero, PLOT.height, "param-zero"));

    // The current output: every dot sits on it, which is the panel's one
    // structural claim and the quickest way to see a bug here.
    const level = toY(snapshot.trace.output);
    marks.push(line(0, level, PLOT.width, level, "output-level"));

    const isChosen = (curve: ParameterCurve) =>
      selected !== null && sameParameter(curve.ref, selected);
    const chosen = curves.filter(isChosen);
    const rest = curves.filter((curve) => !isChosen(curve));

    // The highlighted curve is drawn over the thicket, and the dots over
    // everything, so neither can be buried by a later parameter.
    for (const curve of [...rest, ...chosen]) {
      const points = curve.values
        .map((value, index) => `${toX(value).toFixed(2)},${toY(curve.outputs[index]).toFixed(2)}`)
        .join(" ");
      const polyline = document.createElementNS(SVG, "polyline");
      polyline.setAttribute("points", points);
      polyline.setAttribute(
        "class",
        isChosen(curve) ? "param-curve is-selected" : "param-curve",
      );
      marks.push(polyline);
    }
    for (const curve of [...rest, ...chosen]) {
      const dot = document.createElementNS(SVG, "circle");
      dot.setAttribute("cx", toX(curve.current).toFixed(2));
      dot.setAttribute("cy", toY(curve.output).toFixed(2));
      dot.setAttribute("r", isChosen(curve) ? "4.5" : "2.5");
      dot.setAttribute("class", isChosen(curve) ? "param-dot is-selected" : "param-dot");
      marks.push(dot);
    }

    for (const value of [outputLimit, -outputLimit]) {
      const tick = document.createElementNS(SVG, "text");
      tick.setAttribute("x", "3");
      tick.setAttribute("y", String(toY(value) + (value > 0 ? 9 : -3)));
      tick.setAttribute("class", "slice-tick");
      tick.textContent = value > 0 ? `+${value}` : String(value);
      marks.push(tick);
    }
    const label = document.createElementNS(SVG, "text");
    label.setAttribute("x", String(PLOT.width - 3));
    label.setAttribute("y", String(level - 4));
    label.setAttribute("text-anchor", "end");
    label.setAttribute("class", "slice-tick");
    label.textContent = `ŷ = ${format(snapshot.trace.output)}`;
    marks.push(label);

    svg.replaceChildren(...marks);
    svg.setAttribute(
      "aria-label",
      `${curves.length} parameters swept one at a time; every curve passes ` +
        `through ŷ = ${format(snapshot.trace.output)} at its current value.`,
    );

    bar.style.backgroundImage = gradientBar(scale, -scale.limit, scale.limit, "to right");
    bar.setAttribute("aria-label", `parameter value, from −${scale.limit} to ${scale.limit}`);
    low.textContent = `−${scale.limit}`;
    high.textContent = `+${scale.limit}`;
    name.textContent = "parameter value";

    slider.min = String(-scale.limit);
    slider.max = String(scale.limit);
    slider.step = String(scale.limit / 500);
    slider.disabled = selected === null;
    if (selected) {
      const current = curves.find((curve) => sameParameter(curve.ref, selected!));
      // Not while it is being dragged: writing the value back mid-gesture
      // fights the thumb.
      if (current && document.activeElement !== slider) {
        slider.value = String(current.current);
      }
      slider.setAttribute("aria-label", `value of ${describeParameter(selected)}`);
      hint.textContent =
        `Dragging the slider edits ${describeParameter(selected)} and moves its ` +
        `dot along the highlighted curve; every other curve moves too, since ` +
        `they are drawn with this parameter held.`;
    } else {
      slider.value = "0";
      slider.setAttribute("aria-label", "parameter value");
      hint.textContent =
        "Each line is one parameter swept across the scale above, everything " +
        "else held. Select an edge or a node to highlight and edit one.";
    }
  }

  return {
    render,
    destroy() {
      slider.removeEventListener("input", onInput);
      container.replaceChildren();
    },
  };
}

function line(x1: number, y1: number, x2: number, y2: number, className: string): SVGElement {
  const node = document.createElementNS(SVG, "line");
  node.setAttribute("x1", x1.toFixed(2));
  node.setAttribute("y1", y1.toFixed(2));
  node.setAttribute("x2", x2.toFixed(2));
  node.setAttribute("y2", y2.toFixed(2));
  node.setAttribute("class", className);
  return node;
}
