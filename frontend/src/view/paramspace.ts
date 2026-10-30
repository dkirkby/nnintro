/**
 * Parameters → output: the selected unit's parameters' effect on the
 * prediction, at once.
 *
 * One line per parameter of the selected unit — its bias and every weight
 * feeding it, and nothing else — each the output as that parameter alone is
 * swept across the parameter scale with everything else (the other
 * parameters and the probe) held. A dot marks where each parameter actually
 * is, and since every curve passes through the current output at the current
 * value, the dots all land on the horizontal line. A line's slope under its
 * dot is ∂ŷ/∂θ, and a bend is a ReLU switching on or off.
 *
 * The parameter colour scale is the x axis, and under it one slider per
 * parameter on that same axis — the bias first, then the weights in the
 * order their edges appear from top to bottom in the network diagram. The
 * plot, the colour bar and every slider share one grid column, so a dot's x
 * position and its slider's thumb are the same place.
 *
 * The background is the output scale standing up, as in the slices, so a
 * curve's height and the colour behind it are two readings of the same
 * prediction.
 */

import type { ParameterCurve } from "../model/curves.js";
import {
  describeParameter,
  describeUnit,
  parameterSymbol,
  type ParameterRef,
} from "../model/types.js";
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
  /** The selected unit's parameters, bias first: one curve each. */
  curves: ParameterCurve[];
  /** The parameter scale: the x axis, and the colour bar under it. */
  paramScale: ColorScale;
  /** The output scale, painted up the background. */
  outputScale: ColorScale;
  /** Half-height of the output axis, covering every curve. */
  outputLimit: number;
}

export interface ParamSpaceOptions {
  container: HTMLElement;
  /**
   * Where this panel's explanation is written. Owned by the caller, not
   * created here: it lives in the panel's `?` popup, which outlives any one
   * checkpoint's view (see `view/help.ts`).
   */
  legend: HTMLElement;
  /**
   * What the vertical axis is: `ŷ` for a regression, `z` for a classifier,
   * whose output is a logit rather than a prediction of the target.
   */
  outputName: string;
  onEdit: (ref: ParameterRef, value: number) => void;
}

/** One parameter's row of controls, reused across renders while it lasts. */
interface SliderRow {
  ref: ParameterRef;
  name: HTMLElement;
  slider: HTMLInputElement;
  readout: HTMLElement;
}

export function createParamSpaceView(options: ParamSpaceOptions): ParamSpaceView {
  const { container, legend, outputName, onEdit } = options;

  // One grid, so the plot, the colour bar and every slider span the same
  // column: the axis is shared, not merely similar.
  const grid = document.createElement("div");
  grid.className = "param-editor";

  const svg = document.createElementNS(SVG, "svg");
  svg.setAttribute("viewBox", `0 0 ${PLOT.width} ${PLOT.height}`);
  svg.setAttribute("class", "param-plot");
  svg.setAttribute("role", "img");

  const bar = document.createElement("div");
  bar.className = "param-axis";
  bar.setAttribute("role", "img");
  const low = document.createElement("span");
  const high = document.createElement("span");
  const name = document.createElement("span");
  name.className = "param-axis-name";
  bar.append(low, name, high);

  container.replaceChildren(grid);

  /**
   * Rebuilt only when the selected unit's parameters change, so dragging a
   * thumb never replaces the element under the pointer.
   */
  let rows: SliderRow[] = [];

  function rowsFor(refs: ParameterRef[]): SliderRow[] {
    const wanted = refs.map(describeParameter).join("|");
    if (rows.map((row) => describeParameter(row.ref)).join("|") === wanted) {
      return rows;
    }
    for (const row of rows) row.slider.removeEventListener("input", onInput);
    rows = refs.map((ref) => {
      const label = document.createElement("span");
      label.className = "param-row-name";
      label.textContent = parameterSymbol(ref);
      label.title = describeParameter(ref);

      const slider = document.createElement("input");
      slider.type = "range";
      slider.className = "param-slider";
      slider.dataset.param = describeParameter(ref);
      slider.addEventListener("input", onInput);

      const readout = document.createElement("span");
      readout.className = "param-row-value";

      return { ref, name: label, slider, readout };
    });
    grid.replaceChildren(
      svg,
      bar,
      ...rows.flatMap((row) => [row.name, row.slider, row.readout]),
    );
    return rows;
  }

  /** The row a slider belongs to, so one listener serves them all. */
  function onInput(event: Event): void {
    const row = rows.find((candidate) => candidate.slider === event.currentTarget);
    if (!row) return;
    const next = Number(row.slider.value);
    if (Number.isFinite(next)) onEdit(row.ref, next);
  }

  function render(snapshot: Snapshot, frame: ParamSpaceFrame): void {
    const { curves, paramScale: scale, outputScale, outputLimit } = frame;

    // The background is the output scale over the axis this plot actually
    // spans, which is taller than that scale's own limit; values past it
    // saturate, exactly as they do on a node.
    svg.style.backgroundImage = gradientBar(outputScale, -outputLimit, outputLimit, "to top");

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

    // Curves first, then dots, then the symbols, so nothing a reader needs
    // in order to tell two parameters apart can be buried by a later one.
    for (const curve of curves) {
      const points = curve.values
        .map((value, index) => `${toX(value).toFixed(2)},${toY(curve.outputs[index]).toFixed(2)}`)
        .join(" ");
      const polyline = document.createElementNS(SVG, "polyline");
      polyline.setAttribute("points", points);
      polyline.setAttribute("class", "param-curve");
      polyline.setAttribute("data-param", describeParameter(curve.ref));
      const title = document.createElementNS(SVG, "title");
      title.textContent = `${describeParameter(curve.ref)} swept`;
      polyline.append(title);
      marks.push(polyline);
    }
    for (const curve of curves) {
      const dot = document.createElementNS(SVG, "circle");
      dot.setAttribute("cx", toX(curve.current).toFixed(2));
      dot.setAttribute("cy", toY(curve.output).toFixed(2));
      dot.setAttribute("r", "3.5");
      dot.setAttribute("class", "param-dot");
      dot.setAttribute("data-param", describeParameter(curve.ref));
      marks.push(dot);
    }
    // The output line can sit anywhere, including hard against an edge, so
    // the symbols take the roomier side of it and the ŷ label the other.
    const roomAbove = level > PLOT.height / 2;
    // Clear of the two value ticks, which own the left edge of the top and
    // bottom rows: a symbol for a parameter sitting at the end of the axis
    // would otherwise be drawn straight through one.
    const symbolY = clamp(roomAbove ? level - 7 : level + 14, 20, PLOT.height - 14);
    const labelY = clamp(roomAbove ? level + 14 : level - 6, 10, PLOT.height - 4);

    for (const curve of curves) {
      // Beside the dot, which is on the output line, so each symbol sits at
      // its own parameter's value along the axis — pulled in from the ends,
      // where a parameter at the limit would otherwise be half cut off.
      const symbol = document.createElementNS(SVG, "text");
      symbol.setAttribute("x", clamp(toX(curve.current), 9, PLOT.width - 9).toFixed(2));
      symbol.setAttribute("y", symbolY.toFixed(2));
      symbol.setAttribute("text-anchor", "middle");
      symbol.setAttribute("class", "param-symbol");
      symbol.setAttribute("data-param", describeParameter(curve.ref));
      symbol.textContent = parameterSymbol(curve.ref);
      marks.push(symbol);
    }

    for (const value of [outputLimit, -outputLimit]) {
      const tick = document.createElementNS(SVG, "text");
      tick.setAttribute("x", "3");
      tick.setAttribute("y", String(toY(value) + (value > 0 ? 9 : -3)));
      tick.setAttribute("class", "slice-tick");
      tick.textContent = value > 0 ? `+${value}` : String(value);
      marks.push(tick);
    }
    // Opposite the symbols, so a parameter sitting near the end of the axis
    // cannot collide with this.
    const label = document.createElementNS(SVG, "text");
    label.setAttribute("x", String(PLOT.width - 3));
    label.setAttribute("y", labelY.toFixed(2));
    label.setAttribute("text-anchor", "end");
    label.setAttribute("class", "slice-tick");
    label.textContent = `${outputName} = ${format(snapshot.trace.output)}`;
    marks.push(label);

    svg.replaceChildren(...marks);
    svg.setAttribute(
      "aria-label",
      `${curves.length} parameters of ${describeUnit(snapshot.selection)} swept ` +
        `one at a time; every curve passes through ${outputName} = ` +
        `${format(snapshot.trace.output)} at its current value.`,
    );

    bar.style.backgroundImage = gradientBar(scale, -scale.limit, scale.limit, "to right");
    bar.setAttribute("aria-label", `parameter value, from −${scale.limit} to ${scale.limit}`);
    low.textContent = `−${scale.limit}`;
    high.textContent = `+${scale.limit}`;
    name.textContent = "parameter value";

    for (const row of rowsFor(curves.map((curve) => curve.ref))) {
      const curve = curves.find(
        (candidate) => describeParameter(candidate.ref) === describeParameter(row.ref),
      );
      row.slider.min = String(-scale.limit);
      row.slider.max = String(scale.limit);
      row.slider.step = String(scale.limit / 500);
      row.slider.setAttribute("aria-label", `value of ${describeParameter(row.ref)}`);
      if (!curve) continue;
      // Not while it is being dragged: writing the value back mid-gesture
      // fights the thumb.
      if (document.activeElement !== row.slider) {
        row.slider.value = String(curve.current);
      }
      row.readout.textContent = format(curve.current);
    }

    legend.textContent =
      `Each line is one parameter of ${describeUnit(snapshot.selection)} — its ` +
      `bias b and the weights wᵢ feeding it — swept across the scale above, ` +
      `everything else held. Drag a slider to edit that parameter and move ` +
      `its dot along its own curve; every other curve moves too, since they ` +
      `are drawn with it held. Select another node or edge to change units.`;
  }

  return {
    render,
    destroy() {
      for (const row of rows) row.slider.removeEventListener("input", onInput);
      rows = [];
      container.replaceChildren();
    },
  };
}

function clamp(value: number, low: number, high: number): number {
  return Math.max(low, Math.min(high, value));
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
