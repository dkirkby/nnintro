/**
 * The output space: what the network does along one input at a time.
 *
 * Each plot sweeps one input across its whole range with the other held
 * where the probe left it, so the two together are the cross-section of the
 * input space through the probe. The vertical axis is the shared colour
 * scale and the background is that scale as a gradient, which makes height
 * and colour two readings of the same number: the curve meets its own
 * colour, and the point where it crosses the probe's line is the colour the
 * input space shows underneath the probe.
 *
 * It also gives the colour scale the numbers it otherwise lacks, which is
 * what the ticks are for.
 */

import type { Slice } from "../model/surface.js";
import type { DatasetSpec } from "../model/types.js";
import type { Snapshot } from "../state.js";
import type { ColorScale } from "./colormap.js";
import { format } from "./network.js";

const SVG = "http://www.w3.org/2000/svg";
const PLOT = { width: 240, height: 140 };

export interface OutputSpaceView {
  render(snapshot: Snapshot, frame: OutputSpaceFrame): void;
  destroy(): void;
}

export interface OutputSpaceFrame {
  /** One slice per input, in axis order. */
  slices: Slice[];
  scale: ColorScale;
}

export interface OutputSpaceOptions {
  container: HTMLElement;
  dataset: DatasetSpec;
}

export function createOutputSpaceView(options: OutputSpaceOptions): OutputSpaceView {
  const { container, dataset } = options;
  if (dataset.domain.length !== 2) {
    // One plot per input, each captioned with the one input being held: a
    // third input would need the caption, and the panel, rethought.
    throw new Error(`the output space is drawn for 2 inputs, not ${dataset.domain.length}`);
  }
  const plots = dataset.domain.map((_, axis) => createSlicePlot(dataset, axis));
  const legend = document.createElement("p");
  legend.className = "legend";

  const wrapper = document.createElement("div");
  wrapper.className = "output-space";
  wrapper.append(...plots.map((plot) => plot.element));
  container.replaceChildren(wrapper, legend);

  function render(snapshot: Snapshot, frame: OutputSpaceFrame): void {
    for (const plot of plots) plot.draw(snapshot, frame);
    const offScale = frame.slices.some((slice) =>
      slice.values.some((value) => Math.abs(value) > frame.scale.limit),
    );
    legend.textContent =
      `Height and background are the same ±${frame.scale.limit} scale as the ` +
      `input space. The line marks the probe.` +
      (offScale ? " Part of a curve runs past the scale and is cut off." : "");
  }

  return {
    render,
    destroy() {
      container.replaceChildren();
    },
  };
}

interface SlicePlot {
  element: HTMLElement;
  draw(snapshot: Snapshot, frame: OutputSpaceFrame): void;
}

function createSlicePlot(dataset: DatasetSpec, axis: number): SlicePlot {
  const name = dataset.inputNames[axis] ?? `x${axis + 1}`;
  const [lo, hi] = dataset.domain[axis];

  const caption = document.createElement("figcaption");
  caption.className = "slice-caption";

  const gradient = document.createElement("div");
  gradient.className = "slice-gradient";

  const svg = document.createElementNS(SVG, "svg");
  svg.setAttribute("viewBox", `0 0 ${PLOT.width} ${PLOT.height}`);
  svg.setAttribute("class", "slice-marks");
  svg.setAttribute("role", "img");

  const plot = document.createElement("div");
  plot.className = "slice-plot";
  plot.append(gradient, svg);

  const ends = document.createElement("div");
  ends.className = "slice-axis";
  for (const text of [String(lo), name, String(hi)]) {
    const span = document.createElement("span");
    span.textContent = text;
    ends.append(span);
  }

  const element = document.createElement("figure");
  element.className = "slice";
  element.append(caption, plot, ends);

  const toX = (value: number) => ((value - lo) / (hi - lo)) * PLOT.width;

  function draw(snapshot: Snapshot, frame: OutputSpaceFrame): void {
    const { scale } = frame;
    const slice = frame.slices[axis];
    const toY = (value: number) =>
      PLOT.height / 2 - (value / scale.limit) * (PLOT.height / 2);

    // The background is the colour scale itself, upright: the top of the
    // plot is +limit, which is where the gradient's last stop belongs.
    gradient.style.backgroundImage =
      `linear-gradient(to top, ${scale.stops(-scale.limit, scale.limit).join(", ")})`;

    const held = dataset.inputNames[1 - axis] ?? `x${2 - axis}`;
    caption.textContent = `ŷ(${name}) at ${held} = ${format(slice.at[1 - axis])}`;

    const marks: SVGElement[] = [];
    marks.push(
      rule(0, toY(0), PLOT.width, toY(0), "slice-zero"),
      rule(toX(snapshot.x[axis]), 0, toX(snapshot.x[axis]), PLOT.height, "slice-probe"),
    );

    const points = slice.inputs
      .map((input, index) => `${toX(input).toFixed(2)},${toY(slice.values[index]).toFixed(2)}`)
      .join(" ");
    // A casing under the curve, so it stays legible at either end of the
    // gradient without being drawn in a colour that would mean a value.
    for (const className of ["slice-curve-casing", "slice-curve"]) {
      const line = document.createElementNS(SVG, "polyline");
      line.setAttribute("points", points);
      line.setAttribute("class", className);
      marks.push(line);
    }

    const dot = document.createElementNS(SVG, "circle");
    dot.setAttribute("cx", toX(snapshot.x[axis]).toFixed(2));
    dot.setAttribute("cy", toY(snapshot.trace.output).toFixed(2));
    dot.setAttribute("r", "4");
    dot.setAttribute("class", "slice-point");
    marks.push(dot);

    for (const value of [scale.limit, 0, -scale.limit]) {
      const tick = document.createElementNS(SVG, "text");
      tick.setAttribute("x", "3");
      // Nudged inside the box at the ends, where the label would clip.
      tick.setAttribute("y", (toY(value) + (value > 0 ? 9 : value < 0 ? -3 : -3)).toFixed(1));
      tick.setAttribute("class", "slice-tick");
      tick.textContent = value > 0 ? `+${value}` : String(value);
      marks.push(tick);
    }

    svg.replaceChildren(...marks);
    svg.setAttribute(
      "aria-label",
      `${dataset.targetName} against ${name} with ${held} held at ` +
        `${format(slice.at[1 - axis])}: ${format(slice.values[0])} at ${name} = ${lo}, ` +
        `${format(snapshot.trace.output)} at the probe, ` +
        `${format(slice.values[slice.values.length - 1])} at ${name} = ${hi}.`,
    );
  }

  return { element, draw };
}

function rule(x1: number, y1: number, x2: number, y2: number, className: string): SVGElement {
  const line = document.createElementNS(SVG, "line");
  line.setAttribute("x1", x1.toFixed(2));
  line.setAttribute("y1", y1.toFixed(2));
  line.setAttribute("x2", x2.toFixed(2));
  line.setAttribute("y2", y2.toFixed(2));
  line.setAttribute("class", className);
  return line;
}
