/**
 * Inputs → output: the map from the two inputs to the prediction, drawn three
 * ways that share their axes.
 *
 * The square in the corner is ŷ(x₁, x₂). Below it, sharing its x₁ axis, is
 * ŷ(x₁) with x₂ held where the probe is; to its left, sharing its x₂ axis, is
 * ŷ(x₂) with x₁ held, rotated so that x₂ still runs upward and ŷ grows away
 * from the map. The crosshairs are one pair of lines that crosses all three:
 * where they meet is the probe, and each slice meets its own curve there.
 *
 * The map carries its own two axes as thin dashed lines through the origin,
 * so a position on it can be read without counting from the edges.
 *
 * The slice backgrounds are the output scale, so a slice's height and its
 * background are two readings of the same number. The inputs have no colour
 * of their own: their axes carry their numbers.
 *
 * Canvas for the dense surface, SVG above it for everything pickable.
 */

import type { Slice, Surface } from "../model/surface.js";
import type { DatasetSpec } from "../model/types.js";
import type { Probe, Snapshot } from "../state.js";
import { gradientBar, type ColorScale, type Palette } from "./colormap.js";

const SVG = "http://www.w3.org/2000/svg";
/** The map is MAP×MAP user units and each slice MAP×SLICE, so the grid's
 *  1 : 4 column and row ratios scale both without distorting either. */
const MAP = 300;
const SLICE = 75;

export interface InputOutputView {
  render(snapshot: Snapshot, frame: InputOutputFrame): void;
  destroy(): void;
}

/** Everything the panel needs that it is not allowed to compute itself. */
export interface InputOutputFrame {
  surface: Surface;
  /** One slice per input, in axis order. */
  slices: Slice[];
  palette: Palette;
  showSamples: boolean;
}

export interface InputOutputOptions {
  container: HTMLElement;
  dataset: DatasetSpec;
  onProbe: (probe: Probe) => void;
}

export function createInputOutputView(options: InputOutputOptions): InputOutputView {
  const { container, dataset, onProbe } = options;
  if (dataset.domain.length !== 2) {
    throw new Error(`this panel draws 2 inputs, not ${dataset.domain.length}`);
  }
  const [[x1lo, x1hi], [x2lo, x2hi]] = dataset.domain;

  const canvas = document.createElement("canvas");
  canvas.width = MAP;
  canvas.height = MAP;
  canvas.className = "io-canvas";
  const context = canvas.getContext("2d");

  const mapSvg = element("svg", "io-marks", `0 0 ${MAP} ${MAP}`);
  mapSvg.setAttribute("role", "img");

  const map = document.createElement("div");
  map.className = "io-map";
  map.append(canvas, mapSvg);

  const xAxis = axisStrip("x", dataset.inputNames[0] ?? "x₁", x1lo, x1hi);
  const yAxis = axisStrip("y", dataset.inputNames[1] ?? "x₂", x2lo, x2hi);
  const below = slicePanel("x1");
  const left = slicePanel("x2");

  const grid = document.createElement("div");
  grid.className = "io-grid";
  grid.append(left.element, yAxis, map, xAxis, below.element);
  container.replaceChildren(grid);

  const toPixelX = (x: number) => ((x - x1lo) / (x1hi - x1lo)) * MAP;
  const toPixelY = (y: number) => MAP - ((y - x2lo) / (x2hi - x2lo)) * MAP;
  const fromPixelX = (px: number) => x1lo + (px / MAP) * (x1hi - x1lo);
  const fromPixelY = (py: number) => x2lo + ((MAP - py) / MAP) * (x2hi - x2lo);

  function pick(event: PointerEvent): void {
    const rect = mapSvg.getBoundingClientRect();
    const px = ((event.clientX - rect.left) / rect.width) * MAP;
    const py = ((event.clientY - rect.top) / rect.height) * MAP;
    onProbe({
      kind: "free",
      x: [
        Math.min(x1hi, Math.max(x1lo, fromPixelX(px))),
        Math.min(x2hi, Math.max(x2lo, fromPixelY(py))),
      ],
    });
  }

  function onPointerDown(event: PointerEvent): void {
    mapSvg.setPointerCapture(event.pointerId);
    pick(event);
  }
  function onPointerMove(event: PointerEvent): void {
    if (!mapSvg.hasPointerCapture(event.pointerId)) return;
    pick(event);
  }
  mapSvg.addEventListener("pointerdown", onPointerDown);
  mapSvg.addEventListener("pointermove", onPointerMove);

  function drawSurface(surface: Surface, scale: ColorScale): void {
    if (!context) return;
    const cell = MAP / surface.resolution;
    let index = 0;
    for (let row = 0; row < surface.resolution; row += 1) {
      for (let column = 0; column < surface.resolution; column += 1) {
        context.fillStyle = scale.paint(surface.values[index]);
        index += 1;
        // The extra pixel hides the seams between neighbouring cells.
        context.fillRect(column * cell, row * cell, cell + 1, cell + 1);
      }
    }
  }

  function render(snapshot: Snapshot, frame: InputOutputFrame): void {
    const { surface, slices, palette, showSamples } = frame;
    drawSurface(surface, palette.output);

    const px = toPixelX(snapshot.x[0]);
    const py = toPixelY(snapshot.x[1]);

    const marks: SVGElement[] = [];
    // The two input axes, where each input is zero. Thin and dashed, under
    // everything else: they are a reference for reading a position off the
    // map, not a quantity, and a domain that excludes zero simply has none.
    if (x1lo < 0 && 0 < x1hi) {
      const zero = toPixelX(0);
      marks.push(line(zero, 0, zero, MAP, "map-axis"));
    }
    if (x2lo < 0 && 0 < x2hi) {
      const zero = toPixelY(0);
      marks.push(line(0, zero, MAP, zero, "map-axis"));
    }
    if (showSamples) {
      dataset.samples.y.forEach((y, n) => {
        const mark = document.createElementNS(SVG, "circle");
        mark.setAttribute("cx", toPixelX(dataset.samples.x[n][0]).toFixed(2));
        mark.setAttribute("cy", toPixelY(dataset.samples.x[n][1]).toFixed(2));
        mark.setAttribute("r", "3.2");
        // The target on the same scale as the surface behind it, so a dot
        // that disappears into its background is one the model already fits.
        mark.setAttribute("fill", palette.output.css(y));
        mark.setAttribute("class", dataset.samples.isTest[n] ? "sample is-test" : "sample");
        marks.push(mark);
      });
    }
    marks.push(...crosshair(0, py, MAP, py), ...crosshair(px, 0, px, MAP));
    mapSvg.replaceChildren(...marks);
    mapSvg.setAttribute(
      "aria-label",
      `${dataset.targetName} over both inputs. Probe at ` +
        `${snapshot.x[0].toFixed(2)}, ${snapshot.x[1].toFixed(2)}, where ` +
        `${dataset.targetName} is ${snapshot.trace.output.toFixed(3)}.`,
    );

    below.draw({
      slice: slices[0],
      scale: palette.output,
      along: px,
      output: snapshot.trace.output,
      label:
        `${dataset.targetName} against ${dataset.inputNames[0]} with ` +
        `${dataset.inputNames[1]} held at ${snapshot.x[1].toFixed(3)}.`,
    });
    left.draw({
      slice: slices[1],
      scale: palette.output,
      along: py,
      output: snapshot.trace.output,
      label:
        `${dataset.targetName} against ${dataset.inputNames[1]} with ` +
        `${dataset.inputNames[0]} held at ${snapshot.x[0].toFixed(3)}.`,
    });
  }

  return {
    render,
    destroy() {
      mapSvg.removeEventListener("pointerdown", onPointerDown);
      mapSvg.removeEventListener("pointermove", onPointerMove);
      container.replaceChildren();
    },
  };
}

interface SliceDraw {
  slice: Slice;
  scale: ColorScale;
  /** Where the probe sits along the shared axis, in that plot's units. */
  along: number;
  output: number;
  label: string;
}

/**
 * One slice plot. `axis` names the input that varies, which also fixes the
 * orientation: "x1" lies under the map and shares its horizontal axis, "x2"
 * sits to its left, rotated a quarter turn so x₂ still runs upward and ŷ
 * grows leftward, away from the map.
 */
function slicePanel(axis: "x1" | "x2"): {
  element: HTMLElement;
  draw(draw: SliceDraw): void;
} {
  const horizontal = axis === "x1";
  const width = horizontal ? MAP : SLICE;
  const height = horizontal ? SLICE : MAP;

  const gradient = document.createElement("div");
  gradient.className = "slice-gradient";
  const svg = element("svg", "slice-marks", `0 0 ${width} ${height}`);
  svg.setAttribute("role", "img");

  const element_ = document.createElement("div");
  element_.className = `io-slice io-slice-${axis}`;
  element_.append(gradient, svg);

  return {
    element: element_,
    draw({ slice, scale, along, output, label }) {
      // The value axis runs across the SLICE-deep side of the plot: down the
      // page under the map, leftwards beside it.
      const toValue = (value: number) =>
        SLICE / 2 - (Math.max(-1, Math.min(1, value / scale.limit)) * SLICE) / 2;
      const [lo, hi] = [slice.inputs[0], slice.inputs[slice.inputs.length - 1]];
      const toInput = (input: number) =>
        horizontal
          ? ((input - lo) / (hi - lo)) * MAP
          : MAP - ((input - lo) / (hi - lo)) * MAP;
      const at = (input: number, value: number): [number, number] =>
        horizontal ? [toInput(input), toValue(value)] : [toValue(value), toInput(input)];

      gradient.style.backgroundImage = gradientBar(
        scale,
        -scale.limit,
        scale.limit,
        horizontal ? "to top" : "to left",
      );

      const marks: SVGElement[] = [];
      const zero = toValue(0);
      marks.push(
        horizontal
          ? line(0, zero, MAP, zero, "slice-zero")
          : line(zero, 0, zero, MAP, "slice-zero"),
      );

      const points = slice.inputs
        .map((input, index) => {
          const [x, y] = at(input, slice.values[index]);
          return `${x.toFixed(2)},${y.toFixed(2)}`;
        })
        .join(" ");
      // A casing under the curve, so it stays legible at either end of the
      // gradient without being drawn in a colour that would mean a value.
      for (const className of ["slice-curve-casing", "slice-curve"]) {
        const curve = document.createElementNS(SVG, "polyline");
        curve.setAttribute("points", points);
        curve.setAttribute("class", className);
        marks.push(curve);
      }

      // The crosshair continues through the slice, and the curve meets it at
      // the prediction.
      marks.push(
        ...(horizontal
          ? crosshair(along, 0, along, SLICE)
          : crosshair(0, along, SLICE, along)),
      );
      const [cx, cy] = horizontal ? [along, toValue(output)] : [toValue(output), along];
      const dot = document.createElementNS(SVG, "circle");
      dot.setAttribute("cx", cx.toFixed(2));
      dot.setAttribute("cy", cy.toFixed(2));
      dot.setAttribute("r", "3.5");
      dot.setAttribute("class", "slice-point");
      marks.push(dot);

      // The output scale's only numbers. The narrow plot gets zero alone;
      // three labels across SLICE units would collide.
      const ticks: number[] = horizontal ? [scale.limit, 0, -scale.limit] : [0];
      for (const value of ticks) {
        const tick = document.createElementNS(SVG, "text");
        const [tx, ty] = horizontal
          ? [3, toValue(value) + (value > 0 ? 8 : -3)]
          : [toValue(value) + 3, height - 3];
        tick.setAttribute("x", String(tx));
        tick.setAttribute("y", String(ty));
        tick.setAttribute("class", "slice-tick");
        tick.textContent = value > 0 ? `+${value}` : String(value);
        marks.push(tick);
      }

      svg.replaceChildren(...marks);
      svg.setAttribute("aria-label", label);
    },
  };
}

/**
 * One axis: its name and its two ends. The inputs carry no colour of their
 * own — these axes are where their values are read — so this is a label
 * strip, not a bar.
 */
function axisStrip(orientation: "x" | "y", name: string, lo: number, hi: number): HTMLElement {
  const strip = document.createElement("div");
  strip.className = `axis-strip axis-strip-${orientation}`;
  for (const text of [String(lo), name, String(hi)]) {
    const span = document.createElement("span");
    span.textContent = text;
    strip.append(span);
  }
  return strip;
}

/** A line plus the casing that keeps it legible over a saturated surface. */
function crosshair(x1: number, y1: number, x2: number, y2: number): SVGElement[] {
  return [
    line(x1, y1, x2, y2, "crosshair-casing"),
    line(x1, y1, x2, y2, "crosshair"),
  ];
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

function element(tag: string, className: string, viewBox: string): SVGElement {
  const node = document.createElementNS(SVG, tag);
  node.setAttribute("class", className);
  node.setAttribute("viewBox", viewBox);
  return node;
}
