/**
 * The input-space view: the prediction surface, the probe, and the colour
 * bars that read the two input axes.
 *
 * Canvas for the dense surface, an SVG layer above it for the marks — dense
 * pixels on canvas, pickable things in the DOM. The surface, the axis bars,
 * the samples and the network's nodes all share one colour scale (see
 * `./colormap.ts`), so the colour under the probe is the colour of the output
 * node, and an input node's colour is the one its axis bar shows at that
 * coordinate.
 *
 * Samples are hidden until asked for: the first thing to do with this panel
 * is probe it, and four hundred dots on top of the surface answer the
 * question before it has been put.
 */

import type { Surface } from "../model/surface.js";
import type { DatasetSpec } from "../model/types.js";
import type { Probe, Snapshot } from "../state.js";
import type { ColorScale } from "./colormap.js";

const SIZE = 300;
const SVG = "http://www.w3.org/2000/svg";

export interface InputSpaceView {
  render(snapshot: Snapshot, frame: InputSpaceFrame): void;
  destroy(): void;
}

/** Everything the panel needs that it is not allowed to compute itself. */
export interface InputSpaceFrame {
  surface: Surface;
  scale: ColorScale;
  showSamples: boolean;
}

export interface InputSpaceOptions {
  container: HTMLElement;
  dataset: DatasetSpec;
  onProbe: (probe: Probe) => void;
}

export function createInputSpaceView(options: InputSpaceOptions): InputSpaceView {
  const { container, dataset, onProbe } = options;
  const [[x1lo, x1hi], [x2lo, x2hi]] = dataset.domain;

  const canvas = document.createElement("canvas");
  canvas.width = SIZE;
  canvas.height = SIZE;
  canvas.className = "input-space-canvas";
  const context = canvas.getContext("2d");

  const svg = document.createElementNS(SVG, "svg");
  svg.setAttribute("viewBox", `0 0 ${SIZE} ${SIZE}`);
  svg.setAttribute("class", "input-space-marks");
  svg.setAttribute("role", "img");

  const plot = document.createElement("div");
  plot.className = "input-space-plot";
  plot.append(canvas, svg);

  const xBar = axisBar("x", dataset.inputNames[0] ?? "x₁", x1lo, x1hi);
  const yBar = axisBar("y", dataset.inputNames[1] ?? "x₂", x2lo, x2hi);

  const wrapper = document.createElement("div");
  wrapper.className = "input-space";
  wrapper.append(yBar.element, plot, xBar.element);
  container.replaceChildren(wrapper);

  const toPixelX = (x: number) => ((x - x1lo) / (x1hi - x1lo)) * SIZE;
  const toPixelY = (y: number) => SIZE - ((y - x2lo) / (x2hi - x2lo)) * SIZE;
  const fromPixelX = (px: number) => x1lo + (px / SIZE) * (x1hi - x1lo);
  const fromPixelY = (py: number) => x2lo + ((SIZE - py) / SIZE) * (x2hi - x2lo);

  function pick(event: PointerEvent): void {
    const rect = svg.getBoundingClientRect();
    const px = ((event.clientX - rect.left) / rect.width) * SIZE;
    const py = ((event.clientY - rect.top) / rect.height) * SIZE;
    onProbe({
      kind: "free",
      x: [
        Math.min(x1hi, Math.max(x1lo, fromPixelX(px))),
        Math.min(x2hi, Math.max(x2lo, fromPixelY(py))),
      ],
    });
  }

  function onPointerDown(event: PointerEvent): void {
    svg.setPointerCapture(event.pointerId);
    pick(event);
  }
  function onPointerMove(event: PointerEvent): void {
    if (!svg.hasPointerCapture(event.pointerId)) return;
    pick(event);
  }
  svg.addEventListener("pointerdown", onPointerDown);
  svg.addEventListener("pointermove", onPointerMove);

  /** The output at every cell centre, on the shared scale. */
  function drawSurface(surface: Surface, scale: ColorScale): void {
    if (!context) return;
    const cell = SIZE / surface.resolution;
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

  function render(snapshot: Snapshot, frame: InputSpaceFrame): void {
    const { surface, scale, showSamples } = frame;
    drawSurface(surface, scale);
    xBar.paint(scale);
    yBar.paint(scale);

    const marks: SVGElement[] = [];
    if (showSamples) {
      dataset.samples.y.forEach((y, n) => {
        const mark = document.createElementNS(SVG, "circle");
        mark.setAttribute("cx", toPixelX(dataset.samples.x[n][0]).toFixed(2));
        mark.setAttribute("cy", toPixelY(dataset.samples.x[n][1]).toFixed(2));
        mark.setAttribute("r", "3.2");
        // The target on the same scale as the surface behind it, so a dot
        // that disappears into its background is one the model already fits.
        mark.setAttribute("fill", scale.css(y));
        mark.setAttribute("class", dataset.samples.isTest[n] ? "sample is-test" : "sample");
        marks.push(mark);
      });
    }

    const [px, py] = [toPixelX(snapshot.x[0]), toPixelY(snapshot.x[1])];
    // The local sensitivity arrow: direction and relative magnitude only.
    const sensitivity = snapshot.inputSensitivity;
    const norm = Math.hypot(sensitivity[0], sensitivity[1]) || 1;
    const arrow = document.createElementNS(SVG, "line");
    arrow.setAttribute("x1", px.toFixed(2));
    arrow.setAttribute("y1", py.toFixed(2));
    arrow.setAttribute("x2", (px + (sensitivity[0] / norm) * 34).toFixed(2));
    arrow.setAttribute("y2", (py - (sensitivity[1] / norm) * 34).toFixed(2));
    arrow.setAttribute("class", "sensitivity-arrow");
    marks.push(arrow);

    const probe = document.createElementNS(SVG, "circle");
    probe.setAttribute("cx", px.toFixed(2));
    probe.setAttribute("cy", py.toFixed(2));
    probe.setAttribute("r", "6");
    probe.setAttribute("class", "probe");
    marks.push(probe);

    svg.replaceChildren(...marks);
    svg.setAttribute(
      "aria-label",
      `Input space, ${dataset.targetName} on a colour scale of plus or minus ` +
        `${scale.limit}. Probe at ${snapshot.x[0].toFixed(2)}, ` +
        `${snapshot.x[1].toFixed(2)}.`,
    );
  }

  return {
    render,
    destroy() {
      svg.removeEventListener("pointerdown", onPointerDown);
      svg.removeEventListener("pointermove", onPointerMove);
      container.replaceChildren();
    },
  };
}

/**
 * One axis colour bar: the colour an input of that coordinate takes, which is
 * also the colour its input node takes. The bar spans the dataset's domain,
 * not the whole colour scale, so it reads as part of the axis.
 */
function axisBar(
  orientation: "x" | "y",
  name: string,
  lo: number,
  hi: number,
): { element: HTMLElement; paint: (scale: ColorScale) => void } {
  const element = document.createElement("div");
  element.className = `axis-bar axis-bar-${orientation}`;
  element.setAttribute("role", "img");
  const label = document.createElement("span");
  label.className = "axis-bar-label";
  label.textContent = name;
  element.append(label);
  return {
    element,
    paint(scale) {
      const direction = orientation === "x" ? "to right" : "to top";
      element.style.backgroundImage =
        `linear-gradient(${direction}, ${scale.stops(lo, hi).join(", ")})`;
      element.setAttribute(
        "aria-label",
        `${name} runs from ${lo} to ${hi}, coloured on the shared scale`,
      );
    },
  };
}
