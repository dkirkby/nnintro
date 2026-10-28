/**
 * The input-space view: the prediction surface, the samples, and the probe.
 *
 * Canvas for the dense heatmap, an SVG layer above it for the samples, the
 * probe and the sensitivity arrow — dense pixels on canvas, pickable marks
 * in the DOM.
 */

import { forward } from "../model/evaluate.js";
import type { Activation, DatasetSpec, LayerParameters } from "../model/types.js";
import type { Probe, Snapshot } from "../state.js";

const SIZE = 300;

export interface InputSpaceView {
  render(snapshot: Snapshot): void;
  destroy(): void;
}

export interface InputSpaceOptions {
  container: HTMLElement;
  dataset: DatasetSpec;
  activation: Activation;
  onProbe: (probe: Probe) => void;
}

export function createInputSpaceView(options: InputSpaceOptions): InputSpaceView {
  const { container, dataset, activation, onProbe } = options;
  const [[x1lo, x1hi], [x2lo, x2hi]] = dataset.domain;

  const canvas = document.createElement("canvas");
  canvas.width = SIZE;
  canvas.height = SIZE;
  canvas.className = "input-space-canvas";
  const context = canvas.getContext("2d");

  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${SIZE} ${SIZE}`);
  svg.setAttribute("class", "input-space-marks");
  svg.setAttribute("role", "img");

  const wrapper = document.createElement("div");
  wrapper.className = "input-space";
  wrapper.append(canvas, svg);
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

  function drawHeatmap(parameters: LayerParameters[]): void {
    if (!context) return;
    const resolution = 60;
    const cell = SIZE / resolution;
    const values: number[] = [];
    for (let row = 0; row < resolution; row += 1) {
      for (let column = 0; column < resolution; column += 1) {
        const x = [
          fromPixelX((column + 0.5) * cell),
          fromPixelY((row + 0.5) * cell),
        ];
        values.push(forward(parameters, activation, x).output);
      }
    }
    const lo = Math.min(...values, ...dataset.samples.y);
    const hi = Math.max(...values, ...dataset.samples.y);
    const span = hi - lo || 1;
    let index = 0;
    for (let row = 0; row < resolution; row += 1) {
      for (let column = 0; column < resolution; column += 1) {
        const t = (values[index] - lo) / span;
        index += 1;
        // Prediction uses a sequential ramp; targets use outlined marks, so
        // the two encodings stay distinguishable rather than competing.
        context.fillStyle = `oklch(${(32 + 52 * t).toFixed(1)}% ${(0.02 + 0.1 * t).toFixed(3)} 250)`;
        context.fillRect(column * cell, row * cell, cell + 1, cell + 1);
      }
    }
  }

  function render(snapshot: Snapshot): void {
    drawHeatmap(snapshot.parameters);

    const marks: SVGElement[] = [];
    const lo = Math.min(...dataset.samples.y);
    const hi = Math.max(...dataset.samples.y);
    dataset.samples.y.forEach((y, n) => {
      const mark = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      mark.setAttribute("cx", toPixelX(dataset.samples.x[n][0]).toFixed(2));
      mark.setAttribute("cy", toPixelY(dataset.samples.x[n][1]).toFixed(2));
      mark.setAttribute("r", "3");
      const t = (y - lo) / (hi - lo || 1);
      mark.setAttribute("fill", `oklch(${(40 + 50 * t).toFixed(1)}% 0.13 95)`);
      mark.setAttribute(
        "class",
        dataset.samples.isTest[n] ? "sample is-test" : "sample",
      );
      marks.push(mark);
    });

    const [px, py] = [toPixelX(snapshot.x[0]), toPixelY(snapshot.x[1])];
    // The local sensitivity arrow: direction and relative magnitude only.
    const sensitivity = snapshot.inputSensitivity;
    const norm = Math.hypot(sensitivity[0], sensitivity[1]) || 1;
    const arrow = document.createElementNS("http://www.w3.org/2000/svg", "line");
    arrow.setAttribute("x1", px.toFixed(2));
    arrow.setAttribute("y1", py.toFixed(2));
    arrow.setAttribute("x2", (px + (sensitivity[0] / norm) * 34).toFixed(2));
    arrow.setAttribute("y2", (py - (sensitivity[1] / norm) * 34).toFixed(2));
    arrow.setAttribute("class", "sensitivity-arrow");
    marks.push(arrow);

    const probe = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    probe.setAttribute("cx", px.toFixed(2));
    probe.setAttribute("cy", py.toFixed(2));
    probe.setAttribute("r", "6");
    probe.setAttribute("class", "probe");
    marks.push(probe);

    svg.replaceChildren(...marks);
    svg.setAttribute(
      "aria-label",
      `Input space. Probe at ${snapshot.x[0].toFixed(2)}, ${snapshot.x[1].toFixed(2)}.`,
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
