/**
 * Parameter tensors: the shapes the parameters actually have, laid out the
 * way the network runs — left to right, one group per layer of parameters.
 *
 * Each group is the bias vector drawn above its weight matrix. A column of
 * both is one destination unit, so the bias and the weights feeding the same
 * neuron form one vertical band: exactly what the selection in the network
 * panel means, which is why selecting a unit can highlight it here as a
 * single block. Rows of the weight matrix are the sources it reads from.
 *
 * The slots are light dots, not numbers. This panel answers "how many
 * parameters are there, and how are they arranged" — the values themselves
 * are everywhere else in the explorer.
 */

import type { LayerParameters } from "../model/types.js";
import type { Snapshot } from "../state.js";

const SVG = "http://www.w3.org/2000/svg";
/** One value slot, in user units, which are CSS pixels here: the diagram is
 *  drawn at its natural size rather than stretched to the panel. */
const CELL = 18;
const DOT = 2.8;
/** Between the bias strip and the weight block below it. */
const GAP = 8;
/** Between one layer's group and the next. */
const GROUP_GAP = 26;
/** The caption strip under every group. */
const LABEL = 16;

export interface TensorsView {
  render(snapshot: Snapshot): void;
  destroy(): void;
}

export function createTensorsView(options: { container: HTMLElement }): TensorsView {
  const svg = document.createElementNS(SVG, "svg");
  svg.setAttribute("class", "tensor-map");
  svg.setAttribute("role", "img");
  options.container.replaceChildren(svg);

  function render(snapshot: Snapshot): void {
    const layers = snapshot.parameters;
    const marks: SVGElement[] = [];
    const tallest = Math.max(...layers.map(rowsOf));
    const top = CELL + GAP;
    const height = top + tallest * CELL + LABEL;

    let x = 0;
    layers.forEach((layer, index) => {
      const columns = layer.b.length;
      const rows = rowsOf(layer);
      const width = columns * CELL;
      const selected = snapshot.selection.layer === index ? snapshot.selection.out : null;

      // The band first, so it sits behind the dots and the frames.
      if (selected !== null) {
        marks.push(band(x + selected * CELL, 0, CELL, CELL));
        marks.push(band(x + selected * CELL, top, CELL, rows * CELL));
      }

      marks.push(
        block(x, 0, width, CELL, 1, columns, `1×${columns} biases for ${name(index + 1, layers)}`),
      );
      marks.push(
        block(
          x,
          top,
          width,
          rows * CELL,
          rows,
          columns,
          `${rows}×${columns} weights connecting ${name(index, layers)} to ${name(index + 1, layers)}`,
        ),
      );

      const caption = document.createElementNS(SVG, "text");
      caption.setAttribute("x", (x + width / 2).toFixed(2));
      caption.setAttribute("y", String(height - 4));
      caption.setAttribute("text-anchor", "middle");
      caption.setAttribute("class", "tensor-caption");
      caption.textContent = short(index + 1, layers);
      marks.push(caption);

      x += width + GROUP_GAP;
    });

    const width = Math.max(x - GROUP_GAP, CELL);
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    // Natural size: a schematic of a dozen slots blown up to the panel's
    // width would read as a chart of something.
    svg.setAttribute("width", String(width));
    svg.setAttribute("height", String(height));
    svg.setAttribute(
      "aria-label",
      layers
        .map(
          (layer, index) =>
            `${rowsOf(layer)}×${layer.b.length} weights and ${layer.b.length} biases ` +
            `into ${name(index + 1, layers)}`,
        )
        .join("; "),
    );
    svg.replaceChildren(...marks);
  }

  return {
    render,
    destroy() {
      options.container.replaceChildren();
    },
  };
}

function rowsOf(layer: LayerParameters): number {
  return layer.W[0]?.length ?? 0;
}

/** A tensor's frame, its slots, and the tooltip that names it. */
function block(
  x: number,
  y: number,
  width: number,
  height: number,
  rows: number,
  columns: number,
  label: string,
): SVGElement {
  const group = document.createElementNS(SVG, "g");
  group.setAttribute("class", "tensor-block");
  group.dataset.label = label;

  const frame = document.createElementNS(SVG, "rect");
  frame.setAttribute("x", String(x));
  frame.setAttribute("y", String(y));
  frame.setAttribute("width", String(width));
  frame.setAttribute("height", String(height));
  frame.setAttribute("class", "tensor-frame");
  group.append(frame);

  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      const dot = document.createElementNS(SVG, "circle");
      dot.setAttribute("cx", String(x + (column + 0.5) * CELL));
      dot.setAttribute("cy", String(y + (row + 0.5) * CELL));
      dot.setAttribute("r", String(DOT));
      dot.setAttribute("class", "tensor-dot");
      group.append(dot);
    }
  }

  // An SVG <title> is the tooltip, and the accessible name of the group.
  const title = document.createElementNS(SVG, "title");
  title.textContent = label;
  group.append(title);
  return group;
}

/** The background behind the selected unit's own slots. */
function band(x: number, y: number, width: number, height: number): SVGElement {
  const rect = document.createElementNS(SVG, "rect");
  rect.setAttribute("x", String(x));
  rect.setAttribute("y", String(y));
  rect.setAttribute("width", String(width));
  rect.setAttribute("height", String(height));
  rect.setAttribute("class", "tensor-band");
  return rect;
}

/** An activation layer's name, counting the inputs as layer 0. */
function name(index: number, layers: LayerParameters[]): string {
  if (index === 0) return "the input layer";
  if (index === layers.length) return "the output layer";
  return `hidden layer ${index}`;
}

function short(index: number, layers: LayerParameters[]): string {
  if (index === layers.length) return "output";
  return `hidden ${index}`;
}
