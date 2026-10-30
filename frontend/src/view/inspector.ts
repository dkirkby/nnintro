/**
 * The inspector: what the selected neuron computes, and how the local slope
 * compares with the actual change.
 *
 * The neuron panel carries a plot of the activation function itself with the
 * unit's own (z, a) marked on it, so the arithmetic above it (z = Σ wᵢ aᵢ + b)
 * and the nonlinearity after it are one picture. Editing any of the unit's
 * parameters moves the dot along that curve.
 *
 * Every number appears as a number. Colour and SVG picking are a second
 * channel, never the only one (PLAN.md section 4).
 */

import { activationDerivative, applyActivation } from "../model/evaluate.js";
import type { Activation, Task, UnitRef } from "../model/types.js";
import type { EditRecord, Snapshot } from "../state.js";
import { niceLimit } from "./colormap.js";
import { format } from "./network.js";

const SVG = "http://www.w3.org/2000/svg";
/** Square: both axes carry one scale, so a 45° line really is the identity. */
const PHI = { size: 150, pad: 12 };
/** Odd, so z = 0 is sampled exactly and ReLU's kink is a corner. */
const PHI_SAMPLES = 81;

export interface InspectorOptions {
  container: HTMLElement;
  activation: Activation;
  task: Task;
  targetName: string;
  inputNames: string[];
}

export interface Inspector {
  render(snapshot: Snapshot, edit: EditRecord | null): void;
  destroy(): void;
}

function row(label: string, value: string, hint?: string): HTMLElement {
  const element = document.createElement("div");
  element.className = "row";
  element.innerHTML =
    `<span class="row-label">${label}</span><span class="row-value">${value}</span>`;
  if (hint) {
    const note = document.createElement("span");
    note.className = "row-hint";
    note.textContent = hint;
    element.append(note);
  }
  return element;
}

/**
 * The activation function, with this unit's own (z, a) on it.
 *
 * Both axes span the same range on the same scale, so the identity runs at
 * 45° and so does ReLU's on-arm: the shape in the frame is the function's,
 * not the frame's. That range is the smallest round number that holds this
 * unit's z with a little headroom, so the dot can never sit on the frame or
 * outside it, and the axis only moves a ladder stop when z really outgrows
 * it rather than rescaling under every drag.
 *
 * z and a come straight from the trace, so an edit to the bias — or to any
 * weight feeding this unit — slides the dot along the curve on the next
 * render. The curve itself is the activation function and does not move.
 */
function activationPlot(name: Activation, z: number, a: number): SVGElement {
  const limit = niceLimit([z * 1.05, a * 1.05, 1]);
  const span = PHI.size - 2 * PHI.pad;
  const toX = (value: number) => PHI.pad + ((value + limit) / (2 * limit)) * span;
  const toY = (value: number) => PHI.size - PHI.pad - ((value + limit) / (2 * limit)) * span;

  const marks: SVGElement[] = [];
  marks.push(line(PHI.pad, toY(0), PHI.size - PHI.pad, toY(0), "phi-axis"));
  marks.push(line(toX(0), PHI.pad, toX(0), PHI.size - PHI.pad, "phi-axis"));

  const points: string[] = [];
  for (let i = 0; i < PHI_SAMPLES; i += 1) {
    const value = -limit + (2 * limit * i) / (PHI_SAMPLES - 1);
    points.push(`${toX(value).toFixed(2)},${toY(applyActivation(name, value)).toFixed(2)}`);
  }
  const curve = document.createElementNS(SVG, "polyline");
  curve.setAttribute("points", points.join(" "));
  curve.setAttribute("class", "phi-curve");
  marks.push(curve);

  // Down to the z axis and across to the a axis, so the dot is readable as a
  // pair of values and not only as a position.
  marks.push(line(toX(z), toY(a), toX(z), toY(0), "phi-guide"));
  marks.push(line(toX(z), toY(a), toX(0), toY(a), "phi-guide"));

  const dot = document.createElementNS(SVG, "circle");
  dot.setAttribute("cx", toX(z).toFixed(2));
  dot.setAttribute("cy", toY(a).toFixed(2));
  dot.setAttribute("r", "3.5");
  dot.setAttribute("class", "phi-dot");
  marks.push(dot);

  marks.push(tick("z", PHI.size - PHI.pad, toY(0) - 4, "end"));
  marks.push(tick("a", toX(0) + 4, PHI.pad - 3, "start"));
  // One pair of numbers for both axes, since they share a scale.
  marks.push(tick(`−${limit}`, PHI.pad, PHI.size - 2, "start"));
  marks.push(tick(`+${limit}`, PHI.size - PHI.pad, PHI.size - 2, "end"));

  const svg = document.createElementNS(SVG, "svg");
  svg.setAttribute("viewBox", `0 0 ${PHI.size} ${PHI.size}`);
  svg.setAttribute("class", "phi-plot");
  svg.setAttribute("role", "img");
  svg.setAttribute(
    "aria-label",
    `${name} from −${limit} to ${limit}; this unit sits at z = ${format(z)}, ` +
      `a = ${format(a)}.`,
  );
  svg.dataset.activation = name;
  svg.replaceChildren(...marks);
  return svg;
}

export function createInspector(options: InspectorOptions): Inspector {
  const { container, activation } = options;

  function neuronSection(snapshot: Snapshot, selected: UnitRef): HTMLElement {
    const section = document.createElement("section");
    section.className = "panel";
    const layer = selected.layer;
    const unit = selected.out;
    const z = snapshot.trace.pre[layer + 1][unit];
    const a = snapshot.trace.act[layer + 1][unit];
    const inputs = snapshot.trace.act[layer];
    const weights = snapshot.parameters[layer].W[unit];
    const isOutput = layer === snapshot.parameters.length - 1;
    // The output layer applies no nonlinearity, which is the identity: the
    // plot says so by being a straight 45° line through the origin.
    const phi: Activation = isOutput ? "identity" : activation;

    const heading = document.createElement("h3");
    heading.textContent = `Neuron: layer ${layer + 1}, unit ${unit}`;
    section.append(heading);

    // The arithmetic on the left, the activation plot on the right when the
    // panel is wide enough for both; it wraps underneath when it is not.
    const layout = document.createElement("div");
    layout.className = "neuron-body";
    const math = document.createElement("div");
    math.className = "neuron-math";

    const table = document.createElement("table");
    table.className = "contributions";
    table.innerHTML =
      "<thead><tr><th>term</th><th>wᵢ</th><th>aᵢ</th><th>wᵢ·aᵢ</th></tr></thead>";
    const body = document.createElement("tbody");
    // One row per input only: the bias is not a term of the sum over i, and
    // the line under the table is where it enters.
    weights.forEach((w, i) => {
      const tr = document.createElement("tr");
      tr.innerHTML =
        `<td>i=${i}</td><td>${format(w)}</td>` +
        `<td>${format(inputs[i])}</td><td>${format(w * inputs[i])}</td>`;
      body.append(tr);
    });
    table.append(body);
    math.append(table);

    // What the table adds up to, and what the nonlinearity then does with
    // it: the two steps the plot below draws, written out.
    math.append(
      formula("Preactivation z = Σ wᵢ aᵢ + bias = ", z),
      formula(`Activation a = ${activationName(phi)}(z) = `, a),
    );
    if (isOutput) {
      math.append(note("the output layer applies no nonlinearity"));
    } else {
      // The slope stays because this is the only place the ReLU convention
      // is disclosed: PLAN.md section 4 requires saying that no derivative
      // exists at the kink, and a dead unit is worth naming where it shows.
      const slope = activationDerivative(activation, z);
      math.append(formula("Slope φ′(z) = ", slope));
      if (activation === "relu" && z === 0) {
        math.append(
          note("exactly at the ReLU kink: no derivative exists here, and 0 is a convention"),
        );
      } else if (activation === "relu" && z < 0) {
        math.append(note("this unit is off, so it blocks changes from upstream"));
      }
    }

    layout.append(math, activationPlot(phi, z, a));
    section.append(layout);
    return section;
  }

  function predictionSection(snapshot: Snapshot, edit: EditRecord | null): HTMLElement {
    const section = document.createElement("section");
    section.className = "panel";
    const heading = document.createElement("h3");
    heading.textContent = "Prediction and loss";
    section.append(heading);

    // One grid for the lot, each column as wide as its own widest entry:
    // the values line up without being pushed to the far side of the panel.
    const readouts = document.createElement("div");
    readouts.className = "readouts";
    const isClassifier = options.task === "classification";
    readouts.append(
      // For a classifier the network's scalar output is a logit, never a
      // probability: sigmoid is applied for display and the logit stays
      // visible beside it, because the logit is what the parameters
      // actually produce and what every other panel plots (PLAN.md
      // section 3).
      row(
        isClassifier ? "logit z" : "prediction ŷ",
        format(snapshot.trace.output),
      ),
      ...(snapshot.probability === null
        ? []
        : [
            row(
              "probability p",
              format(snapshot.probability),
              "σ(z), the class-1 probability; z = 0 is p = 0.5",
            ),
          ]),
      row(
        isClassifier ? "class y" : "target y",
        snapshot.target === null ? "—" : format(snapshot.target),
        snapshot.target === null
          ? "this point has no defined target, so it has no loss"
          : undefined,
      ),
      row("point loss ℓ", format(snapshot.pointLoss)),
      // A classifier's sensitivity defaults to the probability's, with the
      // logit's kept beside it and labeled: dp/dx is the quantity a reader
      // can judge, and dz/dx is the one the chain rule produced (PLAN.md
      // section 3).
      ...(snapshot.probabilitySensitivity === null
        ? []
        : [
            row(
              "∂p/∂x",
              snapshot.probabilitySensitivity.map(format).join(",  "),
              `with respect to ${options.inputNames.join(" and ")}`,
            ),
          ]),
      row(
        isClassifier ? "∂z/∂x" : "∂ŷ/∂x",
        snapshot.inputSensitivity.map(format).join(",  "),
        isClassifier
          ? "the logit's own sensitivity, before σ"
          : `with respect to ${options.inputNames.join(" and ")}`,
      ),
      row("mean loss L (train)", format(snapshot.train.meanLoss)),
      row("mean loss L (held out)", format(snapshot.test.meanLoss)),
    );
    section.append(readouts);

    if (edit) {
      // The comparison the whole explorer is built around, evaluated
      // against the frozen before-edit state.
      const delta = edit.after - edit.before;
      const previous = edit.previous;
      const slope =
        previous.pointGradient === null
          ? null
          : edit.ref.kind === "weight"
            ? previous.pointGradient[edit.ref.layer].W[edit.ref.out][edit.ref.in]
            : previous.pointGradient[edit.ref.layer].b[edit.ref.out];
      const predicted = slope === null ? null : slope * delta;
      const actual =
        snapshot.pointLoss === null || previous.pointLoss === null
          ? null
          : snapshot.pointLoss - previous.pointLoss;

      const comparison = document.createElement("div");
      comparison.className = "comparison readouts";
      comparison.append(
        row("Δθ", format(delta)),
        row("predicted Δℓ ≈ (∂ℓ/∂θ)Δθ", format(predicted)),
        row("actual Δℓ", format(actual)),
      );
      if (predicted !== null && actual !== null) {
        comparison.append(row("difference", format(actual - predicted),
          "first order is a local approximation; it degrades as Δθ grows"));
      }
      section.append(comparison);
    }
    return section;
  }

  return {
    render(snapshot, edit) {
      // Something is always selected, so there is no empty branch here.
      container.replaceChildren(
        predictionSection(snapshot, edit),
        neuronSection(snapshot, snapshot.selection),
      );
    },

    destroy() {
      container.replaceChildren();
    },
  };
}

/** One line of the neuron's arithmetic: the expression, then its value. */
function formula(text: string, value: number): HTMLElement {
  const element = document.createElement("p");
  element.className = "formula";
  element.append(text);
  const number = document.createElement("span");
  number.className = "formula-value";
  number.textContent = format(value);
  element.append(number);
  return element;
}

/** A caveat under a line of arithmetic, in the hints' own quieter voice. */
function note(text: string): HTMLElement {
  const element = document.createElement("p");
  element.className = "formula-note";
  element.textContent = text;
  return element;
}

/** The activation's name as it is written in the mathematics. */
function activationName(name: Activation): string {
  return name === "relu" ? "ReLU" : name;
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

function tick(text: string, x: number, y: number, anchor: string): SVGElement {
  const node = document.createElementNS(SVG, "text");
  node.setAttribute("x", x.toFixed(2));
  node.setAttribute("y", y.toFixed(2));
  node.setAttribute("text-anchor", anchor);
  node.setAttribute("class", "slice-tick");
  node.textContent = text;
  return node;
}
