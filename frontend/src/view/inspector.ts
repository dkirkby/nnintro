/**
 * The inspector: what the selected neuron computes, what the selected
 * parameter does, and how the local slope compares with the actual change.
 *
 * Every number appears as a number. Colour and SVG picking are a second
 * channel, never the only one (PLAN.md section 4).
 */

import { activationDerivative } from "../model/evaluate.js";
import { describeRipple, type Ripple } from "../model/ripple.js";
import {
  describeParameter,
  readParameter,
  type Activation,
  type ParameterRef,
} from "../model/types.js";
import type { EditRecord, Snapshot } from "../state.js";
import { format } from "./network.js";

export interface InspectorOptions {
  container: HTMLElement;
  activation: Activation;
  targetName: string;
  inputNames: string[];
  onEdit: (ref: ParameterRef, value: number) => void;
  onRestore: (ref: ParameterRef) => void;
}

export interface Inspector {
  render(snapshot: Snapshot, edit: EditRecord | null, ripple: Ripple | null): void;
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

export function createInspector(options: InspectorOptions): Inspector {
  const { container, activation, onEdit, onRestore } = options;

  function neuronSection(snapshot: Snapshot, ref: ParameterRef): HTMLElement {
    const section = document.createElement("section");
    section.className = "panel";
    const layer = ref.layer;
    const unit = ref.out;
    const z = snapshot.trace.pre[layer + 1][unit];
    const a = snapshot.trace.act[layer + 1][unit];
    const inputs = snapshot.trace.act[layer];
    const weights = snapshot.parameters[layer].W[unit];
    const bias = snapshot.parameters[layer].b[unit];
    const isOutput = layer === snapshot.parameters.length - 1;

    const heading = document.createElement("h3");
    heading.textContent = `Neuron: layer ${layer + 1}, unit ${unit}`;
    section.append(heading);

    const table = document.createElement("table");
    table.className = "contributions";
    table.innerHTML =
      "<caption>z = Σ wᵢ aᵢ + b</caption>" +
      "<thead><tr><th>term</th><th>w</th><th>a</th><th>w·a</th></tr></thead>";
    const body = document.createElement("tbody");
    weights.forEach((w, i) => {
      const tr = document.createElement("tr");
      tr.innerHTML =
        `<td>input ${i}</td><td>${format(w)}</td>` +
        `<td>${format(inputs[i])}</td><td>${format(w * inputs[i])}</td>`;
      body.append(tr);
    });
    const biasRow = document.createElement("tr");
    biasRow.innerHTML = `<td>bias</td><td>—</td><td>—</td><td>${format(bias)}</td>`;
    body.append(biasRow);
    table.append(body);
    section.append(table);

    section.append(row("preactivation z", format(z)));
    if (isOutput) {
      section.append(
        row("activation a", format(a), "the output layer applies no nonlinearity"),
      );
    } else {
      const slope = activationDerivative(activation, z);
      section.append(row(`a = ${activation}(z)`, format(a)));
      const atKink = activation === "relu" && z === 0;
      section.append(
        row(
          "φ′(z)",
          format(slope),
          atKink
            ? "exactly at the ReLU kink: no derivative exists here, and 0 is a convention"
            : activation === "relu" && z < 0
              ? "this unit is off, so it blocks changes from upstream"
              : undefined,
        ),
      );
    }
    return section;
  }

  function parameterSection(snapshot: Snapshot, ref: ParameterRef): HTMLElement {
    const section = document.createElement("section");
    section.className = "panel";
    const value = readParameter(snapshot.parameters, ref);

    const heading = document.createElement("h3");
    heading.textContent = `Parameter: ${describeParameter(ref)}`;
    section.append(heading);

    const controls = document.createElement("div");
    controls.className = "editor";

    const number = document.createElement("input");
    number.type = "number";
    number.step = "0.01";
    number.value = String(value);
    number.className = "value-input";
    number.setAttribute("aria-label", `value of ${describeParameter(ref)}`);

    // An adaptive range that still permits typed values beyond it.
    const extent = Math.max(2, Math.abs(value) * 2);
    const slider = document.createElement("input");
    slider.type = "range";
    slider.min = String(-extent);
    slider.max = String(extent);
    slider.step = String(extent / 500);
    slider.value = String(value);
    slider.setAttribute("aria-label", `slider for ${describeParameter(ref)}`);

    const commit = (raw: string) => {
      const next = Number(raw);
      if (!Number.isFinite(next)) {
        number.setCustomValidity("must be a finite number");
        number.reportValidity();
        return;
      }
      number.setCustomValidity("");
      onEdit(ref, next);
    };
    slider.addEventListener("input", () => commit(slider.value));
    number.addEventListener("change", () => commit(number.value));

    const restore = document.createElement("button");
    restore.type = "button";
    restore.textContent = "Restore this parameter";
    restore.addEventListener("click", () => onRestore(ref));

    controls.append(slider, number, restore);
    section.append(controls);

    const gradient = snapshot.pointGradient;
    if (gradient) {
      const slope =
        ref.kind === "weight"
          ? gradient[ref.layer].W[ref.out][ref.in]
          : gradient[ref.layer].b[ref.out];
      section.append(row("∂ℓ/∂θ (this example)", format(slope)));
    }
    const trainSlope =
      ref.kind === "weight"
        ? snapshot.train.gradient[ref.layer].W[ref.out][ref.in]
        : snapshot.train.gradient[ref.layer].b[ref.out];
    section.append(
      row("∂L/∂θ (mean loss, train split)", format(trainSlope),
        "the slope of the mean loss itself, not an average of per-example magnitudes"),
    );
    return section;
  }

  function predictionSection(snapshot: Snapshot, edit: EditRecord | null): HTMLElement {
    const section = document.createElement("section");
    section.className = "panel";
    const heading = document.createElement("h3");
    heading.textContent = "Prediction and loss";
    section.append(heading);

    section.append(row("prediction ŷ", format(snapshot.trace.output)));
    section.append(
      row(
        "target y",
        snapshot.target === null ? "—" : format(snapshot.target),
        snapshot.target === null
          ? "this point has no defined target, so it has no loss"
          : undefined,
      ),
    );
    section.append(row("point loss ℓ", format(snapshot.pointLoss)));
    section.append(
      row(
        "∂ŷ/∂x",
        snapshot.inputSensitivity.map(format).join(",  "),
        `with respect to ${options.inputNames.join(" and ")}`,
      ),
    );
    section.append(row("mean loss L (train)", format(snapshot.train.meanLoss)));
    section.append(row("mean loss L (held out)", format(snapshot.test.meanLoss)));

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
      comparison.className = "comparison";
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
    render(snapshot, edit, ripple) {
      const children: HTMLElement[] = [predictionSection(snapshot, edit)];
      if (snapshot.selection) {
        children.push(parameterSection(snapshot, snapshot.selection));
        children.push(neuronSection(snapshot, snapshot.selection));
      } else {
        const hint = document.createElement("section");
        hint.className = "panel";
        hint.innerHTML =
          "<h3>Nothing selected</h3><p>Select an edge to edit its weight, " +
          "or a node to edit its bias.</p>";
        children.push(hint);
      }
      if (ripple) {
        const section = document.createElement("section");
        section.className = "panel";
        section.innerHTML = "<h3>Downstream effect</h3>";
        section.append(row("reach", describeRipple(ripple)));
        const list = document.createElement("table");
        list.className = "contributions";
        list.innerHTML =
          "<thead><tr><th>node</th><th>before</th><th>after</th><th>Δ</th></tr></thead>";
        const body = document.createElement("tbody");
        for (const stage of ripple.stages) {
          for (const node of stage.nodes) {
            const tr = document.createElement("tr");
            tr.className = node.changed ? "" : "is-blocked";
            tr.innerHTML =
              `<td>L${node.layer}·${node.index}</td><td>${format(node.before)}</td>` +
              `<td>${format(node.after)}</td><td>${node.changed ? format(node.delta) : "unchanged"}</td>`;
            body.append(tr);
          }
        }
        list.append(body);
        section.append(list);
        children.push(section);
      }
      container.replaceChildren(...children);
    },
  };
}
