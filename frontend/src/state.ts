/**
 * Explorer state: an immutable baseline, an editable working copy, and one
 * versioned evaluation snapshot that every view consumes.
 *
 * Two rules hold this together:
 *  - The snapshot is recomputed synchronously on every change. Animation
 *    reads it; animation never produces it.
 *  - An edit never clears the selection. Students edit the thing they have
 *    selected and then look at it, so losing the selection on edit would
 *    break the core interaction (a Stage 0 exit condition).
 *
 * The selection is a *unit*, not a single parameter, and there is no empty
 * selection: a neuron and everything feeding it is the smallest thing worth
 * looking at, and the explorer opens on the output unit.
 */

import {
  forward,
  hiddenActivation,
  inputSensitivity,
  pointGradient,
  pointLoss,
  probabilitySensitivity,
  sigmoid,
  splitObjective,
  type SplitObjective,
  type Trace,
} from "./model/evaluate.js";
import { randomParameters } from "./model/random.js";
import {
  cloneParameters,
  describeParameter,
  readParameter,
  sameParameter,
  writeParameter,
  sameUnit,
  type Activation,
  type Checkpoint,
  type LayerParameters,
  type ParameterRef,
  type UnitRef,
} from "./model/types.js";

/**
 * Where the probe opens, in each input, clamped into that input's own
 * domain. Not the centre: the map draws its axes through the origin, and a
 * probe sitting on them would hide them and zero out every input term of
 * the first layer's sums.
 */
export const OPENING_PROBE = 0.5;

export type Provenance =
  | { kind: "pretrained"; checkpointId: string }
  | { kind: "random"; seed: number }
  | { kind: "modified"; from: Provenance; edits: number };

export function describeProvenance(provenance: Provenance): string {
  switch (provenance.kind) {
    case "pretrained":
      return `Pre-trained (${provenance.checkpointId})`;
    case "random":
      return `Random (seed ${provenance.seed})`;
    case "modified": {
      const edits = provenance.edits === 1 ? "1 edit" : `${provenance.edits} edits`;
      return `Modified from ${describeProvenance(provenance.from)} — ${edits}`;
    }
  }
}

/** A probe is either a dataset sample (which has a target) or a free point. */
export type Probe =
  | { kind: "sample"; index: number }
  | { kind: "free"; x: number[] };

export interface Snapshot {
  /** Increments on every change; stale async work compares against it. */
  version: number;
  parameters: LayerParameters[];
  provenance: Provenance;
  /** The selected unit: its bias and every weight feeding it. Never null. */
  selection: UnitRef;
  probe: Probe;
  x: number[];
  trace: Trace;
  /** d(output)/dx. For a classifier this is the logit sensitivity. */
  inputSensitivity: number[];
  /** Present only for a classifier, and labeled as such in the UI. */
  probability: number | null;
  probabilitySensitivity: number[] | null;
  /** Null when the probe has no defined target; never invented. */
  target: number | null;
  pointLoss: number | null;
  pointGradient: LayerParameters[] | null;
  train: SplitObjective;
  test: SplitObjective;
}

/** What one edit did, retained so the UI can show actual deltas. */
export interface EditRecord {
  ref: ParameterRef;
  before: number;
  after: number;
  /** The snapshot from immediately before the edit, frozen for comparison. */
  previous: Snapshot;
}

export interface Explorer {
  readonly checkpoint: Checkpoint;
  snapshot(): Snapshot;
  lastEdit(): EditRecord | null;
  subscribe(listener: (snapshot: Snapshot) => void): () => void;

  selectUnit(unit: UnitRef): void;
  setProbe(probe: Probe): void;
  setParameter(ref: ParameterRef, value: number): void;

  loadPretrained(): void;
  randomize(seed?: number): void;
  restoreParameter(ref: ParameterRef): void;
  restoreBaseline(): void;

  canUndo(): boolean;
  canRedo(): boolean;
  undo(): void;
  redo(): void;
}

interface HistoryEntry {
  parameters: LayerParameters[];
  provenance: Provenance;
}

export function createExplorer(checkpoint: Checkpoint): Explorer {
  const activation: Activation = hiddenActivation(checkpoint.architecture);
  const { dataset } = checkpoint;

  let baseline: HistoryEntry = {
    parameters: cloneParameters(checkpoint.parameters),
    provenance: { kind: "pretrained", checkpointId: checkpoint.checkpointId },
  };
  let parameters = cloneParameters(baseline.parameters);
  let provenance: Provenance = baseline.provenance;
  // The output unit, so the panel opens on the parameters nearest the
  // prediction: the ones whose sweeps move it most.
  let selection: UnitRef = {
    layer: checkpoint.parameters.length - 1,
    out: 0,
  };
  // The input space opens with no samples drawn, so the probe starts off the
  // origin and off sample 0: away from the axes it draws, with both inputs
  // non-zero, so every term of the first neuron's sum is a real number. The
  // first thing to do with that panel is move this.
  let probe: Probe = {
    kind: "free",
    x: dataset.domain.map(([lo, hi]) => Math.min(hi, Math.max(lo, OPENING_PROBE))),
  };
  let version = 0;
  let edit: EditRecord | null = null;

  const past: HistoryEntry[] = [];
  const future: HistoryEntry[] = [];
  const listeners = new Set<(snapshot: Snapshot) => void>();
  let current = evaluate();

  function probePoint(): { x: number[]; target: number | null } {
    if (probe.kind === "sample") {
      return {
        x: dataset.samples.x[probe.index],
        target: dataset.samples.y[probe.index],
      };
    }
    // A free probe gets a target only where the generator defines one at
    // every point. A stochastic classification dataset does not, and an
    // arbitrary point is not silently given a label.
    const target = dataset.hasAnalyticTarget ? analyticTarget(probe.x) : null;
    return { x: probe.x, target };
  }

  function analyticTarget(x: number[]): number | null {
    if (dataset.id === "tilted-plane") return 0.6 * x[0] - 0.4 * x[1] + 0.2;
    return null;
  }

  function evaluate(): Snapshot {
    const { x, target } = probePoint();
    const trace = forward(parameters, activation, x);
    const sensitivity = inputSensitivity(parameters, activation, trace);
    const isClassifier = dataset.task === "classification";
    return {
      version,
      parameters,
      provenance,
      selection,
      probe,
      x,
      trace,
      inputSensitivity: sensitivity,
      probability: isClassifier ? sigmoid(trace.output) : null,
      probabilitySensitivity: isClassifier
        ? probabilitySensitivity(trace.output, sensitivity)
        : null,
      target,
      pointLoss:
        target === null ? null : pointLoss(trace.output, target, dataset.task),
      pointGradient:
        target === null
          ? null
          : pointGradient(parameters, activation, trace, target, dataset.task),
      train: splitObjective(parameters, activation, dataset.task, dataset.samples, "train"),
      test: splitObjective(parameters, activation, dataset.task, dataset.samples, "test"),
    };
  }

  function commit(): void {
    version += 1;
    current = evaluate();
    for (const listener of listeners) listener(current);
  }

  /** Replace the working parameters, recording the previous state. */
  function replace(next: LayerParameters[], nextProvenance: Provenance): void {
    past.push({ parameters, provenance });
    future.length = 0;
    parameters = next;
    provenance = nextProvenance;
    // Selection deliberately survives: see the module comment.
    commit();
  }

  function markModified(): Provenance {
    return provenance.kind === "modified"
      ? { ...provenance, edits: provenance.edits + 1 }
      : { kind: "modified", from: provenance, edits: 1 };
  }

  return {
    checkpoint,
    snapshot: () => current,
    lastEdit: () => edit,
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },

    selectUnit(unit) {
      if (sameUnit(unit, selection)) return;
      selection = unit;
      edit = null;
      commit();
    },

    setProbe(next) {
      probe = next;
      edit = null;
      commit();
    },

    setParameter(ref, value) {
      if (!Number.isFinite(value)) {
        throw new RangeError(
          `${describeParameter(ref)} must be finite, got ${value}`,
        );
      }
      const before = readParameter(parameters, ref);
      if (before === value) return;
      // Recorded BEFORE commit: `replace` notifies subscribers, and they
      // read `lastEdit()` to build the ripple and the predicted-vs-actual
      // comparison. Assigning after would hand them the previous edit.
      edit = { ref, before, after: value, previous: current };
      replace(writeParameter(parameters, ref, value), markModified());
    },

    loadPretrained() {
      baseline = {
        parameters: cloneParameters(checkpoint.parameters),
        provenance: { kind: "pretrained", checkpointId: checkpoint.checkpointId },
      };
      edit = null;
      replace(cloneParameters(baseline.parameters), baseline.provenance);
    },

    randomize(seed = Math.floor(Math.random() * 1e9)) {
      // The actual parameters are recorded, not just the seed.
      baseline = {
        parameters: randomParameters(
          checkpoint.architecture.layerSizes,
          activation,
          seed,
        ),
        provenance: { kind: "random", seed },
      };
      edit = null;
      replace(cloneParameters(baseline.parameters), baseline.provenance);
    },

    restoreParameter(ref) {
      const value = readParameter(baseline.parameters, ref);
      const before = readParameter(parameters, ref);
      if (before === value) return;
      // Recorded BEFORE commit: `replace` notifies subscribers, and they
      // read `lastEdit()` to build the ripple and the predicted-vs-actual
      // comparison. Assigning after would hand them the previous edit.
      edit = { ref, before, after: value, previous: current };
      replace(writeParameter(parameters, ref, value), markModified());
    },

    restoreBaseline() {
      edit = null;
      replace(cloneParameters(baseline.parameters), baseline.provenance);
    },

    canUndo: () => past.length > 0,
    canRedo: () => future.length > 0,

    undo() {
      const entry = past.pop();
      if (!entry) return;
      future.push({ parameters, provenance });
      parameters = entry.parameters;
      provenance = entry.provenance;
      edit = null;
      commit();
    },

    redo() {
      const entry = future.pop();
      if (!entry) return;
      past.push({ parameters, provenance });
      parameters = entry.parameters;
      provenance = entry.provenance;
      edit = null;
      commit();
    },
  };
}

export { sameParameter, sameUnit };
