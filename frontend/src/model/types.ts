/**
 * The asset contract, mirrored from `src/nnintro/export.py`.
 *
 * Parameter orientation is destination-first throughout: `W[out][in]`,
 * matching `torch.nn.Linear`. The renderer indexes edges source-first, and
 * `src/view/network.ts` is the only place that translates.
 */

export type Task = "regression" | "classification";
export type Activation = "relu" | "tanh" | "identity";

/** Per-layer parameters, `W` with shape [out][in] and `b` with shape [out]. */
export interface LayerParameters {
  W: number[][];
  b: number[];
}

/** A single scalar parameter: what an edit targets and an inspector selects. */
export type ParameterRef =
  | { kind: "weight"; layer: number; out: number; in: number }
  | { kind: "bias"; layer: number; out: number };

export interface DatasetSamples {
  x: number[][];
  y: number[];
  isTest: boolean[];
}

export interface DatasetSpec {
  id: string;
  /** The short name a control wears; `description` is the formula. */
  label: string;
  task: Task;
  description: string;
  generatorVersion: number;
  domain: [number, number][];
  inputNames: string[];
  targetName: string;
  hasAnalyticTarget: boolean;
  preprocessing: null;
  samples: DatasetSamples;
}

export interface ArchitectureSpec {
  layerSizes: number[];
  hiddenActivation: Activation | null;
  outputTransform: "identity" | "sigmoid";
  loss: "squared-error" | "bce-with-logits";
  weightLayout: string;
  reluDerivativeAtZero: number | null;
}

export interface GoldenProbe {
  id: string;
  label: string;
  x: number[];
  pre: number[][];
  act: number[][];
  output: number;
  probability?: number;
  target: number | null;
  pointLoss: number | null;
  pointGradient: LayerParameters[] | null;
  inputSensitivity: number[];
}

export interface SplitGradient {
  count: number;
  meanLoss: number;
  gradient: LayerParameters[];
}

export interface Checkpoint {
  schemaVersion: string;
  checkpointId: string;
  contentHash: string;
  dataset: DatasetSpec;
  architecture: ArchitectureSpec;
  parameters: LayerParameters[];
  training: Record<string, unknown>;
  metrics: { trainLoss: number; testLoss: number };
  selection: unknown;
  splitGradients: { train: SplitGradient; test: SplitGradient };
  goldenProbes: GoldenProbe[];
}

export interface ManifestEntry {
  checkpointId: string;
  dataset: string;
  task: Task;
  layerSizes: number[];
  hiddenActivation: Activation | null;
  url: string;
  contentHash: string;
  metrics: { trainLoss: number; testLoss: number };
}

/** One dataset's own properties, shared by all of its checkpoints. */
export interface ManifestDataset {
  id: string;
  label: string;
  task: Task;
  description: string;
  generatorVersion: number;
  hasAnalyticTarget: boolean;
  sampleCount: number;
}

export interface Manifest {
  schemaVersion: string;
  datasets: ManifestDataset[];
  checkpoints: ManifestEntry[];
}

export type Split = "train" | "test";

/** Deep copy of a parameter set, for the baseline/working-copy split. */
export function cloneParameters(parameters: LayerParameters[]): LayerParameters[] {
  return parameters.map((layer) => ({
    W: layer.W.map((row) => row.slice()),
    b: layer.b.slice(),
  }));
}

export function readParameter(
  parameters: LayerParameters[],
  ref: ParameterRef,
): number {
  return ref.kind === "weight"
    ? parameters[ref.layer].W[ref.out][ref.in]
    : parameters[ref.layer].b[ref.out];
}

/** Returns a new parameter set; never mutates the one passed in. */
export function writeParameter(
  parameters: LayerParameters[],
  ref: ParameterRef,
  value: number,
): LayerParameters[] {
  const next = cloneParameters(parameters);
  if (ref.kind === "weight") next[ref.layer].W[ref.out][ref.in] = value;
  else next[ref.layer].b[ref.out] = value;
  return next;
}

export function sameParameter(a: ParameterRef, b: ParameterRef): boolean {
  if (a.kind !== b.kind || a.layer !== b.layer || a.out !== b.out) return false;
  return a.kind === "weight" && b.kind === "weight" ? a.in === b.in : true;
}

export function describeParameter(ref: ParameterRef): string {
  return ref.kind === "weight"
    ? `W[${ref.layer}][${ref.out}][${ref.in}]`
    : `b[${ref.layer}][${ref.out}]`;
}

/**
 * A destination unit: one neuron's own parameters, which is what the network
 * panel selects. Selecting a unit selects its bias `b[layer][out]` *and*
 * every weight `W[layer][out][·]` feeding it — the whole of `z = Σ wᵢaᵢ + b`
 * for that neuron, and nothing else.
 */
export interface UnitRef {
  layer: number;
  out: number;
}

export function sameUnit(a: UnitRef, b: UnitRef): boolean {
  return a.layer === b.layer && a.out === b.out;
}

/** The unit a parameter belongs to: the neuron it helps compute. */
export function unitOf(ref: ParameterRef): UnitRef {
  return { layer: ref.layer, out: ref.out };
}

export function inUnit(ref: ParameterRef, unit: UnitRef): boolean {
  return sameUnit(unitOf(ref), unit);
}

/**
 * A unit's parameters in the order the UI lists them: the bias first, then
 * each incoming weight by source index. Source index 0 is the top node of
 * the previous layer in the network diagram (nnet-svg lays a layer out in
 * index order, downward), so this is top-to-bottom as drawn.
 */
export function unitParameters(
  parameters: LayerParameters[],
  unit: UnitRef,
): ParameterRef[] {
  const refs: ParameterRef[] = [{ kind: "bias", layer: unit.layer, out: unit.out }];
  const row = parameters[unit.layer]?.W[unit.out] ?? [];
  row.forEach((_, input) => {
    refs.push({ kind: "weight", layer: unit.layer, out: unit.out, in: input });
  });
  return refs;
}

export function describeUnit(unit: UnitRef): string {
  return `layer ${unit.layer + 1} unit ${unit.out}`;
}

const SUBSCRIPTS = "₀₁₂₃₄₅₆₇₈₉";

/**
 * The short name a parameter wears next to its curve and its slider: `b` for
 * the bias, `wᵢ` for the weight from source unit `i`. Unambiguous only
 * within one unit, which is the only place it is used.
 */
export function parameterSymbol(ref: ParameterRef): string {
  if (ref.kind === "bias") return "b";
  const digits = String(ref.in)
    .split("")
    .map((digit) => SUBSCRIPTS[Number(digit)])
    .join("");
  return `w${digits}`;
}
