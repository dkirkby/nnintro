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

export interface Manifest {
  schemaVersion: string;
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
