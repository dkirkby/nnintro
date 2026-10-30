/**
 * Loading and validating published assets.
 *
 * The explorer enumerates the manifest rather than assuming the catalog is
 * covered: a control must never be able to reach a configuration that has
 * no checkpoint (PLAN.md section 3).
 */

import type {
  Activation,
  Checkpoint,
  Manifest,
  ManifestDataset,
  ManifestEntry,
} from "./types.js";

const BASE = "data/";

export async function fetchManifest(base = BASE): Promise<Manifest> {
  const response = await fetch(`${base}manifest.json`);
  if (!response.ok) throw new Error(`manifest: HTTP ${response.status}`);
  return validateManifest(await response.json());
}

export async function fetchCheckpoint(url: string, base = BASE): Promise<Checkpoint> {
  const response = await fetch(`${base}${url}`);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return validateCheckpoint(await response.json());
}

export function validateManifest(value: unknown): Manifest {
  const manifest = value as Manifest;
  if (!manifest?.checkpoints?.length) throw new Error("manifest lists no checkpoints");
  if (!manifest.datasets?.length) throw new Error("manifest lists no datasets");
  return manifest;
}

/**
 * Structural validation, run before anything consumes a checkpoint: shapes
 * must agree with the declared architecture and every number must be
 * finite. A malformed asset should fail here with a clear message, not as
 * a NaN surfacing in the middle of a figure.
 */
export function validateCheckpoint(value: unknown): Checkpoint {
  const checkpoint = value as Checkpoint;
  const sizes = checkpoint?.architecture?.layerSizes;
  if (!Array.isArray(sizes) || sizes.length < 2) {
    throw new Error("checkpoint has no usable layerSizes");
  }
  if (checkpoint.parameters.length !== sizes.length - 1) {
    throw new Error(
      `expected ${sizes.length - 1} parameter layers, got ${checkpoint.parameters.length}`,
    );
  }
  checkpoint.parameters.forEach((layer, index) => {
    const expectedOut = sizes[index + 1];
    const expectedIn = sizes[index];
    if (layer.W.length !== expectedOut || layer.b.length !== expectedOut) {
      throw new Error(
        `layer ${index}: expected ${expectedOut} rows (destination-first), ` +
          `got W=${layer.W.length}, b=${layer.b.length}`,
      );
    }
    layer.W.forEach((row, out) => {
      if (row.length !== expectedIn) {
        throw new Error(
          `layer ${index} row ${out}: expected ${expectedIn} columns, got ${row.length}; ` +
            "a transposed W would look exactly like this",
        );
      }
      row.forEach((weight, inIndex) => {
        if (!Number.isFinite(weight)) {
          throw new Error(`layer ${index} W[${out}][${inIndex}] is not finite`);
        }
      });
    });
    layer.b.forEach((bias, out) => {
      if (!Number.isFinite(bias)) throw new Error(`layer ${index} b[${out}] is not finite`);
    });
  });
  const samples = checkpoint.dataset?.samples;
  if (!samples || samples.x.length !== samples.y.length) {
    throw new Error("dataset samples are missing or ragged");
  }
  if (samples.isTest.length !== samples.y.length) {
    throw new Error("split membership does not cover every sample");
  }
  return checkpoint;
}

/**
 * A configuration the controls can ask for: a dataset, the hidden widths,
 * and the hidden activation. `widths` is the hidden layers only, so `[]` is
 * "no hidden layer" and `[4, 2]` is 2 → 4 → 2 → 1; its length is the depth.
 * `activation` is carried even at depth 0, where no hidden layer reads it,
 * so that toggling depth off and back on returns the previous choice.
 */
export interface Selection {
  dataset: string;
  widths: number[];
  activation: Activation;
}

export interface CatalogIndex {
  /** The datasets, in the order the pipeline declares them. */
  datasets: ManifestDataset[];
  /** Hidden depths with at least one checkpoint, ascending. */
  depths: number[];
  /** Hidden widths with at least one checkpoint, ascending. */
  widths: number[];
  /** Hidden activations with at least one checkpoint, in canonical order. */
  activations: Activation[];
  dataset(id: string): ManifestDataset | null;
  /** The checkpoint for a selection, or null if the catalog has no such entry. */
  find(selection: Selection): ManifestEntry | null;
  has(selection: Selection): boolean;
}

/** The order activations are offered in, simplest last. */
const ACTIVATION_ORDER: Activation[] = ["relu", "tanh", "identity"];

/**
 * What a selection is keyed by. A network with no hidden layer has no
 * hidden activation — the manifest stores null — so every depth-0 entry
 * keys as "none" and the remembered activation cannot miss it.
 */
function keyOf(dataset: string, widths: number[], activation: Activation | null): string {
  const hidden = widths.length > 0 ? activation : null;
  return `${dataset}|${widths.join("-")}|${hidden ?? "none"}`;
}

/**
 * The selectable dimensions, enumerated from the manifest rather than
 * assumed: the controls offer exactly the stops the published catalog
 * covers, and a gap disables a stop instead of producing a failed fetch
 * (PLAN.md section 3).
 */
export function createCatalogIndex(manifest: Manifest): CatalogIndex {
  const byKey = new Map<string, ManifestEntry>();
  for (const entry of manifest.checkpoints) {
    const widths = entry.layerSizes.slice(1, -1);
    byKey.set(keyOf(entry.dataset, widths, entry.hiddenActivation), entry);
  }

  const depths = new Set<number>();
  const widths = new Set<number>();
  const activations = new Set<Activation>();
  for (const entry of manifest.checkpoints) {
    const hidden = entry.layerSizes.slice(1, -1);
    depths.add(hidden.length);
    for (const width of hidden) widths.add(width);
    if (entry.hiddenActivation) activations.add(entry.hiddenActivation);
  }

  const datasets = manifest.datasets ?? [];
  const ascending = (a: number, b: number) => a - b;
  return {
    datasets,
    depths: [...depths].sort(ascending),
    widths: [...widths].sort(ascending),
    activations: ACTIVATION_ORDER.filter((name) => activations.has(name)),
    dataset: (id) => datasets.find((entry) => entry.id === id) ?? null,
    find: (selection) =>
      byKey.get(keyOf(selection.dataset, selection.widths, selection.activation)) ?? null,
    has(selection) {
      return this.find(selection) !== null;
    },
  };
}

/** The checkpoint a selection names, loaded once and then reused. */
export function createCheckpointCache(base = BASE): (url: string) => Promise<Checkpoint> {
  const loaded = new Map<string, Promise<Checkpoint>>();
  return (url: string) => {
    // The promise is cached, not the result, so two switches in flight at
    // once cannot both fetch the same file.
    const existing = loaded.get(url);
    if (existing) return existing;
    const pending = fetchCheckpoint(url, base);
    loaded.set(url, pending);
    return pending;
  };
}
