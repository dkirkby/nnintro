/**
 * Loading and validating published assets.
 *
 * The explorer enumerates the manifest rather than assuming the catalog is
 * covered: a control must never be able to reach a configuration that has
 * no checkpoint (PLAN.md section 3).
 */

import type { Checkpoint, Manifest } from "./types.js";

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
