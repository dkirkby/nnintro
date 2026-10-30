/** Loads the published snapshot that the tests check against. */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import type { Checkpoint, Manifest } from "../src/model/types.js";

/** Resolved from the project root, so it holds under any test environment
 *  (a jsdom `import.meta.url` is an http: URL, not a file: one). */
const DATA = resolve(process.cwd(), "public/data");

export function loadManifest(): Manifest {
  return JSON.parse(readFileSync(resolve(DATA, "manifest.json"), "utf8"));
}

export function loadCheckpoint(url: string): Checkpoint {
  return JSON.parse(readFileSync(resolve(DATA, url), "utf8"));
}

/**
 * The checkpoint the single-configuration tests are written against: one
 * hidden layer of four ReLU units on the tilted plane, which is what the
 * explorer opens on.
 *
 * Pinned by name rather than taken as the manifest's first entry: the
 * catalog is ordered by architecture now, so entry zero is the 2 → 1 linear
 * model, which has no hidden layer for a test about hidden units to find.
 */
export const PRIMARY = "tilted-plane__2-4-1__relu";

/** The same architecture on a classifier, whose output is a logit. */
export const CLASSIFIER = "gaussian-clusters__2-4-1__relu";

export function checkpointNamed(id: string): Checkpoint {
  const entry = loadManifest().checkpoints.find((row) => row.checkpointId === id);
  if (!entry) throw new Error(`the published snapshot has no checkpoint ${id}`);
  return loadCheckpoint(entry.url);
}

/** Every checkpoint in the published snapshot. */
export function allCheckpoints(): Checkpoint[] {
  return loadManifest().checkpoints.map((entry) => loadCheckpoint(entry.url));
}
