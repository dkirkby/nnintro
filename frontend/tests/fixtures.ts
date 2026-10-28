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

/** Every checkpoint in the published snapshot. */
export function allCheckpoints(): Checkpoint[] {
  return loadManifest().checkpoints.map((entry) => loadCheckpoint(entry.url));
}
