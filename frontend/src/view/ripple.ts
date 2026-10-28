/**
 * The downstream pulse.
 *
 * Three rules from PLAN.md section 4 are load-bearing here:
 *
 *  - **The new state is already computed.** This animates an explanation of
 *    a change that has happened; it is never the source of the values.
 *  - **A pulse means "influence travelled here", nothing more.** Zipline's
 *    carrier oscillation, speed and transverse amplitude carry no
 *    quantitative meaning. Deltas go in labels, never in the waveform.
 *  - **Only edges that actually carried a change are animated.** A node
 *    that is structurally downstream but unmoved is marked as blocked, so
 *    the animation never implies that everything reachable changed.
 *
 * A new edit cancels an obsolete run: `play` takes a generation token and
 * any run whose token is stale destroys its controllers and resolves.
 */

import { createZiplineGroup } from "@dkirkby/zipline";
import type { ZiplineGroup } from "@dkirkby/zipline";
import type { DenseNetworkLayout } from "nnet-svg";

import type { Ripple } from "../model/ripple.js";
import { edgeKeyFromModel } from "./adapter.js";

/** Pulse shape. N must be one of zipline's precomputed tables. */
const PULSE = { f: 0.35, N: 2, alpha: 0.05, r: 1.5 } as const;
/** Seconds per layer hop. */
const HOP_SECONDS = 0.45;

export interface RippleAnimator {
  /** Animate one ripple. Resolves when it finishes or is cancelled. */
  play(ripple: Ripple): Promise<void>;
  /** Abandon any run in progress and tear down its controllers. */
  cancel(): void;
  destroy(): void;
  readonly running: boolean;
  /** Increments on every play or cancel; stale runs compare against it. */
  readonly generation: number;
  /** True when the animation is suppressed in favour of static deltas. */
  readonly reducedMotion: boolean;
}

export interface RippleOptions {
  overlay: () => SVGGElement;
  layout: () => DenseNetworkLayout;
  /** Injected so tests need no matchMedia; defaults to the real query. */
  prefersReducedMotion?: () => boolean;
}

function defaultReducedMotion(): boolean {
  return (
    typeof window !== "undefined" &&
    typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches
  );
}

export function createRippleAnimator(options: RippleOptions): RippleAnimator {
  const prefersReducedMotion = options.prefersReducedMotion ?? defaultReducedMotion;
  let generation = 0;
  /**
   * Every group currently mounted. Runs overlap: a stale one is still
   * awaiting its own `play()` when the run that replaced it mounts the next
   * group, so ownership has to be explicit — each controller is destroyed
   * exactly once, by whichever of the two reaches it first.
   */
  const live = new Set<ZiplineGroup>();
  let running = false;
  let destroyed = false;

  /** Edge geometry, keyed the same way the adapter keys edges. */
  function geometry(): Map<string, { x1: number; y1: number; x2: number; y2: number }> {
    const map = new Map<string, { x1: number; y1: number; x2: number; y2: number }>();
    for (const edge of options.layout().edges) {
      map.set(
        edgeKeyFromModel(edge.sourceLayerIndex, edge.sourceNodeIndex, edge.targetNodeIndex),
        { x1: edge.x1, y1: edge.y1, x2: edge.x2, y2: edge.y2 },
      );
    }
    return map;
  }

  function release(group: ZiplineGroup): void {
    if (!live.delete(group)) return;
    group.destroy();
  }

  function teardown(): void {
    for (const group of Array.from(live)) release(group);
    running = false;
  }

  function markStatic(ripple: Ripple): void {
    // The reduced-motion and post-run presentation: the overlay states
    // which nodes moved and which were blocked, without any motion.
    const overlay = options.overlay();
    overlay.replaceChildren();
    const positions = new Map<string, { x: number; y: number }>();
    for (const item of options.layout().items) {
      if (item.kind !== "node") continue;
      positions.set(`${item.layerIndex}:${item.nodeIndex}`, { x: item.x, y: item.y });
    }
    for (const stage of ripple.stages) {
      for (const node of stage.nodes) {
        const at = positions.get(`${node.layer}:${node.index}`);
        if (!at) continue;
        const marker = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        marker.setAttribute("cx", String(at.x));
        marker.setAttribute("cy", String(at.y));
        marker.setAttribute("r", "13");
        marker.setAttribute("class", node.changed ? "ripple-changed" : "ripple-blocked");
        overlay.append(marker);
      }
    }
  }

  async function play(ripple: Ripple): Promise<void> {
    if (destroyed) return;
    generation += 1;
    const token = generation;
    teardown();
    running = true;

    markStatic(ripple);
    if (prefersReducedMotion()) {
      running = false;
      return;
    }

    const edges = geometry();
    for (const stage of ripple.stages) {
      // Only edges that carried a change are drawn: a blocked path shows
      // no pulse, which is the point.
      const lines = stage.edges
        .filter((edge) => edge.carried)
        .map((edge) => {
          const at = edges.get(edgeKeyFromModel(edge.layer, edge.source, edge.target));
          return at ? { id: edgeKeyFromModel(edge.layer, edge.source, edge.target), ...at } : null;
        })
        .filter((line): line is NonNullable<typeof line> => line !== null);
      if (lines.length === 0) continue;

      if (token !== generation || destroyed) return;
      const group = createZiplineGroup({ ...PULSE, dt: HOP_SECONDS, lines }).mount(
        options.overlay(),
      );
      live.add(group);
      await group.play();
      release(group);
      // The await yields, so a newer edit may have landed meanwhile. It owns
      // `running` from here: a stale run leaves quietly rather than
      // reporting that nothing is animating.
      if (token !== generation || destroyed) return;
    }
    if (token === generation) running = false;
  }

  return {
    play,
    cancel() {
      generation += 1;
      teardown();
    },
    destroy() {
      destroyed = true;
      generation += 1;
      teardown();
      try {
        options.overlay().replaceChildren();
      } catch {
        // The view may already be gone; nothing to clear.
      }
    },
    get running() {
      return running;
    },
    get generation() {
      return generation;
    },
    get reducedMotion() {
      return prefersReducedMotion();
    },
  };
}
