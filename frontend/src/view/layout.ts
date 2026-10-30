/**
 * Where the panels sit, as arithmetic.
 *
 * The workspace is a canvas of twelve columns and rows of a fixed height,
 * and every panel is a rectangle on it. This module owns the geometry and
 * nothing else: no DOM, no events, no storage. `view/workspace.ts` turns a
 * layout into elements and pointer gestures back into layouts, and the rules
 * below are testable without either.
 *
 * Two rules do all the work:
 *
 * - **Nothing overlaps.** A panel dropped on top of another pushes it down,
 *   and whatever that one lands on is pushed in turn.
 * - **Nothing floats.** After every change each panel rises until it meets
 *   the one above it, so dragging a panel out of the middle closes the hole
 *   rather than leaving the canvas full of gaps nobody asked for.
 *
 * Compaction is vertical only. A panel keeps the column it was dropped in
 * and loses the empty space above it, which is what makes a dashboard of
 * this kind predictable: the horizontal position is the reader's decision
 * and the vertical one is a consequence of it.
 *
 * The panel being dragged is **pinned**: it is placed first and never pushed,
 * so the thing under the pointer is the one thing on the canvas that goes
 * exactly where it is put.
 */

/** The canvas is twelve columns wide, at every window width. */
export const COLUMNS = 12;

/**
 * A minimised panel's height, in rows.
 *
 * It is exactly its title bar; `view/workspace.ts` derives the bar's CSS
 * height from this number rather than the other way round, so the two
 * cannot drift apart.
 */
export const MINIMISED_ROWS = 3;

export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface PanelLayout extends Rect {
  id: string;
  /**
   * Minimised panels keep `h`: it is the height to come back to, not the
   * height on screen. `heightOf` is what the geometry uses.
   */
  minimised: boolean;
}

export type Layout = PanelLayout[];

/** What a panel's content can live with, in grid units. */
export interface Limits {
  minW: number;
  maxW: number;
  minH: number;
  maxH: number;
}

/** The rows a panel actually occupies, which minimising changes. */
export function heightOf(panel: PanelLayout): number {
  return panel.minimised ? MINIMISED_ROWS : panel.h;
}

export function overlaps(a: PanelLayout, b: PanelLayout): boolean {
  if (a.id === b.id) return false;
  if (a.x + a.w <= b.x || b.x + b.w <= a.x) return false;
  return a.y + heightOf(a) > b.y && b.y + heightOf(b) > a.y;
}

/** The rows the whole canvas needs, for the height of the element. */
export function rowsUsed(layout: Layout): number {
  return layout.reduce((total, panel) => Math.max(total, panel.y + heightOf(panel)), 0);
}

export function find(layout: Layout, id: string): PanelLayout | null {
  return layout.find((panel) => panel.id === id) ?? null;
}

/**
 * A rectangle its panel can live in: within its own limits, and on the
 * canvas rather than half off the side of it.
 */
export function clampRect(rect: Rect, limits: Limits): Rect {
  const w = Math.min(Math.max(Math.round(rect.w), limits.minW), Math.min(limits.maxW, COLUMNS));
  const h = Math.min(Math.max(Math.round(rect.h), limits.minH), limits.maxH);
  return {
    w,
    h,
    x: Math.min(Math.max(Math.round(rect.x), 0), COLUMNS - w),
    y: Math.max(Math.round(rect.y), 0),
  };
}

/**
 * Who gets out of whose way: the pinned panel first, then down the canvas.
 *
 * Only `separate` wants this order. Compaction wants the canvas's own
 * order, because a panel that is placed first is placed at the top, and
 * the panel under the pointer has no business rising above the ones it was
 * dropped below.
 */
function ordered(layout: Layout, pinned: string | null): Layout {
  return [...layout].sort((a, b) => {
    if (a.id === pinned) return -1;
    if (b.id === pinned) return 1;
    return a.y - b.y || a.x - b.x;
  });
}

/** Placement order: down the canvas, and the pinned panel wins its row. */
function byRow(layout: Layout, pinned: string | null): Layout {
  return [...layout].sort((a, b) => {
    if (a.y !== b.y) return a.y - b.y;
    if (a.id === pinned) return -1;
    if (b.id === pinned) return 1;
    return a.x - b.x;
  });
}

/** The row a panel is centred on, which is what decides which way it moves. */
function centre(panel: PanelLayout): number {
  return panel.y + heightOf(panel) / 2;
}

/**
 * Get whatever the pinned panel landed on out of its way, and whatever
 * that lands on in turn.
 *
 * Which way a panel is moved is the difference between a canvas that can
 * be rearranged and one that can only be stacked. A panel the pinned one
 * has come down *onto* goes down. A panel the pinned one has come down
 * *past* — its centre is now above the pinned panel's — goes up into the
 * room the pinned panel left, which is what makes dragging a panel below
 * its neighbour actually swap the two. Without that the pinned panel is
 * simply floated back where it came from and nothing can ever be reordered
 * downward.
 *
 * Lifting is only offered where there is room above; otherwise the panel
 * goes down as before. `compact` settles whatever is left, so an early
 * exit from the guard below costs tidiness rather than correctness.
 */
function separate(layout: Layout, pinned: string | null): Layout {
  const panels = layout.map((panel) => ({ ...panel }));
  const limit = panels.length * panels.length + 8;
  for (let pass = 0; pass < limit; pass += 1) {
    const order = ordered(panels, pinned);
    let clashed = false;
    for (let above = 0; above < order.length && !clashed; above += 1) {
      for (let below = above + 1; below < order.length; below += 1) {
        const stay = order[above];
        const move = order[below];
        if (!overlaps(stay, move)) continue;
        const lifted = stay.y - heightOf(move);
        move.y =
          stay.id === pinned && centre(move) < centre(stay) && lifted >= 0
            ? lifted
            : stay.y + heightOf(stay);
        clashed = true;
        break;
      }
    }
    if (!clashed) return panels;
  }
  return panels;
}

/**
 * Float every panel up until it meets the one above it, or the top.
 *
 * Each panel is first pushed clear of whatever is already placed and then
 * floated up, so this settles a layout rather than assuming one: a stored
 * arrangement whose panels overlap comes out of here not overlapping.
 */
export function compact(layout: Layout, pinned: string | null = null): Layout {
  const placed: PanelLayout[] = [];
  for (const panel of byRow(layout, pinned)) {
    const risen = { ...panel };
    while (placed.some((other) => overlaps(risen, other))) risen.y += 1;
    while (
      risen.y > 0 &&
      !placed.some((other) => overlaps({ ...risen, y: risen.y - 1 }, other))
    ) {
      risen.y -= 1;
    }
    placed.push(risen);
  }
  // Back into the caller's own order, so a layout is comparable with the
  // one it came from and the DOM is not reshuffled for nothing.
  return layout.map((panel) => placed.find((other) => other.id === panel.id) ?? panel);
}

/** Settle a layout after one panel has been given a new rectangle. */
function settle(layout: Layout, pinned: string): Layout {
  return compact(separate(layout, pinned), pinned);
}

/** Put one panel where it was dropped, and let the rest get out of the way. */
export function moveTo(layout: Layout, id: string, x: number, y: number): Layout {
  const panel = find(layout, id);
  if (!panel) return layout;
  const next = layout.map((other) =>
    other.id === id
      ? {
          ...other,
          x: Math.min(Math.max(Math.round(x), 0), COLUMNS - other.w),
          y: Math.max(Math.round(y), 0),
        }
      : other,
  );
  return settle(next, id);
}

/**
 * The panel directly above or below this one, among those sharing a column
 * with it. Nothing else can be in its way.
 */
function neighbour(layout: Layout, panel: PanelLayout, direction: 1 | -1): PanelLayout | null {
  const candidates = layout.filter(
    (other) =>
      other.id !== panel.id &&
      other.x < panel.x + panel.w &&
      panel.x < other.x + other.w &&
      (direction === 1 ? other.y > panel.y : other.y < panel.y),
  );
  if (candidates.length === 0) return null;
  return candidates.reduce((best, other) =>
    direction === 1 ? (other.y < best.y ? other : best) : (other.y > best.y ? other : best),
  );
}

/**
 * Move a panel past the one above or below it.
 *
 * The vertical step a keyboard wants, and it is not one row: nothing
 * floats, so a panel nudged a single row down is floated straight back and
 * a reader would be holding the key against a layout that cannot move. The
 * only vertical move that means anything is past the neighbour, so that is
 * what the arrow keys do. With no neighbour that way there is nowhere to
 * go, and the layout comes back unchanged.
 */
export function stepRow(layout: Layout, id: string, direction: 1 | -1): Layout {
  const panel = find(layout, id);
  if (!panel) return layout;
  const other = neighbour(layout, panel, direction);
  if (!other) return layout;
  const y =
    direction === 1
      ? other.y + heightOf(other)
      : Math.max(0, other.y - heightOf(panel));
  return moveTo(layout, id, panel.x, y);
}

/** Resize one panel within what its own content can live with. */
export function resizeTo(
  layout: Layout,
  id: string,
  size: { w: number; h: number },
  limits: Limits,
): Layout {
  const panel = find(layout, id);
  if (!panel || panel.minimised) return layout;
  const rect = clampRect({ ...panel, ...size }, limits);
  const next = layout.map((other) => (other.id === id ? { ...other, ...rect } : other));
  return settle(next, id);
}

/**
 * Fold a panel down to its title bar, or unfold it again.
 *
 * The height it had is kept rather than recomputed, so a panel comes back
 * the size the reader made it rather than the size it shipped with.
 */
export function setMinimised(layout: Layout, id: string, minimised: boolean): Layout {
  const panel = find(layout, id);
  if (!panel || panel.minimised === minimised) return layout;
  const next = layout.map((other) =>
    other.id === id ? { ...other, minimised } : other,
  );
  return settle(next, id);
}

/** Every panel expanded, in the places it shipped with. */
export function defaultLayout(specs: { id: string; rect: Rect }[]): Layout {
  return compact(
    specs.map((spec) => ({ id: spec.id, ...spec.rect, minimised: false })),
  );
}

interface Stored {
  version: number;
  panels: PanelLayout[];
}

/** The format version, bumped when a stored layout stops meaning this. */
export const LAYOUT_VERSION = 1;

export function serialise(layout: Layout): string {
  const stored: Stored = { version: LAYOUT_VERSION, panels: layout };
  return JSON.stringify(stored);
}

/**
 * A stored layout, or null to say "use the defaults".
 *
 * Null for anything that does not describe exactly the panels this build
 * has: a different version, a missing panel, one nobody knows, a number
 * that is not a number. A layout is a convenience, so a stale one is worth
 * nothing and a half-applied one would be worse than starting again —
 * adding a panel in a later release therefore costs every reader their
 * arrangement once, which is the cheapest honest answer.
 */
export function parseLayout(text: string | null, ids: string[]): Layout | null {
  if (!text) return null;
  let stored: Stored;
  try {
    stored = JSON.parse(text) as Stored;
  } catch {
    return null;
  }
  if (stored?.version !== LAYOUT_VERSION || !Array.isArray(stored.panels)) return null;
  if (stored.panels.length !== ids.length) return null;

  const seen = new Set<string>();
  const layout: Layout = [];
  for (const panel of stored.panels) {
    if (typeof panel?.id !== "string" || !ids.includes(panel.id) || seen.has(panel.id)) {
      return null;
    }
    seen.add(panel.id);
    const numbers = [panel.x, panel.y, panel.w, panel.h];
    if (!numbers.every((value) => Number.isFinite(value) && value >= 0)) return null;
    if (typeof panel.minimised !== "boolean") return null;
    if (panel.w < 1 || panel.x + panel.w > COLUMNS || panel.h < 1) return null;
    layout.push({
      id: panel.id,
      x: Math.round(panel.x),
      y: Math.round(panel.y),
      w: Math.round(panel.w),
      h: Math.round(panel.h),
      minimised: panel.minimised,
    });
  }
  // In the order this build declares, not the order it was written in.
  const inOrder = ids.map((id) => layout.find((panel) => panel.id === id)!);
  // A stored layout is only ever as trustworthy as the rules it came from,
  // so it is settled on the way in rather than taken at its word.
  return compact(separate(inOrder, null), null);
}
