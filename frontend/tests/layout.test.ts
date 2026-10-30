/**
 * The workspace's geometry: what pushes what, and what floats where.
 *
 * Two rules, and every test here is one of them: nothing overlaps, and
 * nothing floats. The DOM half is `workspace.dom.test.ts`; none of this
 * needs a document.
 */
import { describe, expect, it } from "vitest";

import {
  COLUMNS,
  MINIMISED_ROWS,
  clampRect,
  compact,
  defaultLayout,
  find,
  heightOf,
  moveTo,
  overlaps,
  parseLayout,
  resizeTo,
  rowsUsed,
  serialise,
  setMinimised,
  stepRow,
  type Layout,
  type Limits,
} from "../src/view/layout.js";

const LIMITS: Limits = { minW: 2, maxW: COLUMNS, minH: 4, maxH: 40 };

/** Three panels in two columns, the shape the explorer itself opens in. */
function threeUp(): Layout {
  return defaultLayout([
    { id: "left", rect: { x: 0, y: 0, w: 6, h: 10 } },
    { id: "right", rect: { x: 6, y: 0, w: 6, h: 6 } },
    { id: "below", rect: { x: 0, y: 10, w: 6, h: 5 } },
  ]);
}

/** Every pair, so "nothing overlaps" is checked as the claim it is. */
function anyOverlap(layout: Layout): boolean {
  return layout.some((a) => layout.some((b) => overlaps(a, b)));
}

function rect(layout: Layout, id: string) {
  const panel = find(layout, id)!;
  return { x: panel.x, y: panel.y, w: panel.w, h: heightOf(panel) };
}

describe("the default layout", () => {
  it("OVERLAPS NOWHERE and leaves no gap at the top", () => {
    const layout = threeUp();
    expect(anyOverlap(layout)).toBe(false);
    expect(Math.min(...layout.map((panel) => panel.y))).toBe(0);
    expect(rowsUsed(layout)).toBe(15);
  });

  it("closes a hole its own numbers left", () => {
    // Declared at row 30 with nothing between: a default is a position, not
    // a promise about the rows above it.
    const layout = defaultLayout([
      { id: "top", rect: { x: 0, y: 0, w: 6, h: 4 } },
      { id: "loose", rect: { x: 0, y: 30, w: 6, h: 4 } },
    ]);
    expect(rect(layout, "loose").y).toBe(4);
  });
});

describe("moving a panel", () => {
  it("PUSHES WHAT IT LANDS ON DOWN, and what that lands on in turn", () => {
    const layout = moveTo(threeUp(), "right", 0, 0);
    // "right" is where it was put; the column it took is clear beneath it.
    expect(rect(layout, "right")).toEqual({ x: 0, y: 0, w: 6, h: 6 });
    expect(rect(layout, "left").y).toBe(6);
    expect(rect(layout, "below").y).toBe(16);
    expect(anyOverlap(layout)).toBe(false);
  });

  it("closes the hole it came out of", () => {
    const layout = moveTo(threeUp(), "left", 6, 20);
    // Nothing is left above "below", so it rises to the top of its column.
    expect(rect(layout, "below").y).toBe(0);
    // And the panel that was dropped low floats onto what is under it.
    expect(rect(layout, "left").y).toBe(6);
  });

  it("CANNOT BE DRAGGED OFF THE CANVAS, either side", () => {
    const far = moveTo(threeUp(), "right", 99, 3);
    expect(rect(far, "right").x).toBe(COLUMNS - 6);
    const left = moveTo(threeUp(), "right", -4, 3);
    expect(rect(left, "right").x).toBe(0);
    expect(moveTo(threeUp(), "right", 6, -5).find((p) => p.y < 0)).toBeUndefined();
  });

  it("LETS A PANEL BE DROPPED BELOW ITS NEIGHBOUR, which pushing alone never would", () => {
    // Dragged down past the middle of the panel beneath it: that one comes
    // UP into the room left behind rather than being shoved further down.
    // Pushing only ever downward would float the dragged panel back where
    // it came from, and nothing on the canvas could be reordered downward.
    const layout = moveTo(threeUp(), "left", 0, 12);
    expect(rect(layout, "below").y).toBe(0);
    expect(rect(layout, "left").y).toBe(5);
    expect(anyOverlap(layout)).toBe(false);
  });

  it("settles rather than circling, however the panels are stacked", () => {
    let layout = threeUp();
    for (const id of ["left", "right", "below", "left"]) {
      layout = moveTo(layout, id, 0, 0);
      expect(anyOverlap(layout)).toBe(false);
    }
  });
});

describe("resizing a panel", () => {
  it("IS HELD TO WHAT ITS CONTENT CAN LIVE WITH", () => {
    const grown = resizeTo(threeUp(), "right", { w: 99, h: 99 }, LIMITS);
    expect(rect(grown, "right").w).toBe(LIMITS.maxW);
    expect(rect(grown, "right").h).toBe(LIMITS.maxH);
    const shrunk = resizeTo(threeUp(), "right", { w: 0, h: 0 }, LIMITS);
    expect(rect(shrunk, "right").w).toBe(LIMITS.minW);
    expect(rect(shrunk, "right").h).toBe(LIMITS.minH);
  });

  it("stays in its own row rather than jumping to the top", () => {
    // The panel being worked on is placed before the others so that it
    // keeps its column, which must not also mean it is placed at the top:
    // growing the bottom panel once sent it above the two it sat under.
    const layout = resizeTo(threeUp(), "below", { w: 6, h: 8 }, LIMITS);
    expect(rect(layout, "below").y).toBe(10);
    expect(rect(layout, "left").y).toBe(0);
    expect(anyOverlap(layout)).toBe(false);
  });

  it("pushes its neighbours down when it grows into them", () => {
    const layout = resizeTo(threeUp(), "left", { w: 6, h: 14 }, LIMITS);
    expect(rect(layout, "below").y).toBe(14);
  });

  it("widening cannot push a panel off the canvas", () => {
    const layout = resizeTo(threeUp(), "right", { w: 12, h: 6 }, LIMITS);
    expect(rect(layout, "right").x).toBe(0);
    expect(rect(layout, "right").w).toBe(12);
  });

  it("leaves a minimised panel alone: there is nothing to resize", () => {
    const folded = setMinimised(threeUp(), "left", true);
    expect(resizeTo(folded, "left", { w: 3, h: 20 }, LIMITS)).toBe(folded);
  });
});

describe("minimising a panel", () => {
  it("GIVES BACK EVERY ROW BUT ITS TITLE BAR", () => {
    const layout = setMinimised(threeUp(), "left", true);
    expect(heightOf(find(layout, "left")!)).toBe(MINIMISED_ROWS);
    expect(rect(layout, "below").y).toBe(MINIMISED_ROWS);
    expect(anyOverlap(layout)).toBe(false);
  });

  it("comes back the height it was, not the height it shipped with", () => {
    const resized = resizeTo(threeUp(), "left", { w: 6, h: 21 }, LIMITS);
    const folded = setMinimised(resized, "left", true);
    // The height is kept while it is folded away: it is where to come back
    // to, not what is on screen.
    expect(find(folded, "left")!.h).toBe(21);
    expect(heightOf(find(setMinimised(folded, "left", false), "left")!)).toBe(21);
  });
});

describe("stepping a panel by the keyboard", () => {
  it("MOVES IT PAST ITS NEIGHBOUR, since a row at a time moves nothing", () => {
    const down = stepRow(threeUp(), "left", 1);
    expect(rect(down, "below").y).toBe(0);
    expect(rect(down, "left").y).toBe(5);
    expect(rect(stepRow(down, "left", -1), "left").y).toBe(0);
  });

  it("leaves it where it is when there is no neighbour that way", () => {
    const layout = threeUp();
    expect(stepRow(layout, "below", 1)).toEqual(layout);
    expect(stepRow(layout, "right", -1)).toEqual(layout);
  });

  it("only counts panels that share a column with it", () => {
    // "right" is beside "left", not above or below it, so neither is in
    // the other's way whatever their rows say.
    const layout = moveTo(threeUp(), "right", 6, 8);
    expect(stepRow(layout, "right", -1)).toEqual(layout);
  });
});

describe("clampRect", () => {
  it("rounds to the grid and keeps the panel on it", () => {
    expect(clampRect({ x: 2.4, y: 3.6, w: 5.5, h: 7.2 }, LIMITS)).toEqual({
      x: 2,
      y: 4,
      w: 6,
      h: 7,
    });
    expect(clampRect({ x: 11, y: 0, w: 6, h: 6 }, LIMITS).x).toBe(COLUMNS - 6);
  });
});

describe("a stored layout", () => {
  const ids = ["left", "right", "below"];

  it("ROUND-TRIPS what the reader arranged", () => {
    const arranged = setMinimised(moveTo(threeUp(), "right", 0, 0), "below", true);
    expect(parseLayout(serialise(arranged), ids)).toEqual(arranged);
  });

  it("IS REFUSED WHOLE unless it describes exactly these panels", () => {
    const good = JSON.parse(serialise(threeUp()));
    const of = (change: (value: Record<string, unknown>) => void) => {
      const copy = JSON.parse(JSON.stringify(good));
      change(copy);
      return parseLayout(JSON.stringify(copy), ids);
    };
    // A layout is a convenience: a stale one is worth nothing, and a
    // half-applied one would be worse than starting again.
    expect(of((value) => (value.version = 99))).toBeNull();
    expect(of((value) => (value.panels as unknown[]).pop())).toBeNull();
    expect(of((value) => ((value.panels as { id: string }[])[0].id = "nobody"))).toBeNull();
    expect(of((value) => ((value.panels as { h: unknown }[])[0].h = "tall"))).toBeNull();
    expect(of((value) => ((value.panels as { x: number }[])[0].x = 11))).toBeNull();
    expect(of((value) => ((value.panels as { minimised: unknown }[])[0].minimised = 1))).toBeNull();
    expect(parseLayout("not json at all", ids)).toBeNull();
    expect(parseLayout(null, ids)).toBeNull();
  });

  it("is settled on the way in rather than taken at its word", () => {
    const overlapping = JSON.stringify({
      version: 1,
      panels: [
        { id: "left", x: 0, y: 0, w: 6, h: 10, minimised: false },
        { id: "right", x: 0, y: 2, w: 6, h: 6, minimised: false },
        { id: "below", x: 0, y: 40, w: 6, h: 5, minimised: false },
      ],
    });
    const layout = parseLayout(overlapping, ids)!;
    expect(anyOverlap(layout)).toBe(false);
    expect(rect(layout, "right").y).toBe(10);
    expect(rect(layout, "below").y).toBe(16);
  });

  it("comes back in the order this build declares its panels", () => {
    const shuffled = JSON.stringify({
      version: 1,
      panels: JSON.parse(serialise(threeUp())).panels.reverse(),
    });
    expect(parseLayout(shuffled, ids)!.map((panel) => panel.id)).toEqual(ids);
  });
});

describe("compact", () => {
  it("is vertical only: a panel keeps the column it was dropped in", () => {
    const layout = compact([
      { id: "a", x: 6, y: 4, w: 3, h: 4, minimised: false },
    ]);
    expect(rect(layout, "a")).toEqual({ x: 6, y: 0, w: 3, h: 4 });
  });

  it("keeps the caller's own order, so nothing is reshuffled for nothing", () => {
    const layout = threeUp();
    expect(compact(layout).map((panel) => panel.id)).toEqual(
      layout.map((panel) => panel.id),
    );
  });
});
