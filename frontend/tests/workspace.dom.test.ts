/**
 * @vitest-environment jsdom
 *
 * The panels as a canvas: folding, dragging, resizing, and remembering.
 *
 * The geometry itself is `layout.test.ts`. These are the parts that need a
 * document — that a gesture on a title bar becomes a move, that a fold
 * leaves the bar and nothing else, and that the arrangement the reader made
 * is the one they get back.
 *
 * jsdom lays nothing out, so the canvas has no width to measure and
 * `nominalWidth` stands in for one: at 1200 over twelve columns, a column
 * is a round hundred pixels and a drag is arithmetic rather than a guess.
 */
import { beforeEach, describe, expect, it } from "vitest";

import { COLUMNS, MINIMISED_ROWS, type Limits } from "../src/view/layout.js";
import {
  GAP,
  ROW_HEIGHT,
  createWorkspace,
  type PanelSpec,
  type Workspace,
} from "../src/view/workspace.js";

const NOMINAL = 1200;
const COLUMN = NOMINAL / COLUMNS;

const LIMITS: Limits = { minW: 2, maxW: COLUMNS, minH: 4, maxH: 40 };

const PANELS: PanelSpec[] = [
  {
    id: "left",
    title: "The left one",
    body: `<p id="left-body">left</p>`,
    rect: { x: 0, y: 0, w: 6, h: 10 },
    limits: LIMITS,
  },
  {
    id: "right",
    title: "The right one",
    body: `<p id="right-body">right</p>`,
    rect: { x: 6, y: 0, w: 6, h: 6 },
    limits: LIMITS,
  },
  {
    id: "below",
    title: "The one below",
    body: `<p id="below-body">below</p>`,
    rect: { x: 0, y: 10, w: 6, h: 5 },
    limits: LIMITS,
  },
];

/** A `Storage` that lives and dies with one test. */
function fakeStorage(): Storage {
  const held = new Map<string, string>();
  return {
    get length() {
      return held.size;
    },
    clear: () => held.clear(),
    getItem: (key: string) => held.get(key) ?? null,
    key: (index: number) => [...held.keys()][index] ?? null,
    removeItem: (key: string) => void held.delete(key),
    setItem: (key: string, value: string) => void held.set(key, value),
  };
}

describe("the workspace", () => {
  let container: HTMLElement;
  let storage: Storage;
  let workspace: Workspace;

  beforeEach(() => {
    container = document.createElement("div");
    document.body.replaceChildren(container);
    storage = fakeStorage();
    workspace = createWorkspace({
      container,
      panels: PANELS,
      storage,
      nominalWidth: NOMINAL,
    });
  });

  function section(id: string): HTMLElement {
    return container.querySelector<HTMLElement>(`[data-panel="${id}"]`)!;
  }

  function bar(id: string): HTMLElement {
    return section(id).querySelector<HTMLElement>(".panel-bar")!;
  }

  /**
   * What the DOM says, back in grid units.
   *
   * The styles are what a reader actually gets — a percentage across and
   * pixels down — so they are read off the element rather than from the
   * layout the workspace is holding, and converted here.
   */
  function box(id: string): { x: number; y: number; w: number; h: number } {
    const style = section(id).style;
    const across = (value: string) =>
      (Number(/(-?[\d.]+)%/.exec(value)?.[1] ?? "0") * COLUMNS) / 100;
    return {
      x: Math.round(across(style.left)),
      y: Number(style.top.replace("px", "")) / ROW_HEIGHT,
      w: Math.round(across(style.width)),
      h: (Number(style.height.replace("px", "")) + GAP) / ROW_HEIGHT,
    };
  }

  function rows(id: string): number {
    return box(id).h;
  }

  /** A pointer gesture, in the events jsdom can actually construct. */
  function gesture(from: Element, dx: number, dy: number): void {
    from.dispatchEvent(
      new window.MouseEvent("pointerdown", { bubbles: true, clientX: 0, clientY: 0 }),
    );
    document.dispatchEvent(
      new window.MouseEvent("pointermove", { bubbles: true, clientX: dx, clientY: dy }),
    );
    document.dispatchEvent(new window.MouseEvent("pointerup", { bubbles: true }));
  }

  function press(id: string, key: string, shiftKey = false): void {
    bar(id).dispatchEvent(new window.KeyboardEvent("keydown", { bubbles: true, key, shiftKey }));
  }

  it("BUILDS ONE PANEL PER SPEC, each a bar over a body", () => {
    expect(container.querySelectorAll(".panel")).toHaveLength(PANELS.length);
    expect(bar("left").querySelector("h2")!.textContent).toBe("The left one");
    expect(workspace.body("left").querySelector("#left-body")).not.toBeNull();
    // The bar is a handle, and a handle a keyboard can reach.
    expect(bar("left").tabIndex).toBe(0);
    expect(box("left")).toEqual({ x: 0, y: 0, w: 6, h: 10 });
    expect(box("below").y).toBe(10);
  });

  it("FOLDS A PANEL DOWN TO ITS TITLE BAR, and the rest rise into the room", () => {
    const fold = section("left").querySelector<HTMLButtonElement>(".panel-fold")!;
    fold.click();
    expect(section("left").classList.contains("is-minimised")).toBe(true);
    expect(rows("left")).toBe(MINIMISED_ROWS);
    expect(fold.getAttribute("aria-expanded")).toBe("false");
    expect(fold.getAttribute("aria-label")).toBe("Expand The left one");
    expect(box("below").y).toBe(MINIMISED_ROWS);

    fold.click();
    expect(rows("left")).toBe(10);
    expect(box("below").y).toBe(10);
  });

  it("MOVES A PANEL BY ITS TITLE BAR, pushing what it lands on down", () => {
    gesture(bar("right"), -6 * COLUMN, 0);
    expect(box("right").x).toBe(0);
    expect(box("right").y).toBe(0);
    // "left" was there first and is pushed below the panel dropped on it.
    expect(box("left").y).toBe(6);
    expect(box("below").y).toBe(16);
  });

  it("does not start a drag from a button on the bar", () => {
    const fold = section("left").querySelector<HTMLButtonElement>(".panel-fold")!;
    gesture(fold, 4 * COLUMN, 4 * ROW_HEIGHT);
    expect(box("left").x).toBe(0);
    expect(box("left").y).toBe(0);
  });

  it("RESIZES FROM THE CORNER, within what the panel can live with", () => {
    gesture(section("right").querySelector(".panel-grip")!, 0, 6 * ROW_HEIGHT);
    expect(rows("right")).toBe(12);

    // Past the floor in both directions at once: the limits hold.
    gesture(section("right").querySelector(".panel-grip")!, -20 * COLUMN, -40 * ROW_HEIGHT);
    expect(rows("right")).toBe(LIMITS.minH);
    expect(box("right").x).toBe(6);
    expect(box("right").w).toBe(LIMITS.minW);
  });

  it("moves and resizes from the keyboard, for a reader with no pointer", () => {
    // Vertically past the neighbour, which is the only vertical move a
    // canvas that compacts actually has: "left" and "below" share a
    // column, so one press swaps them.
    press("left", "ArrowDown");
    expect(box("below").y).toBe(0);
    expect(box("left").y).toBe(5);
    press("left", "ArrowUp");
    expect(box("left").y).toBe(0);
    expect(box("below").y).toBe(10);
    // And with nothing below it, there is nowhere to go.
    press("below", "ArrowDown");
    expect(box("below").y).toBe(10);

    // Sideways by a column.
    press("right", "ArrowLeft");
    expect(box("right").x).toBe(5);

    press("right", "ArrowDown", true);
    expect(rows("right")).toBe(7);
    press("right", "ArrowLeft", true);
    expect(box("right").w).toBe(5);
  });

  it("KEEPS THE ARRANGEMENT BETWEEN SESSIONS", () => {
    section("left").querySelector<HTMLButtonElement>(".panel-fold")!.click();
    gesture(bar("below"), 6 * COLUMN, 0);
    const arranged = container.querySelectorAll(".panel").length;
    expect(storage.getItem("nnintro.layout")).toContain('"minimised":true');

    workspace.destroy();
    workspace = createWorkspace({
      container,
      panels: PANELS,
      storage,
      nominalWidth: NOMINAL,
    });
    expect(container.querySelectorAll(".panel")).toHaveLength(arranged);
    expect(workspace.isRestored()).toBe(true);
    expect(section("left").classList.contains("is-minimised")).toBe(true);
    expect(box("below").x).toBe(6);
  });

  it("starts from the defaults when the stored layout is not about these panels", () => {
    storage.setItem(
      "nnintro.layout",
      JSON.stringify({
        version: 1,
        panels: [{ id: "a-panel-from-another-build", x: 0, y: 0, w: 4, h: 4, minimised: false }],
      }),
    );
    workspace.destroy();
    workspace = createWorkspace({
      container,
      panels: PANELS,
      storage,
      nominalWidth: NOMINAL,
    });
    expect(workspace.isRestored()).toBe(false);
    expect(rows("left")).toBe(10);
  });

  it("RESETS to every panel expanded and where it shipped, and forgets", () => {
    section("left").querySelector<HTMLButtonElement>(".panel-fold")!.click();
    gesture(bar("right"), -6 * COLUMN, 3 * ROW_HEIGHT);
    expect(storage.getItem("nnintro.layout")).not.toBeNull();

    workspace.reset();
    expect(storage.getItem("nnintro.layout")).toBeNull();
    expect(section("left").classList.contains("is-minimised")).toBe(false);
    expect(box("left")).toEqual({ x: 0, y: 0, w: 6, h: 10 });
    expect(box("right").x).toBe(6);
    expect(box("below").y).toBe(10);
  });

  it("renames a panel whose name is part of what it shows", () => {
    workspace.setTitle("right", "Neuron: layer 2, unit 0");
    expect(bar("right").querySelector("h2")!.textContent).toBe("Neuron: layer 2, unit 0");
    expect(
      section("right").querySelector(".panel-fold")!.getAttribute("aria-label"),
    ).toBe("Minimise Neuron: layer 2, unit 0");
  });

  it("gives the canvas the height of what is on it", () => {
    expect(container.style.height).toBe(`${15 * ROW_HEIGHT}px`);
    section("below").querySelector<HTMLButtonElement>(".panel-fold")!.click();
    expect(container.style.height).toBe(`${(10 + MINIMISED_ROWS) * ROW_HEIGHT}px`);
  });

  it("works with nowhere to store a layout, which is a private window", () => {
    workspace.destroy();
    workspace = createWorkspace({
      container,
      panels: PANELS,
      storage: null,
      nominalWidth: NOMINAL,
    });
    section("left").querySelector<HTMLButtonElement>(".panel-fold")!.click();
    expect(rows("left")).toBe(MINIMISED_ROWS);
    expect(workspace.isRestored()).toBe(false);
  });
});
