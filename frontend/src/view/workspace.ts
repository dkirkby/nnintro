/**
 * The panels as a canvas the reader arranges.
 *
 * Every panel is a card on a twelve-column grid: drag it by its title bar,
 * resize it by the corner, fold it down to the bar alone. The arrangement
 * is the reader's, so it is kept between sessions and `Reset layout` puts
 * it back. `view/layout.ts` owns the geometry — what pushes what, and what
 * floats where — and this module owns the elements and the gestures.
 *
 * The point of folding a panel away is that the explorer can be built up a
 * piece at a time. The figure is five views of one computation and all five
 * at once is a wall; a reader who has just met the network diagram can keep
 * the rest minimised until they are wanted, and the one panel being worked
 * in can have the window.
 *
 * Geometry is in grid units everywhere, converted at the edges:
 * **horizontally in percentages**, so a window resize rescales the canvas
 * without the layout being recomputed, and **vertically in rows** of
 * `ROW_HEIGHT`, so a panel's height is a number the reader can land on.
 */

import {
  COLUMNS,
  MINIMISED_ROWS,
  compact,
  defaultLayout,
  find,
  heightOf,
  moveTo,
  parseLayout,
  resizeTo,
  rowsUsed,
  serialise,
  stepRow,
  setMinimised,
  type Layout,
  type Limits,
  type Rect,
} from "./layout.js";

/** One row, in pixels. The grid is this fine vertically and no finer. */
export const ROW_HEIGHT = 16;
/** The space between two panels, taken out of each one's own box. */
export const GAP = 12;
/**
 * The title bar's height, which is what a minimised panel is.
 *
 * Derived from the row grid rather than chosen in the stylesheet, so a bar
 * that is not exactly `MINIMISED_ROWS` tall cannot happen: the value is
 * published as a custom property and `styles.css` reads it.
 */
const BAR_HEIGHT = MINIMISED_ROWS * ROW_HEIGHT - GAP;

/**
 * The canvas width assumed where there is nothing to measure — a container
 * that is not laid out, which in practice means jsdom. It only has to make
 * the pointer arithmetic defined; a real browser measures.
 */
const NOMINAL_WIDTH = 1200;

/** A panel the workspace owns: what it is called, and what it can be. */
export interface PanelSpec {
  id: string;
  title: string;
  /** The body's markup, mounted once and then owned by the caller. */
  body: string;
  /** Where it sits before anyone moves it, and what it may be resized to. */
  rect: Rect;
  limits: Limits;
}

export interface Workspace {
  /** The section itself, for anything that decorates a panel (help popups). */
  panel(id: string): HTMLElement;
  /** Where a view mounts: everything below the title bar. */
  body(id: string): HTMLElement;
  /** For a panel whose name is part of what it shows, like the neuron. */
  setTitle(id: string, title: string): void;
  /**
   * Give every expanded panel the height its own content asks for.
   *
   * A default height is a guess about a window nobody has seen, and the
   * plots scale with their width: the same panel is taller on a wide
   * screen. Measuring once the views have rendered is the only way the
   * figure opens without a scrollbar in it.
   */
  fit(): void;
  /** Back to every panel expanded and where it shipped, fitted to content. */
  reset(): void;
  /** Whether the arrangement came from storage rather than the defaults. */
  isRestored(): boolean;
  /** The current arrangement, for tests and for anything that saves it. */
  layout(): Layout;
  destroy(): void;
}

export interface WorkspaceOptions {
  container: HTMLElement;
  panels: PanelSpec[];
  /**
   * Where the arrangement is kept. `localStorage` normally; null turns
   * persistence off, which is what a test wants unless it is testing it.
   */
  storage?: Storage | null;
  storageKey?: string;
  /** Called after any change the reader made, for a status line. */
  onChange?: (layout: Layout) => void;
  /** Only used where the container has no width to measure. */
  nominalWidth?: number;
}

export const LAYOUT_STORAGE_KEY = "nnintro.layout";

export function createWorkspace(options: WorkspaceOptions): Workspace {
  const { container, panels: specs } = options;
  const storageKey = options.storageKey ?? LAYOUT_STORAGE_KEY;
  const storage = options.storage === undefined ? safeStorage() : options.storage;
  const ids = specs.map((spec) => spec.id);
  const limits = new Map(specs.map((spec) => [spec.id, spec.limits]));

  container.classList.add("workspace");
  container.style.setProperty("--panel-bar", `${BAR_HEIGHT}px`);

  const sections = new Map<string, HTMLElement>();
  const bodies = new Map<string, HTMLElement>();
  const titles = new Map<string, HTMLElement>();
  const folds = new Map<string, HTMLButtonElement>();

  for (const spec of specs) {
    const section = document.createElement("section");
    section.className = `panel panel-${spec.id}`;
    section.dataset.panel = spec.id;

    const bar = document.createElement("div");
    bar.className = "panel-bar";
    // The bar is the handle, so it is also the keyboard's: a panel can be
    // moved and resized without a pointer at all.
    bar.tabIndex = 0;
    bar.setAttribute("role", "group");
    bar.setAttribute(
      "aria-label",
      `${spec.title} panel. Arrow keys move it, shift and arrow keys resize it.`,
    );

    const title = document.createElement("h2");
    title.textContent = spec.title;

    const fold = document.createElement("button");
    fold.type = "button";
    fold.className = "panel-fold";
    fold.setAttribute("aria-expanded", "true");

    bar.append(title, fold);

    const body = document.createElement("div");
    body.className = "panel-body";
    body.innerHTML = spec.body;

    const grip = document.createElement("div");
    grip.className = "panel-grip";
    grip.setAttribute("aria-hidden", "true");

    section.append(bar, body, grip);
    container.append(section);

    sections.set(spec.id, section);
    bodies.set(spec.id, body);
    titles.set(spec.id, title);
    folds.set(spec.id, fold);

    bar.addEventListener("pointerdown", (event) => startDrag(spec.id, event));
    bar.addEventListener("keydown", (event) => onKey(spec.id, event));
    bar.addEventListener("dblclick", (event) => {
      if ((event.target as Element).closest("button")) return;
      toggle(spec.id);
    });
    fold.addEventListener("click", () => toggle(spec.id));
    grip.addEventListener("pointerdown", (event) => startResize(spec.id, event));
  }

  const restored = parseLayout(read(), ids);
  let current = restored ?? defaultLayout(specs.map(({ id, rect }) => ({ id, rect })));
  apply(current);

  /** A gesture in progress: what it started from, so moves never drift. */
  interface Gesture {
    id: string;
    from: Layout;
    rect: Rect;
    x: number;
    y: number;
  }
  let gesture: Gesture | null = null;

  function columnWidth(): number {
    const width = container.getBoundingClientRect().width;
    return (width || options.nominalWidth || NOMINAL_WIDTH) / COLUMNS;
  }

  function apply(layout: Layout): void {
    current = layout;
    for (const panel of layout) {
      const section = sections.get(panel.id);
      if (!section) continue;
      const rows = heightOf(panel);
      section.style.left = `${((panel.x / COLUMNS) * 100).toFixed(4)}%`;
      section.style.width = `calc(${((panel.w / COLUMNS) * 100).toFixed(4)}% - ${GAP}px)`;
      section.style.top = `${panel.y * ROW_HEIGHT}px`;
      section.style.height = `${rows * ROW_HEIGHT - GAP}px`;
      section.classList.toggle("is-minimised", panel.minimised);
      const fold = folds.get(panel.id)!;
      fold.setAttribute("aria-expanded", String(!panel.minimised));
      fold.textContent = panel.minimised ? "▸" : "▾";
      const name = titles.get(panel.id)?.textContent ?? panel.id;
      fold.setAttribute("aria-label", `${panel.minimised ? "Expand" : "Minimise"} ${name}`);
      fold.title = fold.getAttribute("aria-label")!;
    }
    // The canvas is as tall as what is on it: panels are positioned, so
    // nothing else can give the page its height.
    container.style.height = `${rowsUsed(layout) * ROW_HEIGHT}px`;
  }

  function settle(layout: Layout): void {
    apply(layout);
    save();
    options.onChange?.(layout);
  }

  function toggle(id: string): void {
    const panel = find(current, id);
    if (!panel) return;
    settle(setMinimised(current, id, !panel.minimised));
  }

  function startDrag(id: string, event: PointerEvent): void {
    if ((event.target as Element)?.closest("button")) return;
    begin(id, event, "move");
  }

  function startResize(id: string, event: PointerEvent): void {
    const panel = find(current, id);
    if (!panel || panel.minimised) return;
    begin(id, event, "resize");
  }

  function begin(id: string, event: PointerEvent, kind: "move" | "resize"): void {
    const panel = find(current, id);
    if (!panel) return;
    event.preventDefault();
    gesture = {
      id,
      from: current,
      rect: { x: panel.x, y: panel.y, w: panel.w, h: panel.h },
      x: event.clientX,
      y: event.clientY,
    };
    sections.get(id)?.classList.add(kind === "move" ? "is-dragging" : "is-resizing");
    container.classList.add("is-arranging");
    const onMove = kind === "move" ? dragTo : resizeBy;
    // On the document rather than on the panel: the pointer leaves a small
    // panel immediately, and a capture that a synthetic event cannot take
    // would make this untestable.
    document.addEventListener("pointermove", onMove);
    document.addEventListener("pointerup", finish, { once: true });

    function finish(): void {
      document.removeEventListener("pointermove", onMove);
      sections.get(id)?.classList.remove("is-dragging", "is-resizing");
      container.classList.remove("is-arranging");
      gesture = null;
      save();
      options.onChange?.(current);
    }
  }

  /**
   * Every move is computed from where the gesture began, never from the
   * last frame: the layout under the pointer is settled as it goes, and
   * accumulating deltas against a moving layout would wander.
   */
  function dragTo(event: PointerEvent): void {
    if (!gesture) return;
    const columns = Math.round((event.clientX - gesture.x) / columnWidth());
    const rows = Math.round((event.clientY - gesture.y) / ROW_HEIGHT);
    apply(moveTo(gesture.from, gesture.id, gesture.rect.x + columns, gesture.rect.y + rows));
  }

  function resizeBy(event: PointerEvent): void {
    if (!gesture) return;
    const columns = Math.round((event.clientX - gesture.x) / columnWidth());
    const rows = Math.round((event.clientY - gesture.y) / ROW_HEIGHT);
    apply(
      resizeTo(
        gesture.from,
        gesture.id,
        { w: gesture.rect.w + columns, h: gesture.rect.h + rows },
        limits.get(gesture.id)!,
      ),
    );
  }

  const STEPS: Record<string, [number, number]> = {
    ArrowLeft: [-1, 0],
    ArrowRight: [1, 0],
    ArrowUp: [0, -1],
    ArrowDown: [0, 1],
  };

  /**
   * Arrows move the panel, shift and arrows resize it.
   *
   * Sideways is a column at a time, which is what a column is for.
   * Vertically it is past the neighbour rather than a row at a time: a
   * panel nudged one row down floats straight back, so a row step would
   * leave a reader holding the key against a layout that never moves
   * (`stepRow`). Resizing is a row at a time, where one row is exactly the
   * resolution the reader is choosing between.
   */
  function onKey(id: string, event: KeyboardEvent): void {
    const step = STEPS[event.key];
    if (!step) return;
    const panel = find(current, id);
    if (!panel) return;
    event.preventDefault();
    const [dx, dy] = step;
    if (event.shiftKey) {
      settle(
        resizeTo(current, id, { w: panel.w + dx, h: panel.h + dy }, limits.get(id)!),
      );
      return;
    }
    settle(dy === 0 ? moveTo(current, id, panel.x + dx, panel.y) : stepRow(current, id, dy as 1 | -1));
  }

  /**
   * The rows one panel's content needs: its bar, its body, and the gap that
   * every box gives back to the canvas.
   *
   * A body with nothing laid out measures zero — jsdom, or a panel in a
   * container that is not displayed — and a panel is never fitted to
   * nothing, so that case is left alone.
   */
  function rowsForContent(id: string): number | null {
    const body = bodies.get(id);
    if (!body) return null;
    const content = body.scrollHeight;
    if (!content) return null;
    return Math.ceil((BAR_HEIGHT + content + GAP) / ROW_HEIGHT);
  }

  /**
   * Measuring has to happen at the size the panel is *going* to be.
   *
   * Panels slide to their new places, and a box in the middle of a 120ms
   * width transition measures at the width it is passing through: fitting
   * against that gives every panel the height it needed a moment ago. The
   * transition is suppressed, the new geometry forced through layout, and
   * only then is the content measured.
   */
  function settled(work: () => void): void {
    container.classList.add("is-settling");
    // Reading a layout property is what makes the pending styles real.
    void container.offsetHeight;
    work();
    void container.offsetHeight;
    container.classList.remove("is-settling");
  }

  function fitAll(): void {
    settled(() => {
      let layout = current;
      for (const panel of current) {
        if (panel.minimised) continue;
        const rows = rowsForContent(panel.id);
        if (rows === null) continue;
        layout = resizeTo(layout, panel.id, { w: panel.w, h: rows }, limits.get(panel.id)!);
      }
      apply(layout);
    });
  }

  function read(): string | null {
    try {
      return storage?.getItem(storageKey) ?? null;
    } catch {
      return null;
    }
  }

  function save(): void {
    try {
      storage?.setItem(storageKey, serialise(current));
    } catch {
      // A layout nobody can store is still a layout that works.
    }
  }

  return {
    panel: (id) => sections.get(id)!,
    body: (id) => bodies.get(id)!,
    setTitle(id, title) {
      const element = titles.get(id);
      if (element) element.textContent = title;
      // The fold button's label names the panel, so it follows the title.
      apply(current);
    },
    fit() {
      fitAll();
      save();
    },
    reset() {
      try {
        storage?.removeItem(storageKey);
      } catch {
        // Nothing stored is exactly the state reset is aiming for.
      }
      settled(() => apply(defaultLayout(specs.map(({ id, rect }) => ({ id, rect })))));
      fitAll();
      options.onChange?.(current);
    },
    isRestored: () => restored !== null,
    layout: () => compact(current),
    destroy() {
      container.replaceChildren();
      container.classList.remove("workspace", "is-arranging");
      container.style.removeProperty("height");
    },
  };
}

/** `localStorage`, or nothing where a browser refuses to hand it over. */
function safeStorage(): Storage | null {
  try {
    return globalThis.localStorage ?? null;
  } catch {
    return null;
  }
}
