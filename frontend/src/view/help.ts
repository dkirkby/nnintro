/**
 * A panel's explanation, out of the way until it is asked for.
 *
 * Each panel keeps a `?` in its heading; clicking it opens that panel's
 * legend in a popup. The popup is a child of `document.body`, never of the
 * panel, and is placed **beside** the panel — to its right if there is room,
 * otherwise to its left, otherwise below it. It therefore never covers the
 * thing it explains, which is the whole point: the text is read against the
 * plot, not instead of it.
 *
 * The legend element is moved into the popup rather than copied, so whoever
 * writes its text — `main.ts` for two of the panels, `paramspace.ts` for its
 * own — keeps doing so and the popup is always current, open or not.
 */

/** Clear of the panel's border and of the viewport's edges. */
const MARGIN = 12;

export interface Help {
  /** Whether the popup is open, for tests and for the caller's own state. */
  isOpen(): boolean;
  close(): void;
  destroy(): void;
}

export function createHelp(panel: HTMLElement, body: HTMLElement, name: string): Help {
  const heading = panel.querySelector("h2");
  if (!heading) throw new Error(`panel "${name}" has no heading to hang help from`);

  const button = document.createElement("button");
  button.type = "button";
  button.className = "help-button";
  button.textContent = "?";
  button.setAttribute("aria-expanded", "false");
  button.setAttribute("aria-label", `About ${name}`);
  button.title = `About ${name}`;
  heading.append(button);
  heading.classList.add("has-help");

  const popup = document.createElement("div");
  popup.className = "help-popup";
  popup.setAttribute("role", "dialog");
  popup.setAttribute("aria-label", name);
  popup.hidden = true;

  const title = document.createElement("h3");
  title.textContent = name;

  const close = document.createElement("button");
  close.type = "button";
  close.className = "help-close";
  close.textContent = "×";
  close.setAttribute("aria-label", "Close");

  const head = document.createElement("div");
  head.className = "help-head";
  head.append(title, close);

  body.classList.add("help-body");
  popup.append(head, body);
  document.body.append(popup);

  /**
   * Beside the panel, anchored to the `?` so a tall panel's help still
   * opens next to the question rather than at its top. Fixed, so the
   * clamping below is against the viewport and nothing can push it off.
   */
  function place(): void {
    const area = panel.getBoundingClientRect();
    const anchor = button.getBoundingClientRect();
    const { width, height } = popup.getBoundingClientRect();
    const view = {
      width: globalThis.innerWidth || width,
      height: globalThis.innerHeight || height,
    };

    let left = area.right + MARGIN;
    let top = anchor.top;
    if (left + width > view.width - MARGIN) {
      const mirrored = area.left - MARGIN - width;
      if (mirrored >= MARGIN) {
        left = mirrored;
      } else {
        // No room either side: under the panel, which still leaves the
        // panel itself visible.
        left = area.left;
        top = area.bottom + MARGIN;
      }
    }
    popup.style.left = `${clamp(left, MARGIN, view.width - width - MARGIN)}px`;
    popup.style.top = `${clamp(top, MARGIN, view.height - height - MARGIN)}px`;
  }

  function open(): void {
    if (!popup.hidden) return;
    popup.hidden = false;
    button.setAttribute("aria-expanded", "true");
    place();
    // While it is open the popup has to follow the panel, which a scroll or
    // a resize moves out from under it.
    globalThis.addEventListener("scroll", place, true);
    globalThis.addEventListener("resize", place);
    document.addEventListener("pointerdown", onOutside, true);
    document.addEventListener("keydown", onKey);
    close.focus();
  }

  function shut(restoreFocus: boolean): void {
    if (popup.hidden) return;
    popup.hidden = true;
    button.setAttribute("aria-expanded", "false");
    globalThis.removeEventListener("scroll", place, true);
    globalThis.removeEventListener("resize", place);
    document.removeEventListener("pointerdown", onOutside, true);
    document.removeEventListener("keydown", onKey);
    if (restoreFocus) button.focus();
  }

  function onOutside(event: Event): void {
    const target = event.target;
    if (!(target instanceof Node)) return;
    if (popup.contains(target) || button.contains(target)) return;
    shut(false);
  }

  function onKey(event: KeyboardEvent): void {
    if (event.key === "Escape") shut(true);
  }

  function onToggle(): void {
    if (popup.hidden) open();
    else shut(true);
  }

  button.addEventListener("click", onToggle);
  close.addEventListener("click", () => shut(true));

  return {
    isOpen: () => !popup.hidden,
    close: () => shut(false),
    destroy() {
      shut(false);
      button.removeEventListener("click", onToggle);
      button.remove();
      popup.remove();
    },
  };
}

function clamp(value: number, low: number, high: number): number {
  return Math.max(low, Math.min(Math.max(low, high), value));
}
