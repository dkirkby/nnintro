/**
 * The one colour map, shared by every value the explorer paints.
 *
 * Inputs, hidden activations, the output, the prediction surface behind the
 * input space and the two axis colour bars all read on this map, so a colour
 * means the same number wherever it appears: the output node's fill is the
 * same colour as the surface under the probe, and an input node's fill is the
 * colour the axis bar shows at that coordinate. Weights and contributions use
 * the same map with their own limit, because they are a different quantity.
 *
 * Zero is the neutral surface colour and the map is linear in the value, so
 * sign is readable and "nearly nothing" looks like nearly nothing.
 *
 * Two renderings of the same map:
 *
 * - `css` mixes the theme's own custom properties, so a light/dark change
 *   repaints the DOM without any re-render (the rule set in `styles.css`).
 * - `paint` returns a concrete colour, because a canvas cannot resolve
 *   custom properties. The tokens are read once per scale, so a theme change
 *   reaches the canvas on the next repaint.
 */

/** A colour in OKLab, the space both renderings interpolate in. */
interface Lab {
  L: number;
  a: number;
  b: number;
}

/** The light-theme tokens, used when the stylesheet is not reachable. */
const FALLBACK: Record<Token, Lab> = {
  "--pos": fromOklch(55, 0.16, 250),
  "--neg": fromOklch(58, 0.17, 30),
  "--surface": fromOklch(99, 0.004, 250),
  "--nonfinite": fromOklch(60, 0.2, 330),
};

type Token = "--pos" | "--neg" | "--surface" | "--nonfinite";

/** Rounded limits, so probing does not make the colours shift underfoot. */
const LADDER = [1, 1.5, 2, 3, 5, 7.5];

export interface ColorScale {
  /** `±limit` is full saturation; beyond it the colour clamps. */
  readonly limit: number;
  /** Theme-reactive colour for the DOM. */
  css(value: number): string;
  /** Concrete colour for a canvas. */
  paint(value: number): string;
  /**
   * `steps` colour stops spanning `[from, to]`, for a CSS gradient. The axis
   * bars span the input domain rather than the full limit, which is why the
   * range is a parameter.
   */
  stops(from: number, to: number, steps?: number): string[];
}

/**
 * The smallest round number that covers every value given, so the limit (and
 * therefore every colour) only moves when the data really outgrows it.
 */
export function niceLimit(...groups: Iterable<number>[]): number {
  let largest = 0;
  for (const group of groups) {
    for (const value of group) {
      if (!Number.isFinite(value)) continue;
      const magnitude = Math.abs(value);
      if (magnitude > largest) largest = magnitude;
    }
  }
  if (!(largest > 0)) return 1;
  const decade = 10 ** Math.floor(Math.log10(largest));
  for (const step of LADDER) {
    // The slack absorbs the rounding in the decade itself, so a value that
    // is exactly a ladder stop does not push the limit up a notch.
    if (step * decade >= largest * (1 - 1e-12)) return step * decade;
  }
  return 10 * decade;
}

export function createColorScale(limit: number, root?: Element): ColorScale {
  const tokens = resolveTokens(root);
  const span = Math.abs(limit) || 1;
  const fraction = (value: number) => Math.min(1, Math.abs(value) / span);

  function css(value: number): string {
    if (!Number.isFinite(value)) return "var(--nonfinite)";
    const token = value >= 0 ? "--pos" : "--neg";
    const percent = (100 * fraction(value)).toFixed(1);
    return `color-mix(in oklab, var(${token}) ${percent}%, var(--surface))`;
  }

  function paint(value: number): string {
    if (!Number.isFinite(value)) return format(tokens["--nonfinite"]);
    const ink = tokens[value >= 0 ? "--pos" : "--neg"];
    return format(mix(ink, tokens["--surface"], fraction(value)));
  }

  return {
    limit: span,
    css,
    paint,
    stops(from, to, steps = 17) {
      const list: string[] = [];
      for (let i = 0; i < steps; i += 1) {
        list.push(css(from + ((to - from) * i) / (steps - 1)));
      }
      return list;
    },
  };
}

/** `color-mix(in oklab, ink <weight>, base)`, done by hand for the canvas. */
function mix(ink: Lab, base: Lab, weight: number): Lab {
  return {
    L: base.L + (ink.L - base.L) * weight,
    a: base.a + (ink.a - base.a) * weight,
    b: base.b + (ink.b - base.b) * weight,
  };
}

function format(color: Lab): string {
  return `oklab(${color.L.toFixed(2)}% ${color.a.toFixed(4)} ${color.b.toFixed(4)})`;
}

function fromOklch(lightness: number, chroma: number, hue: number): Lab {
  const radians = (hue * Math.PI) / 180;
  return {
    L: lightness,
    a: chroma * Math.cos(radians),
    b: chroma * Math.sin(radians),
  };
}

function resolveTokens(root?: Element): Record<Token, Lab> {
  const element = root ?? globalThis.document?.documentElement;
  if (!element) return FALLBACK;
  let computed: CSSStyleDeclaration;
  try {
    computed = getComputedStyle(element);
  } catch {
    return FALLBACK;
  }
  const resolved = { ...FALLBACK };
  for (const token of Object.keys(FALLBACK) as Token[]) {
    const parsed = parseOklch(computed.getPropertyValue(token));
    if (parsed) resolved[token] = parsed;
  }
  return resolved;
}

/**
 * The tokens are authored as `oklch(L% C H)`, and a custom property's
 * computed value is the text as written, so this parses exactly that form and
 * falls back rather than guessing at anything else.
 */
function parseOklch(text: string): Lab | null {
  const match = /^\s*oklch\(\s*([\d.]+)%\s+([\d.]+)\s+([\d.]+)\s*\)\s*$/.exec(text);
  if (!match) return null;
  return fromOklch(Number(match[1]), Number(match[2]), Number(match[3]));
}
