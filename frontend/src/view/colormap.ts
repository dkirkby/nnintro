/**
 * Three diverging colour maps, one per kind of quantity.
 *
 * A single map across inputs, parameters and outputs made every panel look
 * alike and invited comparisons between numbers that have nothing to do with
 * each other, so the families are now visually distinct:
 *
 * - **input** (brown/teal) — the two input coordinates: the input nodes and
 *   the colour bars along the two axes of the input-to-output map.
 * - **param** (magenta/green) — weights and biases: every edge, and the
 *   hidden nodes, which are filled by their bias while still showing their
 *   activation as a number.
 * - **output** (red/blue) — the prediction: the surface, the two slice
 *   backgrounds, the samples and the output node.
 *
 * Within a family a colour still means one number, so the output node is the
 * colour of the surface under the probe and an input node is the colour its
 * axis bar shows at that coordinate. Across families it means nothing, which
 * is the point.
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

/** Which quantity a scale paints; see the module comment. */
export type Family = "input" | "param" | "output";

/** The light-theme tokens, used when the stylesheet is not reachable. */
const FALLBACK: Record<string, Lab> = {
  "--input-pos": fromOklch(55, 0.1, 195),
  "--input-neg": fromOklch(50, 0.09, 70),
  "--param-pos": fromOklch(55, 0.15, 145),
  "--param-neg": fromOklch(55, 0.19, 345),
  "--output-pos": fromOklch(55, 0.16, 250),
  "--output-neg": fromOklch(58, 0.17, 30),
  "--surface": fromOklch(99, 0.004, 250),
  "--nonfinite": fromOklch(60, 0.2, 330),
};

/** Rounded limits, so probing does not make the colours shift underfoot. */
const LADDER = [1, 1.5, 2, 3, 5, 7.5];

export interface ColorScale {
  /** Which family's two inks this scale mixes. */
  readonly family: Family;
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

export function createColorScale(
  limit: number,
  family: Family,
  root?: Element,
): ColorScale {
  const inks = [`--${family}-neg`, `--${family}-pos`] as const;
  const tokens = resolveTokens([...inks, "--surface", "--nonfinite"], root);
  const span = Math.abs(limit) || 1;
  const fraction = (value: number) => Math.min(1, Math.abs(value) / span);

  function css(value: number): string {
    if (!Number.isFinite(value)) return "var(--nonfinite)";
    const token = inks[value >= 0 ? 1 : 0];
    const percent = (100 * fraction(value)).toFixed(1);
    return `color-mix(in oklab, var(${token}) ${percent}%, var(--surface))`;
  }

  function paint(value: number): string {
    if (!Number.isFinite(value)) return format(tokens["--nonfinite"]);
    const ink = tokens[inks[value >= 0 ? 1 : 0]];
    return format(mix(ink, tokens["--surface"], fraction(value)));
  }

  return {
    family,
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

function resolveTokens(
  wanted: readonly string[],
  root?: Element,
): Record<string, Lab> {
  const resolved = { ...FALLBACK };
  const element = root ?? globalThis.document?.documentElement;
  if (!element) return resolved;
  let computed: CSSStyleDeclaration;
  try {
    computed = getComputedStyle(element);
  } catch {
    return resolved;
  }
  for (const token of wanted) {
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

/** The three scales a render works with, built together in `main.ts`. */
export interface Palette {
  input: ColorScale;
  param: ColorScale;
  output: ColorScale;
}

/**
 * A CSS gradient across part of a scale, for the axis bars and the legend
 * strips. `direction` is a CSS gradient direction, and the stops run from
 * `from` to `to` in that direction.
 */
export function gradientBar(
  scale: ColorScale,
  from: number,
  to: number,
  direction: string,
): string {
  return `linear-gradient(${direction}, ${scale.stops(from, to).join(", ")})`;
}
