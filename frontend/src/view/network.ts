/**
 * The network panel: an nnet-svg rendering plus the overlays the explorer
 * owns (bias badges, selection, and the ripple layer).
 *
 * All index translation goes through `./adapter.js`. This module owns the
 * DOM and nothing mathematical: it reads a snapshot and draws it.
 */

import { createDenseNetworkSvg, layoutDenseNetwork } from "nnet-svg";
import type { EdgeRef, NodeRef } from "nnet-svg";

import type { Snapshot } from "../state.js";
import { sameUnit, type UnitRef } from "../model/types.js";
import { createColorScale, niceLimit, type ColorScale, type Palette } from "./colormap.js";
import {
  edgeContribution,
  edgeKey,
  edgeUnit,
  edgeWeight,
  nodeBias,
  nodeKey,
  nodeParameter,
  nodeUnit,
} from "./adapter.js";

/**
 * What the edges show. Node fills do not vary with it: a node is always
 * filled by what it owns -- an input by its value, a hidden unit by its
 * bias, the output by the prediction -- each on its own family's map (see
 * `./colormap.ts`). The number on a node is always its activation, so a
 * hidden unit shows its bias as colour and its activation as text.
 */
export type DisplayMode = "weights" | "contributions";

export const DISPLAY_MODES: { id: DisplayMode; label: string; legend: string }[] = [
  {
    id: "weights",
    label: "Weights",
    legend: "Edge colour and width: weight w.",
  },
  {
    id: "contributions",
    label: "Contributions",
    legend: "Edge colour and width: the contribution w·a for this input.",
  },
];

/** The half of the legend that does not change with the mode. */
export const NODE_LEGEND =
  "Node fill: a hidden unit's bias, or ŷ at the output — two different " +
  "colour maps; inputs are unfilled. Node text is always the activation. " +
  "A dotted grey edge is too small for the scale to show. With the samples " +
  "shown and the probe on one of them, the output scale stands up in the " +
  "gutter on the right — the same red/blue scale the map and the slices " +
  "use — marked with the prediction on its left and that sample's target " +
  "on its right, so the two numbers the loss is made of are read off one " +
  "axis. On a classifier the target is a class label rather than a logit, " +
  "so its marker sits at the end of the scale its class wants, exactly as " +
  "a sample is drawn on the map.";

export interface NetworkViewOptions {
  container: HTMLElement;
  layerSizes: number[];
  /**
   * The unit the click landed on: a node's own unit, or the unit an edge
   * feeds. Null for an input node, which owns no parameters — the one click
   * that selects nothing, since there is no way to deselect.
   */
  onSelect: (unit: UnitRef | null) => void;
  /** Scoped per instance so several explorers can share a page. */
  idPrefix: string;
}

export interface NetworkView {
  /**
   * The three scales are built once per render by the caller. `showSamples`
   * is the reader's display preference, which the output bar is gated on
   * along with the probe actually being a sample: see `outputBar`.
   */
  render(
    snapshot: Snapshot,
    mode: DisplayMode,
    palette: Palette,
    showSamples: boolean,
  ): void;
  svg(): SVGSVGElement;
  destroy(): void;
}

/**
 * The frame, and the part of it the network is laid out in.
 *
 * The network gets everything but a gutter down the right, which holds the
 * output bar. The gutter is reserved whether or not the bar is drawn: a
 * frame that grew when a sample was selected would resize the whole diagram
 * at the moment the reader was looking at one node of it.
 */
const VIEW_BOX = { width: 680, height: 380 };
const BAR_GUTTER = 56;
const NETWORK_BOX = { width: VIEW_BOX.width - BAR_GUTTER, height: VIEW_BOX.height };
const SVG = "http://www.w3.org/2000/svg";

/**
 * Below this fraction of the scale an edge's colour is indistinguishable
 * from the panel, so it is drawn as a dotted grey line instead of a colour
 * nobody can see. A zero weight is the common case and still has to be
 * visible, and selectable.
 */
const FAINT_FRACTION = 0.05;

/**
 * Thin at nothing, thick at the limit. The floor is generous because the
 * view box is wider than the panel that renders it, so a stroke of 1 unit
 * lands under a pixel: a small weight has to be a readable band, not a
 * hairline in a colour that is already pale.
 */
function strokeWidth(value: number, limit: number): number {
  return 2.2 + 4.8 * Math.min(1, Math.abs(value) / limit);
}

/** The rails drawn along a selected edge, one each side. */
const OUTLINE_WIDTH = 1;

/**
 * The output bar, in the frame's own units: the colour strip itself, then
 * how far a marker reaches past it, then where the rotated label sits.
 *
 * Its middle is the frame's middle, which is also the output node's own
 * row, so the zero of the scale is level with the node whose value it
 * reads. The strip is drawn as a stack of bands rather than as an SVG
 * gradient, because each band can then be a `color-mix` of the theme's own
 * tokens -- the same colour the node and the surface get -- and a theme
 * change repaints it with no re-render.
 */
const BAR = {
  x: NETWORK_BOX.width + 12,
  width: 12,
  top: 40,
  height: 300,
  bands: 32,
};
const MARKER = 9;
/** Half the height of a marker's base: a slim triangle, not an arrowhead. */
const MARKER_HALF = 5;
const BAR_LABEL_X = BAR.x + BAR.width + MARKER + 9;
/**
 * Half the rotated label's own length, which is how far it may be pulled
 * back from an end of the bar.
 *
 * A word centred on the very top or bottom row would hang half its length
 * past the scale it names, into the corner of the panel — and that is not
 * the rare case: a classifier's target is always at one end or the other.
 */
const BAR_LABEL_HALF = 20;

export function createNetworkView(options: NetworkViewOptions): NetworkView {
  const { container, layerSizes, onSelect, idPrefix } = options;
  // Pure geometry, no DOM: the selection outline and the click targets are
  // drawn from the same layout the renderer uses.
  const layout = layoutDenseNetwork({ layers: layerSizes, viewBox: NETWORK_BOX });
  let root: SVGSVGElement | null = null;

  const lastLayer = layerSizes.length - 1;

  function render(
    snapshot: Snapshot,
    mode: DisplayMode,
    palette: Palette,
    showSamples: boolean,
  ): void {
    const { parameters, trace, selection } = snapshot;
    /** One unit is selected: its node, and every edge feeding that node. */
    const isSelectedUnit = (unit: UnitRef | null) =>
      unit !== null && sameUnit(unit, selection);
    // Contributions are still a parameter-family quantity -- an edge is an
    // edge -- but they need their own limit, since w·a is not w.
    const contributionScale = createColorScale(
      niceLimit(
        parameters.flatMap((layer, l) => layer.W.flatMap((row) => row.map((w, i) => w * trace.act[l][i]))),
      ),
      "param",
    );

    /** What a node owns, which is not what it displays. */
    function nodeFill(node: NodeRef): string {
      // An input owns nothing the network chose: its value is a coordinate,
      // and the plot axes carry those. It is left unfilled.
      if (node.layerIndex === 0) return "var(--surface)";
      if (node.layerIndex === lastLayer) {
        return palette.output.css(trace.act[lastLayer][node.nodeIndex]);
      }
      const bias = nodeBias(parameters, node);
      return bias === null ? "var(--surface)" : palette.param.css(bias);
    }

    const svg = createDenseNetworkSvg({
      layers: layerSizes,
      viewBox: NETWORK_BOX,
      idPrefix,
      classPrefix: "dn",
      title: "Dense neural network",
      desc:
        `A ${layerSizes.join("-")} network. Select a node, or any edge into ` +
        "it, to work on that unit: its bias and every weight feeding it.",
      // Always the activation: the bias is the colour, not the number.
      nodeValue: (node: NodeRef) => trace.act[node.layerIndex]?.[node.nodeIndex],
      edgeValue: (edge: EdgeRef) =>
        mode === "contributions"
          ? edgeContribution(parameters, trace.act, edge)
          : edgeWeight(parameters, edge),
      nodeAttrs: (node) => {
        const ref = nodeParameter(node);
        const selected = isSelectedUnit(nodeUnit(node));
        return {
          fill: nodeFill(node),
          class: selected ? "is-selected" : "",
          "data-key": nodeKey(node),
          tabindex: ref ? "0" : null,
          role: ref ? "button" : null,
          "aria-label": ref
            ? `layer ${node.layerIndex} unit ${node.nodeIndex}, activation ` +
              `${format(node.value)}, bias ${format(nodeBias(parameters, node))}`
            : `input ${node.nodeIndex}, ${format(node.value)}`,
        };
      },
      edgeAttrs: (edge) => {
        const selected = isSelectedUnit(edgeUnit(edge));
        const scale = mode === "contributions" ? contributionScale : palette.param;
        const value = edge.value ?? 0;
        const faint = Math.abs(value) < FAINT_FRACTION * scale.limit;
        return {
          // A faint edge's colour is overridden in CSS by `.is-faint`, which
          // a presentation attribute cannot outrank.
          stroke: scale.css(value),
          "stroke-width": strokeWidth(value, scale.limit).toFixed(2),
          class: [selected ? "is-selected" : "", faint ? "is-faint" : ""]
            .filter(Boolean)
            .join(" "),
          "data-key": edgeKey(edge),
          tabindex: "0",
          role: "button",
          "aria-label":
            `weight from layer ${edge.sourceLayerIndex} unit ${edge.sourceNodeIndex} ` +
            `to unit ${edge.targetNodeIndex}, ${format(edge.value)}`,
        };
      },
      // A visible badge, not a tooltip: biases are part of the computation
      // and hiding them behind a hover makes them look optional.
      nodeLabel: (node) => (node.value === undefined ? null : format(node.value)),
      nodeLabelAttrs: () => ({ class: "dn-node-label" }),
      edgeTitle: (edge) =>
        `${mode === "contributions" ? "contribution" : "weight"} ${format(edge.value)}`,
      nodeTitle: (node) =>
        node.layerIndex === 0
          ? `input ${format(node.value)}`
          : `activation ${format(node.value)}, bias ${format(nodeBias(parameters, node))}`,
    });

    // The network was laid out in `NETWORK_BOX`; the frame it is drawn in
    // is wider by the bar's gutter, which nothing is laid out in.
    svg.setAttribute(
      "viewBox",
      `0 0 ${VIEW_BOX.width} ${VIEW_BOX.height}`,
    );

    svg.addEventListener("click", handleActivate);
    svg.addEventListener("keydown", handleKey);
    decorate(svg, snapshot, mode, mode === "contributions" ? contributionScale : palette.param);
    outputBar(svg, snapshot, palette.output, showSamples);

    container.replaceChildren(svg);
    root = svg;
  }

  /**
   * Two layers the renderer does not provide: rails alongside each selected
   * edge — every edge into the selected unit — under the edges so each
   * weight's own colour stays on top, and a wide transparent line over each
   * edge so a hairline is still clickable. The targets go between the edges
   * and the nodes, or they would swallow clicks meant for a node.
   */
  function decorate(
    svg: SVGSVGElement,
    snapshot: Snapshot,
    mode: DisplayMode,
    scale: ColorScale,
  ): void {
    const edges = svg.querySelector(".dn-edges");
    if (!edges) return;
    const { selection, parameters, trace } = snapshot;

    const outlines = document.createElementNS(SVG, "g");
    outlines.setAttribute("class", "selection-outline");
    outlines.setAttribute("aria-hidden", "true");
    const targets = document.createElementNS(SVG, "g");
    targets.setAttribute("class", "edge-targets");
    targets.setAttribute("aria-hidden", "true");

    for (const edge of layout.edges) {
      const value =
        mode === "contributions"
          ? edgeContribution(parameters, trace.act, edge)
          : edgeWeight(parameters, edge);
      const faint = Math.abs(value ?? 0) < FAINT_FRACTION * scale.limit;

      if (sameUnit(edgeUnit(edge), selection)) {
        // Two thin rails alongside, rather than one wider line underneath:
        // the edge keeps its own colour and thickness, and the selection
        // reads as an outline instead of a darkened edge.
        const offset = (strokeWidth(value ?? 0, scale.limit) + OUTLINE_WIDTH) / 2;
        for (const rail of rails(edge, offset)) {
          rail.setAttribute("class", faint ? "edge-outline is-faint" : "edge-outline");
          outlines.append(rail);
        }
      }

      const target = document.createElementNS(SVG, "line");
      setLine(target, edge);
      target.setAttribute("class", "edge-target");
      target.setAttribute("data-key", edgeKey(edge));
      const title = document.createElementNS(SVG, "title");
      title.textContent =
        `${mode === "contributions" ? "contribution" : "weight"} ${format(value)}`;
      target.append(title);
      targets.append(target);
    }

    svg.insertBefore(outlines, edges);
    edges.after(targets);
  }

  /**
   * The output scale stood up in the right-hand gutter, with the prediction
   * marked on its left and the probed sample's target on its right.
   *
   * The output node is already the prediction -- its fill and its badge --
   * and the bar is the axis that fill came off, so the two numbers the loss
   * is made of are a pair of marks on one scale rather than two printed
   * numbers the reader has to relate. Which is also why the bar is this
   * panel's and not the inputs panel's: the comparison belongs where the
   * prediction is produced.
   *
   * Drawn only when the samples are visible AND the probe is on one of
   * them, which is both of the conditions the reader can see. A target
   * exists for a free probe too on a generator with an analytic one, but
   * marking it would put a number on the bar with nothing on screen to say
   * where it came from; a sample's target is a dot the reader put the probe
   * on. Samples hidden is the same argument from the other side: the dot
   * the mark refers to would not be drawn.
   *
   * A classifier's target is a class label rather than a point on the logit
   * axis the bar spans, so its marker goes to the end of the scale its class
   * wants -- exactly the convention the map uses for the same sample
   * (`view/inputoutput.ts`), and the reason the limit itself is computed
   * without the targets (PLAN.md section 3).
   */
  function outputBar(
    svg: SVGSVGElement,
    snapshot: Snapshot,
    scale: ColorScale,
    showSamples: boolean,
  ): void {
    const { target, probe, probability, trace } = snapshot;
    if (!showSamples || probe.kind !== "sample" || target === null) return;
    const output = trace.act[lastLayer][0];
    const isClassifier = probability !== null;
    // The class label is not on this axis; the end of the scale its class
    // wants is where the map draws the same sample.
    const targetMark = isClassifier
      ? (target > 0.5 ? scale.limit : -scale.limit)
      : target;

    const group = document.createElementNS(SVG, "g");
    group.setAttribute("class", "output-bar");
    group.setAttribute("role", "img");
    group.setAttribute(
      "aria-label",
      `output scale from −${scale.limit} to ${scale.limit}: ` +
        `prediction ${format(output)}, ` +
        (isClassifier
          ? `target class ${format(target)}, marked at the end of the scale`
          : `target ${format(target)}`),
    );

    // Bottom to top, so the band a value lands in is the colour the value
    // itself gets: the same orientation as the slices, where the output
    // grows upward.
    const band = BAR.height / BAR.bands;
    for (let index = 0; index < BAR.bands; index += 1) {
      const centre = scale.limit * (1 - (2 * (index + 0.5)) / BAR.bands);
      const rect = document.createElementNS(SVG, "rect");
      rect.setAttribute("x", String(BAR.x));
      rect.setAttribute("y", (BAR.top + index * band).toFixed(2));
      rect.setAttribute("width", String(BAR.width));
      // A hair of overlap, or the seams between bands show as pale lines.
      rect.setAttribute("height", (band + 0.5).toFixed(2));
      rect.setAttribute("fill", scale.css(centre));
      rect.setAttribute("class", "output-bar-band");
      group.append(rect);
    }
    // A frame, because the bands either side of zero are the panel's own
    // colour: without it the bar would look broken in the middle.
    const frame = document.createElementNS(SVG, "rect");
    frame.setAttribute("x", String(BAR.x));
    frame.setAttribute("y", String(BAR.top));
    frame.setAttribute("width", String(BAR.width));
    frame.setAttribute("height", String(BAR.height));
    frame.setAttribute("class", "output-bar-frame");
    group.append(frame);

    group.append(
      marker(barY(output, scale), "left", `prediction ${format(output)}`),
      marker(
        barY(targetMark, scale),
        "right",
        isClassifier ? `target class ${format(target)}` : `target ${format(target)}`,
      ),
    );
    // Rotated, like the tensor panel's row labels: a horizontal "target"
    // would need a gutter wider than the bar it labels.
    group.append(
      rotatedLabel("target", BAR_LABEL_X, labelRow(barY(targetMark, scale))),
    );
    svg.append(group);
  }

  /** The label's row: its marker's, kept far enough in to fit beside it. */
  function labelRow(y: number): number {
    return Math.min(
      BAR.top + BAR.height - BAR_LABEL_HALF,
      Math.max(BAR.top + BAR_LABEL_HALF, y),
    );
  }

  /** One marker: a triangle whose tip touches the bar at its own row. */
  function marker(y: number, side: "left" | "right", title: string): SVGElement {
    const tip = side === "left" ? BAR.x : BAR.x + BAR.width;
    const base = side === "left" ? tip - MARKER : tip + MARKER;
    const node = document.createElementNS(SVG, "polygon");
    node.setAttribute(
      "points",
      `${base.toFixed(2)},${(y - MARKER_HALF).toFixed(2)} ` +
        `${base.toFixed(2)},${(y + MARKER_HALF).toFixed(2)} ` +
        `${tip.toFixed(2)},${y.toFixed(2)}`,
    );
    node.setAttribute("class", `output-marker is-${side}`);
    const caption = document.createElementNS(SVG, "title");
    caption.textContent = title;
    node.append(caption);
    return node;
  }

  /**
   * Where a value sits on the bar, pinned to its ends beyond the scale.
   *
   * The scale is the dataset's own targets and a prediction is free to
   * leave them — `Randomize all` leaves them all at once — so a marker
   * can only be honest about which end a value ran past. The value itself
   * stays in the marker's title and on the node's badge.
   */
  function barY(value: number, scale: ColorScale): number {
    const fraction = (scale.limit - value) / (2 * scale.limit);
    return BAR.top + BAR.height * Math.min(1, Math.max(0, fraction));
  }

  /**
   * `undefined` for a click that missed everything, as distinct from `null`
   * for an input node, which is a hit on something that owns no parameters.
   */
  function unitFromTarget(target: EventTarget | null): UnitRef | null | undefined {
    if (!(target instanceof Element)) return undefined;
    const key = target.closest("[data-key]")?.getAttribute("data-key");
    if (!key) return undefined;
    const [kind, a, b, c] = key.split(":");
    if (kind === "e") {
      // The unit the edge feeds, not the edge: clicking one weight selects
      // the whole of the destination neuron's z = Σ wᵢaᵢ + b.
      return edgeUnit({ sourceLayerIndex: Number(a), targetNodeIndex: Number(c) });
    }
    return nodeUnit({ layerIndex: Number(a), nodeIndex: Number(b) });
  }

  function handleActivate(event: Event): void {
    const unit = unitFromTarget(event.target);
    if (unit === undefined) return;
    onSelect(unit);
  }

  function handleKey(event: KeyboardEvent): void {
    if (event.key !== "Enter" && event.key !== " ") return;
    const unit = unitFromTarget(event.target);
    if (unit === undefined) return;
    event.preventDefault();
    onSelect(unit);
  }

  return {
    render,
    svg: () => {
      if (!root) throw new Error("render() before using the svg");
      return root;
    },
    destroy() {
      root?.removeEventListener("click", handleActivate);
      root?.removeEventListener("keydown", handleKey);
      container.replaceChildren();
      root = null;
    },
  };
}

export function format(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  if (!Number.isFinite(value)) return Number.isNaN(value) ? "NaN" : value > 0 ? "+∞" : "−∞";
  if (value === 0) return "0";
  const magnitude = Math.abs(value);
  if (magnitude >= 1e4 || magnitude < 1e-3) return value.toExponential(2);
  return value.toFixed(magnitude >= 100 ? 1 : 3);
}

/**
 * Text turned on its side, centred on the row it labels.
 *
 * The same device as the tensor panel's row labels, for the same reason: a
 * horizontal word here would need a gutter wider than the thing it names.
 */
function rotatedLabel(text: string, x: number, y: number): SVGElement {
  const node = document.createElementNS(SVG, "text");
  node.setAttribute("x", x.toFixed(2));
  node.setAttribute("y", y.toFixed(2));
  node.setAttribute("text-anchor", "middle");
  node.setAttribute("transform", `rotate(-90 ${x.toFixed(2)} ${y.toFixed(2)})`);
  node.setAttribute("class", "output-bar-label");
  node.textContent = text;
  return node;
}

interface Segment {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

/** The segment shifted `offset` to either side, along its own normal. */
function rails(edge: Segment, offset: number): SVGElement[] {
  const dx = edge.x2 - edge.x1;
  const dy = edge.y2 - edge.y1;
  const length = Math.hypot(dx, dy) || 1;
  const nx = (-dy / length) * offset;
  const ny = (dx / length) * offset;
  return [1, -1].map((side) => {
    const line = document.createElementNS(SVG, "line");
    setLine(line, {
      x1: edge.x1 + nx * side,
      y1: edge.y1 + ny * side,
      x2: edge.x2 + nx * side,
      y2: edge.y2 + ny * side,
    });
    return line;
  });
}

function setLine(line: SVGElement, edge: Segment): void {
  line.setAttribute("x1", edge.x1.toFixed(2));
  line.setAttribute("y1", edge.y1.toFixed(2));
  line.setAttribute("x2", edge.x2.toFixed(2));
  line.setAttribute("y2", edge.y2.toFixed(2));
}
