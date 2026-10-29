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
import { sameParameter, type ParameterRef } from "../model/types.js";
import { createColorScale, niceLimit, type ColorScale, type Palette } from "./colormap.js";
import {
  edgeContribution,
  edgeKey,
  edgeParameter,
  edgeWeight,
  nodeBias,
  nodeKey,
  nodeParameter,
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
  "A dotted grey edge is too small for the scale to show.";

export interface NetworkViewOptions {
  container: HTMLElement;
  layerSizes: number[];
  onSelect: (ref: ParameterRef | null) => void;
  /** Scoped per instance so several explorers can share a page. */
  idPrefix: string;
}

export interface NetworkView {
  /** The three scales are built once per render by the caller. */
  render(snapshot: Snapshot, mode: DisplayMode, palette: Palette): void;
  svg(): SVGSVGElement;
  destroy(): void;
}

const VIEW_BOX = { width: 680, height: 380 };
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

export function createNetworkView(options: NetworkViewOptions): NetworkView {
  const { container, layerSizes, onSelect, idPrefix } = options;
  // Pure geometry, no DOM: the selection outline and the click targets are
  // drawn from the same layout the renderer uses.
  const layout = layoutDenseNetwork({ layers: layerSizes, viewBox: VIEW_BOX });
  let root: SVGSVGElement | null = null;

  const lastLayer = layerSizes.length - 1;

  function render(snapshot: Snapshot, mode: DisplayMode, palette: Palette): void {
    const { parameters, trace, selection } = snapshot;
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
      viewBox: VIEW_BOX,
      idPrefix,
      classPrefix: "dn",
      title: "Dense neural network",
      desc:
        `A ${layerSizes.join("-")} network. Select an edge to edit its weight, ` +
        "or a node to edit its bias.",
      // Always the activation: the bias is the colour, not the number.
      nodeValue: (node: NodeRef) => trace.act[node.layerIndex]?.[node.nodeIndex],
      edgeValue: (edge: EdgeRef) =>
        mode === "contributions"
          ? edgeContribution(parameters, trace.act, edge)
          : edgeWeight(parameters, edge),
      nodeAttrs: (node) => {
        const ref = nodeParameter(node);
        const selected = ref && selection && sameParameter(ref, selection);
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
        const ref = edgeParameter(edge);
        const selected = selection && sameParameter(ref, selection);
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

    svg.addEventListener("click", handleActivate);
    svg.addEventListener("keydown", handleKey);
    decorate(svg, snapshot, mode, mode === "contributions" ? contributionScale : palette.param);

    container.replaceChildren(svg);
    root = svg;
  }

  /**
   * Two layers the renderer does not provide: the selected edge's outline,
   * under the edges so the weight's own colour stays on top, and a wide
   * transparent line over each edge so a hairline is still clickable. The
   * targets go between the edges and the nodes, or they would swallow
   * clicks meant for a node.
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
      const ref = edgeParameter(edge);
      const value =
        mode === "contributions"
          ? edgeContribution(parameters, trace.act, edge)
          : edgeWeight(parameters, edge);
      const faint = Math.abs(value ?? 0) < FAINT_FRACTION * scale.limit;

      if (selection && sameParameter(ref, selection)) {
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

  function refFromTarget(target: EventTarget | null): ParameterRef | null | undefined {
    if (!(target instanceof Element)) return undefined;
    const key = target.closest("[data-key]")?.getAttribute("data-key");
    if (!key) return undefined;
    const [kind, a, b, c] = key.split(":");
    if (kind === "e") {
      return edgeParameter({
        sourceLayerIndex: Number(a),
        sourceNodeIndex: Number(b),
        targetNodeIndex: Number(c),
      });
    }
    return nodeParameter({ layerIndex: Number(a), nodeIndex: Number(b) });
  }

  function handleActivate(event: Event): void {
    const ref = refFromTarget(event.target);
    if (ref === undefined) return;
    onSelect(ref);
  }

  function handleKey(event: KeyboardEvent): void {
    if (event.key !== "Enter" && event.key !== " ") return;
    const ref = refFromTarget(event.target);
    if (ref === undefined) return;
    event.preventDefault();
    onSelect(ref);
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
