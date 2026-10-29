/**
 * The network panel: an nnet-svg rendering plus the overlays the explorer
 * owns (bias badges, selection, and the ripple layer).
 *
 * All index translation goes through `./adapter.js`. This module owns the
 * DOM and nothing mathematical: it reads a snapshot and draws it.
 */

import { createDenseNetworkSvg } from "nnet-svg";
import type { EdgeRef, NodeRef } from "nnet-svg";

import type { Snapshot } from "../state.js";
import { sameParameter, type ParameterRef } from "../model/types.js";
import { createColorScale, niceLimit, type Palette } from "./colormap.js";
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
  "Node fill: an input's value, a hidden unit's bias, or ŷ — a different " +
  "colour map each. Node text is always the activation.";

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

export function createNetworkView(options: NetworkViewOptions): NetworkView {
  const { container, layerSizes, onSelect, idPrefix } = options;
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
      if (node.layerIndex === 0) {
        return palette.input.css(trace.act[0][node.nodeIndex]);
      }
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
        const magnitude = Math.min(1, Math.abs(edge.value ?? 0) / scale.limit);
        return {
          stroke: scale.css(edge.value ?? 0),
          // The floor is in the width, not the colour: a weight near zero
          // stays a visible hairline without being given a colour it has
          // not earned.
          "stroke-width": (1 + 3.25 * magnitude).toFixed(2),
          class: selected ? "is-selected" : "",
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

    container.replaceChildren(svg);
    root = svg;
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
