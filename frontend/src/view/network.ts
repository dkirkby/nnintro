/**
 * The network panel: an nnet-svg rendering plus the overlays the explorer
 * owns (bias badges, selection, and the ripple layer).
 *
 * All index translation goes through `./adapter.js`. This module owns the
 * DOM and nothing mathematical: it reads a snapshot and draws it.
 */

import { createDenseNetworkSvg, layoutDenseNetwork } from "nnet-svg";
import type { DenseNetworkLayout, EdgeRef, NodeRef } from "nnet-svg";

import type { Snapshot } from "../state.js";
import { sameParameter, type ParameterRef } from "../model/types.js";
import { createColorScale, niceLimit, type ColorScale } from "./colormap.js";
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
 * What the colours and widths mean right now.
 *
 * These are separate modes rather than one overloaded scale: a single
 * colour ramp carrying activation, weight, contribution and sensitivity at
 * once cannot be read (PLAN.md section 4).
 */
export type DisplayMode = "activations" | "weights" | "contributions";

/**
 * Colours come from the one shared map in `./colormap.ts`. Node fills in
 * "activations" mode use the scale the input space passes in, so a node and
 * the surface behind the probe are the same colour for the same number;
 * weights and contributions are a different quantity and get their own
 * limit on the same map.
 */

export const DISPLAY_MODES: { id: DisplayMode; label: string; legend: string }[] = [
  {
    id: "activations",
    label: "Activations",
    legend: "Node fill: activation a, on the input space's colour scale. Edge width: |w|.",
  },
  {
    id: "weights",
    label: "Weights",
    legend: "Edge colour and width: weight w. Node fill: bias b.",
  },
  {
    id: "contributions",
    label: "Contributions",
    legend: "Edge colour and width: the contribution w·a for this input.",
  },
];

export interface NetworkViewOptions {
  container: HTMLElement;
  layerSizes: number[];
  onSelect: (ref: ParameterRef | null) => void;
  /** Scoped per instance so several explorers can share a page. */
  idPrefix: string;
}

export interface NetworkView {
  /** `valueScale` is the shared activation scale, owned by the caller. */
  render(snapshot: Snapshot, mode: DisplayMode, valueScale: ColorScale): void;
  /** Geometry for the ripple overlay, in SVG user units. */
  layout(): DenseNetworkLayout;
  overlay(): SVGGElement;
  svg(): SVGSVGElement;
  destroy(): void;
}

const VIEW_BOX = { width: 680, height: 380 };

export function createNetworkView(options: NetworkViewOptions): NetworkView {
  const { container, layerSizes, onSelect, idPrefix } = options;
  const layout = layoutDenseNetwork({ layers: layerSizes, viewBox: VIEW_BOX });
  let root: SVGSVGElement | null = null;
  let overlayGroup: SVGGElement | null = null;

  function render(snapshot: Snapshot, mode: DisplayMode, valueScale: ColorScale): void {
    const { parameters, trace, selection } = snapshot;
    const allWeights = parameters.flatMap((layer) => layer.W.flat());
    const allBiases = parameters.flatMap((layer) => layer.b);
    // Biases share the weight limit: in "weights" mode the node fill and the
    // edge colour are then the same quantity on the same scale.
    const weightScale = createColorScale(niceLimit(allWeights, allBiases));
    const contributionScale = createColorScale(
      niceLimit(
        parameters.flatMap((layer, l) => layer.W.flatMap((row) => row.map((w, i) => w * trace.act[l][i]))),
      ),
    );

    const svg = createDenseNetworkSvg({
      layers: layerSizes,
      viewBox: VIEW_BOX,
      idPrefix,
      classPrefix: "dn",
      title: "Dense neural network",
      desc:
        `A ${layerSizes.join("-")} network. Select an edge to edit its weight, ` +
        "or a node to edit its bias.",
      nodeValue: (node: NodeRef) =>
        mode === "weights" ? nodeBias(parameters, node) : trace.act[node.layerIndex]?.[node.nodeIndex],
      edgeValue: (edge: EdgeRef) =>
        mode === "contributions"
          ? edgeContribution(parameters, trace.act, edge)
          : edgeWeight(parameters, edge),
      nodeAttrs: (node) => {
        const ref = nodeParameter(node);
        const selected = ref && selection && sameParameter(ref, selection);
        const scale = mode === "weights" ? weightScale : valueScale;
        return {
          fill: node.value === undefined ? "var(--surface)" : scale.css(node.value),
          class: selected ? "is-selected" : "",
          "data-key": nodeKey(node),
          tabindex: ref ? "0" : null,
          role: ref ? "button" : null,
          "aria-label": ref
            ? `bias of layer ${node.layerIndex} unit ${node.nodeIndex}, ${format(node.value)}`
            : `input ${node.nodeIndex}, ${format(node.value)}`,
        };
      },
      edgeAttrs: (edge) => {
        const ref = edgeParameter(edge);
        const selected = selection && sameParameter(ref, selection);
        const scale = mode === "contributions" ? contributionScale : weightScale;
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
          : `${mode === "weights" ? "bias" : "activation"} ${format(node.value)}`,
    });

    svg.addEventListener("click", handleActivate);
    svg.addEventListener("keydown", handleKey);

    overlayGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
    overlayGroup.setAttribute("class", "ripple-overlay");
    overlayGroup.setAttribute("aria-hidden", "true");
    svg.append(overlayGroup);

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
    layout: () => layout,
    overlay: () => {
      if (!overlayGroup) throw new Error("render() before using the overlay");
      return overlayGroup;
    },
    svg: () => {
      if (!root) throw new Error("render() before using the svg");
      return root;
    },
    destroy() {
      root?.removeEventListener("click", handleActivate);
      root?.removeEventListener("keydown", handleKey);
      container.replaceChildren();
      root = null;
      overlayGroup = null;
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
