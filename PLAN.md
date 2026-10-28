# Dense Neural Network Explorer — Implementation Plan

Status: proposed plan, ready for iteration. Initial scope agreed through five design questions.

## 1. Purpose and boundaries

Build an interactive explorer for students with a science background to develop a solid understanding of dense neural networks: how inputs, weights, biases, activation functions, predictions, and loss relate.

The central experiment is: **change one thing, predict what will happen, and inspect the consequences.** Students can load random or pre-trained parameters, follow one input through the network, edit a parameter, and compare local derivative predictions with actual changes.

Confirmed requirements:

- Supervised regression and classification datasets.
- A curated set of architectures, with saved pre-trained parameters for each dataset.
- Architecture visualization using [nnet-svg](https://github.com/dkirkby/nnet-svg), with downstream ripple animations using [zipline](https://github.com/dkirkby/zipline).
- Controls for hidden-layer count, layer widths, and activation functions.
- Easy reset to random or pre-trained parameters, followed by individual weight/bias edits.
- Equal emphasis on input sensitivity and loss sensitivity to parameters.
- No explanation or animation of backpropagation mechanics; a separate tool will cover that topic.
- Training and checkpoint generation occur offline in PyTorch, never in the browser.
- Input-distribution tracing and 2D loss landscapes are later stages.

Proposed defaults below are implementation recommendations, not additional confirmed requirements. Start with a static browser application, two input features, scalar outputs, and small fully visible networks. Exclude browser training, optimizer controls, CNNs, recurrent networks, attention, and arbitrary model import from the initial release.

## 2. Learning outcomes

Students should be able to:

1. Explain a neuron as a weighted sum plus a bias followed by a nonlinearity.
2. Distinguish a weight, a weighted contribution, a preactivation, and an activation.
3. Predict which parts of a feed-forward network can change after editing one parameter.
4. Explain why depth without nonlinear activations remains an affine mapping.
5. Relate input sensitivity to local changes in prediction.
6. Interpret a loss derivative with respect to a weight or bias as a local slope, with sign and magnitude.
7. Distinguish prediction from target, single-example loss from dataset-average loss, and local sensitivity from finite changes.
8. Recognize that architecture limits what a model can represent, while parameters determine the particular mapping.

Later outcomes: explain nonlinear transformations of distributions and interpret a loss slice as a restricted view of a much larger parameter space.

## 3. Dataset and model catalog

### Shared dimensions

Use two input features for every initial dataset. This is a pedagogical choice, not a mathematical requirement: it makes input-space plots directly interpretable and permits reuse of every architecture. Use one scalar output for both regression and binary classification, with task-specific output interpretation and loss. Defer multiclass classification.

Keep model coordinates explicit. Prefer synthetic inputs already expressed on a common domain, such as [-1, 1]². If preprocessing is introduced, display it and apply the chain rule when reporting derivatives in original input units.

### Proposed initial datasets

| Task | Dataset | Teaching purpose |
| --- | --- | --- |
| Regression | Tilted plane, y = 0.6x₁ − 0.4x₂ + 0.2 | Weights and bias have an immediately interpretable role |
| Regression | Smooth curved surface, y = sin(πx₁) cos(πx₂) | Nonlinear representation and approximation |
| Binary classification | Two separated Gaussian clusters | Linear decision boundaries and probabilities |
| Binary classification | XOR quadrants | A simple problem requiring nonlinear features |

Use fixed seeded samples and fixed training/held-out splits. Store samples, targets, domain, generator version, and split membership. Start without adjustable noise or sample counts: those controls would change the training problem without changing its checkpoint. Optional noisy variants must be separate versioned catalog entries.

A selected dataset point has its observed target. For a freely dragged probe, show prediction and sensitivity; show point loss only when a target is explicitly defined. An analytic regression target may be available from the generator. Do not silently assign a label to an arbitrary point in a stochastic classification dataset.

### Proposed architecture family

Use hidden depth 0, 1, or 2, and each hidden width 2 or 4. This gives seven architectures:

- 2 → 1, with no hidden layer.
- 2 → 2 → 1 and 2 → 4 → 1.
- 2 → 2 → 2 → 1, 2 → 2 → 4 → 1, 2 → 4 → 2 → 1, and 2 → 4 → 4 → 1.

Use discrete depth and width sliders with labeled stops; every reachable combination must exist in the catalog. This retains the desired architecture controls while avoiding unsupported checkpoints. Show every node and edge, without ellipses.

Offer ReLU, tanh, and identity as a shared hidden-layer activation. Identity provides a useful linear baseline. Defer independent per-layer activation choices and sigmoid hidden units until there is a teaching reason to expand the catalog.

For four datasets, this means 4 × (1 + 6 × 3) = **76 checkpoint variants**, initially one curated checkpoint per variant. The no-hidden-layer model has no hidden activation choice. A small catalog can contain imperfect fits: linear XOR is an intentional limitation, not a failed asset.

Regression uses an identity output and squared error. Binary classification uses a scalar logit internally, sigmoid for displayed probability, and numerically stable binary cross-entropy from logits. Display logit and probability distinctly.

## 4. Main workspace and interactions

Use three coordinated areas: input-space view, network view, and inspector. Place dataset/model controls above them and prediction/loss summaries nearby. On narrow screens, stack panels without losing the current selection.

### Input-space view

- Plot samples in the two-feature plane, colored by target or class.
- Show a prediction heatmap; add a p = 0.5 decision contour for classification.
- Select a sample or drag a free probe with keyboard-accessible alternatives.
- Keep target and prediction color encodings distinguishable and labeled.
- Show training and held-out mean losses separately; accuracy is secondary to loss.

### Network view and neuron inspector

- Nodes expose preactivation z and activation a; edges expose weight w.
- Select an edge to edit its weight; select a node to edit its bias.
- Include a visible bias badge or equivalent affordance, rather than hiding biases in tooltips.
- Selecting a neuron expands its computation: zⱼ = Σᵢ wⱼᵢaᵢ + bⱼ, then aⱼ = φ(zⱼ).
- Show each incoming contribution wⱼᵢaᵢ, the bias contribution, and the activation curve with its current operating point.
- Add explicit display modes for activations, weights, contributions, and sensitivities. Do not overload one color scale with several meanings at once.
- Provide numeric values, signs, legends, and a parameter table as alternatives to color and SVG picking.

### Editing and model provenance

Provide numeric inputs plus sliders with useful adaptive ranges; permit typed values beyond the default range, subject to finite-number validation. Include undo, redo, restore-selected-parameter, and restore-baseline actions.

Maintain an immutable baseline and an editable working copy. A baseline is either a catalog checkpoint or a seeded random initialization. Show “Random”, “Pre-trained”, or “Modified from …”, including the random seed or checkpoint identifier.

- **Load pre-trained:** replace the working parameters with the exact matching catalog checkpoint.
- **Randomize all:** create a new deterministic seeded initialization; use documented activation-aware initialization and randomized biases. Record the actual parameters, not just a seed.
- **Change dataset or architecture:** create the matching fresh baseline in the current random/pre-trained mode, retain the previous state in undo history, and visibly announce the reset. Do not silently resize matrices or reuse parameters for a different problem.
- **Change hidden activation:** by default load or initialize the matching variant. Also provide an explicit “keep parameters” experiment for activation-only comparisons; this becomes a modified model, not a pre-trained model for the new activation.

Random initialization is for exploration, not a claim of comparable training quality. Comparing pre-trained architectures also changes their parameters; label that distinction.

### Downstream ripple

Compute the new state immediately and animate the explanation separately. Preserve before/after values so students can inspect actual deltas. A weight edit starts at its destination neuron; a bias edit starts at that neuron; an input edit starts at the input node. Animate downstream dependencies layer by layer.

Distinguish structurally affected nodes from nodes whose numerical values actually changed. ReLU can block a change, and different contributions can cancel. The ripple should not imply every reachable neuron changed.

Use zipline pulses to indicate the sequence of influence. Carrier oscillation, speed, and transverse amplitude have no physical or probabilistic meaning here. Encode quantitative deltas in labels and documented color scales. Provide replay, pause, and reduced-motion/static-delta modes. New edits cancel obsolete animations.

## 5. Gradients without a backpropagation lesson

Treat derivatives as observable properties of the current model, not as a procedural visualization of their computation. Implement analytic derivative calculations internally, but omit backward-flow animations, chain-rule traversal lessons, optimizer steps, and training controls.

### Input sensitivity

For the selected input, display ∂ŷ/∂x₁ and ∂ŷ/∂x₂. For classification, default to probability sensitivity and offer logit sensitivity with an explicit label. Draw a local sensitivity arrow in input space and show the first-order estimate Δŷ ≈ ∇ₓŷ · Δx alongside the actual change for a small input perturbation.

### Parameter sensitivity

For a selected weight or bias θ, show ∂ℓ/∂θ for the selected labeled example and ∂L/∂θ for the active dataset split. Clearly label the objective used by any gradient overlay. Do not average per-example gradient magnitudes: differentiate the mean loss itself.

Show ΔL ≈ (∂L/∂θ)Δθ beside the actual loss change, evaluated relative to a frozen before-edit state. A small one-parameter line plot can make the local slope tangible without introducing the later 2D landscape feature.

Use fixed, labeled comparison scales where possible. Distinguish small gradients caused by low error from small probability sensitivities caused by saturation. At ReLU's kink, disclose the chosen derivative convention (zero at z = 0) and explain that a unique mathematical derivative does not exist there. Mark non-finite states rather than silently clamping reported results.

## 6. Internal representations

In the first release, tracing a single input must expose its full activation vector at each hidden layer. Show a compact signed bar chart or numeric strip per layer, plus the selected neuron's activation curve. Keep preactivations available alongside activations.

A subsequent enhancement within the core release can propagate the fixed dataset and plot two chosen hidden coordinates, highlighting the selected sample in every view. For width four, use an explicitly labeled coordinate pair and retain the full vector inspector. Do not present a two-coordinate projection as the complete representation. Defer PCA and other embeddings unless they add clear value.

## 7. Technical design and package integration

Proposed stack: TypeScript, Vite, and a thin React interface, with pure numerical modules independent of the UI. No inference service or GPU is required at this scale. Use SVG for networks and vector overlays; Canvas is appropriate for dense prediction heatmaps. Pin package versions after an integration spike.

Maintain separate modules for catalog loading, numerical evaluation, derivative calculations, state/history, network rendering, animation, and plots. All views consume one versioned evaluation snapshot. Animation must never become the source of mathematical state.

### Existing package capabilities

The repository READMEs reviewed for this plan describe:

- **nnet-svg:** `createDenseNetworkSvg`, a pure `layoutDenseNetwork` geometry API, semantic node/edge indices, numeric-value and styling callbacks, and inspectable SVG data attributes.
- **zipline:** mounting into existing SVG, adopting line elements, synchronized groups, masked animations, and pause/seek/stop lifecycle controls.

Use an adapter to map nnet-svg semantic references to model coordinates. Use geometry to place a zipline overlay or adopt lines where ownership is unambiguous. Scope IDs to the explorer instance. Ensure replacing a network destroys old animation controllers and removes listeners.

The current nnet-svg examples index edge values source-first, while the proposed canonical model storage is destination-first, matching PyTorch Linear weights. Centralize this translation and test it with an asymmetric matrix.

### Candidate upstream improvements, only if justified by the spike

1. A stable node/edge handle or geometry lookup API, if current public layout/data attributes are insufficient.
2. An update-in-place rendering controller, if rebuilding SVG loses focus, selection, or animation state during edits.
3. Explicit bias annotation hooks, if application overlays prove fragile.
4. A shared external clock or grouped cancellation contract, if existing zipline controls cannot support clean layer sequencing and scrubbing.

Prefer adapters and additive changes first. Breaking changes are allowed when they materially simplify the design, but require a minimal failing use case, a proposed contract, migration examples, and corresponding package tests. Keep model math and gradient semantics out of both visualization packages.

## 8. Offline PyTorch asset pipeline

Implement reproducible scripts to generate datasets, train every catalog variant, evaluate them, and export browser assets. Store training settings, generator/training seeds, software versions, preprocessing, split definitions, validation metrics, and checkpoint provenance.

Use multiple offline training attempts where necessary to curate useful examples. Record selection criteria and held-out performance; do not imply that every architecture can solve every dataset. Verify catalog coverage automatically before publishing assets.

Versioned model assets should include:

- Schema version and dataset, architecture, activation, and checkpoint IDs.
- Full layer sizes, output transform, loss definition, and preprocessing.
- For each layer, W with shape [out, in] and b with shape [out], using documented row-major ordering.
- Numeric precision and explicit finite-value requirements.
- Training metadata, evaluation metrics, and content hashes.
- Golden probes containing inputs, all preactivations/activations, logits or predictions, losses, input sensitivities, and parameter gradients.

A manifest lists valid configurations and asset URLs. Cache and load checkpoints on demand. Keep dataset samples fixed across architecture comparisons. Browser-side derivatives must be recomputed after edits; saved gradients are validation fixtures, not runtime substitutes.

Define loss precisely: regression point loss ℓ = (ŷ − y)²; dataset loss L is the arithmetic mean over the chosen split. Classification uses BCE from logits with the same averaging convention. This prevents hidden factors of two or split-dependent scaling.

## 9. Later stage: distribution tracing

Let students define a bivariate Gaussian through mean, two positive standard deviations, and correlation |ρ| < 1. Visualize its input samples and covariance ellipse, then push the same seeded samples through every layer of the fixed network.

- Maintain sample identities, colors, and selections across layers.
- Use coordinate-pair plots for multivariate hidden layers and histograms for scalar outputs; retain marginal summaries for all coordinates.
- Show empirical means and covariances, labeled as summaries rather than complete distribution descriptions.
- Reuse the same samples after parameter edits to avoid Monte Carlo noise masquerading as a model change.
- Explicitly show that nonlinear transformations generally do not preserve Gaussian shape. ReLU may create a point mass at zero.
- Distinguish an empirical pushforward distribution from a first-order Gaussian approximation; the latter is optional and must be labeled.
- Mark samples outside the dataset domain. Input variability is not model uncertainty, and class-probability distributions are not distributions of labels unless labels are separately sampled.

Acceptance: affine-network moments match the analytic transformed Gaussian within Monte Carlo error, and nonlinear examples visibly depart from a Gaussian approximation.

## 10. Later stage: 2D loss landscapes

Begin with a slice through two selected scalar parameters, holding every other parameter fixed. For reference parameters θ₀ and directions u, v, evaluate L(θ₀ + αu + βv). Axis labels must name the actual parameters and offsets.

Freeze the reference model, directions, dataset split, and ranges until the user explicitly rebases. Moving along the slice updates the working model and marker consistently; unrelated parameter changes invalidate the slice. The heatmap shows the actual mean loss for that fixed objective, not an interpolated checkpoint score.

Show contours, the current location, and the two corresponding gradient components. Offer matched random/pre-trained markers only if both parameter vectors actually lie in the displayed plane. Later, support seeded normalized orthogonal directions or a plane containing a random-to-trained displacement, with their construction clearly documented.

Use a worker, cancellable grid jobs, and coarse-then-fine rendering if measurement shows they are needed. Explain that a slice is only a restriction of a high-dimensional landscape; apparent minima need not be full-space minima. Do not draw an optimization path when no such trajectory was recorded.

Acceptance: the center equals the independently computed objective, selected grid points match direct evaluation, and projected gradient components match finite differences away from nondifferentiable points.

## 11. Delivery milestones

| Stage | Deliverables | Exit condition |
| --- | --- | --- |
| 0 — Numerical and visual spike | One dataset, one network, PyTorch export, SVG adapter, editable weight and bias, downstream pulse | Browser forward pass and derivatives match golden fixtures; edits preserve selection and cancel stale animations |
| 1 — Core explorer | Four datasets, full curated catalog, discrete architecture/activation controls, random/pre-trained baselines, input tracing, parameter editing, loss summaries | Every offered configuration loads; all neuron computations are inspectable; no browser training |
| 2 — Sensitivity and teaching quality | Both gradient views, local-versus-actual changes, representation strips, undo/redo, accessible controls, short guided experiments | Students can distinguish input sensitivity, example-loss gradients, and mean-loss gradients without a backprop explanation |
| 3 — Distribution tracing | Gaussian controls, fixed seeded samples, coordinated layer views and output histograms | Affine and nonlinear validation checks pass; projections and distribution summaries are labeled accurately |
| 4 — Loss slices | Two-parameter slices, frozen reference, contours and projected gradients | Slice values, coordinate semantics, invalidation, and cancellation are verified |

Stages 0–2 define the initial release. Add later stages independently after the core interaction is useful.

## 12. Verification and quality targets

Numerical checks:

- Compare browser forward states and analytic derivatives against PyTorch golden fixtures for every activation and task.
- Use centered finite differences as an independent derivative check, avoiding ReLU kinks and using scale-aware step sizes.
- Verify matrix orientation, stable sigmoid/BCE at extreme logits, identity-network equivalence to an affine mapping, and mean-loss reduction.
- Verify that edits cannot change upstream activations and that random/checkpoint restoration is exact.

Interaction checks:

- Exercise every manifest configuration, model provenance transition, undo operation, and activation-only experiment.
- Verify probe targeting rules and loss availability for unlabeled inputs.
- Check keyboard parameter editing, focus retention, numeric alternatives, color legends, and reduced motion.
- Test rapid edits and dataset switches for stale evaluation results or orphaned animation controllers.

Provisional performance target: selected-input results respond within 100 ms on a representative laptop. Refresh dense plots separately if necessary, with visible pending state and no mismatched model snapshots. Benchmark the largest offered network and dataset before increasing catalog size.

Teaching review: ask several science students to explain one neuron, predict the effect of a bias edit, and interpret a gradient. Revise labels and interaction order when the visuals encourage incorrect explanations.

## 13. Suggested guided experiments

1. Recover the tilted plane: inspect the weights and bias of a trained no-hidden-layer model.
2. Turn a neuron off: adjust a bias until a ReLU output becomes zero and inspect the downstream effect.
3. Compare identity with nonlinear activations on XOR while separating parameter-preserving experiments from pre-trained comparisons.
4. Follow one example through the network, inspecting weighted contributions and hidden vectors.
5. Compare a small parameter change with the derivative prediction, then increase the change and observe the local approximation break down.
6. Find a parameter change that helps one example but worsens the mean loss.

## 14. Sources and implementation decisions to revisit

Package integration is grounded in the repository READMEs retrieved while preparing this plan:

- https://github.com/dkirkby/nnet-svg/blob/main/README.md
- https://github.com/dkirkby/zipline/blob/main/README.md

These are mutable main-branch documents, not pinned implementation dependencies. Verify exact types and behavior against pinned commits during Stage 0. No package changes have been made as part of this plan.

Revisit after the spike: the four-dataset selection, the seven-architecture family, whether width four is sufficiently expressive for the curved surface, default initialization scales, whether activation should be global or per layer, and the smallest necessary upstream API changes. These decisions should follow classroom utility and measured behavior rather than expand scope preemptively.
