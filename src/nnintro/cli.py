"""Command line surface for the offline pipeline."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from . import datasets
from .export import (
    build_checkpoint,
    build_manifest,
    checkpoint_id,
    content_hash,
    publish,
    write_checkpoint,
)
from .model import ACTIVATIONS, DenseNet, mean_loss
from .train import TrainConfig, curate

# The architecture family of PLAN.md section 3: hidden depth 0, 1 or 2 and
# each hidden width 2 or 4, which is seven architectures.
HIDDEN_WIDTHS = (2, 4)
ARCHITECTURES: list[list[int]] = [
    [2, 1],
    *([2, w, 1] for w in HIDDEN_WIDTHS),
    *([2, a, b, 1] for a in HIDDEN_WIDTHS for b in HIDDEN_WIDTHS),
]


def build_catalog() -> list[dict]:
    """Every configuration the UI controls can select: 4 x 19 = 76 entries.

    Generated rather than written out, because the controls enumerate the
    same product and a hand-maintained list would drift from them. A network
    with no hidden layer has no hidden activation to vary, so it contributes
    one entry rather than three: 1 + 2x3 + 4x3 = 19 per dataset.
    """
    entries = []
    for dataset in datasets.BUILDERS:
        for sizes in ARCHITECTURES:
            activations = ACTIVATIONS if len(sizes) > 2 else ("identity",)
            for activation in activations:
                entries.append(
                    {"dataset": dataset, "sizes": sizes, "activation": activation}
                )
    return entries


CATALOG = build_catalog()

DATA_DIR = Path("data")
SNAPSHOT_DIR = Path("frontend/public/data")


def _build_one(entry: dict, steps: int | None, attempts: int) -> dict:
    """Curate and export one catalog entry, in its own process.

    Returns a record rather than raising: one configuration that cannot be
    curated is a gap in the catalog, which the caller reports and the UI
    disables, not a reason to abandon the other seventy-five.
    """
    # One thread per worker: these are 2x4 matmuls, so torch's intra-op
    # threads cost more in coordination than the arithmetic they split, and
    # they fight the process pool for the same cores.
    torch.set_num_threads(1)
    dataset = datasets.build(entry["dataset"])
    config = TrainConfig(
        sizes=entry["sizes"],
        activation=entry["activation"],
        **({"steps": steps} if steps else {}),
    )
    name = checkpoint_id(entry["dataset"], entry["sizes"], entry["activation"])
    try:
        result, selection = curate(dataset, config, attempts=attempts)
    except RuntimeError as error:
        return {"checkpointId": name, "payload": None, "error": str(error)}
    payload = build_checkpoint(result, dataset, selection)
    return {
        "checkpointId": name,
        "payload": payload,
        "error": None,
        "summary": (
            f"train {result.train_loss:.3e}  test {result.test_loss:.3e}  "
            f"(seed {selection['chosen_init_seed']}, "
            f"{selection['accepted_count']}/{selection['attempts']} accepted)"
        ),
    }


def _build_all(
    outdir: Path, steps: int | None, attempts: int, workers: int
) -> tuple[list[dict], list[dict]]:
    payloads: list[dict] = []
    gaps: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_build_one, entry, steps, attempts) for entry in CATALOG]
        for index, future in enumerate(futures, start=1):
            record = future.result()
            name = record["checkpointId"]
            if record["error"]:
                gaps.append(record)
                print(f"[{index}/{len(CATALOG)}] GAP {name}: {record['error']}")
                continue
            path = write_checkpoint(record["payload"], outdir)
            payloads.append(record["payload"])
            print(f"[{index}/{len(CATALOG)}] {name}: {record['summary']} -> {path}")

    manifest = build_manifest(payloads)
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"manifest: {len(payloads)} checkpoint(s) -> {outdir / 'manifest.json'}")
    if gaps:
        print(f"{len(gaps)} configuration(s) have no checkpoint:")
        for record in gaps:
            print(f"  {record['checkpointId']}")
    return payloads, gaps


def cmd_train(args: argparse.Namespace) -> int:
    _, gaps = _build_all(Path(args.outdir), args.steps, args.attempts, args.workers)
    # A gap is reportable: the controls must then disable that stop, and a
    # silent zero exit would let one slip into a published snapshot.
    return 1 if gaps else 0


def cmd_publish(args: argparse.Namespace) -> int:
    written = publish(Path(args.src), Path(args.dest))
    for path in written:
        print(f"published {path}")
    if not written:
        print("nothing published")
        return 1
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Re-check a published snapshot against a fresh evaluation.

    Confirms that the stored hash matches the stored content, that every
    golden probe reproduces from the stored parameters alone, and that the
    manifest covers exactly the checkpoints present.
    """
    directory = Path(args.dir)
    manifest = json.loads((directory / "manifest.json").read_text())
    listed = {c["checkpointId"] for c in manifest["checkpoints"]}
    present = {p.stem for p in directory.glob("*.json") if p.name != "manifest.json"}
    if listed != present:
        print(f"FAIL manifest/file mismatch: {listed ^ present}")
        return 1
    # The frontend reads the manifest's `datasets` array to label and group
    # its dataset control, so every dataset a checkpoint names has to appear
    # there: a missing entry makes that dataset unselectable.
    described = {entry["id"] for entry in manifest.get("datasets", [])}
    used = {entry["dataset"] for entry in manifest["checkpoints"]}
    if described != used:
        print(f"FAIL manifest datasets do not cover its checkpoints: {described ^ used}")
        return 1
    # Catalog coverage, which PLAN.md section 11 asks to be checked
    # automatically before assets are published: every combination the
    # controls enumerate must be here, because the controls offer exactly
    # the product the catalog defines and a missing one disables a stop.
    expected = {
        checkpoint_id(entry["dataset"], entry["sizes"], entry["activation"])
        for entry in build_catalog()
    }
    if not expected <= listed:
        print(f"FAIL catalog is not covered; missing: {sorted(expected - listed)}")
        return 1
    print(f"catalog covered: {len(expected)} configuration(s)")

    failures = 0
    for entry in manifest["checkpoints"]:
        payload = json.loads((directory / entry["url"]).read_text())
        name = payload["checkpointId"]

        if content_hash(payload) != payload["contentHash"]:
            print(f"FAIL {name}: content hash does not match content")
            failures += 1
        expected_id = checkpoint_id(
            payload["dataset"]["id"],
            payload["architecture"]["layerSizes"],
            payload["architecture"]["hiddenActivation"] or "none",
        )
        if expected_id != name:
            print(f"FAIL {name}: identifier disagrees with its own contents")
            failures += 1

        model = _model_from_payload(payload)
        task = payload["dataset"]["task"]
        for probe in payload["goldenProbes"]:
            x = torch.as_tensor(probe["x"], dtype=torch.float64)
            trace = model.trace(x)
            if abs(float(trace.output.detach()) - probe["output"]) > 1e-12:
                print(f"FAIL {name}/{probe['id']}: output does not reproduce")
                failures += 1
            for layer, (stored, got) in enumerate(zip(probe["act"], trace.act)):
                if not np.allclose(stored, got.tolist(), rtol=0, atol=1e-12):
                    print(f"FAIL {name}/{probe['id']}: activations differ at {layer}")
                    failures += 1

        samples = payload["dataset"]["samples"]
        x_all = torch.as_tensor(samples["x"], dtype=torch.float64)
        y_all = torch.as_tensor(samples["y"], dtype=torch.float64)
        is_test = np.array(samples["isTest"], dtype=bool)
        for split, mask in (("train", ~is_test), ("test", is_test)):
            loss = float(
                mean_loss(model(x_all[mask]), y_all[torch.as_tensor(mask)], task).detach()
            )
            stored = payload["splitGradients"][split]["meanLoss"]
            if abs(loss - stored) > 1e-12:
                print(f"FAIL {name}: {split} mean loss {loss} != stored {stored}")
                failures += 1
        print(f"checked {name}: {len(payload['goldenProbes'])} probes")

    if failures:
        print(f"{failures} failure(s)")
        return 1
    print("all checks passed")
    return 0


def _model_from_payload(payload: dict) -> DenseNet:
    """Rebuild a model from the stored parameters alone."""
    arch = payload["architecture"]
    model = DenseNet(
        arch["layerSizes"], arch["hiddenActivation"] or "identity", seed=0
    )
    with torch.no_grad():
        for layer, stored in zip(model.layers, payload["parameters"]):
            layer.weight.copy_(torch.as_tensor(stored["W"], dtype=torch.float64))
            layer.bias.copy_(torch.as_tensor(stored["b"], dtype=torch.float64))
    return model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nnintro", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="train the catalog and export assets")
    p_train.add_argument("--outdir", default=str(DATA_DIR))
    p_train.add_argument("--steps", type=int, default=None)
    p_train.add_argument(
        "--attempts",
        type=int,
        default=96,
        help="training attempts to curate from (see train.SELECTION_CRITERIA)",
    )
    p_train.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 2),
        help="catalog entries to curate in parallel",
    )
    p_train.add_argument(
        "--only",
        default=None,
        help="restrict to catalog entries whose id contains this substring",
    )
    p_train.set_defaults(func=cmd_train)

    p_publish = sub.add_parser(
        "publish", help="copy working output to the tracked snapshot"
    )
    p_publish.add_argument("--src", default=str(DATA_DIR))
    p_publish.add_argument("--dest", default=str(SNAPSHOT_DIR))
    p_publish.set_defaults(func=cmd_publish)

    p_verify = sub.add_parser("verify", help="re-check a published snapshot")
    p_verify.add_argument("--dir", default=str(SNAPSHOT_DIR))
    p_verify.set_defaults(func=cmd_verify)

    args = parser.parse_args(argv)
    if getattr(args, "only", None):
        global CATALOG
        CATALOG = [
            entry
            for entry in CATALOG
            if args.only
            in checkpoint_id(entry["dataset"], entry["sizes"], entry["activation"])
        ]
        if not CATALOG:
            print(f"no catalog entry matches {args.only!r}")
            return 1
    return args.func(args)
