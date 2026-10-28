"""Command line surface for the offline pipeline."""

from __future__ import annotations

import argparse
import json
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
from .model import DenseNet, mean_loss
from .train import TrainConfig, curate

# The Stage 0 spike trains one variant. Stage 1 grows this to the full
# catalog of PLAN.md section 3; every entry a UI control can reach must
# appear here, which is what `verify` checks against the manifest.
CATALOG = [
    {"dataset": "tilted-plane", "sizes": [2, 4, 1], "activation": "relu"},
]

DATA_DIR = Path("data")
SNAPSHOT_DIR = Path("frontend/public/data")


def _build_all(outdir: Path, steps: int | None, attempts: int) -> list[dict]:
    payloads = []
    for entry in CATALOG:
        dataset = datasets.build(entry["dataset"])
        config = TrainConfig(
            sizes=entry["sizes"],
            activation=entry["activation"],
            **({"steps": steps} if steps else {}),
        )
        result, selection = curate(dataset, config, attempts=attempts)
        payload = build_checkpoint(result, dataset, selection)
        path = write_checkpoint(payload, outdir)
        print(
            f"{payload['checkpointId']}: "
            f"train {result.train_loss:.3e}  test {result.test_loss:.3e}  "
            f"(seed {selection['chosen_init_seed']}, "
            f"{selection['accepted_count']}/{selection['attempts']} accepted) -> {path}"
        )
        payloads.append(payload)
    manifest = build_manifest(payloads)
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"manifest: {len(payloads)} checkpoint(s) -> {outdir / 'manifest.json'}")
    return payloads


def cmd_train(args: argparse.Namespace) -> int:
    _build_all(Path(args.outdir), args.steps, args.attempts)
    return 0


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
        default=24,
        help="training attempts to curate from (see train.SELECTION_CRITERIA)",
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
    return args.func(args)
