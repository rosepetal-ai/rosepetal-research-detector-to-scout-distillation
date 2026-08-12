#!/usr/bin/env python3
"""Image-level anomaly scoring with a trained RT-Scout student (PyTorch).

Stage 3 of the paper's pipeline. At inference the student runs
teacher-free: one forward pass, one scalar score — the sigmoid of the
decoupled image head. Per-query boxes are available as an optional
diagnostic output (``--include-boxes``); they are training scaffolding,
not the deployed decision output.

Preprocessing (must match training exactly):

* aspect-preserving letterbox to ``image_size`` x ``image_size``
  (default 256), zero-padded, centered;
* pixel values scaled to ``[0, 1]``;
* NO ImageNet mean/std normalization (the pipeline never applies it).

Usage (CLI)::

    python -m detector_to_scout.infer \\
        --checkpoint runs/distill_seed1337/final_state.pt \\
        --images img1.png img2.png \\
        --device cuda \\
        --out scores.jsonl \\
        --include-boxes

    # or score a whole manifest:
    python -m detector_to_scout.infer \\
        --checkpoint runs/distill_seed1337/final_state.pt \\
        --manifest data/val.jsonl --device cuda --out scores.jsonl

Usage (API)::

    from detector_to_scout.infer import ScoutPredictor
    predictor = ScoutPredictor("runs/distill_seed1337/final_state.pt", device="cuda")
    result = predictor.score_image("img1.png")   # {"image_score": ..., "boxes": [...]}

Exit codes: 0 = all images scored; 2 = invalid args / missing files.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data import load_manifest, resolve_path
from .model import RTScout, load_model_from_checkpoint


class ScoutPredictor:
    """Thin API wrapper: load once, score many.

    Parameters
    ----------
    checkpoint:
        Path to a ``final_state.pt`` written by ``train.py`` (payload
        format), or a bare state_dict (then pass ``params=`` with the
        config's ``model:`` block).
    device:
        ``"cuda"`` / ``"cpu"`` / ``torch.device``. ``"cuda"`` raises if
        no GPU is reachable — no silent CPU fallback.
    """

    def __init__(
        self,
        checkpoint: str,
        *,
        device: str = "cpu",
        params: dict[str, Any] | None = None,
    ) -> None:
        if str(device) == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device='cuda' requested but no GPU is reachable")
        self.device = torch.device(device)
        self.model: RTScout = load_model_from_checkpoint(
            checkpoint, device=self.device, params=params
        )

    def score_image(self, image: str | Path | np.ndarray) -> dict[str, Any]:
        """Score one image (path or HWC array). Returns
        ``{"image_score": float, "boxes": [...], "latency_ms": float}``."""

        if isinstance(image, (str, Path)):
            from PIL import Image as PILImage

            arr = np.asarray(PILImage.open(image).convert("RGB"))
        else:
            arr = image
        t0 = time.perf_counter()
        out = self.model.forward_infer(arr)
        out["latency_ms"] = (time.perf_counter() - t0) * 1000.0
        return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", required=True, help="final_state.pt from train.py")
    p.add_argument("--images", nargs="*", default=[], help="Image files to score")
    p.add_argument("--manifest", default=None, help="Or: images manifest JSONL to score")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cpu")
    p.add_argument("--out", default=None, help="Output JSONL (default: stdout)")
    p.add_argument(
        "--include-boxes",
        action="store_true",
        help="Include the optional per-query diagnostic boxes in the output "
        "(letterboxed 256-px canvas coordinates).",
    )
    args = p.parse_args(argv)

    if not args.images and not args.manifest:
        print("ERROR: provide --images or --manifest", file=sys.stderr)
        return 2
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}", file=sys.stderr)
        return 2

    try:
        predictor = ScoutPredictor(str(ckpt_path), device=args.device)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    jobs: list[tuple[str, Path]] = []
    if args.manifest:
        manifest_path = Path(args.manifest)
        if not manifest_path.exists():
            print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
            return 2
        base_dir = manifest_path.resolve().parent
        for rec in load_manifest(manifest_path):
            jobs.append(
                (str(rec["image_id"]), resolve_path(str(rec["image_path"]), base_dir=base_dir))
            )
    for img in args.images:
        ip = Path(img)
        if not ip.exists():
            print(f"ERROR: image not found: {ip}", file=sys.stderr)
            return 2
        jobs.append((str(ip), ip))

    out_fh = Path(args.out).open("w") if args.out else sys.stdout
    try:
        for image_id, img_path in jobs:
            result = predictor.score_image(img_path)
            record: dict[str, Any] = {
                "image_id": image_id,
                "image_path": str(img_path),
                "image_score": float(result["image_score"]),
                "latency_ms": float(result["latency_ms"]),
            }
            if args.include_boxes:
                record["boxes"] = result["boxes"]
            out_fh.write(json.dumps(record) + "\n")
        if args.out:
            print(f"Wrote {len(jobs)} scores -> {args.out}")
    finally:
        if args.out:
            out_fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
