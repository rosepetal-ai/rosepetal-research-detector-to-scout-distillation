#!/usr/bin/env python3
"""Generate a tiny synthetic dataset + synthetic teacher predictions.

Produces everything the end-to-end smoke chain needs:

* ``<out>/images/`` — synthetic textured images. Normal ("ok") images
  are noise + soft blobs; defective ("nok") images additionally carry a
  bright rectangular defect patch.
* ``<out>/train.jsonl`` / ``<out>/val.jsonl`` — images manifests
  (``image_id`` / ``image_path`` / ``label``), image paths relative to
  the manifest's directory.
* ``<out>/teacher_predictions.jsonl`` — synthetic teacher predictions
  in the GENERIC contract consumed by ``build_targets.py``: on "nok"
  images a jittered high-confidence box around the true defect; on "ok"
  images one or two low-confidence background boxes (mirroring the
  paper's observation that detector teachers emit boxes on every
  image).
* ``<out>/smoke_config.yaml`` — a ready-to-run training config pointing
  at the generated manifests (``pretrained: false`` so the smoke run
  needs no network access; the paper recipe in
  ``configs/distill_default.yaml`` keeps ``pretrained: true``).

Usage::

    python examples/make_dummy_dataset.py --out /tmp/scout_dummy
    python -m detector_to_scout.build_targets \\
        --manifest /tmp/scout_dummy/train.jsonl \\
        --predictions /tmp/scout_dummy/teacher_predictions.jsonl \\
        --teacher synthetic_teacher \\
        --out /tmp/scout_dummy/targets/teacher_targets.jsonl
    python -m detector_to_scout.train \\
        --config /tmp/scout_dummy/smoke_config.yaml \\
        --run-dir /tmp/scout_dummy/run --max-steps 20 --device cuda
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np


def _make_image(
    rng: np.random.Generator,
    *,
    height: int,
    width: int,
    defective: bool,
) -> tuple[np.ndarray, list[float] | None]:
    """Synthesize one image; return (HWC uint8, defect box xyxy or None)."""

    base = rng.normal(loc=110.0, scale=18.0, size=(height, width, 3))
    # A couple of soft background blobs so "ok" images have structure.
    yy, xx = np.mgrid[0:height, 0:width]
    for _ in range(3):
        cy = rng.uniform(0.2, 0.8) * height
        cx = rng.uniform(0.2, 0.8) * width
        sigma = rng.uniform(0.08, 0.2) * max(height, width)
        blob = 25.0 * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma**2)))
        base += blob[..., None]

    defect_box: list[float] | None = None
    if defective:
        bw = int(rng.uniform(0.12, 0.3) * width)
        bh = int(rng.uniform(0.12, 0.3) * height)
        x0 = int(rng.uniform(0, width - bw - 1))
        y0 = int(rng.uniform(0, height - bh - 1))
        patch = rng.normal(loc=235.0, scale=10.0, size=(bh, bw, 3))
        base[y0 : y0 + bh, x0 : x0 + bw] = patch
        defect_box = [float(x0), float(y0), float(x0 + bw), float(y0 + bh)]
    return np.clip(base, 0, 255).astype(np.uint8), defect_box


def _teacher_boxes(
    rng: np.random.Generator,
    *,
    defect_box: list[float] | None,
    width: int,
    height: int,
) -> list[dict]:
    """Synthetic teacher predictions in the generic contract.

    Defective image: a jittered high-confidence box around the true
    defect. Normal image: 1-2 low-confidence background boxes — real
    detector teachers emit boxes on every image (paper Section 3.1),
    and the synthetic teacher mirrors that density.
    """

    boxes: list[dict] = []
    if defect_box is not None:
        x0, y0, x1, y1 = defect_box
        jitter = rng.uniform(-4.0, 4.0, size=4)
        boxes.append(
            {
                "xyxy": [
                    float(max(0.0, x0 + jitter[0])),
                    float(max(0.0, y0 + jitter[1])),
                    float(min(width - 1.0, x1 + jitter[2])),
                    float(min(height - 1.0, y1 + jitter[3])),
                ],
                "score": float(rng.uniform(0.72, 0.98)),
                "label": 0,
            }
        )
    n_bg = int(rng.integers(1, 3))
    for _ in range(n_bg):
        bw = rng.uniform(0.05, 0.15) * width
        bh = rng.uniform(0.05, 0.15) * height
        bx = rng.uniform(0, width - bw - 1)
        by = rng.uniform(0, height - bh - 1)
        boxes.append(
            {
                "xyxy": [float(bx), float(by), float(bx + bw), float(by + bh)],
                "score": float(rng.uniform(0.02, 0.15)),
                "label": 0,
            }
        )
    return boxes


SMOKE_CONFIG_TEMPLATE = """\
# Auto-generated smoke config (make_dummy_dataset.py). Offline-safe:
# pretrained: false. The paper recipe lives in configs/distill_default.yaml.
model:
  backbone: timm:resnet18
  image_size: 256
  feature_dim: 256
  num_queries: 100
  num_heads: 8
  num_encoder_layers: 2
  num_decoder_layers: 2
  multi_scale: true
  score_threshold: 0.05
  min_area: 4
  top_k: 32
  timm:
    pretrained: false
    in_chans: 3

manifests:
  train: {train_manifest}
  val: {val_manifest}

teacher_targets:
  - {targets_path}

training:
  max_steps: 20
  batch_size: 4
  learning_rate: 1.0e-4
  weight_decay: 1.0e-4
  seed: 1337
  loss:
    class_weight: 1.0
    l1_weight: 5.0
    giou_weight: 2.0
    image_weight: 1.0
"""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--n-train", type=int, default=32)
    p.add_argument("--n-val", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    from PIL import Image as PILImage

    out_dir = Path(args.out)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    py_rng = random.Random(args.seed)

    # Mixed image sizes to exercise the letterbox path.
    sizes = [(256, 320), (320, 256), (300, 300)]

    teacher_lines: list[str] = []
    for split, n in (("train", args.n_train), ("val", args.n_val)):
        manifest_lines: list[str] = []
        for i in range(n):
            defective = i % 2 == 1  # 50/50 ok/nok
            height, width = sizes[i % len(sizes)]
            img, defect_box = _make_image(rng, height=height, width=width, defective=defective)
            image_id = f"{split}_{i:04d}"
            rel_path = f"images/{image_id}.png"
            PILImage.fromarray(img).save(out_dir / rel_path)
            manifest_lines.append(
                json.dumps(
                    {
                        "image_id": image_id,
                        "image_path": rel_path,
                        "label": "nok" if defective else "ok",
                    }
                )
            )
            # Teacher predictions for TRAINING images only (leakage
            # guard — never build targets from eval images).
            if split == "train":
                teacher_lines.append(
                    json.dumps(
                        {
                            "image_path": rel_path,
                            "image_id": image_id,
                            "boxes": _teacher_boxes(
                                rng, defect_box=defect_box, width=width, height=height
                            ),
                        }
                    )
                )
        (out_dir / f"{split}.jsonl").write_text("\n".join(manifest_lines) + "\n")
        print(f"wrote {n} records -> {out_dir / (split + '.jsonl')}")

    (out_dir / "teacher_predictions.jsonl").write_text("\n".join(teacher_lines) + "\n")
    print(f"wrote {len(teacher_lines)} teacher predictions -> {out_dir / 'teacher_predictions.jsonl'}")

    smoke_cfg = SMOKE_CONFIG_TEMPLATE.format(
        train_manifest=str((out_dir / "train.jsonl").resolve()),
        val_manifest=str((out_dir / "val.jsonl").resolve()),
        targets_path=str((out_dir / "targets" / "teacher_targets.jsonl").resolve()),
    )
    (out_dir / "smoke_config.yaml").write_text(smoke_cfg)
    print(f"wrote smoke config -> {out_dir / 'smoke_config.yaml'}")
    _ = py_rng  # reserved for future stochastic layout choices
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
