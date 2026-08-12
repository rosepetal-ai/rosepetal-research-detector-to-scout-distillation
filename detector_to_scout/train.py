#!/usr/bin/env python3
"""Distill a box-supervised teacher into the compact RT-Scout student.

Stage 2 of the paper's pipeline. Consumes an images manifest (train +
val JSONL) and one or more teacher-targets JSONL files (output of
``build_targets.py``); trains the multi-scale student with Hungarian-
matched pseudo-box supervision + image-level BCE; evaluates image-level
AUROC on the val manifest; writes all artifacts into ``--run-dir``.

Paper recipe (``configs/distill_default.yaml``): 30,000 steps, batch
size 16, AdamW (lr 1e-4, weight decay 1e-4), fp32, 256 px letterboxed
inputs, seed 1337, loss weights class 1.0 / L1 5.0 / GIoU 2.0 /
image-BCE 1.0.

Outputs (under ``--run-dir``)::

    train_log.jsonl     per-step losses + matching stats
    last_state.pt       periodic resume checkpoint (model+optimizer+RNG)
    final_state.pt      final weights + model params (payload format)
    predictions.jsonl   per-val-image score + optional boxes
    metrics.json        image AUROC + loss summary
    summary.json        resolved recipe, seed, device, wall time

Usage::

    python -m detector_to_scout.train \\
        --config configs/distill_default.yaml \\
        --teacher-targets targets/teacher_targets.jsonl \\
        --run-dir runs/distill_seed1337 \\
        --device cuda

Fine-tuning on a new dataset: point the config's manifests at your new
data and warm-start from a prior checkpoint with
``--init-checkpoint runs/prior/final_state.pt`` (loads model weights
only; optimizer and step counter start fresh).

Exit codes: 0 = training + eval finished; 2 = invalid args / inputs.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from .data import (
    build_supervision_batch,
    load_image_rgb,
    load_manifest,
    load_teacher_targets,
)
from .model import DEFAULT_IMAGE_SIZE, RTScout, distillation_loss

LOGGER = logging.getLogger("detector_to_scout.train")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(preference: str) -> torch.device:
    """Resolve the compute device. ``cuda`` hard-fails if unavailable —
    no silent CPU fallback."""

    if preference == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda requested but no GPU is reachable")
        return torch.device("cuda")
    if preference == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_image_auroc(records: list[dict[str, Any]], scores: list[float]) -> float | None:
    """Rank-based (Mann-Whitney) image AUROC against the manifest's
    ``label`` field (``nok`` = positive). ``None`` if one class is
    absent. Matches ``sklearn.metrics.roc_auc_score``."""

    y = np.asarray([1 if r.get("label") == "nok" else 0 for r in records], dtype=np.int64)
    s = np.asarray(scores, dtype=np.float32)
    if y.sum() == 0 or y.sum() == len(y):
        return None
    order = np.argsort(s)
    y_sorted = y[order]
    n_pos = y_sorted.sum()
    n_neg = len(y_sorted) - n_pos
    pos_ranks = np.where(y_sorted == 1)[0] + 1
    auc = (pos_ranks.sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc) if not np.isnan(auc) else None


def build_model(cfg: dict[str, Any]) -> RTScout:
    mp = dict(cfg.get("model") or {})
    params = {
        "backbone": mp.get("backbone", "timm:resnet18"),
        "image_size": int(mp.get("image_size", DEFAULT_IMAGE_SIZE)),
        "feature_dim": int(mp.get("feature_dim", 256)),
        "num_queries": int(mp.get("num_queries", 100)),
        "num_heads": int(mp.get("num_heads", 8)),
        "num_encoder_layers": int(mp.get("num_encoder_layers", 2)),
        "num_decoder_layers": int(mp.get("num_decoder_layers", 2)),
        "top_k": int(mp.get("top_k", 32)),
        "score_threshold": float(mp.get("score_threshold", 0.05)),
        "min_area": int(mp.get("min_area", 4)),
        "multi_scale": bool(mp.get("multi_scale", True)),
        "timm": mp.get("timm") or {},
    }
    return RTScout(params)


def _resolve_cfg_path(raw: str, *, config_dir: Path) -> Path:
    """Resolve a path from the config: absolute, CWD-relative, or
    config-dir-relative (first match wins)."""

    p = Path(raw)
    if p.is_absolute():
        return p
    if p.exists():
        return p
    candidate = config_dir / p
    if candidate.exists():
        return candidate
    return p  # let the caller produce the not-found error


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", required=True, help="YAML config (see configs/)")
    p.add_argument("--run-dir", required=True, help="Output directory for all artifacts")
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--seed", type=int, default=None, help="Override training.seed")
    p.add_argument("--max-steps", type=int, default=None, help="Override training.max_steps")
    p.add_argument("--limit", type=int, default=None, help="Limit manifest rows (smoke)")
    p.add_argument(
        "--teacher-targets",
        action="append",
        default=[],
        help="Teacher targets JSONL (repeatable; merged first-seen-wins). "
        "Overrides the config's teacher_targets list when given.",
    )
    p.add_argument(
        "--init-checkpoint",
        default=None,
        help="Warm-start fine-tuning: load model weights from a prior "
        "final_state.pt (payload or bare state_dict). Optimizer and step "
        "counter start fresh.",
    )
    p.add_argument(
        "--resume-from",
        default=None,
        help="Resume an interrupted run from a last_state.pt (restores model, "
        "optimizer, RNG state and step counter).",
    )
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=2000,
        help="Save last_state.pt every N steps (0 disables).",
    )
    args = p.parse_args(argv)

    cfg_path = Path(args.config).resolve()
    if not cfg_path.exists():
        print(f"ERROR: config not found: {cfg_path}", file=sys.stderr)
        return 2
    cfg = yaml.safe_load(cfg_path.read_text())
    config_dir = cfg_path.parent

    manifests_cfg = cfg.get("manifests") or {}
    train_manifest = _resolve_cfg_path(str(manifests_cfg.get("train", "")), config_dir=config_dir)
    val_manifest = _resolve_cfg_path(str(manifests_cfg.get("val", "")), config_dir=config_dir)
    if not train_manifest.exists():
        print(f"ERROR: train manifest not found: {train_manifest}", file=sys.stderr)
        return 2
    if not val_manifest.exists():
        print(f"ERROR: val manifest not found: {val_manifest}", file=sys.stderr)
        return 2

    train_records = load_manifest(train_manifest, limit=args.limit)
    val_records = load_manifest(val_manifest, limit=args.limit)
    train_base_dir = train_manifest.resolve().parent
    val_base_dir = val_manifest.resolve().parent

    target_paths = [Path(t) for t in args.teacher_targets]
    if not target_paths:
        target_paths = [
            _resolve_cfg_path(str(t), config_dir=config_dir)
            for t in (cfg.get("teacher_targets") or [])
        ]
    if not target_paths:
        print(
            "ERROR: no teacher targets: pass --teacher-targets or set "
            "teacher_targets: in the config",
            file=sys.stderr,
        )
        return 2
    for tp in target_paths:
        if not tp.exists():
            print(f"ERROR: teacher targets not found: {tp}", file=sys.stderr)
            return 2
    teacher_index = load_teacher_targets(target_paths)
    LOGGER.info("loaded %d teacher targets from %d file(s)", len(teacher_index), len(target_paths))

    training_cfg = cfg.get("training") or {}
    seed = int(args.seed if args.seed is not None else training_cfg.get("seed", 1337))
    set_seed(seed)
    device = resolve_device(args.device)
    LOGGER.info("device=%s seed=%d", device, seed)

    image_size = int((cfg.get("model") or {}).get("image_size", DEFAULT_IMAGE_SIZE))
    model = build_model(cfg).to(device)
    LOGGER.info(
        "model built: %d checkpoint tensor elements (params + buffers)",
        model.count_parameters(),
    )

    # --- Warm start for fine-tuning ---
    if args.init_checkpoint:
        init_path = Path(args.init_checkpoint)
        if not init_path.exists():
            print(f"ERROR: --init-checkpoint not found: {init_path}", file=sys.stderr)
            return 2
        ckpt = torch.load(init_path, map_location="cpu", weights_only=False)
        state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        model.load_state_dict(state)
        LOGGER.info("warm-started model weights from %s", init_path)

    max_steps = int(args.max_steps if args.max_steps is not None else training_cfg.get("max_steps", 30000))
    batch_size = int(training_cfg.get("batch_size", 16))
    lr = float(training_cfg.get("learning_rate", 1e-4))
    weight_decay = float(training_cfg.get("weight_decay", 1e-4))
    loss_cfg = training_cfg.get("loss") or {}
    class_weight = float(loss_cfg.get("class_weight", 1.0))
    l1_weight = float(loss_cfg.get("l1_weight", 5.0))
    giou_weight = float(loss_cfg.get("giou_weight", 2.0))
    image_weight = float(loss_cfg.get("image_weight", 1.0))

    params_iter = (q for q in model.parameters() if q.requires_grad)
    optimizer = torch.optim.AdamW(params_iter, lr=lr, weight_decay=weight_decay)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    train_log_path = run_dir / "train_log.jsonl"
    rng = random.Random(seed)
    step = 0
    losses: list[float] = []

    # --- Resume from a previous checkpoint, if requested ---
    last_ckpt_path = run_dir / "last_state.pt"
    resume_path: Path | None = None
    if args.resume_from:
        resume_path = Path(args.resume_from)
    elif last_ckpt_path.exists():
        resume_path = last_ckpt_path

    if resume_path and resume_path.exists():
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        if "model_state" in ckpt and "optimizer_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            step = int(ckpt.get("step", 0))
            rng_state = ckpt.get("rng_state")
            if rng_state is not None:
                rng.setstate(tuple(tuple(x) if isinstance(x, list) else x for x in rng_state))
            np_rng_state = ckpt.get("np_rng_state")
            if np_rng_state is not None:
                np.random.set_state(np_rng_state)
            torch_rng_state = ckpt.get("torch_rng_state")
            if torch_rng_state is not None:
                torch.set_rng_state(torch.tensor(torch_rng_state, dtype=torch.uint8))
            cuda_rng_state = ckpt.get("cuda_rng_state")
            if cuda_rng_state is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state(torch.tensor(cuda_rng_state, dtype=torch.uint8))
            LOGGER.info("resumed from %s at step=%d", resume_path, step)
            train_log = train_log_path.open("a")
        else:
            LOGGER.warning("resume file lacks optimizer state; starting fresh")
            train_log = train_log_path.open("w")
    else:
        train_log = train_log_path.open("w")

    # --- Training loop (fp32, epoch-shuffled streaming batches) ---
    model.train()
    t0 = time.time()
    while step < max_steps:
        rng.shuffle(train_records)
        for i in range(0, len(train_records), batch_size):
            batch = train_records[i : i + batch_size]
            if not batch:
                continue
            images, targets, _ = build_supervision_batch(
                batch,
                teacher_index=teacher_index,
                image_size=image_size,
                base_dir=train_base_dir,
            )
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad(set_to_none=True)
            logits = model.forward_logits(images)
            losses_dict = distillation_loss(
                logits,
                targets,
                class_weight=class_weight,
                l1_weight=l1_weight,
                giou_weight=giou_weight,
                image_weight=image_weight,
            )
            loss = losses_dict["loss"]
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

            if step % args.log_every == 0:
                LOGGER.info(
                    "step=%d loss=%.4f cls=%.4f l1=%.4f giou=%.4f image=%.4f n_matched=%d match=%s",
                    step,
                    float(loss.detach().cpu()),
                    float(losses_dict["loss_class"].detach().cpu()),
                    float(losses_dict["loss_l1"].detach().cpu()),
                    float(losses_dict["loss_giou"].detach().cpu()),
                    float(losses_dict["loss_image"].detach().cpu()),
                    int(losses_dict["n_matched"].detach().cpu()),
                    losses_dict["match_algorithm"],
                )
            train_log.write(
                json.dumps(
                    {
                        "step": step,
                        "loss": float(loss.detach().cpu()),
                        "loss_class": float(losses_dict["loss_class"].detach().cpu()),
                        "loss_l1": float(losses_dict["loss_l1"].detach().cpu()),
                        "loss_giou": float(losses_dict["loss_giou"].detach().cpu()),
                        "loss_image": float(losses_dict["loss_image"].detach().cpu()),
                        "n_matched": int(losses_dict["n_matched"].detach().cpu()),
                        "match_algorithm": losses_dict["match_algorithm"],
                    }
                )
                + "\n"
            )
            train_log.flush()
            step += 1

            # Periodic checkpoint — atomic write via temp+rename.
            if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
                tmp = run_dir / "last_state.pt.tmp"
                payload = {
                    "step": step,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "rng_state": rng.getstate(),
                    "np_rng_state": np.random.get_state(),
                    "torch_rng_state": torch.get_rng_state().tolist(),
                    "cuda_rng_state": (
                        torch.cuda.get_rng_state().tolist() if torch.cuda.is_available() else None
                    ),
                    "seed": seed,
                    "max_steps": max_steps,
                }
                torch.save(payload, tmp)
                tmp.replace(last_ckpt_path)
                LOGGER.info("checkpoint saved: step=%d -> %s", step, last_ckpt_path.name)

            if step >= max_steps:
                break
    train_log.close()
    train_wall_s = time.time() - t0

    # --- Evaluation: score the val manifest, write predictions ---
    model.eval()
    predictions: list[dict[str, Any]] = []
    val_scores: list[float] = []
    with torch.no_grad():
        for rec in val_records:
            rgb = load_image_rgb(rec, base_dir=val_base_dir)
            t1 = time.perf_counter()
            out = model.forward_infer(rgb)
            t_ms = (time.perf_counter() - t1) * 1000.0
            predictions.append(
                {
                    "image_id": rec["image_id"],
                    "image_path": rec["image_path"],
                    "image_score": float(out["image_score"]),
                    "boxes": out["boxes"],
                    "latency_ms": float(t_ms),
                }
            )
            val_scores.append(float(out["image_score"]))

    auroc = compute_image_auroc(val_records, val_scores)

    (run_dir / "predictions.jsonl").write_text(
        "\n".join(json.dumps(q) for q in predictions) + ("\n" if predictions else "")
    )
    metrics = {
        "image_auroc": auroc,
        "n_train_records": len(train_records),
        "n_val_records": len(val_records),
        "final_train_loss": losses[-1] if losses else None,
        "mean_train_loss_last_50": float(np.mean(losses[-50:])) if losses else None,
        "max_steps": max_steps,
        "seed": seed,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # --- Final checkpoint (payload format: weights + build params) ---
    ckpt_path = run_dir / "final_state.pt"
    payload = model.state_dict_payload()
    payload["step"] = step
    payload["seed"] = seed
    torch.save(payload, ckpt_path)

    summary = {
        "config_path": str(cfg_path),
        "run_dir": str(run_dir),
        "seed": seed,
        "device": str(device),
        "teacher_targets": [str(t) for t in target_paths],
        "init_checkpoint": args.init_checkpoint,
        "max_steps": max_steps,
        "n_steps_run": step,
        "batch_size": batch_size,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "loss": {
            "class_weight": class_weight,
            "l1_weight": l1_weight,
            "giou_weight": giou_weight,
            "image_weight": image_weight,
        },
        "n_model_tensor_elements": model.count_parameters(),
        "train_wall_s": train_wall_s,
        "metrics": metrics,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    LOGGER.info(
        "done — run_dir=%s steps=%d auroc=%s wall=%.1fs", run_dir, step, auroc, train_wall_s
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
