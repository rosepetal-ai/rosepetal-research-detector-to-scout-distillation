#!/usr/bin/env python3
"""Build distillation targets from a box-supervised teacher's predictions.

Stage 1 of the paper's pipeline: harvest a detector's cached
predictions into per-image pseudo-box target records. The teacher is
queried ONCE (by you, outside this repo) to produce a predictions
JSONL; this script reshapes those predictions into the targets JSONL
consumed by ``train.py``. No teacher model is loaded here or at any
later stage.

Teacher-predictions contract (generic — ANY detector can be the teacher)
------------------------------------------------------------------------
One JSON object per line::

    {"image_path": "images/cat_000.png",
     "boxes": [{"xyxy": [x0, y0, x1, y1], "score": 0.93, "label": 0}]}

* ``image_path`` (str, required) — must match the ``image_path`` in the
  images manifest (or provide a matching ``image_id``, see below).
* ``image_id`` (str, optional) — used for alignment when present;
  defaults to ``image_path``.
* ``boxes`` (list, required; may be empty) — each box carries:
    * exactly one geometry key:
        * ``"xyxy": [x0, y0, x1, y1]`` — absolute pixel coordinates in
          the ORIGINAL image (x0 < x1, y0 < y1), or
        * ``"cxcywh": [cx, cy, w, h]`` — normalized to ``[0, 1]`` by the
          original image width/height (DETR-family raw output format).
          Conversion to pixels needs the image size: provide ``width``
          and ``height`` fields on the prediction record, or the script
          opens the image to read them.
    * ``"score"`` (float, required) — confidence in ``[0, 1]``
      (clamped). Stored in the target record; NOTE: the trainer
      supervises matched queries with hard 1.0 targets (paper §3.2) —
      scores are kept for provenance and future soft-target work.
    * ``"label"`` (int or str, optional) — class id; stored as a string
      for provenance, not used by the loss (the student is
      class-agnostic: defect vs background).

Adapter notes (thin, format-only)
---------------------------------
* **RF-DETR-style outputs** (normalized cxcywh + per-box confidence):
  emit each detection as ``{"cxcywh": [cx, cy, w, h], "score": conf,
  "label": cls}`` and add the record-level ``width`` / ``height`` of
  the original image, e.g.::

      rec = {"image_path": path, "width": W, "height": H,
             "boxes": [{"cxcywh": list(b), "score": float(s), "label": int(c)}
                       for b, s, c in zip(pred_boxes, pred_scores, pred_labels)]}

* **RT-DETR / COCO-style outputs** (absolute ``[x, y, w, h]`` pixel
  boxes, e.g. COCO detection dumps): convert xywh -> xyxy::

      rec = {"image_path": path,
             "boxes": [{"xyxy": [x, y, x + w, y + h], "score": float(s),
                        "label": int(c)} for (x, y, w, h), s, c in dets]}

Merging multiple teachers
-------------------------
Single-teacher is the default (one ``--predictions`` file). Passing
several ``--predictions`` files requires the explicit
``--merge first-seen-wins`` flag: per image, the record from the FIRST
file that provides one is kept; later files only fill gaps. This is the
merge used for the paper's measured runs — and, as the paper discloses
(Section 3.2), when the first teacher covers every image the second
contributes no effective supervision. Per-teacher and true-union merges
are future work; no other merge policy is implemented.

Leakage warning
---------------
Build targets for TRAINING images only (``--manifest`` should be your
train manifest). Never let evaluation images into the targets file or
the training pool.

Usage::

    python -m detector_to_scout.build_targets \\
        --manifest data/train.jsonl \\
        --predictions teacher_predictions.jsonl \\
        --teacher my_detector \\
        --out targets/teacher_targets.jsonl

Exit codes: 0 = targets written; 2 = invalid args / malformed input.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from .data import resolve_path


def _load_manifest_index(manifest_path: Path) -> dict[str, dict[str, Any]]:
    """Manifest records keyed by ``image_id`` AND by ``image_path``."""

    out: dict[str, dict[str, Any]] = {}
    with manifest_path.open() as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{manifest_path}: line {lineno} JSON decode failed: {exc}") from exc
            image_path = rec.get("image_path")
            image_id = rec.get("image_id") or image_path
            if image_id:
                out[str(image_id)] = rec
            if image_path:
                out.setdefault(str(image_path), rec)
    return out


def _image_size_for(
    pred: dict[str, Any],
    *,
    manifest_dir: Path,
) -> tuple[float, float]:
    """Original (width, height) for normalized-box conversion."""

    if "width" in pred and "height" in pred:
        return float(pred["width"]), float(pred["height"])
    from PIL import Image as PILImage

    img_path = resolve_path(str(pred["image_path"]), base_dir=manifest_dir)
    with PILImage.open(img_path) as im:
        return float(im.width), float(im.height)


def _box_to_xyxy(
    box: dict[str, Any],
    *,
    pred: dict[str, Any],
    manifest_dir: Path,
    size_cache: dict[str, tuple[float, float]],
) -> list[float] | None:
    """Convert one contract box to pixel xyxy; None if malformed."""

    if "xyxy" in box:
        bbox = box["xyxy"]
        if not bbox or len(bbox) != 4:
            return None
        return [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
    if "cxcywh" in box:
        bbox = box["cxcywh"]
        if not bbox or len(bbox) != 4:
            return None
        key = str(pred.get("image_id") or pred["image_path"])
        if key not in size_cache:
            size_cache[key] = _image_size_for(pred, manifest_dir=manifest_dir)
        img_w, img_h = size_cache[key]
        cx, cy, w, h = (float(v) for v in bbox)
        return [
            (cx - 0.5 * w) * img_w,
            (cy - 0.5 * h) * img_h,
            (cx + 0.5 * w) * img_w,
            (cy + 0.5 * h) * img_h,
        ]
    return None


def _transform_record(
    pred: dict[str, Any],
    *,
    teacher: str,
    manifest_rec: dict[str, Any],
    manifest_dir: Path,
    size_cache: dict[str, tuple[float, float]],
) -> dict[str, Any]:
    """Convert one teacher-prediction line into a target record."""

    boxes_raw = pred.get("boxes") or []
    boxes_out: list[dict[str, Any]] = []
    label_counter: Counter[str] = Counter()
    max_score = 0.0
    for b in boxes_raw:
        bbox = _box_to_xyxy(b, pred=pred, manifest_dir=manifest_dir, size_cache=size_cache)
        if bbox is None:
            continue
        score = float(b.get("score", 0.0))
        if score < 0.0:
            score = 0.0
        if score > 1.0:
            score = 1.0
        label = str(b.get("label", ""))
        boxes_out.append({"bbox": bbox, "score": score, "label": label})
        label_counter[label] += 1
        if score > max_score:
            max_score = score
    top_label = label_counter.most_common(1)[0][0] if label_counter else None
    image_id = str(pred.get("image_id") or pred["image_path"])
    # Canonicalize to the manifest's image_id so train.py's lookup works
    # even when predictions carried only image_path.
    manifest_id = manifest_rec.get("image_id") or manifest_rec.get("image_path")
    if manifest_id:
        image_id = str(manifest_id)
    image_path = str(pred.get("image_path") or manifest_rec.get("image_path") or "")
    return {
        "image_id": image_id,
        "image_path": image_path,
        "teacher": teacher,
        "boxes": boxes_out,
        "summary": {
            "n_boxes": len(boxes_out),
            "max_score": float(max_score),
            "top_label": top_label,
        },
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--manifest", required=True, help="Images manifest JSONL (TRAIN split only!)")
    p.add_argument(
        "--predictions",
        action="append",
        required=True,
        help="Teacher predictions JSONL (generic contract). Repeatable; "
        "more than one file requires --merge first-seen-wins.",
    )
    p.add_argument("--teacher", required=True, help="Teacher name tag stored in each record")
    p.add_argument("--out", required=True, help="Output teacher_targets.jsonl path")
    p.add_argument(
        "--merge",
        choices=("first-seen-wins",),
        default=None,
        help="Required when multiple --predictions files are given. The only "
        "implemented policy is 'first-seen-wins' (the paper's merge): per "
        "image, the FIRST file providing a record wins; later files only "
        "fill gaps. NOTE: if the first file covers every image, later "
        "files contribute nothing (disclosed in the paper, Section 3.2).",
    )
    args = p.parse_args(argv)

    manifest_path = Path(args.manifest)
    prediction_paths = [Path(pp) for pp in args.predictions]
    out_path = Path(args.out)

    if not manifest_path.is_file():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    for pp in prediction_paths:
        if not pp.is_file():
            print(f"ERROR: predictions not found: {pp}", file=sys.stderr)
            return 2
    if len(prediction_paths) > 1 and args.merge != "first-seen-wins":
        print(
            "ERROR: multiple --predictions files require an explicit "
            "--merge first-seen-wins (the only implemented policy; see --help).",
            file=sys.stderr,
        )
        return 2

    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_index = _load_manifest_index(manifest_path)
    manifest_dir = manifest_path.resolve().parent
    size_cache: dict[str, tuple[float, float]] = {}

    # First-seen-wins across files AND lines: the first record seen for
    # an image_id is kept.
    merged: dict[str, dict[str, Any]] = {}
    per_file_counts: list[dict[str, Any]] = []
    for pp in prediction_paths:
        n_seen = 0
        n_kept = 0
        with pp.open() as fh_in:
            for lineno, line in enumerate(fh_in, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    pred = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(
                        f"ERROR: {pp}: line {lineno} JSON decode failed: {exc}",
                        file=sys.stderr,
                    )
                    return 2
                if "image_path" not in pred:
                    print(
                        f"ERROR: {pp}: line {lineno} missing required key 'image_path'",
                        file=sys.stderr,
                    )
                    return 2
                key = str(pred.get("image_id") or pred["image_path"])
                manifest_rec = manifest_index.get(key) or manifest_index.get(
                    str(pred["image_path"])
                )
                if manifest_rec is None:
                    continue  # not a training image — skip (leakage guard)
                n_seen += 1
                canonical = str(manifest_rec.get("image_id") or manifest_rec.get("image_path"))
                if canonical in merged:
                    continue  # first-seen-wins
                try:
                    record = _transform_record(
                        pred,
                        teacher=args.teacher,
                        manifest_rec=manifest_rec,
                        manifest_dir=manifest_dir,
                        size_cache=size_cache,
                    )
                except (FileNotFoundError, KeyError, ValueError, TypeError) as exc:
                    print(f"ERROR: {pp}: line {lineno}: {exc}", file=sys.stderr)
                    return 2
                merged[canonical] = record
                n_kept += 1
        per_file_counts.append({"file": str(pp), "matched_manifest": n_seen, "kept": n_kept})

    written = 0
    n_with_targets = 0
    with out_path.open("w") as fh_out:
        for record in merged.values():
            fh_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            if record["summary"]["n_boxes"] > 0:
                n_with_targets += 1

    n_manifest = len({id(v) for v in manifest_index.values()})
    summary = {
        "n_images": written,
        "n_with_targets": n_with_targets,
        "n_manifest_records": n_manifest,
        "teacher": args.teacher,
        "manifest": str(manifest_path),
        "predictions_sources": per_file_counts,
        "merge": args.merge or "single-teacher",
        "out_path": str(out_path),
    }
    summary_path = out_path.parent / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {written} teacher-target records ({n_with_targets} with boxes) -> {out_path}")
    if written < n_manifest:
        print(
            f"NOTE: {n_manifest - written} manifest images have no teacher record; "
            "they will be trained with background-only supervision."
        )
    print(f"Summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
