"""Data loading for detector-to-scout distillation.

Two JSONL files drive training:

1. **Images manifest** (``train.jsonl`` / ``val.jsonl``) — one record
   per image::

       {"image_id": "cat_000", "image_path": "images/cat_000.png",
        "label": "ok"}

   * ``image_path`` — absolute, or relative (resolved against the
     current working directory first, then against the manifest file's
     own directory).
   * ``label`` — ``"ok"`` (normal) or ``"nok"`` (defective). This is
     the binary image label consumed by the image-level BCE term.
   * ``image_id`` — optional; defaults to ``image_path``. Must match
     the ``image_id`` used in the targets JSONL.

2. **Targets JSONL** (output of ``build_targets.py``) — one record per
   training image::

       {"image_id": "cat_000", "image_path": "images/cat_000.png",
        "teacher": "my_detector",
        "boxes": [{"bbox": [x0, y0, x1, y1], "score": 0.93, "label": "0"}],
        "summary": {"n_boxes": 1, "max_score": 0.93, "top_label": "0"}}

   ``bbox`` is xyxy in ORIGINAL image pixel coordinates; the trainer
   letterboxes boxes and image together.

Training-side letterboxing uses PIL bilinear resizing (mirroring the
measured training path); inference-side letterboxing lives in
``model.letterbox_infer`` and uses torch bilinear (mirroring the
measured inference path). Pixel values are scaled to ``[0, 1]``; there
is NO ImageNet mean/std normalization anywhere in the pipeline.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path + JSONL helpers
# ---------------------------------------------------------------------------


def resolve_path(rel: str, *, base_dir: Path | None = None) -> Path:
    """Resolve an image path from a manifest record.

    Order: absolute path as-is; else relative to the current working
    directory; else relative to ``base_dir`` (the manifest's own
    directory). Raises ``FileNotFoundError`` if none exists — missing
    images are an error, never silently substituted.
    """

    p = Path(rel)
    if p.is_absolute():
        if p.exists():
            return p
        raise FileNotFoundError(f"image not found: {p}")
    if p.exists():
        return p
    if base_dir is not None:
        candidate = base_dir / p
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"image not found: {rel!r} (tried CWD and manifest dir {base_dir})"
    )


def load_jsonl(path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of dicts (skipping blank lines)."""

    rows: list[dict[str, Any]] = []
    with Path(path).open() as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: line {lineno} JSON decode failed: {exc}") from exc
            if limit and len(rows) >= limit:
                break
    return rows


def load_manifest(path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    """Load an images manifest; fill in default ``image_id`` values."""

    rows = load_jsonl(path, limit=limit)
    for rec in rows:
        if "image_path" not in rec:
            raise ValueError(f"{path}: manifest record missing 'image_path': {rec}")
        rec.setdefault("image_id", str(rec["image_path"]))
    return rows


def load_teacher_targets(paths: list[Path]) -> dict[str, dict[str, Any]]:
    """Load and merge targets JSONL files into ``image_id -> record``.

    Merge policy is **first-seen-wins**: when several files provide a
    record for the same ``image_id``, the record from the earliest file
    (and earliest line) is kept; later files only fill gaps. This is
    exactly the merge used for the paper's measured runs — and, as the
    paper discloses, when the first teacher covers every image the
    second contributes no effective supervision.
    """

    out: dict[str, dict[str, Any]] = {}
    for p in paths:
        with Path(p).open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                out.setdefault(str(rec["image_id"]), rec)
    return out


# ---------------------------------------------------------------------------
# Letterbox (training side, PIL bilinear — mirrors the measured trainer)
# ---------------------------------------------------------------------------


def letterbox_train_with_boxes(
    image: np.ndarray,
    boxes: np.ndarray | None,
    *,
    target: int,
) -> tuple[np.ndarray, np.ndarray | None, tuple[int, int, float, int, int]]:
    """Aspect-preserving letterbox; return canvas + adjusted boxes + metadata.

    Returns the float32 ``(target, target, 3)`` canvas in ``[0, 1]``,
    the boxes in canvas pixel space (or None), and
    ``(new_h, new_w, scale, pad_y, pad_x)``. Degenerate boxes
    (width or height <= 1 px after scaling) are dropped.
    """

    from PIL import Image as PILImage

    if image.ndim == 2:
        image = image[..., None]
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    h, w = image.shape[:2]
    scale = float(target) / float(max(h, w, 1))
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    resized = np.array(
        PILImage.fromarray(image.astype(np.uint8)).resize((new_w, new_h), resample=PILImage.BILINEAR),
        dtype=np.uint8,
    )
    canvas = np.zeros((target, target, 3), dtype=np.uint8)
    pad_y = (target - new_h) // 2
    pad_x = (target - new_w) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    canvas_f = canvas.astype(np.float32) / 255.0

    boxes_adj: np.ndarray | None = None
    if boxes is not None and len(boxes) > 0:
        # Boxes are xyxy in original image pixel space. Scale + shift to
        # canvas pixel space, clamp, drop degenerate boxes.
        boxes_adj = boxes.copy().astype(np.float32)
        boxes_adj[:, [0, 2]] = boxes_adj[:, [0, 2]] * scale + pad_x
        boxes_adj[:, [1, 3]] = boxes_adj[:, [1, 3]] * scale + pad_y
        boxes_adj[:, 0::2] = np.clip(boxes_adj[:, 0::2], 0, target - 1)
        boxes_adj[:, 1::2] = np.clip(boxes_adj[:, 1::2], 0, target - 1)
        valid = (boxes_adj[:, 2] - boxes_adj[:, 0] > 1.0) & (boxes_adj[:, 3] - boxes_adj[:, 1] > 1.0)
        boxes_adj = boxes_adj[valid]
    return canvas_f, boxes_adj, (new_h, new_w, scale, pad_y, pad_x)


def xyxy_to_cxcywh_norm(boxes: np.ndarray, *, image_size: int) -> torch.Tensor:
    """Canvas-pixel xyxy -> normalized cxcywh in ``[0, 1]``."""

    if boxes.shape[0] == 0:
        return torch.zeros((0, 4), dtype=torch.float32)
    out = np.zeros_like(boxes)
    out[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2.0
    out[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2.0
    out[:, 2] = boxes[:, 2] - boxes[:, 0]
    out[:, 3] = boxes[:, 3] - boxes[:, 1]
    out = out / float(image_size)
    out = np.clip(out, 0.0, 1.0)
    return torch.from_numpy(out.astype(np.float32))


# ---------------------------------------------------------------------------
# Supervision batch builder
# ---------------------------------------------------------------------------


def load_image_rgb(record: dict[str, Any], *, base_dir: Path | None = None) -> np.ndarray:
    """Load a manifest record's image as an HWC uint8 RGB array."""

    from PIL import Image as PILImage

    img_path = resolve_path(str(record["image_path"]), base_dir=base_dir)
    return np.asarray(PILImage.open(img_path).convert("RGB"))


def build_supervision_batch(
    records: list[dict[str, Any]],
    *,
    teacher_index: dict[str, dict[str, Any]],
    image_size: int,
    base_dir: Path | None = None,
) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]], list[dict[str, Any]]]:
    """Load images + build the target list consumed by ``distillation_loss``.

    For each manifest record:

    * the image is letterboxed to ``image_size`` (PIL bilinear,
      zero-padded, centered) and scaled to ``[0, 1]``;
    * the teacher-target boxes (xyxy, original pixel space) are looked
      up by ``image_id``, projected into canvas space, converted to
      normalized cxcywh — and supervised with **hard** scores of 1.0
      (teacher confidences are stored in the target records but not
      propagated into the loss; paper Section 3.2);
    * the binary image label comes from the MANIFEST's ``label`` field
      (``"nok"`` -> 1.0, anything else -> 0.0), never from teacher-box
      presence (detector teachers emit boxes on every image).

    Images with no teacher-target record (or an empty box list) get an
    empty target set: every query is supervised toward background.
    """

    images: list[torch.Tensor] = []
    targets: list[dict[str, torch.Tensor]] = []
    metas: list[dict[str, Any]] = []
    for rec in records:
        rgb = load_image_rgb(rec, base_dir=base_dir)

        t = teacher_index.get(str(rec["image_id"]))
        if t is None or not t.get("boxes"):
            gt: np.ndarray | None = None
        else:
            gt = np.asarray([b["bbox"] for b in t["boxes"]], dtype=np.float32)

        canvas_f, boxes_adj, lb_meta = letterbox_train_with_boxes(rgb, gt, target=image_size)
        chw = np.transpose(canvas_f, (2, 0, 1)).astype(np.float32)
        images.append(torch.from_numpy(chw))

        record_label = str(rec.get("label", "")).lower()
        image_label = 1.0 if record_label == "nok" else 0.0

        if boxes_adj is None or len(boxes_adj) == 0:
            target_entry = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "scores": torch.zeros((0,), dtype=torch.float32),
                "image_label": torch.tensor([image_label], dtype=torch.float32),
            }
        else:
            # Hard classification targets: every retained box is
            # supervised with score = 1.0 (paper Section 3.2 — the
            # as-measured trainer did exactly this).
            target_scores = torch.ones((boxes_adj.shape[0],), dtype=torch.float32)
            target_entry = {
                "boxes": xyxy_to_cxcywh_norm(boxes_adj, image_size=image_size),
                "scores": target_scores,
                "image_label": torch.tensor([image_label], dtype=torch.float32),
            }
        targets.append(target_entry)
        metas.append(
            {
                "image_id": rec["image_id"],
                "image_path": rec["image_path"],
                "label": rec.get("label"),
                "letterbox": lb_meta,
            }
        )
    return torch.stack(images, dim=0), targets, metas


__all__ = [
    "build_supervision_batch",
    "letterbox_train_with_boxes",
    "load_image_rgb",
    "load_jsonl",
    "load_manifest",
    "load_teacher_targets",
    "resolve_path",
    "xyxy_to_cxcywh_norm",
]
