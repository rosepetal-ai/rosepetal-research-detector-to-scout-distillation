---
title: "Data and Targets"
description: "Image manifests, the teacher-predictions format any detector can produce, adapters for RF-DETR-style and COCO-style outputs, and build_targets: options, output records and the first-seen-wins merge."
sidebar_position: 1
---

# Data and Targets

Training needs two kinds of files: image manifests with OK/NOK labels, and a targets file built from a detector's predictions over the training images.

## Image manifests

A manifest is a JSONL file, one JSON object per image. Write one for training and one for validation.

```json
{"image_id": "grid_0001", "image_path": "images/grid_0001.png", "label": "ok"}
{"image_id": "grid_0107", "image_path": "images/grid_0107.png", "label": "nok"}
```

| Field | Required | Meaning |
|-------|----------|---------|
| `image_path` | yes | Absolute, or relative to the current directory, or relative to the manifest's directory, tried in that order. A path that resolves nowhere stops the run with `image not found` when the image is loaded. |
| `label` | yes | `nok` for a defective image, `ok` for a normal one. It is the target of the image-level loss. Training reads it in any letter case and counts any value other than `nok`, or a missing label, as normal; the validation AUROC counts only the exact value `nok` as defective. |
| `image_id` | no | Identifier of the image; defaults to `image_path`. The targets file is joined to the manifest by this value. |

Keep the validation and test images out of the training manifest, and fill `label` on every record: since the teacher draws boxes on normal images too, the presence of boxes says nothing about the label.

For MVTec AD, walk the category folders and label `good` images `ok` and defect images `nok`. Do not train on the images you will evaluate on: if you plan to evaluate on MVTec AD's official test split, none of its images may enter the training manifest or the targets.

## Teacher predictions

Run your box-supervised detector once over the **training images only** and write one JSON object per image. Any detector works, as long as its output is written in this form:

```json
{"image_path": "images/grid_0107.png",
 "boxes": [{"xyxy": [412.0, 105.5, 468.2, 160.0], "score": 0.93, "label": 0}]}
```

| Field | Meaning |
|-------|---------|
| `image_path` | Required. Must match the manifest's `image_path`, unless `image_id` matches. A line without it stops `build_targets` |
| `image_id` | Optional. Used to find the image in the manifest when present; defaults to `image_path` |
| `width`, `height` | Optional. Size of the original image, needed to convert `cxcywh` boxes; without them the tool opens the image to read it |
| `boxes` | The detector's boxes for this image; may be empty |
| `boxes[].xyxy` | `[x0, y0, x1, y1]` in pixels of the original image. Each box needs this or `cxcywh`; a box with neither, or without four values, is skipped |
| `boxes[].cxcywh` | `[cx, cy, w, h]` normalized to `[0, 1]` by the original width and height |
| `boxes[].score` | Confidence, clamped to `[0, 1]`. Stored in the target record; the training loss does not use it |
| `boxes[].label` | Class id, stored as text; not used by the loss (the student only tells defect from background) |

Do not threshold the detector's output: keep its low-confidence boxes. In the paper, the teachers emit boxes on every image, normal ones included, so normal images get low-confidence target boxes instead of an empty set.

### Adapters

RF-DETR-style output (normalized `cxcywh` and a confidence per box), with the original image size on the record:

```python
rec = {"image_path": path, "width": W, "height": H,
       "boxes": [{"cxcywh": [float(v) for v in b], "score": float(s), "label": int(c)}
                 for b, s, c in zip(pred_boxes, pred_scores, pred_labels)]}
```

RT-DETR or COCO-style output (absolute `[x, y, w, h]` boxes in pixels), converted to `xyxy`:

```python
rec = {"image_path": path,
       "boxes": [{"xyxy": [x, y, x + w, y + h], "score": float(s), "label": int(c)}
                 for (x, y, w, h), s, c in dets]}
```

A `cxcywh` box must be normalized by the original image size; boxes with centers in pixels go in `xyxy` instead.

## Building the targets

```bash
python -m detector_to_scout.build_targets \
    --manifest data/train.jsonl \
    --predictions teacher_predictions.jsonl \
    --teacher my_detector \
    --out targets/teacher_targets.jsonl
```

| Option | Meaning |
|--------|---------|
| `--manifest` | Required. The **training** manifest |
| `--predictions` | Required. A teacher-predictions file; repeat it for several teachers |
| `--teacher` | Required. Name stored in every record |
| `--out` | Required. Output JSONL file; its folder is created if missing |
| `--merge` | `first-seen-wins`, required when more than one `--predictions` file is given; the only merge implemented |

Predictions for images that are not in the manifest are skipped. That is the guard against leakage: pass the training manifest, and evaluation images cannot reach the targets. A manifest that contains evaluation images defeats it.

Each output line is one image:

```json
{"image_id": "grid_0107", "image_path": "images/grid_0107.png", "teacher": "my_detector",
 "boxes": [{"bbox": [412.0, 105.5, 468.2, 160.0], "score": 0.93, "label": "0"}],
 "summary": {"n_boxes": 1, "max_score": 0.93, "top_label": "0"}}
```

`bbox` is `xyxy` in pixels of the original image, whatever the input format was, and `image_id` is the manifest's, so the trainer finds the record even when the predictions only had `image_path`. The tool also writes `summary.json` next to the output file (images written, images with boxes, manifest records, the files read and how many of their lines matched), prints how many records it wrote, and notes the manifest images that have no teacher record: training supervises those toward background only. It exits with code 2 on a missing file, a malformed line or several `--predictions` without `--merge`.

### Several teachers

```bash
python -m detector_to_scout.build_targets \
    --manifest data/train.jsonl \
    --predictions teacher_a_predictions.jsonl \
    --predictions teacher_b_predictions.jsonl \
    --merge first-seen-wins \
    --teacher teacher_a_plus_b \
    --out targets/merged_targets.jsonl
```

For each image, the record of the first file that has one is kept; later files only fill images the earlier ones lack. The paper's runs merged RF-DETR-S (first) and RT-DETRv2 (second) this way, and since RF-DETR-S covered every training image, RT-DETRv2 contributed nothing: the measured recipe is effectively single-teacher (paper §3.2). Passing several `--teacher-targets` files to `train` merges them by the same rule. Per-teacher and union merges are not implemented.
