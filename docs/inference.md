---
title: "Inference and Export"
description: "Scoring images with a trained RT-Scout student in PyTorch, its preprocessing, ONNX export with a parity check, ONNX Runtime usage, and choosing an operating threshold on validation scores."
sidebar_position: 3
---

# Inference and Export

At inference the student runs without its teacher: one forward pass per image and one score, the sigmoid of the image head, in `[0, 1]`, higher meaning more likely defective. The per-query boxes are also available, as diagnostics only.

## Preprocessing

`infer`, `ScoutPredictor` and the validation scoring at the end of training all prepare an image with `letterbox_infer` (in `detector_to_scout/model.py`):

1. Load the image as RGB (a grayscale array is repeated over 3 channels).
2. Resize it so its longest side is 256 pixels, keeping the aspect ratio, with PyTorch bilinear interpolation (`align_corners=False`).
3. Center it on a black 256×256 canvas.
4. Scale the pixels to `[0, 1]` as float32, channels first.

There is **no ImageNet mean/std normalization**, although the backbone was initialized from ImageNet: do not add it. Training letterboxes its images the same way but resizes them with PIL's bilinear filter, as the paper's training code did, so the two paths are not bit-identical. A consumer of the ONNX model should reproduce `letterbox_infer`, the path the scores were measured with.

## Scoring with PyTorch

```bash
python -m detector_to_scout.infer \
    --checkpoint runs/distill_seed1337/final_state.pt \
    --images line_cam_001.png line_cam_002.png \
    --device cuda --out scores.jsonl --include-boxes

# or every image of a manifest
python -m detector_to_scout.infer \
    --checkpoint runs/distill_seed1337/final_state.pt \
    --manifest data/val.jsonl --device cuda --out scores.jsonl
```

| Option | Default | Meaning |
|--------|---------|---------|
| `--checkpoint` | required | `final_state.pt` of a training run |
| `--images` | none | Image files to score |
| `--manifest` | none | A manifest whose images to score; with `--images` too, both are scored |
| `--device` | `cpu` | `cpu` or `cuda`; `cuda` stops with an error when no GPU is reachable |
| `--out` | standard output | JSONL file for the scores |
| `--include-boxes` | off | Add the diagnostic boxes to each row |

Each output row has `image_id` (the manifest's, or the file path for `--images`), `image_path`, `image_score`, `latency_ms` and, with `--include-boxes`, `boxes`. The tool exits with code 2 when it is given no images, when the checkpoint, the manifest or an `--images` file does not exist, or when CUDA is not available.

The diagnostic boxes are the queries with the highest confidence, at most `top_k` (32), with a confidence of at least `score_threshold` (0.05) and an area of at least `min_area` (4) pixels, each as `{"xyxy": [x0, y0, x1, y1], "score": s}` in pixels of the 256×256 canvas. They are a training by-product: the paper found they do not localise defects (§7).

From Python:

```python
from detector_to_scout.infer import ScoutPredictor

predictor = ScoutPredictor("runs/distill_seed1337/final_state.pt", device="cuda")
result = predictor.score_image("line_cam_001.png")
print(result["image_score"])   # float in [0, 1], the deployed output
print(result["boxes"])         # diagnostic boxes, canvas pixel coordinates
```

`ScoutPredictor(checkpoint, *, device="cpu", params=None)` loads the student once; `device="cuda"` raises an error when no GPU is reachable. `params` is needed only for a checkpoint saved as a bare state dict: pass the `model:` block of its configuration. `score_image` takes a path or an HWC image array and returns `image_score`, `boxes` and `latency_ms`. Loading a checkpoint never downloads backbone weights.

## ONNX export

```bash
python -m detector_to_scout.export_onnx \
    --checkpoint runs/distill_seed1337/final_state.pt \
    --out runs/distill_seed1337/rt_scout.onnx \
    --verify
```

| Option | Default | Meaning |
|--------|---------|---------|
| `--checkpoint` | required | `final_state.pt` of a training run |
| `--out` | required | Path of the `.onnx` file |
| `--opset` | `17` | ONNX opset |
| `--verify` | off | Compare PyTorch and ONNX Runtime outputs after export |
| `--atol` | `1e-4` | Largest accepted difference for `--verify` |

The export runs on the CPU and checks the file with `onnx.checker`. The graph takes `images`, a float32 `[batch, 3, 256, 256]` tensor already preprocessed, and returns:

| Output | Shape | Contents |
|--------|-------|----------|
| `image_score` | `[batch]` | The anomaly score, the deployed output |
| `query_scores` | `[batch, 100]` | Confidence of each query (diagnostic) |
| `query_boxes` | `[batch, 100, 4]` | Normalized `(cx, cy, w, h)` of each query (diagnostic); multiply by 256 for canvas pixels |

The batch size is dynamic; the image size is fixed. With `--verify`, the tool feeds random tensors at batch sizes 1 and 4 to both PyTorch and ONNX Runtime and fails (exit code 2) if any element of any output differs by `--atol` or more. Verify after every export and after changing the versions of PyTorch, ONNX or ONNX Runtime. Since the check uses random tensors, it does not test your preprocessing: compare a few real images scored in both stacks for that.

### ONNX Runtime

```python
import numpy as np
import onnxruntime as ort
from PIL import Image
from detector_to_scout.model import letterbox_infer

sess = ort.InferenceSession("runs/distill_seed1337/rt_scout.onnx",
                            providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
canvas, _ = letterbox_infer(np.asarray(Image.open("line_cam_001.png").convert("RGB")), target=256)
x = np.transpose(canvas, (2, 0, 1))[None].astype(np.float32)   # (1, 3, 256, 256)
image_score, query_scores, query_boxes = sess.run(None, {"images": x})
print(float(image_score[0]))
```

The student has 6,814,165 tensor elements in its state dict (6.81 M, buffers included), about 26 MiB as fp32: about 4.7 times fewer than the RF-DETR-S teacher (31.8 M) and 3.0 times fewer than RT-DETRv2 (20.5 M). If only the image score is needed, the diagnostic outputs can be ignored. The paper did not measure the student's latency on any hardware.

## Choosing a threshold

The student gives a score, not a decision. Where to cut it is a trade-off between false alarms and missed defects: choose it on validation images, never on the test set, and report the threshold-free AUROC and AUPR next to any thresholded result. The training run's `predictions.jsonl` holds the validation scores without labels; this snippet joins them with the validation manifest by `image_id` (every manifest row needs one) and finds the thresholds that maximise Youden's index and F1:

```python
import json
import numpy as np

preds = [json.loads(l) for l in open("runs/finetune_new_dataset/predictions.jsonl")]
val = {json.loads(l)["image_id"]: json.loads(l)["label"] for l in open("data/new_dataset/val.jsonl")}
y = np.array([1 if val[p["image_id"]] == "nok" else 0 for p in preds])
s = np.array([p["image_score"] for p in preds])

taus = np.unique(s)
best = {"youden": (None, -1.0), "f1": (None, -1.0)}
for tau in taus:
    yhat = (s >= tau).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum()); fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum()); tn = int(((yhat == 0) & (y == 0)).sum())
    tpr = tp / max(tp + fn, 1); fpr = fp / max(fp + tn, 1)
    youden = tpr - fpr
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    if youden > best["youden"][1]: best["youden"] = (float(tau), youden)
    if f1 > best["f1"][1]: best["f1"] = (float(tau), f1)
print("tau* (Youden):", best["youden"], " tau* (F1):", best["f1"])
```

An image is flagged when its score is at or above the chosen threshold. The paper itself tunes no threshold: it reports false-alarm and missed-defect rates at a fixed 0.5, descriptively only.
