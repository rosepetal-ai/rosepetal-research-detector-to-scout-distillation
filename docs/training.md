---
title: "Training"
description: "Distilling teacher targets into the RT-Scout student: the paper recipe, configuration keys, train options and outputs, resuming a run, warm-started fine-tuning, the student architecture and the loss."
sidebar_position: 2
---

# Training

`train` reads the image manifests and the targets file, trains the student, scores the validation images and writes everything to a run folder.

```bash
python -m detector_to_scout.train \
    --config configs/distill_default.yaml \
    --teacher-targets targets/teacher_targets.jsonl \
    --run-dir runs/distill_seed1337 \
    --device cuda
```

The manifests come from the configuration only: set `manifests.train` and `manifests.val` in it before the first run.

## The paper recipe

[`configs/distill_default.yaml`](https://github.com/rosepetal-ai/rosepetal-research-detector-to-scout-distillation/blob/main/configs/distill_default.yaml) is the configuration of the paper's distilled student:

| Setting | Value |
|---------|-------|
| Steps | 30,000 |
| Batch size | 16 |
| Optimizer | AdamW, learning rate 1e-4, weight decay 1e-4 |
| Precision | fp32 |
| Input | 256 px letterbox (aspect kept, centered on black), pixels in `[0, 1]`, no ImageNet mean/std normalization |
| Seed | 1337 |
| Loss weights | class BCE 1.0, box L1 5.0, GIoU 2.0, image BCE 1.0 |
| Matching | Hungarian (SciPy `linear_sum_assignment`), cost `-IoU + 0.5·L1 - 0.5·confidence` |
| Gradient clipping | Global norm 1.0 (fixed in the code) |
| Backbone initialization | ImageNet-pretrained, from `timm` |

### Configuration keys

| Key | Value in `distill_default.yaml` | Meaning |
|-----|--------------------------------|---------|
| `model.backbone` | `timm:resnet18` | `timm:resnet18` or `timm:resnet50` |
| `model.image_size` | `256` | Side of the square input |
| `model.feature_dim` | `256` | Width of the transformer |
| `model.num_queries` | `100` | Learnable decoder queries |
| `model.num_heads` | `8` | Attention heads |
| `model.num_encoder_layers` | `2` | Encoder layers |
| `model.num_decoder_layers` | `2` | Decoder layers |
| `model.multi_scale` | `true` | Three backbone levels; `false` uses one |
| `model.score_threshold` | `0.05` | Lowest query confidence for a diagnostic box at inference |
| `model.min_area` | `4` | Smallest area of a diagnostic box, in pixels of the input canvas |
| `model.top_k` | `32` | Most diagnostic boxes returned per image |
| `model.timm.pretrained` | `true` | Start the backbone from ImageNet weights (downloaded by `timm` on first use) |
| `model.timm.in_chans` | `3` | Input channels |
| `manifests.train` | `data/train.jsonl` | Training manifest (to edit) |
| `manifests.val` | `data/val.jsonl` | Validation manifest (to edit) |
| `teacher_targets` | `[targets/teacher_targets.jsonl]` | Targets files; `--teacher-targets` replaces the list |
| `training.max_steps` | `30000` | Optimizer steps (one batch each) |
| `training.batch_size` | `16` | Images per step |
| `training.learning_rate` | `1.0e-4` | AdamW learning rate |
| `training.weight_decay` | `1.0e-4` | AdamW weight decay |
| `training.seed` | `1337` | Seed of the shuffling and of the initialization |
| `training.loss.class_weight` | `1.0` | Weight of the per-query class BCE |
| `training.loss.l1_weight` | `5.0` | Weight of the box L1 loss |
| `training.loss.giou_weight` | `2.0` | Weight of the GIoU loss |
| `training.loss.image_weight` | `1.0` | Weight of the image-level BCE |

Relative paths in the configuration are looked up in the current directory first, then in the configuration file's directory. Both manifests must exist; the run stops with exit code 2 otherwise.

## Options

| Option | Default | Meaning |
|--------|---------|---------|
| `--config` | required | YAML configuration |
| `--run-dir` | required | Run folder, created if missing |
| `--device` | `auto` | `auto` takes CUDA when available; `cuda` stops with `--device cuda requested but no GPU is reachable` instead of falling back to the CPU; `cpu` |
| `--seed` | `training.seed` | Seed |
| `--max-steps` | `training.max_steps` | Number of steps |
| `--limit` | none | Read only the first N rows of each manifest (smoke runs) |
| `--teacher-targets` | `teacher_targets` | Targets file; repeat it for several, merged first-seen-wins |
| `--init-checkpoint` | none | Start from the weights of an earlier run (see [fine-tuning](#fine-tuning)) |
| `--resume-from` | none | Continue an interrupted run from a `last_state.pt` |
| `--log-every` | `10` | Steps between log lines |
| `--checkpoint-every` | `2000` | Steps between `last_state.pt` saves; `0` disables them |

## Outputs

| File in the run folder | Contents |
|------------------------|----------|
| `train_log.jsonl` | One line per step: `loss`, `loss_class`, `loss_l1`, `loss_giou`, `loss_image`, `n_matched` (queries matched in the batch) and `match_algorithm` (`hungarian`, `greedy` without SciPy, or `none` when no image of the batch had target boxes) |
| `last_state.pt` | Every `--checkpoint-every` steps: weights, optimizer, random-generator states and step, for resuming |
| `final_state.pt` | The trained student: weights (`model_state`), architecture (`model_params`), `step` and `seed`. This is the checkpoint `infer` and `export_onnx` read |
| `predictions.jsonl` | One line per validation image: `image_id`, `image_path`, `image_score`, the diagnostic `boxes` and `latency_ms`. The label is not copied |
| `metrics.json` | `image_auroc` on the validation set (`null` when it holds a single class), record counts, final and mean last-50 training loss, `max_steps`, `seed` |
| `summary.json` | The resolved recipe: configuration path, seed, device, targets files, warm-start checkpoint, steps, batch size, learning rate, weight decay, loss weights, size of the model, training time and the metrics |

The validation images are scored one at a time through the inference path ([Inference and export](./inference.md#preprocessing)). The log prints the size of the model (6,814,165 tensor elements for the paper's architecture) and, every `--log-every` steps, the losses and the matching algorithm.

## Resuming

```bash
python -m detector_to_scout.train --config configs/distill_default.yaml \
    --teacher-targets targets/teacher_targets.jsonl \
    --run-dir runs/distill_seed1337 --resume-from runs/distill_seed1337/last_state.pt \
    --device cuda
```

Resuming restores the weights, the optimizer, the random-generator states and the step counter, and appends to `train_log.jsonl`. A run folder that already holds a `last_state.pt` is resumed even without `--resume-from`: start a new run in a new folder.

## Fine-tuning

To adapt a trained student to a new production line, write the new manifests, run a detector over the new training images, build the targets, and warm-start from the earlier `final_state.pt`:

```bash
python -m detector_to_scout.train \
    --config configs/finetune_example.yaml \
    --teacher-targets targets/new_dataset_targets.jsonl \
    --init-checkpoint runs/distill_seed1337/final_state.pt \
    --run-dir runs/finetune_new_dataset \
    --device cuda
```

`--init-checkpoint` loads the weights only: the optimizer and the step counter start fresh. The model is first built from the configuration, so its `model:` block must describe the same architecture as the checkpoint; with `pretrained: true` the ImageNet weights are still fetched before the checkpoint replaces them. [`configs/finetune_example.yaml`](https://github.com/rosepetal-ai/rosepetal-research-detector-to-scout-distillation/blob/main/configs/finetune_example.yaml) keeps the paper recipe with 5,000 steps and a learning rate of 5e-5; set its manifests first. Without `--init-checkpoint`, the same command trains from ImageNet weights.

Other runs of the README, for reference:

```bash
# Another seed, everything else identical
python -m detector_to_scout.train --config configs/distill_default.yaml \
    --teacher-targets targets/teacher_targets.jsonl \
    --run-dir runs/distill_seed2027 --seed 2027 --device cuda

# Short pilot run to check a new dataset
python -m detector_to_scout.train --config configs/distill_default.yaml \
    --teacher-targets targets/teacher_targets.jsonl \
    --run-dir runs/pilot --max-steps 2000 --device cuda
```

After very few steps every image scores close to 0.5: the image head starts with a zero bias and small weights, so its output starts near zero.

## The student

`RTScout` (in `detector_to_scout/model.py`):

- **Backbone.** ResNet-18 from `timm`, kept through its third stage: three feature maps at strides 4, 8 and 16 with 64, 128 and 256 channels.
- **Tokens.** Each map is projected to 256 channels by a 1×1 convolution, given a 2-D sine position embedding and flattened; at 256 px the three scales give 5,376 tokens.
- **Transformer.** A 2-layer pre-norm encoder over all tokens, and a 2-layer decoder over 100 learnable queries.
- **Per-query heads.** A class logit (initialized to a 0.01 prior) and a 3-layer box MLP that outputs normalized `(cx, cy, w, h)`. They exist to take the distillation losses; their boxes are only diagnostics at inference.
- **Image head.** LayerNorm, Linear, GELU and Linear over the mean of the encoder output. Its sigmoid is the anomaly score, the only deployed output.

The model has 6,814,165 tensor elements in its state dict (parameters and buffers, the paper's counting convention): 6.81 M.

## The loss

For each image of a batch:

1. The target boxes are letterboxed with the image; boxes 1 pixel wide or high or less after scaling are dropped.
2. The 100 queries are matched to the target boxes by the Hungarian algorithm, with the cost `-IoU + 0.5 × L1 - 0.5 × c`, where L1 is the distance between normalized box coordinates and `c` the query's predicted confidence. Without SciPy a greedy matching is used instead, with a warning.
3. Matched queries learn a class target of **1.0**, whatever the teacher's confidence (hard targets, paper §3.2), plus an L1 and a GIoU loss on their box. Unmatched queries learn background (0). An image with no target boxes, including one without a teacher record, has all its queries learn background.

The image head learns a binary cross-entropy against the manifest's label (1 for `nok`, 0 otherwise), never against the presence of boxes: the teacher draws boxes on normal images too. The total is `class_weight × class BCE + l1_weight × L1 + giou_weight × GIoU + image_weight × image BCE`, with gradients clipped to a global norm of 1.0; a step whose loss is not finite is skipped.
