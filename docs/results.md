---
title: "Paper Results"
description: "What the paper Boxes to Scores measured on MVTec AD: the distilled student against the image-label control, the supervision landscape, retention and compression, historical cross-dataset cells, the protocol and its limits."
sidebar_position: 4
---

# Paper Results

Every number on this page comes from the paper ([PDF](./paper.pdf)); the section or table is given in parentheses. They hold for the paper's dataset, pools and protocol, described first.

## What this repository reproduces

The code implements the student, the target construction, the matching and losses, and the training recipe exactly as the paper describes them, and runs the student in PyTorch and ONNX. It does not include the detector teachers, their cached predictions, the paper's training pool and evaluation set, or any trained checkpoint, so the numbers below cannot be reproduced with it alone.

## Protocol

- **Data** (§4.1, §4.2). MVTec AD under a pooled-category protocol: one model for all 15 categories, scored once on one held-out evaluation set of 802 images (614 normal, 188 defective). This is not the per-category protocol of the MVTec AD literature, so the numbers are not comparable with per-category leaderboards. The student and the control train on a box-annotated pool of 3,750 images (2,868 normal, 882 defective); their training and evaluation images are disjoint, checked by image identifier.
- **Teachers** (§3.1, §4.2). RF-DETR-S (31.8 M parameters, DINOv2-S backbone) and RT-DETRv2 (20.5 M, ResNet-18 backbone). Their predictions over the 3,750 images were cached once; merged first-seen-wins, RF-DETR-S was the only effective source of targets (§3.2). The teacher checkpoints that produced the targets were trained on a broader detection split of MVTec AD that contains 711 of the 802 evaluation images: an indirect channel from the evaluation set into the targets, which the paper discloses as a limitation. The teacher rows of the tables below come from separate checkpoints trained on the pool itself.
- **Student** (§3.4). 30,000 steps, batch size 16, AdamW (learning rate 1e-4, weight decay 1e-4), fp32, 256 px, a single seed (1337).
- **Control** (§4.3). The same architecture trained on the same pool with image labels only, boxes withheld, under its original recipe of 1,500 steps, at seeds 1337 and 2027 (a third planned seed was deferred). The rule was registered before the control ran: distillation counts as useful only if it beats every control seed by more than 0.01 AUROC.
- **Metrics** (§4.4). Image AUROC with AUPR; a paired bootstrap over the 802 images (1,000 resamples, 95 % percentile intervals) for the margins. False-alarm and missed-defect rates at a fixed threshold of 0.5 are descriptive only; no threshold is tuned.
- **Parameters** (§3.1). Counted as the sum of the checkpoint's tensor elements, buffers included.

## The control comparison

On the evaluation set (§5.1):

| Model | AUROC | AUPR |
|-------|------:|-----:|
| Distilled student | 0.9614 | 0.9323 |
| Image-label control, seed 1337 | 0.9114 | 0.8311 |
| Image-label control, seed 2027 | 0.9320 | 0.8612 |

The student's AUROC margins are +0.050 over the seed-1337 control (95 % CI 0.030 to 0.071, ahead in 100 % of resamples) and +0.029 over the seed-2027 control (95 % CI 0.010 to 0.048, ahead in 99.8 %): five and about three times the 0.01 bar. The AUPR margins are +0.101 and +0.071. The control's two seeds land in the range of the weak-label scouts trained on much smaller image-label budgets (0.9110 and 0.9339, Table 1): with this architecture, more image labels do not close the gap, and the box-level targets carry the gain.

At the fixed threshold of 0.5 (descriptive), the student gives 342.0 false alarms per 10,000 normal images at a missed-defect rate of 0.1596, against 472.3 and 0.3777 for the seed-1337 control and 260.6 and 0.3670 for the seed-2027 control.

## The supervision landscape

Every row a single-seed measurement on the same 802 images (§5.2, Table 1):

| Model | Supervision | AUROC | AUPR |
|-------|-------------|------:|-----:|
| RF-DETR-S (strongest teacher) | Boxes | 0.9977 | 0.9961 |
| D-FINE-S (reference) | Boxes | 0.9745 | 0.9655 |
| Distilled student | Teacher pseudo-boxes and image labels | 0.9614 | 0.9323 |
| RT-DETRv2 (second teacher) | Boxes | 0.9344 | 0.8966 |
| Weak-label scout, few-defect pool | Image labels, smaller budget | 0.9339 | 0.8715 |
| PatchCore | Normal images only | 0.9242 | 0.8507 |
| Image-label control (seed 1337) | Image labels, full pool | 0.9114 | 0.8311 |
| Weak-label scout, balanced pool | Image labels, smaller budget | 0.9110 | 0.8019 |

The two weak-label scout rows have the same values, on the same 802 images, as the scout of the companion project [Selective Inspection](https://github.com/rosepetal-ai/rosepetal-research-selective-inspection), trained on its few-NOK and balanced pools.

The student does not beat its strongest teacher, and the paper does not claim it does: the claim is that no model trained on less than box-level supervision reaches it on this set. The student never sees a ground-truth box; the box annotation cost sits upstream, in its teachers.

## Retention and compression

(§5.3, Table 2)

| Model | Parameters | AUROC | Share of the strongest teacher's AUROC |
|-------|-----------:|------:|---------------------------------------:|
| RF-DETR-S (teacher) | 31.8 M | 0.9977 | 100 % |
| RT-DETRv2 (teacher) | 20.5 M | 0.9344 | 93.7 % |
| Distilled student | 6.81 M | 0.9614 | 96.4 % |

The student has about 4.7 times fewer parameters than RF-DETR-S and 3.0 times fewer than RT-DETRv2, and scores above the latter. Parameter count is the paper's only efficiency claim: the student's latency and throughput were not measured.

## Other benchmarks

Before a project-wide dataset freeze, the same recipe was run on four other benchmarks, each with teachers trained on that dataset and a test set disjoint from its training pool (§6, Table 3). These cells are historical, single-seed and secondary evidence:

| Benchmark | Image AUROC |
|-----------|------------:|
| MVTec AD (main result) | 0.9614 |
| MPDD | 0.9921 |
| MVTec AD 2 | 0.9703 |
| VisA | 0.8772 (with a disclosed score-polarity caveat) |
| Real-IAD | 0.9426 |

An earlier VisA test set overlapped the training pool and gave an inflated 0.9464; only the disjoint values are reported.

## What the gain is not

From the paper's analysis (§7):

- **Not localisation.** The student's boxes are a training by-product and do not localise defects (box Recall@Budget AUC 0.104); only the image score is deployed.
- **Not ground-truth boxes.** The same architecture trained on ground-truth boxes instead of teacher predictions collapses to 0.5584 (multi-scale) and 0.6128 (single-scale). Unlike ground truth, teacher boxes cover every training image, normal ones included; the collapse is observed, not explained.
- **Not soft-target distillation.** The targets are hard pseudo-boxes; teacher confidences do not enter the loss.

## Limitations

The paper lists them in §9:

- The student is a single training run, compared with two control seeds; the bootstrap intervals cover scoring variability on the fixed set, not training-seed variability.
- The main result comes from one benchmark and one evaluation set; the cross-dataset cells are historical.
- Latency is unprofiled; nothing in the paper is a speed claim.
- The two-teacher merge is effectively single-teacher; per-teacher and union ablations are future work.
- The recipe needs a box-annotated pool and a detector trained on it: annotation cost moves upstream, it is not removed. The teacher's training data matters, and the target-generating teachers overlap the evaluation set as described above.
- The control kept its original 1,500-step recipe; a control trained for as many steps as the student was not run.
- The protocol is pooled-category and not comparable with per-category results.
