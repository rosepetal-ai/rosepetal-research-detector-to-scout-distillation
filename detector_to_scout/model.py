"""RT-Scout student model — compact multi-scale DETR-style anomaly scout.

Self-contained extraction of the model measured in the paper
"Boxes to Scores: Distilling Box-Supervised Defect Detectors into
Compact Image-Level Anomaly Scouts" (the E09 multi-scale distilled
student). No imports from the internal ``rp_detr`` package; only
``torch`` / ``numpy`` / ``timm`` / (optionally) ``scipy``.

Architecture, exactly as implemented and measured:

* Backbone — truncated ResNet-18 via ``timm`` ``features_only`` with
  ``out_indices=(1, 2, 3)``: three feature scales at strides 4/8/16
  with 64/128/256 channels (the final ResNet stage is dropped).
* Per-scale 1x1 projections to a shared width ``d = 256``; each scale
  gets a 2-D sine positional embedding, is flattened, and all scales
  are concatenated along the token axis (5,376 tokens at 256 px input).
* 2-layer pre-norm ``nn.TransformerEncoder`` over the full token set.
* 2-layer ``nn.TransformerDecoder`` over 100 learnable queries
  (zero target + queries added once at the input — the compact
  approximation of DETR's per-layer query addition, mirrored from the
  source implementation).
* Per-query heads (training scaffolding, not deployed outputs):
  a single binary class logit with a RetinaNet-style prior-0.01 bias
  init, and a 3-layer box MLP emitting normalized ``(cx, cy, w, h)``
  through a sigmoid.
* Decoupled image head (the deployed output): LayerNorm -> Linear ->
  GELU -> Linear over the mean-pooled encoder memory; its sigmoid is
  the image-level anomaly score.

The distillation loss below also mirrors the measured implementation:
Hungarian matching (scipy ``linear_sum_assignment``; greedy fallback if
scipy is missing) under the cost ``-IoU + 0.5 * L1 - 0.5 * confidence``,
matched-query BCE + L1 (weight 5.0) + GIoU (weight 2.0), unmatched
queries pushed toward background, plus an image-level BCE (weight 1.0)
against the manifest's binary normal/defective label.

Note on classification targets (paper Section 3.2): the target records
store the teacher's per-box confidence, but the training driver
supervises every matched query with a *hard* target of 1.0 — teacher
confidences are not propagated into the loss. ``distillation_loss``
consumes whatever ``scores`` the caller provides; the trainer in this
repository passes ones, mirroring the measured runs.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

LOGGER = logging.getLogger(__name__)

DEFAULT_IMAGE_SIZE = 256
DEFAULT_NUM_QUERIES = 100
DEFAULT_FEATURE_DIM = 256
DEFAULT_NUM_HEADS = 8
DEFAULT_NUM_ENC_LAYERS = 2
DEFAULT_NUM_DEC_LAYERS = 2
DEFAULT_OUT_STRIDE = 16
DEFAULT_TOP_K = 32
DEFAULT_SCORE_THRESHOLD = 0.05
DEFAULT_MIN_AREA = 4

SUPPORTED_BACKBONES = ("timm:resnet18", "timm:resnet50")


@dataclass
class RTScoutLogits:
    """Raw outputs of :meth:`RTScout.forward_logits`.

    Shapes:

    * ``class_logits``   — ``(B, Q, 1)`` raw binary defect logits.
    * ``boxes``          — ``(B, Q, 4)`` normalized ``(cx, cy, w, h)``
      in ``[0, 1]`` (sigmoid applied inside the forward).
    * ``encoder_memory`` — ``(B, HW, D)`` flattened encoder output.
    * ``image_logits``   — ``(B, 1)`` raw image-level logit from the
      decoupled image head. Its sigmoid is the deployed anomaly score.
    """

    class_logits: torch.Tensor
    boxes: torch.Tensor
    encoder_memory: torch.Tensor
    image_logits: torch.Tensor


# ---------------------------------------------------------------------------
# Position encoding (sine-cosine 2D, standard DETR)
# ---------------------------------------------------------------------------


def sine_pos_embed_2d(h: int, w: int, dim: int, device: torch.device) -> torch.Tensor:
    """Sine-cosine 2D positional embedding of shape ``(H*W, dim)``.

    Half of the channels encode the y coordinate, half encode x.
    Standard DETR-style sin/cos with a 10000 temperature.
    """

    if dim % 2 != 0:
        raise ValueError(f"sine_pos_embed_2d requires even dim, got {dim}")
    half = dim // 2
    if half % 2 != 0:
        # Pad so each axis gets an even number of channels.
        half += 1
    y_embed = torch.linspace(0.0, 1.0, steps=h, device=device).unsqueeze(1).expand(h, w)
    x_embed = torch.linspace(0.0, 1.0, steps=w, device=device).unsqueeze(0).expand(h, w)
    dim_t = torch.arange(half, device=device, dtype=torch.float32)
    dim_t = 10000.0 ** (2.0 * (dim_t // 2) / half)
    pos_y = y_embed.unsqueeze(-1) / dim_t  # (h, w, half)
    pos_x = x_embed.unsqueeze(-1) / dim_t
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos = torch.cat([pos_y, pos_x], dim=-1)  # (h, w, 2*half) >= dim
    pos = pos.reshape(h * w, -1)[:, :dim]
    return pos


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------


def _build_timm_backbone(
    name: str,
    *,
    multi_scale: bool,
    pretrained: bool,
    in_chans: int,
) -> tuple[nn.Module, list[int], list[int]]:
    """Build a timm backbone returning a list of feature maps.

    For the paper's multi-scale student (``multi_scale=True``) the
    backbone is created with ``out_indices=(1, 2, 3)``: for ResNet-18
    this keeps the network through its third stage (the final stage is
    dropped by timm's feature pruning) and yields feature maps at
    strides 4/8/16 with 64/128/256 channels. Single-scale uses
    ``out_indices=(2,)`` (stride 8 for ResNet).
    """

    try:
        import timm  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            f"RT-Scout backbone {name!r} requires `timm` (pip install timm)"
        ) from exc
    timm_name = name.split(":", 1)[1]
    if multi_scale:
        out_indices: tuple[int, ...] = (1, 2, 3)
    else:
        out_indices = (2,)
    backbone = timm.create_model(
        timm_name,
        pretrained=bool(pretrained),
        in_chans=int(in_chans),
        features_only=True,
        out_indices=out_indices,
    )
    channels = list(backbone.feature_info.channels())
    reductions = list(backbone.feature_info.reduction())
    return backbone, channels, reductions


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------


class RTScout(nn.Module):
    """The compact multi-scale DETR-style anomaly scout (the student).

    Parameters
    ----------
    params:
        Dictionary parsed from a YAML config. Recognised keys:

        * ``backbone`` (str) — one of :data:`SUPPORTED_BACKBONES`.
          Paper recipe: ``"timm:resnet18"``.
        * ``image_size`` (int) — square input side. Paper recipe: 256.
        * ``feature_dim`` (int) — transformer model dim. Default 256.
        * ``num_queries`` (int) — learnable queries. Default 100.
        * ``num_heads`` (int) — attention heads. Default 8.
        * ``num_encoder_layers`` / ``num_decoder_layers`` — default 2 / 2.
        * ``multi_scale`` (bool) — paper recipe: ``true``.
        * ``score_threshold`` (float) — minimum sigmoid score for a
          query to surface as an (optional) box output at inference.
        * ``min_area`` (int) — minimum box area in pixels at inference.
        * ``top_k`` (int) — top-K cap on returned boxes at inference.
        * ``timm`` (dict) — extra kwargs (``pretrained``, ``in_chans``)
          forwarded to the backbone builder.
    """

    SUPPORTED_BACKBONES: tuple[str, ...] = SUPPORTED_BACKBONES

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__()
        params = dict(params or {})
        self.params = params
        self.backbone: str = str(params.get("backbone", "timm:resnet18"))
        if self.backbone not in SUPPORTED_BACKBONES:
            raise ValueError(
                f"unknown RT-Scout backbone {self.backbone!r}. Supported: {SUPPORTED_BACKBONES}"
            )
        self.image_size: int = int(params.get("image_size", DEFAULT_IMAGE_SIZE))
        self.feature_dim: int = int(params.get("feature_dim", DEFAULT_FEATURE_DIM))
        self.num_queries: int = int(params.get("num_queries", DEFAULT_NUM_QUERIES))
        self.num_heads: int = int(params.get("num_heads", DEFAULT_NUM_HEADS))
        self.num_encoder_layers: int = int(params.get("num_encoder_layers", DEFAULT_NUM_ENC_LAYERS))
        self.num_decoder_layers: int = int(params.get("num_decoder_layers", DEFAULT_NUM_DEC_LAYERS))
        self.score_threshold: float = float(params.get("score_threshold", DEFAULT_SCORE_THRESHOLD))
        self.min_area: int = int(params.get("min_area", DEFAULT_MIN_AREA))
        self.top_k: int = int(params.get("top_k", DEFAULT_TOP_K))
        self.multi_scale: bool = bool(params.get("multi_scale", True))

        timm_params = dict(params.get("timm") or {})
        self._build_backbone(timm_params)
        self._build_transformer()
        self._build_heads()

    # --- Build helpers ----------------------------------------------------

    def _build_backbone(self, timm_params: dict[str, Any]) -> None:
        backbone, channels, reductions = _build_timm_backbone(
            self.backbone,
            multi_scale=self.multi_scale,
            pretrained=bool(timm_params.get("pretrained", True)),
            in_chans=int(timm_params.get("in_chans", 3)),
        )
        self._backbone_module = backbone
        self._backbone_channels = channels
        self._backbone_reductions = reductions
        # Per-scale 1x1 projection to ``feature_dim`` so the transformer
        # sees a common channel count regardless of the backbone.
        self._input_projs = nn.ModuleList(
            [nn.Conv2d(c, self.feature_dim, kernel_size=1, bias=True) for c in channels]
        )

    def _build_transformer(self) -> None:
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim,
            nhead=self.num_heads,
            dim_feedforward=4 * self.feature_dim,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_encoder_layers)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.feature_dim,
            nhead=self.num_heads,
            dim_feedforward=4 * self.feature_dim,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.num_decoder_layers)
        # Learnable query embeddings — the canonical DETR pattern.
        self.query_embed = nn.Embedding(self.num_queries, self.feature_dim)
        nn.init.normal_(self.query_embed.weight, mean=0.0, std=0.02)

    def _build_heads(self) -> None:
        # Class head: a single defect-vs-background logit per query,
        # with a RetinaNet-style prior-0.01 bias init.
        self.class_head = nn.Linear(self.feature_dim, 1, bias=True)
        prior = 0.01
        nn.init.normal_(self.class_head.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.class_head.bias, float(-math.log((1.0 - prior) / prior)))
        # Box head: a 3-layer MLP outputting normalized (cx, cy, w, h).
        self.box_head = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim, 4),
        )
        for m in self.box_head:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)
        # Decoupled image head (the deployed output). The per-query
        # head's prior-0.01 bias supervises ~99% of queries toward
        # background, which compresses max(sigmoid(class_logits)) near
        # the prior and destroys image-level discrimination — hence a
        # separate head over the mean-pooled encoder memory, with a
        # balanced 0.0 bias init.
        self.image_head = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, 1),
        )
        for m in self.image_head:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

    # --- Forward ---------------------------------------------------------

    def _extract_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Run backbone + project + flatten to ``(B, HW, D)`` tokens.

        All scales are projected to ``feature_dim``, each receives its
        2-D sine position embedding, is flattened, and the scales are
        concatenated along the token axis.
        """

        feats = self._backbone_module(x)
        if not isinstance(feats, (list, tuple)):
            feats = [feats]
        token_chunks: list[torch.Tensor] = []
        device = x.device
        for i, feat in enumerate(feats):
            proj = self._input_projs[i](feat)
            b, c, h, w = proj.shape
            tokens = proj.flatten(2).transpose(1, 2)  # (B, HW, D)
            pos = sine_pos_embed_2d(int(h), int(w), self.feature_dim, device).unsqueeze(0)
            tokens = tokens + pos
            token_chunks.append(tokens)
        if len(token_chunks) == 1:
            return token_chunks[0]
        return torch.cat(token_chunks, dim=1)

    def forward_logits(self, images: torch.Tensor) -> RTScoutLogits:
        """Return raw class + box + image outputs for a batch of images.

        ``images`` is a ``(B, 3, H, W)`` float tensor in ``[0, 1]``
        (letterboxed; NO ImageNet mean/std normalization — see the
        preprocessing spec in the README).
        """

        if images.dim() != 4:
            raise ValueError(f"images must be (B, C, H, W); got {tuple(images.shape)}")
        tokens = self._extract_tokens(images)
        memory = self.encoder(tokens)  # (B, HW, D)
        b = images.shape[0]
        queries = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)  # (B, Q, D)
        # Zero target + learnable queries added once at the input — the
        # compact approximation of DETR's per-layer query addition,
        # mirrored from the measured implementation.
        tgt = torch.zeros_like(queries) + queries
        decoded = self.decoder(tgt, memory)  # (B, Q, D)
        class_logits = self.class_head(decoded)  # (B, Q, 1)
        boxes = torch.sigmoid(self.box_head(decoded))  # (B, Q, 4) in [0, 1]
        # Decoupled image head: mean-pool encoder memory -> MLP -> (B, 1).
        pooled = memory.mean(dim=1)  # (B, D)
        image_logits = self.image_head(pooled)  # (B, 1)
        return RTScoutLogits(
            class_logits=class_logits,
            boxes=boxes,
            encoder_memory=memory,
            image_logits=image_logits,
        )

    @torch.no_grad()
    def forward_infer(self, image: np.ndarray) -> dict[str, Any]:
        """Single-image inference from a raw HWC uint8/float image array.

        Letterboxes the image to ``image_size`` (aspect-preserving,
        zero-padded, centered), scales to ``[0, 1]``, and runs one
        forward pass. Returns a dict with:

        * ``image_score`` — sigmoid of the decoupled image head; the
          deployed anomaly score.
        * ``boxes`` — optional diagnostic boxes: the top-K queries by
          confidence passing the score/area filters, each as
          ``{"xyxy": [x0, y0, x1, y1], "score": s}`` in letterboxed
          canvas pixel coordinates.
        """

        canvas, _meta = letterbox_infer(image, target=self.image_size)
        chw = np.transpose(canvas, (2, 0, 1)).astype(np.float32)
        x = torch.from_numpy(chw).unsqueeze(0)
        device = next(self.parameters()).device
        x = x.to(device)
        logits = self.forward_logits(x)
        scores = torch.sigmoid(logits.class_logits.squeeze(-1)).squeeze(0)  # (Q,)
        boxes = logits.boxes.squeeze(0)  # (Q, 4) cxcywh in [0, 1]
        H = W = int(self.image_size)
        cx = boxes[:, 0] * W
        cy = boxes[:, 1] * H
        bw = boxes[:, 2] * W
        bh = boxes[:, 3] * H
        x0 = (cx - 0.5 * bw).clamp(min=0.0, max=W - 1)
        y0 = (cy - 0.5 * bh).clamp(min=0.0, max=H - 1)
        x1 = (cx + 0.5 * bw).clamp(min=0.0, max=W - 1)
        y1 = (cy + 0.5 * bh).clamp(min=0.0, max=H - 1)
        xyxy = torch.stack([x0, y0, x1, y1], dim=-1)  # (Q, 4)
        score_np = scores.detach().cpu().numpy()
        xyxy_np = xyxy.detach().cpu().numpy()

        # Top-K queries by score; apply score + area filters.
        order = np.argsort(-score_np)
        order = order[: max(0, int(self.top_k))]
        candidates: list[dict[str, Any]] = []
        for idx in order:
            s = float(score_np[idx])
            if s < self.score_threshold:
                continue
            x0f, y0f, x1f, y1f = (float(v) for v in xyxy_np[idx])
            if x1f - x0f <= 0 or y1f - y0f <= 0:
                continue
            area = int(round((x1f - x0f) * (y1f - y0f)))
            if area < self.min_area:
                continue
            candidates.append({"xyxy": [x0f, y0f, x1f, y1f], "score": s})

        image_score = float(torch.sigmoid(logits.image_logits.squeeze(-1)).squeeze(0).item())
        image_score = float(np.clip(image_score, 0.0, 1.0))
        return {"image_score": image_score, "boxes": candidates}

    # --- Persistence -----------------------------------------------------

    def state_dict_payload(self) -> dict[str, Any]:
        """Checkpoint payload with the state dict + the build params."""

        return {
            "model_state": self.state_dict(),
            "model_params": {
                "backbone": self.backbone,
                "image_size": int(self.image_size),
                "feature_dim": int(self.feature_dim),
                "num_queries": int(self.num_queries),
                "num_heads": int(self.num_heads),
                "num_encoder_layers": int(self.num_encoder_layers),
                "num_decoder_layers": int(self.num_decoder_layers),
                "score_threshold": float(self.score_threshold),
                "min_area": int(self.min_area),
                "top_k": int(self.top_k),
                "multi_scale": bool(self.multi_scale),
                "timm": dict(self.params.get("timm") or {}),
            },
        }

    def count_parameters(self) -> int:
        """Sum of checkpoint tensor elements, buffers included.

        This is the parameter-count convention used throughout the
        paper (Section 3): ``sum(t.numel() for t in state_dict())``.
        """

        return int(sum(t.numel() for t in self.state_dict().values()))


def load_model_from_checkpoint(
    checkpoint_path: str,
    *,
    device: torch.device | str = "cpu",
    params: dict[str, Any] | None = None,
) -> RTScout:
    """Rebuild an :class:`RTScout` from a checkpoint file.

    Accepts either the payload format written by ``train.py``
    (``{"model_state": ..., "model_params": ...}``) or a bare
    ``state_dict`` (in which case ``params`` must be provided,
    e.g. from the training config's ``model:`` block).

    Backbone weights come from the checkpoint, so the model is built
    with ``pretrained=False`` — no network access needed at load time.
    """

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
        build_params = dict(ckpt.get("model_params") or {})
        if params:
            build_params.update(params)
    else:
        state = ckpt
        if params is None:
            raise ValueError(
                "checkpoint is a bare state_dict; pass params= (the config's model: block)"
            )
        build_params = dict(params)
    timm_params = dict(build_params.get("timm") or {})
    timm_params["pretrained"] = False  # weights come from the checkpoint
    build_params["timm"] = timm_params
    model = RTScout(build_params)
    model.load_state_dict(state)
    model.to(torch.device(device))
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Inference-side letterbox (torch bilinear — mirrors the measured
# inference path; the training-side letterbox in data.py uses PIL
# bilinear, mirroring the measured training path)
# ---------------------------------------------------------------------------


def letterbox_infer(image: np.ndarray, *, target: int) -> tuple[np.ndarray, tuple[int, int, float]]:
    """Aspect-preserving letterbox of an HWC image to ``(target, target)``.

    Returns ``(canvas_f, (pad_y, pad_x, scale))`` where ``canvas_f`` is
    float32 in ``[0, 1]``. Grayscale inputs are broadcast to 3 channels.
    Resizing uses torch bilinear interpolation (align_corners=False),
    exactly as the measured inference path did.
    """

    if image.ndim == 2:
        image = image[..., None]
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    h, w = image.shape[:2]
    scale = float(target) / float(max(h, w, 1))
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating):
            arr = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        else:
            arr = image.astype(np.uint8)
    else:
        arr = image
    t = torch.from_numpy(arr.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    t = torch.nn.functional.interpolate(t, size=(new_h, new_w), mode="bilinear", align_corners=False)
    resized = t.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
    canvas = np.zeros((target, target, 3), dtype=np.uint8)
    pad_y = (target - new_h) // 2
    pad_x = (target - new_w) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    canvas_f = canvas.astype(np.float32) / 255.0
    return canvas_f, (pad_y, pad_x, scale)


# ---------------------------------------------------------------------------
# Distillation loss (Hungarian matching + matched-query terms + image BCE)
# ---------------------------------------------------------------------------


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    x0 = cx - 0.5 * w
    y0 = cy - 0.5 * h
    x1 = cx + 0.5 * w
    y1 = cy + 0.5 * h
    return torch.stack([x0, y0, x1, y1], dim=-1)


def pairwise_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU between ``a (N, 4)`` and ``b (M, 4)`` in xyxy."""

    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=1e-6)


def generalized_box_iou_loss(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    reduction: str = "none",
    eps: float = 1e-7,
) -> torch.Tensor:
    """GIoU loss between aligned box sets in xyxy, ``loss = 1 - GIoU``.

    Numerically identical re-implementation of
    ``torchvision.ops.generalized_box_iou_loss`` so this repository does
    not depend on torchvision.
    """

    x1, y1, x2, y2 = boxes1.unbind(dim=-1)
    x1g, y1g, x2g, y2g = boxes2.unbind(dim=-1)

    # Intersection + union (element-wise over aligned pairs).
    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)
    intsctk = torch.zeros_like(x1)
    mask = (ykis2 > ykis1) & (xkis2 > xkis1)
    intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk
    iouk = intsctk / (unionk + eps)

    # Smallest enclosing box.
    xc1 = torch.min(x1, x1g)
    yc1 = torch.min(y1, y1g)
    xc2 = torch.max(x2, x2g)
    yc2 = torch.max(y2, y2g)
    area_c = (xc2 - xc1) * (yc2 - yc1)
    miouk = iouk - ((area_c - unionk) / (area_c + eps))
    loss = 1.0 - miouk
    if reduction == "mean":
        return loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    if reduction == "sum":
        return loss.sum()
    return loss


def hungarian_or_greedy_match(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    """Match rows (student queries) to columns (targets) minimizing ``cost``.

    Returns ``(row_idx, col_idx, algorithm)`` where ``algorithm`` is
    ``"hungarian"`` (scipy ``linear_sum_assignment``) if scipy is
    available, otherwise ``"greedy"`` — pick the globally smallest cost
    pair, mark row/column used, repeat. The greedy fallback mirrors the
    source implementation exactly and is logged.
    """

    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(cost)
        return rows, cols, "hungarian"
    except ImportError:
        LOGGER.warning("scipy unavailable; falling back to greedy IoU matching")
    n, m = cost.shape
    used_r = np.zeros(n, dtype=bool)
    used_c = np.zeros(m, dtype=bool)
    rows_l: list[int] = []
    cols_l: list[int] = []
    pairs = sorted(
        ((cost[i, j], i, j) for i in range(n) for j in range(m)),
        key=lambda t: t[0],
    )
    for _, i, j in pairs:
        if used_r[i] or used_c[j]:
            continue
        used_r[i] = True
        used_c[j] = True
        rows_l.append(i)
        cols_l.append(j)
        if len(rows_l) == min(n, m):
            break
    return np.array(rows_l, dtype=np.int64), np.array(cols_l, dtype=np.int64), "greedy"


def distillation_loss(
    student_logits: RTScoutLogits,
    teacher_targets: list[dict[str, Any]],
    *,
    class_weight: float = 1.0,
    l1_weight: float = 5.0,
    giou_weight: float = 2.0,
    image_weight: float = 1.0,
) -> dict[str, Any]:
    """Hungarian-matched class + box distillation loss + image-level BCE.

    Parameters
    ----------
    student_logits:
        Output of :meth:`RTScout.forward_logits`.
    teacher_targets:
        List of length ``B``, one per image. Each entry is a dict with
        keys ``boxes`` (Tensor ``(N, 4)`` in normalized cxcywh),
        ``scores`` (Tensor ``(N,)`` in ``[0, 1]`` — the trainer passes
        hard 1.0 targets, see the module docstring), and
        ``image_label`` (Tensor ``(1,)``, 1.0 = defective / 0.0 =
        normal, from the images manifest). Empty target sets (no boxes)
        contribute only the background term.
    image_weight:
        Weight of the per-image BCE term between the decoupled image
        head and ``image_label``. Paper recipe: 1.0.

    Returns
    -------
    Dict with the total loss, per-term scalars (``loss_class``,
    ``loss_l1``, ``loss_giou``, ``loss_image``), the matched-pair
    count, the no-object count, and the matching algorithm used.
    """

    student_class = student_logits.class_logits  # (B, Q, 1)
    student_boxes = student_logits.boxes  # (B, Q, 4) cxcywh in [0, 1]
    B, Q, _ = student_class.shape
    device = student_class.device
    dtype = student_class.dtype

    total_class = torch.zeros((), device=device, dtype=dtype)
    total_l1 = torch.zeros((), device=device, dtype=dtype)
    total_giou = torch.zeros((), device=device, dtype=dtype)
    n_matched = 0
    n_no_object = 0
    used_algorithm: str | None = None

    for b in range(B):
        target = teacher_targets[b]
        t_boxes = target.get("boxes")
        t_scores = target.get("scores")
        # No-object case: every student query should learn the
        # background class — supervise the class head against zero.
        if t_boxes is None or (hasattr(t_boxes, "numel") and t_boxes.numel() == 0):
            class_target = torch.zeros_like(student_class[b])
            total_class = total_class + nn.functional.binary_cross_entropy_with_logits(
                student_class[b], class_target, reduction="mean"
            )
            n_no_object += 1
            continue
        if not isinstance(t_boxes, torch.Tensor):
            t_boxes = torch.as_tensor(t_boxes, device=device, dtype=dtype)
        else:
            t_boxes = t_boxes.to(device=device, dtype=dtype)
        if not isinstance(t_scores, torch.Tensor):
            t_scores = torch.as_tensor(t_scores, device=device, dtype=dtype)
        else:
            t_scores = t_scores.to(device=device, dtype=dtype)

        # Matching cost: -IoU + 0.5 * L1 - 0.5 * student confidence.
        s_xyxy = box_cxcywh_to_xyxy(student_boxes[b])
        t_xyxy = box_cxcywh_to_xyxy(t_boxes)
        iou = pairwise_iou_xyxy(s_xyxy, t_xyxy)
        l1_cost = torch.cdist(student_boxes[b], t_boxes, p=1)
        s_conf = torch.sigmoid(student_class[b].squeeze(-1)).unsqueeze(1)  # (Q, 1)
        cost = (-iou) + 0.5 * l1_cost - 0.5 * s_conf
        rows, cols, algo = hungarian_or_greedy_match(cost.detach().cpu().numpy())
        used_algorithm = used_algorithm or algo

        # Class supervision: matched queries take the provided target
        # score (the trainer passes 1.0 — hard targets, paper §3.2);
        # unmatched queries are pushed toward 0 (background).
        class_target = torch.zeros_like(student_class[b])
        if rows.size:
            r_t = torch.as_tensor(rows, device=device, dtype=torch.long)
            c_t = torch.as_tensor(cols, device=device, dtype=torch.long)
            class_target[r_t, 0] = t_scores[c_t].clamp(0.0, 1.0)
            matched_pred_boxes = student_boxes[b][r_t]
            matched_target_boxes = t_boxes[c_t]
            total_l1 = total_l1 + nn.functional.l1_loss(
                matched_pred_boxes, matched_target_boxes, reduction="mean"
            )
            total_giou = total_giou + generalized_box_iou_loss(
                box_cxcywh_to_xyxy(matched_pred_boxes),
                box_cxcywh_to_xyxy(matched_target_boxes),
                reduction="mean",
            )
            n_matched += int(rows.size)
        total_class = total_class + nn.functional.binary_cross_entropy_with_logits(
            student_class[b], class_target, reduction="mean"
        )

    # Average per-batch terms.
    denom = max(1, B)
    loss_class = total_class / denom
    loss_l1 = total_l1 / max(1, B - n_no_object)
    loss_giou = total_giou / max(1, B - n_no_object)

    # Image-level BCE on the decoupled head. The label comes from the
    # images manifest (normal/defective), never from teacher-box
    # presence: detector teachers emit boxes on every image, so a
    # box-presence label would collapse to a constant 1. The
    # box-presence fallback below only fires if a target entry carries
    # no ``image_label`` key at all.
    image_loss = torch.zeros((), device=device, dtype=dtype)
    if float(image_weight) != 0.0:
        labels: list[float] = []
        for t in teacher_targets:
            il = t.get("image_label")
            if il is not None:
                if isinstance(il, torch.Tensor):
                    labels.append(float(il.flatten()[0].item()))
                else:
                    labels.append(float(il))
            else:
                tb = t.get("boxes")
                has_boxes = bool(
                    tb is not None
                    and (
                        (hasattr(tb, "numel") and tb.numel() > 0)
                        or (not hasattr(tb, "numel") and len(tb) > 0)
                    )
                )
                labels.append(1.0 if has_boxes else 0.0)
        image_labels = torch.tensor(labels, device=device, dtype=dtype).unsqueeze(-1)  # (B, 1)
        image_loss = nn.functional.binary_cross_entropy_with_logits(
            student_logits.image_logits,
            image_labels,
            reduction="mean",
        )

    total = (
        float(class_weight) * loss_class
        + float(l1_weight) * loss_l1
        + float(giou_weight) * loss_giou
        + float(image_weight) * image_loss
    )
    return {
        "loss": total,
        "loss_class": loss_class.detach(),
        "loss_l1": loss_l1.detach(),
        "loss_giou": loss_giou.detach(),
        "loss_image": image_loss.detach(),
        "n_matched": torch.as_tensor(n_matched, device=device),
        "n_no_object": torch.as_tensor(n_no_object, device=device),
        "match_algorithm": used_algorithm or "none",
    }


__all__ = [
    "DEFAULT_IMAGE_SIZE",
    "DEFAULT_NUM_QUERIES",
    "DEFAULT_FEATURE_DIM",
    "DEFAULT_TOP_K",
    "RTScout",
    "RTScoutLogits",
    "SUPPORTED_BACKBONES",
    "distillation_loss",
    "generalized_box_iou_loss",
    "hungarian_or_greedy_match",
    "letterbox_infer",
    "load_model_from_checkpoint",
    "sine_pos_embed_2d",
]
