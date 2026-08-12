#!/usr/bin/env python3
"""Export a trained RT-Scout student to ONNX (opset 17, dynamic batch).

The exported graph takes a preprocessed batch ``images`` of shape
``(B, 3, S, S)`` (letterboxed, ``[0, 1]``, float32 — see the
preprocessing spec in the README / ``infer.py``) and returns:

* ``image_score``  — ``(B,)``   sigmoid of the decoupled image head;
  the deployed anomaly score.
* ``query_scores`` — ``(B, Q)`` sigmoid per-query confidences
  (diagnostic).
* ``query_boxes``  — ``(B, Q, 4)`` normalized ``(cx, cy, w, h)`` in
  ``[0, 1]`` (diagnostic). Multiply by ``image_size`` to get
  letterboxed-canvas pixel coordinates.

The batch dimension is dynamic on the input and all three outputs.
Preprocessing (letterbox + scaling) happens OUTSIDE the graph — feed
the ONNX model exactly the tensor you would feed the PyTorch model.

Usage::

    python -m detector_to_scout.export_onnx \\
        --checkpoint runs/distill_seed1337/final_state.pt \\
        --out runs/distill_seed1337/rt_scout.onnx \\
        --verify

``--verify`` runs onnxruntime on random inputs at batch sizes 1 and 4
and asserts ``|torch - onnx| < 1e-4`` on every output element.

Exit codes: 0 = exported (and verified, if requested); 2 = failure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .model import RTScout, load_model_from_checkpoint

OPSET_DEFAULT = 17


class OnnxWrapper(nn.Module):
    """Inference wrapper flattening :class:`RTScout` outputs to tensors."""

    def __init__(self, model: RTScout) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.model.forward_logits(images)
        image_score = torch.sigmoid(logits.image_logits.squeeze(-1))  # (B,)
        query_scores = torch.sigmoid(logits.class_logits.squeeze(-1))  # (B, Q)
        query_boxes = logits.boxes  # (B, Q, 4)
        return image_score, query_scores, query_boxes


def export(
    checkpoint: str,
    out_path: str,
    *,
    opset: int = OPSET_DEFAULT,
    verify: bool = False,
    atol: float = 1e-4,
) -> None:
    model = load_model_from_checkpoint(checkpoint, device="cpu")
    model.eval()
    wrapper = OnnxWrapper(model).eval()
    size = int(model.image_size)

    dummy = torch.rand(2, 3, size, size, dtype=torch.float32)
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(out_file),
        input_names=["images"],
        output_names=["image_score", "query_scores", "query_boxes"],
        dynamic_axes={
            "images": {0: "batch"},
            "image_score": {0: "batch"},
            "query_scores": {0: "batch"},
            "query_boxes": {0: "batch"},
        },
        opset_version=int(opset),
        do_constant_folding=True,
    )
    print(f"Exported ONNX (opset {opset}, dynamic batch) -> {out_file}")

    import onnx

    onnx_model = onnx.load(str(out_file))
    onnx.checker.check_model(onnx_model)
    print("onnx.checker: model is well-formed")

    if verify:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(out_file), providers=["CPUExecutionProvider"])
        max_diff_overall = 0.0
        for batch in (1, 4):
            x = torch.rand(batch, 3, size, size, dtype=torch.float32)
            with torch.no_grad():
                t_score, t_qscores, t_boxes = wrapper(x)
            o_score, o_qscores, o_boxes = sess.run(None, {"images": x.numpy()})
            for name, t_val, o_val in (
                ("image_score", t_score, o_score),
                ("query_scores", t_qscores, o_qscores),
                ("query_boxes", t_boxes, o_boxes),
            ):
                diff = float(np.max(np.abs(t_val.numpy() - o_val)))
                max_diff_overall = max(max_diff_overall, diff)
                print(f"verify batch={batch} {name}: max |torch - onnx| = {diff:.2e}")
                if diff >= atol:
                    raise AssertionError(
                        f"parity failure: {name} (batch={batch}) max diff {diff:.2e} >= {atol}"
                    )
        print(f"parity OK: max |torch - onnx| = {max_diff_overall:.2e} < {atol}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", required=True, help="final_state.pt from train.py")
    p.add_argument("--out", required=True, help="Output .onnx path")
    p.add_argument("--opset", type=int, default=OPSET_DEFAULT)
    p.add_argument(
        "--verify",
        action="store_true",
        help="Run onnxruntime and assert |torch - onnx| < --atol on all outputs "
        "at batch sizes 1 and 4.",
    )
    p.add_argument("--atol", type=float, default=1e-4)
    args = p.parse_args(argv)

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        print(f"ERROR: checkpoint not found: {ckpt}", file=sys.stderr)
        return 2
    try:
        export(str(ckpt), args.out, opset=args.opset, verify=args.verify, atol=args.atol)
    except (RuntimeError, AssertionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
