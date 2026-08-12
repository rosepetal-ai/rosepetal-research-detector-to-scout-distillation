"""detector_to_scout — companion code for the paper
"Boxes to Scores: Distilling Box-Supervised Defect Detectors into
Compact Image-Level Anomaly Scouts".

Pipeline: build_targets (teacher predictions -> pseudo-box targets)
-> train (Hungarian-matched distillation + image-label BCE)
-> infer (teacher-free image-level anomaly scoring)
-> export_onnx (opset 17, dynamic batch).
"""

from .model import (
    RTScout,
    RTScoutLogits,
    distillation_loss,
    letterbox_infer,
    load_model_from_checkpoint,
)

__version__ = "1.0.0"

__all__ = [
    "RTScout",
    "RTScoutLogits",
    "distillation_loss",
    "letterbox_infer",
    "load_model_from_checkpoint",
    "__version__",
]
