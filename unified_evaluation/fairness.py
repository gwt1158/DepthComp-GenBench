"""Comparison Protocol V2 fairness artifact validation."""

from __future__ import annotations

import json
from pathlib import Path


REQUIRED_FIELDS = (
    "method", "version", "training_domain", "seed", "fairness_class",
    "controlled_ranking", "unfair_reason", "native_loss",
    "valid_pixel_masking", "zero_coded_unlabelled_pixels_in_loss",
    "external_pretraining", "foundation_model", "official_mixed_training",
    "TTA", "OOD_specific_tuning", "target_overlap", "loss_modified",
    "network_modified", "optimizer_modified", "scheduler_modified",
    "primary_output", "notes",
)

FAIRNESS_CLASSES = {
    "CONTROLLED_NATIVE", "DIRECT_TRANSFER_UNFAIR", "EXTERNAL_PRIOR",
    "OFFICIAL_MIXED_DATA", "OFFICIAL_OOD_TUNED", "NON_LEARNING",
    "BLOCKED_ASSETS", "NOT_MEASURABLE",
}

NEUTRAL_OPTIONAL_FIELDS = {
    "rgb_used": None,
    "sparse_depth_used": None,
    "sparse_mask_used": None,
    "sensor_type": None,
    "sensor_model": None,
    "noise_model": None,
    "camera_model": None,
    "uncertainty_available": False,
    "diffusion_steps": None,
    "diffusion_model": None,
    "ensemble_size": None,
    "processing_resolution": None,
    "guidance_scale": None,
    "latent_initialization": None,
    "seed_policy": None,
    "sparse_conditioning": None,
    "confidence_threshold": None,
    "metric_alignment": None,
    "monocular_prior": None,
    "monocular_model": None,
    "mono_model_input_size": None,
    "scale_alignment": False,
    "shift_alignment": False,
    "alignment_source": None,
    "alignment_per_sample": False,
    "alignment_method": "none",
    "alignment_point_count": None,
    "alignment_scale": None,
    "alignment_shift": None,
    "alignment_fallback": False,
}


def validate(payload):
    if not isinstance(payload, dict):
        raise TypeError("fairness metadata must be a mapping")
    missing = [name for name in REQUIRED_FIELDS if name not in payload]
    if missing:
        raise RuntimeError("fairness metadata missing fields: " + ", ".join(missing))
    if payload["fairness_class"] not in FAIRNESS_CLASSES:
        raise RuntimeError(
            f"unknown fairness_class={payload['fairness_class']!r}")
    if payload["training_domain"] not in ("N", "K", "NK"):
        raise RuntimeError("training_domain must be N, K or NK")
    for name in (
            "controlled_ranking", "valid_pixel_masking",
            "zero_coded_unlabelled_pixels_in_loss", "external_pretraining",
            "official_mixed_training", "TTA", "OOD_specific_tuning",
            "loss_modified", "network_modified", "optimizer_modified",
            "scheduler_modified"):
        if not isinstance(payload[name], bool):
            raise RuntimeError(f"fairness field {name} must be boolean")
    if payload["fairness_class"] != "CONTROLLED_NATIVE" and payload["controlled_ranking"]:
        raise RuntimeError(
            "only CONTROLLED_NATIVE may set controlled_ranking=true")
    normalized = dict(payload)
    for name, value in NEUTRAL_OPTIONAL_FIELDS.items():
        normalized.setdefault(name, value)
    return normalized


def write(output_dir, adapter, args, mode):
    if not hasattr(adapter, "fairness_metadata"):
        raise RuntimeError(
            f"{getattr(adapter, 'METHOD_NAME', 'method')} lacks fairness_metadata()")
    payload = validate(adapter.fairness_metadata(args, mode))
    path = Path(output_dir) / "fairness.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return payload
