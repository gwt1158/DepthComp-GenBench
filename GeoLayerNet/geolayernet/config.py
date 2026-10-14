"""Config: r11 architecture defaults + YAML loader.

The architecture fields in ``GeoLayerConfig`` are FIXED to the adopted-best (r11)
model and must match the checkpoint; do not change them. Training / eval / loss
settings are read from the YAML config files.
"""

from dataclasses import dataclass, asdict
from types import SimpleNamespace


@dataclass
class GeoLayerConfig:
    # architecture (must match the checkpoint — do not change)
    num_layers: int = 4
    rgb_channels: int = 3
    depth_channels: int = 1
    base_channels: int = 64
    norm: str = "batch"
    attention_heads: int = 4
    fusion_type: str = "global_gated"
    relation_steps: int = 2
    # Controlled TIP revision axes.  Defaults preserve every historical
    # checkpoint and the exact pre-revision forward path.
    layer_partition_mode: str = "quantile_ordered"
    layer_partition_seed: int = 42
    dorp_use_order_cues: bool = True
    global_pool: str = "mask_weighted"
    layer_drop_prob: float = 0.25         # train-only; inert at eval
    gate_bias_init: float = -3.0
    residual_scale: float = 0.1
    # sparse layer support (gaussian_bilateral)
    sparse_layer_expand_radius: int = 11
    sparse_layer_sigma_xy: float = 3.0
    sparse_layer_rgb_guided: bool = True
    sparse_layer_rgb_sigma: float = 0.10
    # ---- enhancement switches (ALL default OFF == r11 baseline) ----------- #
    # Every switch below leaves the module set byte-identical to r11 when OFF,
    # so the original checkpoint loads unchanged and proxy_only is unaffected.
    # Phase 1 — coarse densification + proxy enhancement input
    use_coarse_densifier: bool = False           # build the param-free densifier
    coarse_densifier_radii: tuple = (5, 11)      # multi-radius fill scales
    coarse_densifier_rgb_guided: bool = False    # bilateral RGB weighting in fill
    proxy_use_coarse: bool = False               # feed coarse depth+conf to proxy (+2 ch)
    # Phase 2 — selective residual enhancement (param-free reliability gate)
    selective_residual: bool = False
    # output
    output_mode: str = "proxy_only"       # adopted best; "safe" for training


def _ns(d):
    """Recursively wrap a dict into attribute-accessible namespaces."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _ns(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_ns(v) for v in d]
    return d


def load_config(path):
    """Load a YAML config into a dot-accessible namespace (also keeps the raw dict)."""
    import yaml
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    cfg = _ns(raw)
    cfg.raw = raw
    return cfg


def parse_radii(value):
    """Parse a radius list given as tuple/list or a comma-string like '3,7,15'."""
    if value is None:
        return ()
    if isinstance(value, str):
        parts = [p.strip() for p in value.replace(";", ",").split(",") if p.strip()]
        return tuple(int(p) for p in parts)
    if isinstance(value, (list, tuple)):
        return tuple(int(p) for p in value)
    return (int(value),)


def model_config_from_yaml(cfg):
    """Build a GeoLayerConfig (architecture) from a loaded YAML config's model section."""
    m = getattr(cfg, "model", SimpleNamespace())
    gc = GeoLayerConfig()
    mapping = {
        "num_layers": "num_layers",
        "sparse_layer_support_mode": None,  # only gaussian_bilateral supported (baseline)
        "sparse_layer_expand_radius": "sparse_layer_expand_radius",
        "sparse_layer_sigma_xy": "sparse_layer_sigma_xy",
        "sparse_layer_rgb_guided": "sparse_layer_rgb_guided",
        "sparse_layer_rgb_sigma": "sparse_layer_rgb_sigma",
        "residual_scale": "residual_scale",
        "relation_steps": "relation_steps",
        "layer_partition_mode": "layer_partition_mode",
        "layer_partition_seed": "layer_partition_seed",
        "dorp_use_order_cues": "dorp_use_order_cues",
        "gate_bias_init": "gate_bias_init",
        "output_mode": "output_mode",
        # enhancement switches (all default OFF; only set when present in YAML)
        "use_coarse_densifier": "use_coarse_densifier",
        "coarse_densifier_rgb_guided": "coarse_densifier_rgb_guided",
        "proxy_use_coarse": "proxy_use_coarse",
        "selective_residual": "selective_residual",
    }
    for ykey, gkey in mapping.items():
        if gkey is not None and hasattr(m, ykey):
            setattr(gc, gkey, getattr(m, ykey))
    gc.num_layers = int(gc.num_layers)
    if gc.num_layers < 1:
        raise ValueError(f"model.num_layers must be at least 1, got {gc.num_layers}")
    valid_partitions = {
        "quantile_ordered", "uniform_normalized", "random_equal_count"}
    if gc.layer_partition_mode not in valid_partitions:
        raise ValueError(
            "model.layer_partition_mode must be one of "
            f"{sorted(valid_partitions)}, got {gc.layer_partition_mode!r}")
    gc.layer_partition_seed = int(gc.layer_partition_seed)
    gc.dorp_use_order_cues = bool(gc.dorp_use_order_cues)
    # tuple-valued fields accept comma-strings or lists from YAML
    if hasattr(m, "coarse_densifier_radii"):
        gc.coarse_densifier_radii = parse_radii(getattr(m, "coarse_densifier_radii"))
    return gc


def config_to_dict(gc: GeoLayerConfig):
    return asdict(gc)
