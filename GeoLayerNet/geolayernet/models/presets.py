"""r11 loss preset (the adopted-best objective).

These are the loss weights used by the historical sup_r11 training. The post-r11
mechanisms are pinned to 0 (off). Architecture defaults live in config.py /
GeoLayerConfig; these are only the loss lambdas passed to GeoLayerNetLoss.
"""

R11_LOSS = dict(
    lambda_depth=100.0, lambda_final_depth=50.0, lambda_raw_depth=100.0,
    lambda_proxy=10.0, lambda_improve=20.0, lambda_residual_reg=10.0,
    lambda_edge_raw=5.0, lambda_boundary=0.5, lambda_observed=10.0,
    lambda_missing=1.0, lambda_observed_region=0.2, lambda_normal=0.05,
    lambda_ordinal=1.0, lambda_layer_balance=0.01, lambda_layer_ordinal_strong=0.5,
    lambda_layer_reconstruction=0.05, lambda_layer_depth_aux=0.2, lambda_acm_aux=0.2,
    lambda_layer_rgb_reconstruction=0.0, lambda_rgb=10.0, lambda_smooth=0.001,
    lambda_sparse_layer_local_smooth=0.1, lambda_sparse_layer_area=0.01,
    lambda_gate_sparsity=0.01, lambda_uncertainty=1.0, lambda_task=0.0,
    lambda_depth_l2=0.0, lambda_relative_depth=0.0, lambda_nonboundary_smooth=0.0,
    lambda_proxy_nonboundary_smooth=0.0, lambda_geometry_refine_smooth=0.0,
    # post-r11 mechanisms — OFF for the baseline
    lambda_residual_nonedge_tv=0.0, lambda_residual_nonedge_laplacian=0.0,
    lambda_proxy_nonedge_tv=0.0, lambda_gate_nonedge=0.0, lambda_edge_band_raw=0.0,
    # Phase 5 teacher distillation — OFF for the baseline
    lambda_teacher_hf=0.0, lambda_teacher_grad=0.0, lambda_teacher_normal=0.0,
    # optional sharpness supervision — OFF for the baseline
    grad_loss_weight=0.0, boundary_l1_weight=0.0, lap_loss_weight=0.0,
)


def loss_lambdas_from_yaml(cfg):
    """Merge the r11 defaults with the YAML 'loss' section (YAML overrides)."""
    out = dict(R11_LOSS)
    section = getattr(cfg, "loss", None)
    if section is not None:
        for k, v in vars(section).items():
            out[k] = v
    return out
