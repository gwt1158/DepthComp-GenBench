"""Read-only long-horizon BatchNorm diagnostics and replay snapshots.

This module is deliberately method-neutral.  A comparison adapter supplies a
small module-name specification; the shared runner opts in explicitly.  No
hook changes a tensor, parameter, gradient, optimizer state, or BatchNorm
buffer.  In particular, module ``register_full_backward_hook`` is not used:
several released methods apply an in-place activation directly after BN, for
which a full module hook would change autograd view semantics.  Output tensors
and parameters instead receive read-only ``Tensor.register_hook`` callbacks.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import random
import shutil
from collections import deque
from enum import Enum
from pathlib import Path

import numpy as np
import torch


TRACE_SCHEMA = "comparison-confhead-bn-trace-v1"
ROLLING_SCHEMA = "comparison-rolling-replay-state-v1"
FAILURE_BUNDLE_SCHEMA = "comparison-bn-failure-replay-bundle-v1"


class JsonSafeError(TypeError):
    """Strict JSON conversion failed for a nested diagnostic field."""

    def __init__(self, message, *, field_path, python_type, trace_step=None):
        self.field_path = str(field_path)
        self.python_type = python_type
        self.trace_step = None if trace_step is None else int(trace_step)
        detail = (
            f"{message} (field_path={self.field_path!r}, "
            f"python_type={self.python_type!r}"
        )
        if self.trace_step is not None:
            detail += f", trace_step={self.trace_step}"
        detail += ")"
        super().__init__(detail)


def _finite_python_float(value):
    """Map non-finite Python floats to strict-JSON string tokens."""
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "+Infinity" if value > 0 else "-Infinity"
    return float(value)


def _tensor_json_description(tensor):
    """Compact, read-only tensor summary without dumping full storage."""
    with torch.no_grad():
        detached = tensor.detach()
        statistics = tensor_statistics(detached)
        description = {
            "__type__": "tensor",
            "shape": list(detached.shape),
            "dtype": str(detached.dtype).replace("torch.", ""),
            "device": str(detached.device),
            "numel": int(detached.numel()),
            "requires_grad": bool(tensor.requires_grad),
            "finite_count": statistics["finite_count"],
            "nan_count": statistics["nan_count"],
            "positive_infinity_count": statistics["positive_infinity_count"],
            "negative_infinity_count": statistics["negative_infinity_count"],
        }
        if statistics["finite_count"]:
            description.update({
                "minimum_finite_value": statistics["minimum_finite_value"],
                "maximum_finite_value": statistics["maximum_finite_value"],
                "mean_finite_value": statistics["mean_finite_value"],
                "maximum_absolute_finite_value": statistics[
                    "maximum_absolute_finite_value"],
                "finite_l2_norm": statistics["finite_l2_norm"],
            })
        return description


def _ndarray_json_description(array):
    """Compact NumPy array summary; never emit full non-scalar data."""
    flat = np.asarray(array)
    floating = np.issubdtype(flat.dtype, np.floating) or np.issubdtype(
        flat.dtype, np.complexfloating)
    if floating:
        finite_mask = np.isfinite(flat)
        finite_count = int(finite_mask.sum())
        nan_count = int(np.isnan(flat).sum())
        if np.issubdtype(flat.dtype, np.complexfloating):
            positive_infinity_count = int(np.isinf(flat).sum())
            negative_infinity_count = 0
        else:
            positive_infinity_count = int(np.isposinf(flat).sum())
            negative_infinity_count = int(np.isneginf(flat).sum())
        finite_values = flat[finite_mask]
        if np.issubdtype(flat.dtype, np.complexfloating):
            finite_values = np.abs(finite_values)
        finite_values = finite_values.astype(np.float64, copy=False)
    else:
        finite_count = int(flat.size)
        nan_count = positive_infinity_count = negative_infinity_count = 0
        finite_values = flat.astype(np.float64, copy=False) if flat.size else flat
    description = {
        "__type__": "ndarray",
        "shape": list(flat.shape),
        "dtype": str(flat.dtype),
        "numel": int(flat.size),
        "finite_count": finite_count,
        "nan_count": nan_count,
        "positive_infinity_count": positive_infinity_count,
        "negative_infinity_count": negative_infinity_count,
    }
    if finite_count:
        description.update({
            "minimum_finite_value": float(np.min(finite_values)),
            "maximum_finite_value": float(np.max(finite_values)),
            "mean_finite_value": float(np.mean(finite_values)),
            "maximum_absolute_finite_value": float(np.max(np.abs(finite_values))),
            "finite_l2_norm": float(np.linalg.norm(finite_values.ravel())),
        })
    return description


def _json_safe(value, path="record", *, trace_step=None):
    """Recursively build an immutable strict-JSON value (allow_nan=False).

    Large tensors/ndarrays become compact statistics; scalar tensors become
    Python scalars.  The input object graph is never mutated.
    """
    if value is None or isinstance(value, (bool, str)):
        # bool is checked before int because bool is a subclass of int.
        return value
    if isinstance(value, Enum):
        return _json_safe(value.value, path, trace_step=trace_step)
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        with torch.no_grad():
            if int(value.numel()) == 1:
                return _json_safe(
                    value.detach().cpu().item(), path, trace_step=trace_step)
            return _tensor_json_description(value)
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _json_safe(value.item(), path, trace_step=trace_step)
        return _ndarray_json_description(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item(), path, trace_step=trace_step)
    if isinstance(value, float):
        return _finite_python_float(value)
    if isinstance(value, (int,)):
        # JSON numbers cannot represent arbitrary-precision ints larger than
        # the JavaScript safe range, but diagnostic counters stay well below.
        return int(value)
    if isinstance(value, dict):
        return {
            str(key): _json_safe(
                item, f"{path}.{key}", trace_step=trace_step)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _json_safe(item, f"{path}[{index}]", trace_step=trace_step)
            for index, item in enumerate(value)
        ]
    if isinstance(value, set):
        converted = [
            _json_safe(item, f"{path}{{{index}}}", trace_step=trace_step)
            for index, item in enumerate(value)
        ]
        return sorted(
            converted,
            key=lambda item: json.dumps(
                item, sort_keys=True, allow_nan=False, separators=(",", ":")))
    raise JsonSafeError(
        "unsupported diagnostic JSON value",
        field_path=path,
        python_type=type(value).__name__,
        trace_step=trace_step,
    )


def dumps_json_safe(value, *, path="record", trace_step=None, indent=None,
                    separators=(",", ":")):
    """Serialize ``value`` with ``allow_nan=False`` after ``_json_safe``."""
    safe = _json_safe(value, path, trace_step=trace_step)
    kwargs = {"sort_keys": True, "allow_nan": False}
    if indent is None:
        kwargs["separators"] = separators
    else:
        kwargs["indent"] = indent
    return json.dumps(safe, **kwargs)


def _iter_named_tensors(value, prefix=""):
    if torch.is_tensor(value):
        yield prefix or "tensor", value
    elif isinstance(value, dict):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_named_tensors(item, name)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            name = f"{prefix}[{index}]" if prefix else f"[{index}]"
            yield from _iter_named_tensors(item, name)


def cpu_value(value):
    """Detach a nested replay value without changing its source."""
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        # Replay payloads include optimizer.state_dict(), whose ``state`` keys
        # are integer parameter IDs.  Preserve mapping keys exactly so a
        # serialized optimizer can be loaded strictly; JSON conversion belongs
        # only at the reporting boundary, never in the replay state path.
        return {key: cpu_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_value(item) for item in value)
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return repr(value)


def tensor_statistics(tensor):
    """Strict-JSON-safe finite-only statistics for one tensor."""
    detached = tensor.detach()
    count = int(detached.numel())
    floating = detached.is_floating_point() or detached.is_complex()
    finite = (torch.isfinite(detached) if floating
              else torch.ones_like(detached, dtype=torch.bool))
    finite_count = int(finite.sum().item())
    if floating:
        nan_count = int(torch.isnan(detached).sum().item())
        if detached.is_complex():
            positive_inf_count = int(torch.isinf(detached).sum().item())
            negative_inf_count = 0
        else:
            positive_inf_count = int(torch.isposinf(detached).sum().item())
            negative_inf_count = int(torch.isneginf(detached).sum().item())
    else:
        nan_count = positive_inf_count = negative_inf_count = 0
    values = detached[finite]
    if detached.is_complex():
        values = values.abs()
    values = values.double()
    if finite_count:
        minimum = float(values.min().item())
        maximum = float(values.max().item())
        mean = float(values.mean().item())
        std = float(values.std(unbiased=False).item())
        maximum_absolute = float(values.abs().max().item())
        l2 = float(torch.linalg.vector_norm(values).item())
    else:
        minimum = maximum = mean = std = maximum_absolute = l2 = None
    return {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype).replace("torch.", ""),
        "device": str(detached.device),
        "numel": count,
        "finite_count": finite_count,
        "finite_ratio": float(finite_count) / count if count else 1.0,
        "nan_count": nan_count,
        "positive_infinity_count": positive_inf_count,
        "negative_infinity_count": negative_inf_count,
        "minimum_finite_value": minimum,
        "maximum_finite_value": maximum,
        "mean_finite_value": mean,
        "std_finite_value": std,
        "maximum_absolute_finite_value": maximum_absolute,
        "finite_l2_norm": l2,
    }


def collection_statistics(value, prefix=""):
    return {name: tensor_statistics(tensor)
            for name, tensor in _iter_named_tensors(value, prefix)}


def statistics_are_finite(statistics):
    return not any(int(statistics.get(key, 0) or 0) for key in (
        "nan_count", "positive_infinity_count", "negative_infinity_count"))


def capture_rng_state():
    result = {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_states": [],
    }
    if torch.cuda.is_available():
        result["torch_cuda_rng_states"] = torch.cuda.get_rng_state_all()
    return result


def restore_rng_state(state):
    required = {
        "python_random_state", "numpy_random_state", "torch_cpu_rng_state",
        "torch_cuda_rng_states",
    }
    if not isinstance(state, dict) or not required.issubset(state):
        raise RuntimeError("saved RNG state is incomplete")
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.set_rng_state(state["torch_cpu_rng_state"])
    cuda_states = list(state.get("torch_cuda_rng_states") or ())
    if torch.cuda.is_available():
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError(
                "saved CUDA RNG count differs from visible devices: "
                f"saved={len(cuda_states)} visible={torch.cuda.device_count()}")
        torch.cuda.set_rng_state_all(cuda_states)
    elif cuda_states:
        raise RuntimeError("saved RNG used CUDA but replay has no CUDA device")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"stale temporary diagnostic file: {temporary}")
    try:
        torch.save(cpu_value(payload), temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_torch(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _atomic_json(payload, path, *, trace_step=None, root_path="record"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"stale temporary diagnostic file: {temporary}")
    try:
        text = dumps_json_safe(
            payload, path=root_path, trace_step=trace_step, indent=2) + "\n"
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sample_ids(batch):
    if not isinstance(batch, dict):
        return []
    for key in ("sample_id", "name"):
        value = batch.get(key)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value]
        return [str(value)]
    return []


def _batch_metadata(batch):
    result = {"sample_ids": _sample_ids(batch)}
    if not isinstance(batch, dict):
        return result
    for key in ("horizontal_flip", "rgb_path", "sparse_path", "target_path"):
        if key in batch:
            result[key] = cpu_value(batch[key])
    return result


class EpochBatchPlan:
    """Predict the native DataLoader permutation without changing the loader.

    PyTorch consumes one random scalar for the worker base seed before iterating
    RandomSampler.  The simulation uses PyTorch's own sampler classes on a clone
    of the loader generator, avoiding a handwritten randperm approximation.
    Worker RNG streams are intentionally *not* claimed recoverable.  Exact
    augmented tensors are retained in the bounded materialized batch window.
    """

    def __init__(self, loader, epoch):
        self.epoch = int(epoch)
        generator = getattr(loader, "generator", None)
        if generator is None:
            raise RuntimeError("trace requires a DataLoader generator")
        self.generator_before_epoch = generator.get_state().clone()
        clone = torch.Generator()
        clone.set_state(self.generator_before_epoch)
        # Mirrors _BaseDataLoaderIter's base_seed draw.
        torch.empty((), dtype=torch.int64).random_(generator=clone)
        sampler = torch.utils.data.RandomSampler(
            loader.dataset, replacement=False, generator=clone)
        batch_sampler = torch.utils.data.BatchSampler(
            sampler, int(loader.batch_size), bool(loader.drop_last))
        self.batches = [list(map(int, indices)) for indices in batch_sampler]
        self.predicted_generator_after_plan = clone.get_state().clone()
        samples = getattr(loader.dataset, "samples", None)
        if samples is None and isinstance(loader.dataset, torch.utils.data.Subset):
            base = getattr(loader.dataset.dataset, "samples", None)
            if base is not None:
                samples = [base[index] for index in loader.dataset.indices]
        self.sample_ids = None
        if samples is not None:
            self.sample_ids = [str(sample["sample_id"]) for sample in samples]

    def ids_for(self, iteration):
        if self.sample_ids is None or not (0 <= int(iteration) < len(self.batches)):
            return []
        return [self.sample_ids[index] for index in self.batches[int(iteration)]]

    def state_dict(self, loader, iteration, *, include_epoch_plan=True):
        iteration = int(iteration)
        current = self.batches[iteration] if 0 <= iteration < len(self.batches) else []
        next_indices = (
            self.batches[iteration + 1]
            if 0 <= iteration + 1 < len(self.batches) else [])
        generator = getattr(loader, "generator", None)
        result = {
            "schema": "comparison-dataloader-replay-state-v1",
            "epoch": self.epoch,
            "logical_iteration": iteration,
            "batch_count": len(self.batches),
            "current_batch_indices": current,
            "next_batch_indices": next_indices,
            "current_sample_ids": self.ids_for(iteration),
            "next_sample_ids": self.ids_for(iteration + 1),
            "generator_before_epoch": self.generator_before_epoch,
            "predicted_generator_after_plan": self.predicted_generator_after_plan,
            "live_loader_generator_state": (
                generator.get_state().clone() if generator is not None else None),
            "sampler_type": type(loader.sampler).__name__,
            "batch_sampler_type": type(loader.batch_sampler).__name__,
            "num_workers": int(loader.num_workers),
            "worker_rng_capture_supported": False,
            "worker_rng_replay_policy": (
                "exact materialized CPU batches retained from rolling snapshot "
                "to trigger; worker RNG is not fabricated"),
        }
        result["epoch_batch_indices"] = self.batches if include_epoch_plan else None
        result["epoch_plan_included"] = bool(include_epoch_plan)
        return result


def _module_by_suffix(model, requested):
    modules = dict(model.named_modules())
    if requested in modules:
        return requested, modules[requested]
    matches = [(name, module) for name, module in modules.items()
               if name.endswith("." + requested)]
    if len(matches) != 1:
        raise RuntimeError(
            f"trace module {requested!r} resolved to {len(matches)} matches")
    return matches[0]


def batchnorm_record(name, module):
    dtype = None
    if getattr(module, "weight", None) is not None:
        dtype = str(module.weight.dtype).replace("torch.", "")
    elif getattr(module, "running_mean", None) is not None:
        dtype = str(module.running_mean.dtype).replace("torch.", "")
    record = {
        "module": name,
        "num_features": int(module.num_features),
        "eps": float(module.eps),
        "momentum": None if module.momentum is None else float(module.momentum),
        "affine": bool(module.affine),
        "track_running_stats": bool(module.track_running_stats),
        "training": bool(module.training),
        "dtype": dtype,
    }
    for field in ("running_mean", "running_var", "weight", "bias",
                  "num_batches_tracked"):
        value = getattr(module, field, None)
        record[field] = tensor_statistics(value) if torch.is_tensor(value) else None
    return record


def named_batchnorm_buffers(model):
    rows = {}
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            continue
        rows[name] = {
            field: cpu_value(getattr(module, field))
            for field in ("running_mean", "running_var", "num_batches_tracked")
            if torch.is_tensor(getattr(module, field, None))
        }
    return rows


def optimizer_state_statistics(model, optimizer, prefixes):
    prefixes = tuple(str(prefix) for prefix in prefixes)
    rows = {}
    for name, parameter in model.named_parameters():
        if prefixes and not any(name.startswith(prefix) for prefix in prefixes):
            continue
        state = optimizer.state.get(parameter)
        row = {"state_present": state is not None}
        state = state or {}
        missing = []
        for state_name in ("step", "exp_avg", "exp_avg_sq"):
            value = state.get(state_name)
            if torch.is_tensor(value):
                row[state_name] = tensor_statistics(value)
            elif value is not None:
                row[state_name] = value
            else:
                missing.append(state_name)
        row["missing_fields"] = missing
        rows[name] = row
    return rows


def optimizer_magnitude_comparison(rows):
    """Compare confidence-head Adam magnitudes with registered adjacent layers."""
    result = {}
    for field in ("exp_avg", "exp_avg_sq"):
        groups = {"confhead": [], "adjacent": []}
        for parameter_name, state in rows.items():
            statistics = state.get(field)
            if not isinstance(statistics, dict):
                continue
            maximum = statistics.get("maximum_absolute_finite_value")
            if maximum is None:
                continue
            normalized = "." + parameter_name + "."
            group = "confhead" if ".confhead." in normalized else "adjacent"
            groups[group].append(float(maximum))
        target = max(groups["confhead"], default=None)
        adjacent = max(groups["adjacent"], default=None)
        ratio = None
        if target is not None and adjacent is not None:
            ratio = (target / adjacent if adjacent > 0
                     else (None if target == 0 else "Infinity"))
        result[field] = {
            "confhead_maximum_absolute_finite_value": target,
            "adjacent_maximum_absolute_finite_value": adjacent,
            "confhead_to_adjacent_ratio": ratio,
            "confhead_parameter_count": len(groups["confhead"]),
            "adjacent_parameter_count": len(groups["adjacent"]),
        }
    return result


class ConfheadBNTracer:
    """Install exact confhead hooks and classify warning/nonfinite events."""

    def __init__(self, model, optimizer, spec, *, warning_callback,
                 nonfinite_callback):
        self.model = model
        self.optimizer = optimizer
        self.spec = dict(spec)
        self.warning_callback = warning_callback
        self.nonfinite_callback = nonfinite_callback
        thresholds = dict(self.spec.get("thresholds") or {})
        self.activation_thresholds = tuple(
            float(value) for value in thresholds.get(
                "activation_abs", (1e3, 1e6)))
        self.parameter_threshold = float(thresholds.get("parameter_abs", 1e3))
        self.gradient_threshold = float(thresholds.get("gradient_abs", 1e3))
        self.running_var_threshold = float(thresholds.get("running_var", 1e6))
        self.handles = []
        self.context = None
        self.event_sequence = 0
        self.call_counts = {}
        self.first_nonfinite = None
        self.first_nonfinite_evidence = None
        self.first_warning = None
        self._resolved = {}
        self._install()

    def _install(self):
        requested = set(self.spec.get("required_modules") or ())
        observations = dict(self.spec.get("observations") or {})
        requested.update(observations)
        for branch in (self.spec.get("branches") or {}).values():
            requested.update(str(value) for value in branch.values()
                             if isinstance(value, str) and "." in value)
        for requested_name in sorted(requested):
            actual_name, module = _module_by_suffix(self.model, requested_name)
            self._resolved[requested_name] = (actual_name, module)
        static_contract = dict(self.spec.get("batchnorm_static_contract") or {})
        buffer_pointers = []
        for requested_name, expected in static_contract.items():
            actual_name, module = self._resolved[requested_name]
            if not isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                raise RuntimeError(f"trace contract {actual_name} is not BatchNorm")
            for field, expected_value in expected.items():
                actual_value = getattr(module, field)
                if actual_value != expected_value:
                    raise RuntimeError(
                        f"trace BatchNorm contract mismatch {actual_name}.{field}: "
                        f"expected={expected_value!r} actual={actual_value!r}")
            for field in ("weight", "bias", "running_mean", "running_var",
                          "num_batches_tracked"):
                value = getattr(module, field, None)
                if torch.is_tensor(value):
                    buffer_pointers.append((f"{actual_name}.{field}", value.data_ptr()))
        pointer_values = [pointer for _name, pointer in buffer_pointers]
        if len(pointer_values) != len(set(pointer_values)):
            raise RuntimeError("target BatchNorm parameters or buffers unexpectedly alias")
        required = tuple(self.spec.get("required_modules") or ())
        identities = [id(self._resolved[name][1]) for name in required]
        if len(identities) != len(set(identities)):
            raise RuntimeError("required confhead trace modules unexpectedly share an instance")
        for requested_name, (actual_name, module) in self._resolved.items():
            observation = observations.get(requested_name, {})
            self.handles.append(module.register_forward_pre_hook(
                self._pre_hook(requested_name, actual_name, observation)))
            self.handles.append(module.register_forward_hook(
                self._forward_hook(requested_name, actual_name, observation)))
        for name, parameter in self.model.named_parameters():
            if self._is_relevant_parameter(name) and parameter.requires_grad:
                self.handles.append(parameter.register_hook(
                    self._gradient_hook(f"parameter.{name}")))

    def _is_relevant_parameter(self, name):
        prefixes = tuple(self.spec.get("parameter_prefixes") or ())
        return any(name == prefix or name.startswith(prefix + ".")
                   or name.endswith("." + prefix)
                   or ("." + prefix + ".") in name for prefix in prefixes)

    def begin_iteration(self, context, *, epoch, iteration, global_step,
                        batch_metadata=None):
        self.context = context
        self.event_sequence = 0
        self.call_counts = {}
        trace = {
            "schema": TRACE_SCHEMA,
            "epoch": int(epoch),
            "iteration": int(iteration),
            "global_step_before": int(global_step),
            "batch": dict(batch_metadata or {}),
            "events": [],
            "requested_stage_order": "A,B,C,D,E,F,G,H,I,J,K,L",
            "actual_branch_order": "A,E/F/G(rb),B/C/D(ib),H,I,J,K,L",
            "backward_hook_type": "read_only_tensor_register_hook",
            "module_full_backward_hook_used": False,
        }
        context["confhead_bn_trace"] = trace

    def _pre_hook(self, requested_name, actual_name, observation):
        def hook(module, inputs):
            self.call_counts[requested_name] = self.call_counts.get(requested_name, 0) + 1
            self._observe(
                semantic_stage=observation.get("pre_stage", "module_pre"),
                module_name=requested_name, actual_name=actual_name,
                observation="forward_pre_hook", value=inputs, module=module)
            # Read-only tensor hooks provide module-input backward evidence
            # without full backward hooks changing view/in-place semantics.
            for tensor_name, tensor in _iter_named_tensors(
                    inputs, f"{requested_name}.input"):
                if tensor.requires_grad:
                    tensor.register_hook(self._gradient_hook(tensor_name))
        return hook

    def _forward_hook(self, requested_name, actual_name, observation):
        def hook(module, inputs, output):
            del inputs
            self._observe(
                semantic_stage=observation.get("post_stage", "module_post"),
                module_name=requested_name, actual_name=actual_name,
                observation="forward_hook", value=output, module=module)
            for tensor_name, tensor in _iter_named_tensors(
                    output, f"{requested_name}.output"):
                if tensor.requires_grad:
                    tensor.register_hook(self._gradient_hook(tensor_name))
        return hook

    def _gradient_hook(self, name):
        def hook(gradient):
            self._observe(
                semantic_stage="J", module_name=name, actual_name=name,
                observation="backward_tensor_hook", value=gradient,
                module=None, value_role="gradient")
            return None
        return hook

    def _direct_module_statistics(self, module, role):
        if module is None:
            return {}
        values = {}
        iterator = (module.named_parameters(recurse=False)
                    if role == "parameter" else ())
        for name, parameter in iterator:
            values[name] = tensor_statistics(parameter)
        return values

    def _module_gradient_statistics(self, module):
        if module is None:
            return {}
        return {name: tensor_statistics(parameter.grad)
                for name, parameter in module.named_parameters(recurse=False)
                if parameter.grad is not None}

    def _relevant_state_evidence(self):
        """Capture the small confhead state at the instant a threshold fires."""
        parameters = {}
        optimizer_state = {}
        for name, parameter in self.model.named_parameters():
            if not self._is_relevant_parameter(name):
                continue
            parameters[name] = cpu_value(parameter)
            if parameter in self.optimizer.state:
                optimizer_state[name] = cpu_value(
                    self.optimizer.state[parameter])
        prefixes = tuple(self.spec.get("parameter_prefixes") or ())
        return {
            "target_batchnorm_buffers": self.named_batchnorm_buffers(),
            "relevant_parameters": parameters,
            "relevant_optimizer_state": optimizer_state,
            "optimizer_state_statistics": optimizer_state_statistics(
                self.model, self.optimizer, prefixes),
            "trace_so_far": cpu_value(
                (self.context or {}).get("confhead_bn_trace") or {}),
        }

    def _threshold_hits(self, statistics, value_role, bn_state):
        hits = []
        for name, row in statistics.items():
            maximum = row.get("maximum_absolute_finite_value")
            if value_role == "activation" and maximum is not None:
                for threshold in self.activation_thresholds:
                    if maximum > threshold:
                        hits.append({"kind": "activation_abs", "tensor": name,
                                     "threshold": threshold, "value": maximum})
            if value_role == "gradient" and maximum is not None and maximum > self.gradient_threshold:
                hits.append({"kind": "gradient_abs", "tensor": name,
                             "threshold": self.gradient_threshold, "value": maximum})
            if not statistics_are_finite(row):
                hits.append({"kind": "nonfinite", "tensor": name})
        if bn_state:
            running_var = bn_state.get("running_var") or {}
            maximum = running_var.get("maximum_absolute_finite_value")
            if maximum is not None and maximum > self.running_var_threshold:
                hits.append({"kind": "running_var", "tensor": "running_var",
                             "threshold": self.running_var_threshold, "value": maximum})
            if running_var and not statistics_are_finite(running_var):
                hits.append({"kind": "nonfinite", "tensor": "running_var"})
            running_mean = bn_state.get("running_mean") or {}
            if running_mean and not statistics_are_finite(running_mean):
                hits.append({"kind": "nonfinite", "tensor": "running_mean"})
        return hits

    def _observe(self, *, semantic_stage, module_name, actual_name,
                 observation, value, module, value_role="activation"):
        if self.context is None:
            return
        self.event_sequence += 1
        statistics = collection_statistics(value, f"{module_name}.{observation}")
        parameters = self._direct_module_statistics(module, "parameter")
        gradients = self._module_gradient_statistics(module)
        bn_state = (batchnorm_record(actual_name, module)
                    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
                    else None)
        hits = self._threshold_hits(statistics, value_role, bn_state)
        for name, row in parameters.items():
            maximum = row.get("maximum_absolute_finite_value")
            if maximum is not None and maximum > self.parameter_threshold:
                hits.append({"kind": "parameter_abs", "tensor": name,
                             "threshold": self.parameter_threshold, "value": maximum})
            if not statistics_are_finite(row):
                hits.append({"kind": "nonfinite_parameter", "tensor": name})
        for name, row in gradients.items():
            maximum = row.get("maximum_absolute_finite_value")
            if maximum is not None and maximum > self.gradient_threshold:
                hits.append({"kind": "gradient_abs", "tensor": name,
                             "threshold": self.gradient_threshold, "value": maximum})
            if not statistics_are_finite(row):
                hits.append({"kind": "nonfinite_gradient", "tensor": name})
        row = {
            "epoch": self.context["confhead_bn_trace"]["epoch"],
            "iteration": self.context["confhead_bn_trace"]["iteration"],
            "global_step_before": self.context["confhead_bn_trace"][
                "global_step_before"],
            "event_sequence": self.event_sequence,
            "semantic_stage": str(semantic_stage),
            "module": module_name,
            "resolved_module": actual_name,
            "observation": observation,
            "tensor_statistics": statistics,
            "parameter_statistics": parameters,
            "gradient_statistics": gradients,
            "batchnorm": bn_state,
            "threshold_hits": hits,
        }
        self.context["confhead_bn_trace"]["events"].append(row)
        if hits and self.first_warning is None:
            self.first_warning = row
            evidence = {
                "value": cpu_value(value),
                "module_state": (
                    cpu_value(module.state_dict()) if module is not None else {}),
            }
            evidence.update(self._relevant_state_evidence())
            self.warning_callback(row, evidence)
        nonfinite = next((hit for hit in hits if hit["kind"].startswith("nonfinite")), None)
        if nonfinite is not None and self.first_nonfinite is None:
            self.first_nonfinite = row
            self.first_nonfinite_evidence = {
                "event": row,
                "value": cpu_value(value),
                "module_state": (
                    cpu_value(module.state_dict()) if module is not None else {}),
            }
            self.first_nonfinite_evidence.update(
                self._relevant_state_evidence())
            failing_name = str(nonfinite.get("tensor", module_name))
            failing_stats = statistics.get(failing_name)
            if failing_stats is None:
                failing_stats = (bn_state or {}).get(failing_name)
            if failing_stats is None:
                failing_stats = {"trace_event": row}
            self.nonfinite_callback(str(semantic_stage),
                                    f"{module_name}.{failing_name}", failing_stats)

    def boundary(self, stage, *, values=None):
        if self.context is None:
            return
        stage = str(stage)
        if values is not None:
            self._observe(
                semantic_stage=stage, module_name=f"stage_{stage}",
                actual_name=f"stage_{stage}", observation="boundary_value",
                value=values, module=None)
        bn_rows = {}
        for name in self.spec.get("batchnorm_modules") or ():
            actual, module = self._resolved[name]
            bn_rows[name] = batchnorm_record(actual, module)
        prefixes = tuple(self.spec.get("parameter_prefixes") or ())
        parameters = {}
        gradients = {}
        for name, parameter in self.model.named_parameters():
            normalized = name[4:] if name.startswith("net.") else name
            if not any(normalized.startswith(prefix) for prefix in prefixes):
                continue
            parameters[name] = tensor_statistics(parameter)
            if parameter.grad is not None:
                gradients[name] = tensor_statistics(parameter.grad)
        optimizer_rows = optimizer_state_statistics(
            self.model, self.optimizer, prefixes)
        self.event_sequence += 1
        row = {
            "epoch": self.context["confhead_bn_trace"]["epoch"],
            "iteration": self.context["confhead_bn_trace"]["iteration"],
            "global_step_before": self.context["confhead_bn_trace"][
                "global_step_before"],
            "event_sequence": self.event_sequence,
            "semantic_stage": stage,
            "observation": "iteration_boundary",
            "batchnorm": bn_rows,
            "parameter_statistics": parameters,
            "gradient_statistics": gradients,
            "optimizer_state_statistics": optimizer_rows,
            "optimizer_magnitude_comparison": optimizer_magnitude_comparison(
                optimizer_rows),
            "module_call_counts": dict(self.call_counts),
        }
        hits = []
        for role, rows, threshold in (
                ("parameter", parameters, self.parameter_threshold),
                ("gradient", gradients, self.gradient_threshold)):
            for name, stats in rows.items():
                maximum = stats.get("maximum_absolute_finite_value")
                if maximum is not None and maximum > threshold:
                    hits.append({"kind": role + "_abs", "tensor": name,
                                 "threshold": threshold, "value": maximum})
                if not statistics_are_finite(stats):
                    hits.append({"kind": "nonfinite_" + role, "tensor": name})
        for name, state in optimizer_rows.items():
            for field, stats in state.items():
                if isinstance(stats, dict) and not statistics_are_finite(stats):
                    hits.append({"kind": "nonfinite_optimizer_state",
                                 "tensor": f"{name}.{field}"})
        for name, state in bn_rows.items():
            for field in ("running_mean", "running_var", "weight", "bias"):
                stats = state.get(field)
                if stats and not statistics_are_finite(stats):
                    hits.append({"kind": "nonfinite_bn_state",
                                 "tensor": f"{name}.{field}"})
            rv = state.get("running_var") or {}
            maximum = rv.get("maximum_absolute_finite_value")
            if maximum is not None and maximum > self.running_var_threshold:
                hits.append({"kind": "running_var", "tensor": name,
                             "threshold": self.running_var_threshold,
                             "value": maximum})
        row["threshold_hits"] = hits
        self.context["confhead_bn_trace"]["events"].append(row)
        if hits and self.first_warning is None:
            self.first_warning = row
            self.warning_callback(row, self._relevant_state_evidence())
        nonfinite = next((hit for hit in hits if hit["kind"].startswith("nonfinite")), None)
        if nonfinite is not None and self.first_nonfinite is None:
            self.first_nonfinite = row
            self.first_nonfinite_evidence = {
                "event": row,
            }
            self.first_nonfinite_evidence.update(
                self._relevant_state_evidence())
            self.nonfinite_callback(stage, nonfinite["tensor"], {"trace_event": row})

    def named_batchnorm_buffers(self):
        rows = {}
        for name in self.spec.get("batchnorm_modules") or ():
            _actual, module = self._resolved[name]
            rows[name] = {
                field: cpu_value(getattr(module, field))
                for field in ("running_mean", "running_var", "num_batches_tracked")
                if torch.is_tensor(getattr(module, field, None))
            }
        return rows

    def finish_iteration(self, *, terminated_early=False):
        if self.context is not None:
            trace = self.context["confhead_bn_trace"]
            trace["module_call_counts"] = dict(self.call_counts)
            trace["iteration_terminated_early"] = bool(terminated_early)
            expected_bn_calls = {
                name: 1 for name in self.spec.get("batchnorm_modules") or ()}
            violations = {
                name: {"expected": expected, "actual": self.call_counts.get(name, 0)}
                for name, expected in expected_bn_calls.items()
                if self.call_counts.get(name, 0) != expected
            }
            trace["batchnorm_call_count_expected"] = expected_bn_calls
            trace["batchnorm_call_count_violations"] = violations
            actionable_violations = {
                name: row for name, row in violations.items()
                if (not terminated_early) or row["actual"] > row["expected"]
            }
            trace["batchnorm_call_count_validation_complete"] = not bool(
                terminated_early)
            if actionable_violations and self.first_warning is None:
                event = {
                    "epoch": trace["epoch"], "iteration": trace["iteration"],
                    "global_step_before": trace["global_step_before"],
                    "semantic_stage": "H.call_count",
                    "observation": "batchnorm_call_count_warning",
                    "violations": actionable_violations,
                    "threshold_hits": [{"kind": "batchnorm_call_count"}],
                }
                self.first_warning = event
                self.warning_callback(event, self._relevant_state_evidence())
            trace["first_warning"] = self.first_warning
            trace["first_nonfinite"] = self.first_nonfinite
        self.context = None

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


class RollingReplayManager:
    """Persist last-five state snapshots and an exact augmented batch window."""

    def __init__(self, output_dir, *, interval=100, keep=5,
                 materialized_batch_keep=120):
        self.root = Path(output_dir).resolve() / "bn_trace"
        self.rolling_dir = self.root / "rolling"
        self.interval = int(interval)
        self.keep = int(keep)
        if self.interval <= 0 or self.keep <= 0:
            raise ValueError("rolling interval and keep must be positive")
        self.root.mkdir(parents=True, exist_ok=True)
        self.rolling_dir.mkdir(parents=True, exist_ok=True)
        if any(self.rolling_dir.glob("rolling_step_*.pt")):
            raise RuntimeError(
                f"trace rolling directory is not clean: {self.rolling_dir}")
        self.batch_window = deque(maxlen=int(materialized_batch_keep))
        self.epoch_plan = None
        self.current = None
        self.previous_successful = None
        self.successful_before_current = None
        self.first_warning = None
        self.pinned_warning_paths = []

    def begin_epoch(self, loader, epoch):
        self.epoch_plan = EpochBatchPlan(loader, epoch)
        return self.epoch_plan

    def begin_iteration(self, loader, batch, *, epoch, iteration, global_step):
        if self.epoch_plan is None or self.epoch_plan.epoch != int(epoch):
            raise RuntimeError("trace epoch plan was not initialized")
        expected_ids = self.epoch_plan.ids_for(iteration)
        actual_ids = _sample_ids(batch)
        if expected_ids and actual_ids != expected_ids:
            raise RuntimeError(
                "materialized DataLoader batch differs from predicted native "
                f"sampler order at epoch={epoch} iteration={iteration}: "
                f"expected={expected_ids!r} actual={actual_ids!r}")
        loader_state = self.epoch_plan.state_dict(
            loader, iteration, include_epoch_plan=False)
        entry = {
            "epoch": int(epoch),
            "iteration": int(iteration),
            "global_step_before": int(global_step),
            "batch_metadata": _batch_metadata(batch),
            "raw_batch": cpu_value(batch),
            "loader_state": cpu_value(loader_state),
            "successful": False,
        }
        self.batch_window.append(entry)
        self.successful_before_current = self.previous_successful
        self.current = entry
        return loader_state

    def mark_success(self):
        if self.current is not None:
            self.current["successful"] = True
            self.previous_successful = self.current

    def _scheduler_state(self, scheduler):
        if scheduler is None:
            return None
        if hasattr(scheduler, "state_dict"):
            return scheduler.state_dict()
        return {
            "serialization": "attribute_fallback",
            "completed_epochs": getattr(scheduler, "completed_epochs", None),
            "learning_rates": [float(group["lr"])
                               for group in scheduler.optimizer.param_groups],
        }

    def save_snapshot(self, model, optimizer, scheduler, loader, *, epoch,
                      iteration, global_step, tracer, context, source_sha256=None):
        if int(global_step) <= 0 or int(global_step) % self.interval:
            return None
        loader_state = self.epoch_plan.state_dict(loader, iteration)
        optimizer_rows = optimizer_state_statistics(
            model, optimizer, tuple(tracer.spec.get("parameter_prefixes") or ()))
        payload = {
            "schema": ROLLING_SCHEMA,
            "state_timing": "after_successful_optimizer_step_and_zero_grad",
            "epoch": int(epoch),
            "iteration": int(iteration),
            "global_step": int(global_step),
            "model": model.state_dict(),
            "batchnorm_buffers": named_batchnorm_buffers(model),
            "target_batchnorm_buffers": tracer.named_batchnorm_buffers(),
            "optimizer": optimizer.state_dict(),
            "optimizer_state_statistics": optimizer_rows,
            "optimizer_magnitude_comparison": optimizer_magnitude_comparison(
                optimizer_rows),
            "scheduler": self._scheduler_state(scheduler),
            "rng": capture_rng_state(),
            "dataloader": loader_state,
            "current_sample_ids": _sample_ids(self.current["raw_batch"]),
            "next_sample_ids": loader_state.get("next_sample_ids", []),
            "previous_successful_sample_ids": (
                self.successful_before_current["batch_metadata"]["sample_ids"]
                if self.successful_before_current is not None else []),
            "source_sha256": dict(source_sha256 or {}),
            "trace_tail": (context.get("confhead_bn_trace") or {}),
            "epoch_end_scheduler_pending": bool(context.get(
                "epoch_end_scheduler_pending", False)),
        }
        path = self.rolling_dir / f"rolling_step_{int(global_step):09d}.pt"
        if path.exists():
            raise FileExistsError(f"refusing to overwrite rolling state: {path}")
        _atomic_torch_save(payload, path)
        size_bytes = int(path.stat().st_size)
        free_bytes = int(shutil.disk_usage(self.root).free)
        required_free = size_bytes * (self.keep + 2)
        if free_bytes < required_free:
            raise RuntimeError(
                "insufficient free space for required rolling replay states: "
                f"snapshot_bytes={size_bytes} free_bytes={free_bytes} "
                f"minimum_free_bytes={required_free}")
        self._prune()
        return path

    def _rolling_paths(self):
        return sorted(self.rolling_dir.glob("rolling_step_*.pt"))

    def _prune(self):
        paths = self._rolling_paths()
        for path in paths[:-self.keep]:
            path.unlink()

    def _pin_rolling(self, label):
        destination = self.root / label
        destination.mkdir(parents=True, exist_ok=True)
        pinned = []
        for source in self._rolling_paths():
            target = destination / source.name
            if not target.exists():
                try:
                    os.link(source, target)
                except OSError:
                    # Cross-device fallback.  copyfile is intentionally avoided
                    # as snapshots can be large; torch re-save preserves content.
                    payload = load_torch(source)
                    _atomic_torch_save(payload, target)
            pinned.append({
                "path": str(target), "sha256": sha256_file(target),
                "size_bytes": int(target.stat().st_size),
            })
        return pinned

    def capture_warning(self, event, evidence):
        if self.first_warning is not None:
            return
        pinned = self._pin_rolling("first_warning_rolling")
        payload = {
            "schema": "comparison-bn-first-warning-evidence-v1",
            "event": event,
            "evidence": evidence,
            "previous_successful_batch": (
                self.previous_successful if self.previous_successful is not self.current
                else (list(self.batch_window)[-2]
                      if len(self.batch_window) >= 2 else None)),
            "trigger_batch": self.current,
            "batch_history": list(self.batch_window),
            "next_batch_index_info": (
                (self.current or {}).get("loader_state", {})),
            "pinned_rolling_snapshots": pinned,
        }
        evidence_path = self.root / "first_warning_evidence.pt"
        summary_path = self.root / "first_warning.json"
        _atomic_torch_save(payload, evidence_path)
        summary = {
            "schema": payload["schema"],
            "event": event,
            "evidence_path": str(evidence_path),
            "evidence_sha256": sha256_file(evidence_path),
            "pinned_rolling_snapshots": pinned,
        }
        _atomic_json(
            summary, summary_path,
            trace_step=(event or {}).get("global_step_before"),
            root_path="first_warning")
        self.first_warning = summary
        self.pinned_warning_paths = pinned

    def finalize_failure(self, model, optimizer, scheduler, loader, *, epoch,
                         iteration, global_step, start_buffers, tracer, context,
                         error, iteration_start_rng=None):
        # Before-step failures have unchanged parameters/optimizer state, so
        # captured start buffers reconstruct the exact iteration-start model.
        # A stage-L/post-step failure cannot be reversed from the current state;
        # replay must reconstruct its trigger-start state from a rolling state.
        post_optimizer_step = bool(context.get("optimizer_step_completed", False))
        iteration_start_state_exact = not post_optimizer_step
        start_buffers = start_buffers or {}
        iteration_start_model = ({
            name: (start_buffers[name] if name in start_buffers else value)
            for name, value in model.state_dict().items()
        } if iteration_start_state_exact else None)
        pinned_failure = self._pin_rolling("failure_rolling")
        optimizer_rows = optimizer_state_statistics(
            model, optimizer, tuple(tracer.spec.get("parameter_prefixes") or ()))
        payload = {
            "schema": FAILURE_BUNDLE_SCHEMA,
            "formal_experiment": False,
            "epoch": int(epoch),
            "iteration": int(iteration),
            "global_step": int(global_step),
            "failure_stage": getattr(error, "failure_stage", None),
            "failure_tensor": getattr(error, "failure_tensor", None),
            "iteration_start_state_exact": iteration_start_state_exact,
            "iteration_start_state_timing": (
                "reconstructed_before_forward_from_unchanged_parameters_and_captured_buffers"
                if iteration_start_state_exact else
                "unavailable_after_optimizer_step_reconstruct_from_rolling_replay"),
            "iteration_start_model": iteration_start_model,
            "iteration_start_optimizer": (
                optimizer.state_dict() if iteration_start_state_exact else None),
            "iteration_start_scheduler": (
                self._scheduler_state(scheduler)
                if iteration_start_state_exact else None),
            "iteration_start_rng": ((
                iteration_start_rng if iteration_start_rng is not None
                else capture_rng_state()) if iteration_start_state_exact else None),
            "batch_history": list(self.batch_window),
            "previous_successful_batch": (
                list(self.batch_window)[-2] if len(self.batch_window) >= 2 else None),
            "trigger_batch": self.current,
            "next_batch_index_info": (
                self.epoch_plan.state_dict(loader, iteration)
                if self.epoch_plan is not None else None),
            "pinned_failure_rolling_snapshots": pinned_failure,
            "pinned_first_warning_rolling_snapshots": self.pinned_warning_paths,
            "first_warning": self.first_warning,
            "trace": context.get("confhead_bn_trace"),
            "first_nonfinite_evidence": tracer.first_nonfinite_evidence,
            "batchnorm_buffers_after_failure": named_batchnorm_buffers(model),
            "target_batchnorm_buffers_after_failure": (
                tracer.named_batchnorm_buffers()),
            "optimizer_state_statistics_after_failure": optimizer_rows,
            "optimizer_magnitude_comparison_after_failure": (
                optimizer_magnitude_comparison(optimizer_rows)),
        }
        path = self.root / "failure_replay_bundle.pt"
        _atomic_torch_save(payload, path)
        nonfinite_evidence_path = self.root / "first_nonfinite_evidence.pt"
        if tracer.first_nonfinite_evidence is not None:
            _atomic_torch_save(
                tracer.first_nonfinite_evidence, nonfinite_evidence_path)
        summary = {
            "schema": payload["schema"],
            "failure_bundle": str(path),
            "failure_bundle_sha256": sha256_file(path),
            "failure_bundle_size_bytes": int(path.stat().st_size),
            "failure_stage": payload["failure_stage"],
            "failure_tensor": payload["failure_tensor"],
            "epoch": int(epoch),
            "iteration": int(iteration),
            "global_step": int(global_step),
            "batch_history_count": len(payload["batch_history"]),
            "rolling_snapshots": pinned_failure,
            "first_warning": self.first_warning,
            "first_nonfinite_evidence": (
                str(nonfinite_evidence_path)
                if nonfinite_evidence_path.is_file() else None),
            "first_nonfinite_evidence_sha256": (
                sha256_file(nonfinite_evidence_path)
                if nonfinite_evidence_path.is_file() else None),
            "optimizer_state_statistics": payload[
                "optimizer_state_statistics_after_failure"],
            "optimizer_magnitude_comparison": payload[
                "optimizer_magnitude_comparison_after_failure"],
        }
        _atomic_json(
            summary, self.root / "failure_trace.json",
            trace_step=int(global_step), root_path="failure_trace")
        return summary

    def _write_serialization_failure(self, record, *, global_step, error):
        """Persist emergency evidence then keep fail-closed."""
        step = int(global_step)
        field_path = getattr(error, "field_path", None)
        pt_path = self.root / f"trace_serialization_failure_step_{step:09d}.pt"
        json_path = self.root / f"trace_serialization_failure_step_{step:09d}.json"
        _atomic_torch_save({
            "schema": "comparison-bn-trace-serialization-failure-v1",
            "step": step,
            "epoch": record.get("epoch"),
            "iteration": record.get("iteration"),
            "global_step_after": record.get("global_step_after", step),
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "field_path": field_path,
            "python_type": getattr(error, "python_type", None),
            "record": record,
        }, pt_path)
        # Guaranteed-safe minimal JSON: only plain Python scalars/strings.
        minimal = {
            "schema": "comparison-bn-trace-serialization-failure-v1",
            "step": step,
            "epoch": record.get("epoch"),
            "iteration": record.get("iteration"),
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "field_path": None if field_path is None else str(field_path),
            "emergency_pt": str(pt_path),
            "emergency_pt_sha256": sha256_file(pt_path),
        }
        temporary = json_path.with_name(json_path.name + f".tmp.{os.getpid()}")
        if temporary.exists():
            raise FileExistsError(f"stale temporary diagnostic file: {temporary}")
        try:
            temporary.write_text(
                json.dumps(minimal, indent=2, sort_keys=True, allow_nan=False)
                + "\n",
                encoding="utf-8")
            os.replace(temporary, json_path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return {"pt": str(pt_path), "json": str(json_path), "summary": minimal}

    def append_periodic_trace(self, context, global_step, *, force=False):
        if not force and int(global_step) % self.interval:
            return None
        path = self.root / "confhead_trace.jsonl"
        # Shallow copy only; ``_json_safe`` builds a new tree and must not mutate
        # the live in-memory trace attached to ``context``.
        record = dict(context.get("confhead_bn_trace") or {})
        record["global_step_after"] = int(global_step)
        try:
            payload = dumps_json_safe(
                record, path="record", trace_step=int(global_step),
                separators=(",", ":"))
        except Exception as error:
            self._write_serialization_failure(
                record, global_step=global_step, error=error)
            raise
        with path.open("a", encoding="utf-8") as stream:
            stream.write(payload + "\n")
            stream.flush()
        return path


@contextlib.contextmanager
def preserve_batchnorm_buffers(modules):
    """Keep train-mode batch statistics while preserving running buffers.

    The restore must not use ``copy_`` in a forward post-hook.  Native
    BatchNorm autograd retains the running-stat tensor and checks its version at
    backward; mutating that object after forward therefore changes otherwise
    valid backward behavior.  Replacing the registered buffer object with a
    clone preserves the value visible to the next forward while leaving the
    object saved for the current backward version-stable.  This context is
    diagnostic-only (Replay F) and does not alter activations or gradients.
    """
    snapshots = []
    handles = []
    for module in modules:
        if not isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            raise TypeError("preserve_batchnorm_buffers expects BatchNorm modules")
        state = {}

        def pre_hook(current, _inputs, state=state):
            state.clear()
            for field in ("running_mean", "running_var", "num_batches_tracked"):
                value = getattr(current, field, None)
                if torch.is_tensor(value):
                    state[field] = value.detach().clone()

        def post_hook(current, _inputs, output, state=state):
            snapshots.append({
                "attempted_update": {
                    field: cpu_value(getattr(current, field)) for field in state
                },
                "restored": cpu_value(state),
            })
            with torch.no_grad():
                for field, value in state.items():
                    # Assignment retains Module buffer registration.  Do not
                    # mutate the tensor retained by native BN backward.
                    setattr(current, field, value.detach().clone())
            return None

        handles.append(module.register_forward_pre_hook(pre_hook))
        handles.append(module.register_forward_hook(post_hook))
    try:
        yield snapshots
    finally:
        for handle in handles:
            handle.remove()
