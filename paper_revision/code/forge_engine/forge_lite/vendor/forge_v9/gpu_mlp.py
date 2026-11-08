from __future__ import annotations

from dataclasses import dataclass
import copy
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class GpuMlpConfig:
    device: str = "cuda:0"
    batch_size: int = 4096
    predict_batch_size: int = 32768
    epochs: int = 60
    clip: float = 10.0


def _fit_preprocessor(x_fit: np.ndarray, seed: int) -> dict:
    from scipy.stats import skew
    from sklearn.preprocessing import QuantileTransformer, StandardScaler

    x = np.asarray(x_fit, dtype=np.float32)
    with np.errstate(all="ignore"):
        medians = np.nanmedian(x, axis=0).astype(np.float32)
    medians[~np.isfinite(medians)] = 0.0
    filled = np.where(np.isfinite(x), x, medians)
    skewness = np.asarray(skew(filled, axis=0, bias=False,
                               nan_policy="omit"), dtype=float)
    skew_idx = np.flatnonzero(np.isfinite(skewness) & (np.abs(skewness) > .99))
    continuous_idx = np.setdiff1d(np.arange(filled.shape[1]), skew_idx)
    scaler = StandardScaler()
    quantile = QuantileTransformer(
        n_quantiles=min(1000, len(filled)), output_distribution="normal",
        subsample=min(10_000, len(filled)), random_state=int(seed))
    if len(continuous_idx):
        scaler.fit(filled[:, continuous_idx])
    if len(skew_idx):
        quantile.fit(filled[:, skew_idx])
    return {
        "medians": medians,
        "continuous_idx": continuous_idx,
        "skew_idx": skew_idx,
        "scaler": scaler,
        "quantile": quantile,
    }


def _transform(x: np.ndarray, preprocessor: dict, clip: float) -> np.ndarray:
    values = np.asarray(x, dtype=np.float32)
    values = np.where(np.isfinite(values), values, preprocessor["medians"])
    chunks = []
    continuous_idx = preprocessor["continuous_idx"]
    skew_idx = preprocessor["skew_idx"]
    if len(continuous_idx):
        chunks.append(preprocessor["scaler"].transform(
            values[:, continuous_idx]))
    if len(skew_idx):
        chunks.append(preprocessor["quantile"].transform(values[:, skew_idx]))
    transformed = np.column_stack(chunks) if len(chunks) > 1 else chunks[0]
    return np.clip(transformed, -float(clip), float(clip)).astype(np.float32)


def _activation(name: str):
    from torch import nn

    key = str(name).lower()
    if key == "elu":
        return nn.ELU()
    if key == "gelu":
        return nn.GELU()
    if key == "tanh":
        return nn.Tanh()
    return nn.ReLU()


def fit_predict_gpu_mlp(seed: int, x_fit: np.ndarray, y_fit: np.ndarray,
                        x_predict_list: list[np.ndarray], *,
                        runtime: GpuMlpConfig, params: dict,
                        validation: tuple[np.ndarray, np.ndarray] | None = None,
                        ) -> list[np.ndarray]:
    import torch
    from torch import nn

    if not torch.cuda.is_available():
        raise RuntimeError("FORGE_9 CUDA MLP required; CPU fallback forbidden")
    device = torch.device(runtime.device)
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))

    preprocessor = _fit_preprocessor(x_fit, seed)
    x = _transform(x_fit, preprocessor, runtime.clip)
    y = np.asarray(y_fit, dtype=np.int64)
    hidden = tuple(int(v) for v in params.get(
        "hidden_layer_sizes", (64, 32)))
    activation = str(params.get("activation", "relu"))
    dropout = float(params.get("dropout_prob", params.get("dropout", 0.0)))
    batchnorm = bool(params.get("use_batchnorm", False))

    input_dropout = float(params.get("input_dropout", 0.0))
    layers: list[nn.Module] = []
    width = int(x.shape[1])
    if batchnorm:
        layers.append(nn.BatchNorm1d(width))
    if input_dropout > 0:
        layers.append(nn.Dropout(input_dropout))
    layers.extend((nn.Linear(width, hidden[0]), _activation(activation)))
    width = hidden[0]
    for next_width in hidden[1:]:
        if batchnorm:
            layers.append(nn.BatchNorm1d(width))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.extend((nn.Linear(width, next_width), _activation(activation)))
        width = next_width
    layers.append(nn.Linear(width, 2))
    model = nn.Sequential(*layers).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(params.get("learning_rate_init", 1e-3)),
        weight_decay=float(params.get("alpha", 1e-4)),
    )
    criterion = nn.CrossEntropyLoss()
    x_cpu = torch.from_numpy(x).to(device)
    y_cpu = torch.from_numpy(y).to(device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    n_rows = len(y)
    batch_size = min(int(params.get("batch_size", runtime.batch_size)), n_rows)
    epochs = int(params.get("epochs", runtime.epochs))
    patience = int(params.get("early_stopping_rounds", 20))
    best_score = -np.inf
    best_state = None
    stale_epochs = 0
    if validation is not None:
        from sklearn.metrics import average_precision_score

        x_val = _transform(validation[0], preprocessor, runtime.clip)
        y_val = np.asarray(validation[1], dtype=np.int64)

    model.train()
    for _ in range(epochs):
        order = torch.randperm(n_rows, generator=generator).to(device)
        for start in range(0, n_rows, batch_size):
            idx = order[start:start + batch_size]
            xb = x_cpu[idx].to(device, non_blocking=True)
            yb = y_cpu[idx].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        if validation is not None:
            model.eval()
            with torch.inference_mode():
                val_chunks = []
                for start in range(0, len(x_val), runtime.predict_batch_size):
                    xb = torch.from_numpy(
                        x_val[start:start + runtime.predict_batch_size]).to(device)
                    val_chunks.append(torch.softmax(model(xb), dim=1)[:, 1]
                                      .cpu().numpy())
            score = float(average_precision_score(
                y_val, np.concatenate(val_chunks)))
            if score > best_score:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
            model.train()
            if stale_epochs >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    outputs: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for matrix in x_predict_list:
            transformed = _transform(matrix, preprocessor, runtime.clip)
            chunks = []
            for start in range(0, len(transformed),
                               int(runtime.predict_batch_size)):
                xb = torch.from_numpy(
                    transformed[start:start + runtime.predict_batch_size])
                chunks.append(torch.softmax(model(xb.to(device)), dim=1)[:, 1]
                              .cpu().numpy())
            outputs.append(np.concatenate(chunks).astype(np.float32))
    del model, optimizer, x_cpu, y_cpu
    torch.cuda.empty_cache()
    return outputs


_ACTIVE_RUNTIME: GpuMlpConfig | None = None


def fit_member_gpu_with_validation(
    params: dict, seed: int, x_fit: np.ndarray, y_fit: np.ndarray,
    x_val: np.ndarray, y_val: np.ndarray, x_apply: np.ndarray,
) -> list[np.ndarray]:
    if _ACTIVE_RUNTIME is None:
        raise RuntimeError("GPU MLP backend must be installed before fitting")
    return fit_predict_gpu_mlp(
        seed, x_fit, y_fit, [x_val, x_apply], runtime=_ACTIVE_RUNTIME,
        params=params, validation=(x_val, y_val))


def install_gpu_mlp_backend(config: GpuMlpConfig) -> dict:
    import torch
    from forge_v7 import hetero_stack

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing CPU fallback")
    global _ACTIVE_RUNTIME
    _ACTIVE_RUNTIME = config
    original: Callable = hetero_stack.fit_member

    def fit_member(learner: str, params: dict, seed: int,
                   x_fit: np.ndarray, y_fit: np.ndarray,
                   x_predict_list: list[np.ndarray], *, cap_seed: int = 0):
        if learner != "mlp":
            return original(learner, params, seed, x_fit, y_fit,
                            x_predict_list, cap_seed=cap_seed)
        if len(y_fit) > hetero_stack.MLP_FIT_CAP:
            rows = hetero_stack.stratified_cap(
                np.asarray(y_fit), hetero_stack.MLP_FIT_CAP, cap_seed)
            x_fit, y_fit = x_fit[rows], np.asarray(y_fit)[rows]
        return fit_predict_gpu_mlp(seed, x_fit, y_fit, x_predict_list,
                                   runtime=config, params=dict(params or {}))

    hetero_stack.fit_member = fit_member
    return {
        "backend": "pytorch_cuda_configurable",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": config.device,
        "device_name": torch.cuda.get_device_name(torch.device(config.device)),
        "default_batch_size": config.batch_size,
        "default_epochs": config.epochs,
        "cpu_fallback": False,
    }
