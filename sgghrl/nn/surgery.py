from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
from copy import deepcopy
import io
import zipfile

import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.buffers import ReplayBuffer

from ..logging import logger


def set_ppo_params(model: PPO, **kwargs) -> List[str]:
    """Изменить гиперпараметры PPO-модели на лету.

    Поддерживаемые параметры: learning_rate, clip_range, ent_coef,
    gamma, gae_lambda, n_steps, batch_size, n_epochs,
    max_grad_norm, vf_coef.

    Args:
        model: PPO-модель.
        **kwargs: пары параметр=значение.

    Returns:
        Список применённых параметров.
    """
    applied = []
    for key, val in kwargs.items():
        if key == "learning_rate":
            model.learning_rate = val
            model.lr_schedule = lambda _: val
            for pg in model.policy.optimizer.param_groups:
                pg["lr"] = val
        elif key == "clip_range":
            model.clip_range = lambda _: val
        elif key == "ent_coef":
            model.ent_coef = val
        elif key == "gamma":
            model.gamma = val
        elif key == "gae_lambda":
            model.gae_lambda = val
        elif key == "n_steps":
            model.n_steps = val
            model.rollout_buffer.buffer_size = val
            model.rollout_buffer.reset()
        elif key == "batch_size":
            model.batch_size = val
        elif key == "n_epochs":
            model.n_epochs = val
        elif key == "max_grad_norm":
            model.max_grad_norm = val
        elif key == "vf_coef":
            model.vf_coef = val
        elif hasattr(model, key):
            setattr(model, key, val)
        else:
            logger.warning("skip unknown PPO param: %s", key)
            continue
        applied.append(key)
    return applied


def set_sac_params(model: SAC, **kwargs) -> List[str]:
    """Изменить гиперпараметры SAC-модели на лету.

    Поддерживаемые параметры: learning_rate, tau, gamma,
    batch_size, learning_starts, target_entropy, buffer_size.

    Args:
        model: SAC-модель.
        **kwargs: пары параметр=значение.

    Returns:
        Список применённых параметров.
    """
    applied = []
    for key, val in kwargs.items():
        if key == "learning_rate":
            model.learning_rate = val
            model.lr_schedule = lambda _: val
            for pg in model.actor.optimizer.param_groups:
                pg["lr"] = val
            for pg in model.critic.optimizer.param_groups:
                pg["lr"] = val
            if model.ent_coef_optimizer is not None:
                for pg in model.ent_coef_optimizer.param_groups:
                    pg["lr"] = val
        elif key == "tau":
            model.tau = val
        elif key == "gamma":
            model.gamma = val
        elif key == "batch_size":
            model.batch_size = val
        elif key == "learning_starts":
            model.learning_starts = val
        elif key == "target_entropy":
            model.target_entropy = val
        elif key == "buffer_size":
            model.buffer_size = val
            model.replay_buffer = ReplayBuffer(
                val, model.observation_space, model.action_space,
                device=model.device,
                optimize_memory_usage=model.replay_buffer.optimize_memory_usage,
            )
        elif hasattr(model, key):
            setattr(model, key, val)
        else:
            logger.warning("skip unknown SAC param: %s", key)
            continue
        applied.append(key)
    return applied


def _resize_linear(old: nn.Linear, new_in: Optional[int] = None,
                   new_out: Optional[int] = None, fill: float = 0.0) -> nn.Linear:
    in_f = new_in if new_in is not None else old.in_features
    out_f = new_out if new_out is not None else old.out_features
    has_bias = old.bias is not None

    new = nn.Linear(in_f, out_f, bias=has_bias)
    nn.init.constant_(new.weight, fill)
    if has_bias:
        nn.init.constant_(new.bias, fill)

    copy_in = min(old.in_features, in_f)
    copy_out = min(old.out_features, out_f)
    with torch.no_grad():
        new.weight[:copy_out, :copy_in] = old.weight[:copy_out, :copy_in]
        if has_bias:
            new.bias[:copy_out] = old.bias[:copy_out]

    return new


def _find_linear_layers(module: nn.Module) -> List[Tuple[str, nn.Module, str, nn.Linear]]:
    results = []
    for parent_name, parent in module.named_modules():
        for name, child in parent.named_children():
            if isinstance(child, nn.Linear):
                full_name = f"{parent_name}.{name}" if parent_name else name
                results.append((full_name, parent, name, child))
    return results


def _get_submodule(module: nn.Module, path: str) -> nn.Module:
    parts = path.split(".")
    current = module
    for p in parts:
        if p.isdigit():
            current = current[int(p)]
        else:
            current = getattr(current, p)
    return current


def describe_architecture(model) -> Dict[str, Tuple[int, int]]:
    """Описание архитектуры: все Linear-слои с размерами (in, out).

    Args:
        model: SB3-модель (PPO/SAC).

    Returns:
        Словарь {путь_к_слою: (in_features, out_features)}.
    """
    policy = model.policy if hasattr(model, "policy") else model
    result = {}
    for name, _, _, layer in _find_linear_layers(policy):
        result[name] = (layer.in_features, layer.out_features)
    return result


def resize_layer(model, layer_path: str,
                 new_in: Optional[int] = None,
                 new_out: Optional[int] = None,
                 fill: float = 0.0) -> Tuple[int, int]:
    """Изменить размер одного Linear-слоя.

    Существующие веса копируются, новые заполняются fill.

    Args:
        model: SB3-модель.
        layer_path: путь к слою (например "mlp_extractor.policy_net.0").
        new_in: новый in_features (None — без изменений).
        new_out: новый out_features (None — без изменений).
        fill: значение заполнения новых весов.

    Returns:
        Кортеж (старый_размер, новый_размер).
    """
    policy = model.policy if hasattr(model, "policy") else model

    parts = layer_path.rsplit(".", 1)
    if len(parts) == 2:
        parent_path, attr_name = parts
        parent = _get_submodule(policy, parent_path)
    else:
        parent = policy
        attr_name = parts[0]

    old_layer = getattr(parent, attr_name)
    assert isinstance(old_layer, nn.Linear), f"{layer_path} is not nn.Linear"

    old_shape = (old_layer.in_features, old_layer.out_features)
    new_layer = _resize_linear(old_layer, new_in, new_out, fill)
    new_layer = new_layer.to(old_layer.weight.device)
    setattr(parent, attr_name, new_layer)

    new_shape = (new_layer.in_features, new_layer.out_features)
    return old_shape, new_shape


def resize_network(model, new_arch: List[int], fill: float = 0.0,
                   include_groups: Optional[List[str]] = None) -> List[dict]:
    """Изменить архитектуру MLP-сетей модели.

    Args:
        model: SB3-модель.
        new_arch: новые размеры скрытых слоёв (например [128, 64]).
        fill: значение заполнения новых весов.
        include_groups: список префиксов групп для изменения
            (None — все группы с >= 2 слоями).

    Returns:
        Список словарей с информацией об изменениях.
    """
    policy = model.policy if hasattr(model, "policy") else model
    changes = []

    mlp_groups = _find_mlp_groups(policy)

    for group_name, layers in mlp_groups.items():
        if not layers:
            continue

        if include_groups is not None:
            if not any(group_name.startswith(g) for g in include_groups):
                continue
        else:
            if len(layers) < 2:
                continue

        first_in = layers[0][3].in_features
        last_out = layers[-1][3].out_features

        target_sizes = [first_in] + list(new_arch) + [last_out]
        n_target = len(target_sizes) - 1
        n_existing = len(layers)

        if n_target != n_existing:
            logger.warning("%s: layer count mismatch (%d -> %d), skipping",
                           group_name, n_existing, n_target)
            continue

        for i, (full_name, parent, attr_name, old_layer) in enumerate(layers):
            target_in = target_sizes[i]
            target_out = target_sizes[i + 1]

            if old_layer.in_features == target_in and old_layer.out_features == target_out:
                continue

            new_layer = _resize_linear(old_layer, target_in, target_out, fill)
            new_layer = new_layer.to(old_layer.weight.device)
            setattr(parent, attr_name, new_layer)

            changes.append({
                "layer": full_name,
                "old": (old_layer.in_features, old_layer.out_features),
                "new": (target_in, target_out),
            })

    _rebuild_optimizers(model)
    return changes


def _find_mlp_groups(policy: nn.Module) -> Dict[str, List[Tuple]]:
    all_layers = _find_linear_layers(policy)
    groups: Dict[str, List] = {}
    for full_name, parent, attr_name, layer in all_layers:
        # Группируем по префиксу до последней точки перед числом
        # Например: mlp_extractor.policy_net.0 -> mlp_extractor.policy_net
        #           critic.qf0.0 -> critic.qf0
        parts = full_name.split(".")
        group_parts = []
        for p in parts:
            if p.isdigit():
                break
            group_parts.append(p)
        group_name = ".".join(group_parts) if group_parts else full_name
        if group_name not in groups:
            groups[group_name] = []
        groups[group_name].append((full_name, parent, attr_name, layer))
    return groups


def _rebuild_optimizers(model):
    if isinstance(model, PPO):
        lr = model.learning_rate
        lr_val = lr if isinstance(lr, float) else lr(1.0)
        model.policy.optimizer = model.policy.optimizer.__class__(
            model.policy.parameters(), lr=lr_val
        )
    elif isinstance(model, SAC):
        lr = model.learning_rate
        lr_val = lr if isinstance(lr, float) else lr(1.0)
        model.actor.optimizer = model.actor.optimizer.__class__(
            model.actor.parameters(), lr=lr_val
        )
        model.critic.optimizer = model.critic.optimizer.__class__(
            model.critic.parameters(), lr=lr_val
        )
        # critic_target не имеет оптимизатора


def freeze_layers(model, layer_paths: List[str]):
    """Заморозить параметры указанных слоёв (requires_grad=False).

    Args:
        model: SB3-модель.
        layer_paths: пути к замораживаемым слоям.
    """
    policy = model.policy if hasattr(model, "policy") else model
    for path in layer_paths:
        layer = _get_submodule(policy, path)
        for param in layer.parameters():
            param.requires_grad = False


def unfreeze_layers(model, layer_paths: List[str]):
    """Разморозить параметры указанных слоёв (requires_grad=True).

    Args:
        model: SB3-модель.
        layer_paths: пути к размораживаемым слоям.
    """
    policy = model.policy if hasattr(model, "policy") else model
    for path in layer_paths:
        layer = _get_submodule(policy, path)
        for param in layer.parameters():
            param.requires_grad = True


def freeze_all_except(model, keep_paths: List[str]):
    """Заморозить все параметры кроме указанных слоёв.

    Args:
        model: SB3-модель.
        keep_paths: пути к слоям, которые остаются обучаемыми.
    """
    policy = model.policy if hasattr(model, "policy") else model
    for param in policy.parameters():
        param.requires_grad = False
    for path in keep_paths:
        layer = _get_submodule(policy, path)
        for param in layer.parameters():
            param.requires_grad = True


def unfreeze_all(model):
    """Разморозить все параметры модели.

    Args:
        model: SB3-модель.
    """
    policy = model.policy if hasattr(model, "policy") else model
    for param in policy.parameters():
        param.requires_grad = True


def count_parameters(model) -> Dict[str, int]:
    """Подсчитать параметры модели.

    Args:
        model: SB3-модель.

    Returns:
        Словарь с ключами total, trainable, frozen.
    """
    policy = model.policy if hasattr(model, "policy") else model
    total = sum(p.numel() for p in policy.parameters())
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    frozen = total - trainable
    return {"total": total, "trainable": trainable, "frozen": frozen}

def load_sb3_policy_state_dict(path: str) -> Dict[str, torch.Tensor]:
    """Загрузить state_dict политики из SB3 zip-файла.

    Args:
        path: путь к файлу модели (с или без .zip).

    Returns:
        Словарь параметров (state_dict).
    """
    import zipfile, io
    zip_path = path if path.endswith(".zip") else path + ".zip"
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open("policy.pth") as f:
            buf = io.BytesIO(f.read())
            try:
                return torch.load(buf, map_location="cpu", weights_only=True)
            except Exception:
                buf.seek(0)
                return torch.load(buf, map_location="cpu")


def load_state_dict_ignore_mismatched(
    policy, saved_sd: Dict[str, torch.Tensor], fill_value: float = 0.0
) -> list:
    """Загрузить state_dict с игнорированием несовпадений размеров.

    Совпадающие параметры копируются. Несовпадающие — частично
    копируются, оставшиеся элементы заполняются fill_value.

    Args:
        policy: политика SB3 (policy-объект).
        saved_sd: загруженный state_dict.
        fill_value: значение заполнения для новых элементов.

    Returns:
        Список пропущенных параметров [(key, old_shape, new_shape), ...].
    """
    current_sd = policy.state_dict()
    filtered = {}
    skipped = []

    for key in current_sd:
        if key not in saved_sd:
            filtered[key] = current_sd[key]
            skipped.append((key, None, tuple(current_sd[key].shape)))
        elif saved_sd[key].shape == current_sd[key].shape:
            filtered[key] = saved_sd[key]
        else:
            old_t = saved_sd[key]
            new_t = torch.full_like(current_sd[key], fill_value)
            slices = tuple(
                slice(0, min(o, n))
                for o, n in zip(old_t.shape, new_t.shape)
            )
            new_t[slices] = old_t[slices]
            filtered[key] = new_t
            skipped.append((key, tuple(old_t.shape), tuple(new_t.shape)))

    policy.load_state_dict(filtered)
    return skipped