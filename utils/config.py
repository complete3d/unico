import os
from typing import Iterable

from omegaconf import DictConfig, OmegaConf

from .logger import print_log


def log_config_to_file(cfg: DictConfig, pre: str = "cfg", logger=None) -> None:
    resolved = OmegaConf.to_yaml(cfg, resolve=True)
    for line in resolved.splitlines():
        print_log(f"{pre}: {line}", logger=logger)


def save_resolved_config(cfg: DictConfig, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    OmegaConf.save(cfg, output_path, resolve=True)


def load_resume_config(run_dir: str) -> DictConfig:
    cfg_path = os.path.join(run_dir, "config.resolved.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Resume config not found: {cfg_path}")
    return OmegaConf.load(cfg_path)


def _override_key(override: str) -> str:
    key = override
    for prefix in ("++", "+", "~"):
        if key.startswith(prefix):
            key = key[len(prefix):]
            break
    for separator in ("=", "@"):
        if separator in key:
            key = key.split(separator, 1)[0]
            break
    return key


def _filter_resume_overrides(resume_cfg: DictConfig, overrides: Iterable[str]) -> list[str]:
    filtered = []
    missing = object()
    for override in overrides:
        key = _override_key(override)
        if not key:
            continue
        if OmegaConf.select(resume_cfg, key, default=missing) is missing and key not in resume_cfg:
            continue
        filtered.append(override)
    return filtered


def apply_resume_config(cfg: DictConfig, overrides: Iterable[str]) -> DictConfig:
    if not bool(cfg.runtime.resume):
        return cfg

    resume_cfg = load_resume_config(cfg.paths.run_dir)
    filtered_overrides = _filter_resume_overrides(resume_cfg, overrides)
    overlay = OmegaConf.from_dotlist(filtered_overrides)
    return OmegaConf.merge(resume_cfg, overlay)
