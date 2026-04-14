import os

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from tools.test import run_inference
from utils.config import apply_resume_config, save_resolved_config


def _require_helper_script() -> None:
    if os.environ.get("UNICO_USE_HELPER_SCRIPT") != "1":
        raise RuntimeError(
            "Inference must be launched via the helper scripts. "
            "Use `bash scripts/infer.sh ...`."
        )


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    _require_helper_script()
    cfg = apply_resume_config(cfg, HydraConfig.get().overrides.task)
    save_resolved_config(cfg, f"{cfg.paths.run_dir}/config.resolved.yaml")
    run_inference(cfg)


if __name__ == "__main__":
    main()
