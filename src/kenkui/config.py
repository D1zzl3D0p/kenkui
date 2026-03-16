import os
from pathlib import Path

import yaml

from .models import AppConfig, JobConfig

CONFIG_DIR = Path(os.path.expanduser("~/.config/kenkui"))
CONFIG_DIR.mkdir(parents=True, exist_ok=True)


class ConfigManager:
    def __init__(self):
        self.configs_dir = CONFIG_DIR / "configs"
        self.configs_dir.mkdir(exist_ok=True)
        self._default_config: AppConfig | None = None

    @property
    def default_config(self) -> AppConfig:
        if self._default_config is None:
            self._default_config = self._load_default_app_config()
        return self._default_config

    def _get_default_config_path(self) -> Path:
        return CONFIG_DIR / "default.yaml"

    def _load_default_app_config(self) -> AppConfig:
        path = self._get_default_config_path()
        if path.exists():
            try:
                data = yaml.safe_load(path.read_text())
                if data:
                    return AppConfig.from_dict(data)
            except Exception:
                pass
        return AppConfig()

    def load_app_config(self, name: str | None = None) -> AppConfig:
        if name is None:
            return self.default_config

        config_path = Path(name)
        if config_path.exists() and config_path.is_file():
            try:
                data = yaml.safe_load(config_path.read_text())
                if data:
                    return AppConfig.from_dict(data)
            except Exception:
                pass

        path = self.configs_dir / f"{name}.yaml"
        if path.exists():
            try:
                data = yaml.safe_load(path.read_text())
                if data:
                    return AppConfig.from_dict(data)
            except Exception:
                pass
        return self.default_config

    def save_app_config(self, config: AppConfig, name: str | None = None) -> Path:
        if name is None:
            path = self._get_default_config_path()
        else:
            path = self.configs_dir / f"{name}.yaml"

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.dump(config.to_dict(), default_flow_style=False))
        return path

    def load_job_config(self, name: str) -> JobConfig | None:
        path = self.configs_dir / f"{name}.yaml"
        if path.exists():
            try:
                data = yaml.safe_load(path.read_text())
                if data and "job" in data:
                    return JobConfig.from_dict(data["job"])
            except Exception:
                pass
        return None

    def save_job_config(self, job: JobConfig, name: str) -> Path:
        path = self.configs_dir / f"{name}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"job": job.to_dict()}
        path.write_text(yaml.dump(data, default_flow_style=False))
        return path

    def list_configs(self) -> list[str]:
        configs = []
        for f in self.configs_dir.glob("*.yaml"):
            configs.append(f.stem)
        if self._get_default_config_path().exists():
            configs.insert(0, "default")
        return sorted(configs)

    def delete_config(self, name: str) -> bool:
        if name == "default":
            path = self._get_default_config_path()
        else:
            path = self.configs_dir / f"{name}.yaml"

        if path.exists():
            path.unlink()
            return True
        return False


_config_manager: ConfigManager | None = None


def get_config_manager() -> ConfigManager:
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager
