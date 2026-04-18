import os
import stat
import tempfile
from pathlib import Path
import pytest
from kenkui.config import load_provider_credentials, save_provider_credentials, ProviderCredentials


def test_save_and_load_credentials(tmp_path):
    creds_path = tmp_path / "credentials.toml"
    creds = {
        "anthropic": ProviderCredentials(api_key="sk-ant-test", default_model="claude-3-5-sonnet-20241022"),
        "openai": ProviderCredentials(api_key="sk-test", default_model="gpt-4o"),
    }
    save_provider_credentials(creds, creds_path)
    loaded = load_provider_credentials(creds_path)
    assert loaded["anthropic"].api_key == "sk-ant-test"
    assert loaded["openai"].default_model == "gpt-4o"


def test_credentials_file_has_restricted_permissions(tmp_path):
    creds_path = tmp_path / "credentials.toml"
    save_provider_credentials(
        {"anthropic": ProviderCredentials(api_key="secret", default_model="claude-3-5-sonnet-20241022")},
        creds_path,
    )
    mode = stat.S_IMODE(creds_path.stat().st_mode)
    assert mode == 0o600


def test_load_missing_credentials_returns_empty(tmp_path):
    missing = tmp_path / "nonexistent.toml"
    result = load_provider_credentials(missing)
    assert result == {}


def test_inject_credentials_as_env_vars(tmp_path, monkeypatch):
    creds_path = tmp_path / "credentials.toml"
    save_provider_credentials(
        {"anthropic": ProviderCredentials(api_key="sk-ant-test", default_model="claude-opus-4-6")},
        creds_path,
    )
    from kenkui.config import inject_provider_env_vars
    inject_provider_env_vars(load_provider_credentials(creds_path))
    assert os.environ.get("ANTHROPIC_API_KEY") == "sk-ant-test"
