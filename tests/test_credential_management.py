import os
import stat

from kenkui.config import ProviderCredentials, load_provider_credentials, save_provider_credentials


def test_save_and_load_credentials(tmp_path):
    creds_path = tmp_path / "credentials.enc"
    creds = {
        "anthropic": ProviderCredentials(api_key="sk-ant-test", default_model="claude-3-5-sonnet-20241022"),
        "openai": ProviderCredentials(api_key="sk-test", default_model="gpt-4o"),
    }
    save_provider_credentials(creds, creds_path)
    assert b"sk-ant-test" not in creds_path.read_bytes()
    loaded = load_provider_credentials(creds_path)
    assert loaded["anthropic"].api_key == "sk-ant-test"
    assert loaded["openai"].default_model == "gpt-4o"


def test_credentials_file_has_restricted_permissions(tmp_path):
    creds_path = tmp_path / "credentials.enc"
    save_provider_credentials(
        {"anthropic": ProviderCredentials(api_key="secret", default_model="claude-3-5-sonnet-20241022")},
        creds_path,
    )
    mode = stat.S_IMODE(creds_path.stat().st_mode)
    assert mode == 0o600
    key_mode = stat.S_IMODE((tmp_path / "credentials.key").stat().st_mode)
    assert key_mode == 0o600


def test_load_missing_credentials_returns_empty(tmp_path):
    missing = tmp_path / "nonexistent.enc"
    result = load_provider_credentials(missing)
    assert result == {}


def test_inject_credentials_as_env_vars(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("KENKUI_ANTHROPIC_API_KEY", raising=False)
    creds_path = tmp_path / "credentials.enc"
    save_provider_credentials(
        {"anthropic": ProviderCredentials(api_key="sk-ant-test", default_model="claude-opus-4-6")},
        creds_path,
    )
    from kenkui.config import inject_provider_env_vars
    inject_provider_env_vars(load_provider_credentials(creds_path))
    assert os.environ.get("ANTHROPIC_API_KEY") == "sk-ant-test"


def test_openrouter_credentials_are_injected(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("KENKUI_OPENROUTER_API_KEY", raising=False)
    creds_path = tmp_path / "credentials.enc"
    save_provider_credentials(
        {"openrouter": ProviderCredentials(api_key="sk-or-file", default_model="openai/gpt-4.1-mini")},
        creds_path,
    )
    from kenkui.config import inject_provider_env_vars
    inject_provider_env_vars(load_provider_credentials(creds_path))
    assert os.environ.get("OPENROUTER_API_KEY") == "sk-or-file"


def test_kenkui_openrouter_env_overrides_credentials(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("KENKUI_OPENROUTER_API_KEY", "sk-or-env")
    creds_path = tmp_path / "credentials.enc"
    save_provider_credentials(
        {"openrouter": ProviderCredentials(api_key="sk-or-file", default_model="openai/gpt-4.1-mini")},
        creds_path,
    )
    from kenkui.config import inject_provider_env_vars
    inject_provider_env_vars(load_provider_credentials(creds_path))
    assert os.environ.get("OPENROUTER_API_KEY") == "sk-or-env"


def test_standard_env_overrides_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-standard")
    monkeypatch.delenv("KENKUI_OPENROUTER_API_KEY", raising=False)
    creds_path = tmp_path / "credentials.enc"
    save_provider_credentials(
        {"openrouter": ProviderCredentials(api_key="sk-or-file", default_model="openai/gpt-4.1-mini")},
        creds_path,
    )

    from kenkui.config import inject_provider_env_vars
    inject_provider_env_vars(load_provider_credentials(creds_path))

    assert os.environ.get("OPENROUTER_API_KEY") == "sk-or-standard"


def test_legacy_plaintext_credentials_migrate_to_encrypted_file(tmp_path):
    legacy_path = tmp_path / "credentials.toml"
    encrypted_path = tmp_path / "credentials.enc"
    key_path = tmp_path / "credentials.key"
    legacy_path.write_text(
        '[providers.openai]\napi_key = "sk-legacy"\ndefault_model = "gpt-4o"\n',
        encoding="utf-8",
    )

    loaded = load_provider_credentials(
        encrypted_path,
        key_path=key_path,
        legacy_path=legacy_path,
    )

    assert loaded["openai"].api_key == "sk-legacy"
    assert encrypted_path.exists()
    assert key_path.exists()
    assert not legacy_path.exists()
    assert b"sk-legacy" not in encrypted_path.read_bytes()
