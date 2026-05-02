from kenkui.modal.gpu_tiers import resolve_gpu_tier, DEFAULT_GPU, GPU_TIERS


def test_default_gpu_is_t4():
    assert DEFAULT_GPU == "T4"


def test_3b_resolves_t4():
    assert resolve_gpu_tier("llama-3b") == "T4"


def test_7b_resolves_t4():
    assert resolve_gpu_tier("mistral-7b-instruct") == "T4"


def test_8b_resolves_a10g():
    assert resolve_gpu_tier("llama3-8b-instruct") == "A10G"


def test_13b_resolves_a10g():
    assert resolve_gpu_tier("llama-13b") == "A10G"


def test_70b_resolves_a100():
    assert resolve_gpu_tier("llama3.1-70b-instruct") == "A100"


def test_unknown_model_returns_default():
    assert resolve_gpu_tier("some-unknown-model") == DEFAULT_GPU


def test_override_env_var(monkeypatch):
    monkeypatch.setenv("KENKUI_MODAL_NLP_GPU_OVERRIDE", "A100")
    assert resolve_gpu_tier("unknown-model") == "A100"


def test_override_does_not_affect_known_model_without_override(monkeypatch):
    monkeypatch.delenv("KENKUI_MODAL_NLP_GPU_OVERRIDE", raising=False)
    assert resolve_gpu_tier("llama3-8b") == "A10G"
