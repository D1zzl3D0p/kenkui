"""Tests for the HuggingFace authentication module."""

import pytest
from kenkui.huggingface_auth import (
    is_custom_voice,
    is_model_gated,
    GATED_MODELS,
)
from kenkui.utils import DEFAULT_VOICES


class TestIsCustomVoice:
    """Tests for the is_custom_voice function."""

    def test_default_voices_not_custom(self):
        """Test that default built-in voices return False."""
        bundled_voices = []
        for voice in DEFAULT_VOICES:
            assert is_custom_voice(voice, bundled_voices) is False

    def test_bundled_voices_are_custom(self):
        """Test that bundled voices return True."""
        bundled_voices = ["custom1.wav", "custom2.wav"]
        assert is_custom_voice("custom1", bundled_voices) is True
        assert is_custom_voice("custom2", bundled_voices) is True

    def test_local_file_paths_are_custom(self):
        """Test that local file paths return True."""
        bundled_voices = []
        assert is_custom_voice("/path/to/voice.wav", bundled_voices) is True
        assert is_custom_voice("./relative/path/voice.wav", bundled_voices) is True
        assert is_custom_voice("voice.wav", bundled_voices) is True

    def test_hf_urls_are_custom(self):
        """Test that hf:// URLs return True."""
        bundled_voices = []
        assert is_custom_voice("hf://user/repo/voice.wav", bundled_voices) is True
        assert is_custom_voice("hf://org/model/file.wav", bundled_voices) is True

    def test_unknown_voices_are_custom(self):
        """Test that unknown voices return True (to be safe)."""
        bundled_voices = []
        assert is_custom_voice("unknown_voice", bundled_voices) is True
        assert is_custom_voice("random", bundled_voices) is True


class TestIsModelGated:
    """Tests for the is_model_gated function."""

    def test_gated_model_returns_true(self):
        """Test that known gated models return True."""
        assert is_model_gated("kyutai/pocket-tts") is True

    def test_ungated_model_returns_false(self):
        """Test that unknown models return False."""
        assert is_model_gated("unknown/model") is False
        assert is_model_gated("user/repo") is False

    def test_gated_models_dict_is_configurable(self):
        """Test that GATED_MODELS dict can be modified."""
        # Add a test model
        GATED_MODELS["test/model"] = True
        assert is_model_gated("test/model") is True
        
        # Clean up
        del GATED_MODELS["test/model"]
        assert is_model_gated("test/model") is False
