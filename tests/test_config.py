import os
import pytest
from unittest import mock
from config.settings import EnvVar, Config


@pytest.fixture(autouse=True)
def reset_config_state():
    """
    The EnvVar descriptor caches values in `_value` and sets `_loaded` to True
    after the first access. We must reset this state before every test to
    ensure tests are isolated and don't see values from previous tests.
    """
    for attr_name, attr_value in Config.__dict__.items():
        if isinstance(attr_value, EnvVar):
            attr_value._loaded = False
            attr_value._value = None
    yield


class TestEnvVar:
    """Tests for the EnvVar descriptor logic handling casting and defaults."""

    def test_defaults_returned_when_missing(self):
        """Ensure default value is used if env var is unset."""

        class TestObj:
            VAR = EnvVar("TEST_KEY", default="my_default")

        with mock.patch.dict(os.environ, {}, clear=True):
            assert TestObj.VAR == "my_default"

    def test_string_retrieval(self):
        """Ensure basic string values are loaded."""

        class TestObj:
            VAR = EnvVar("TEST_KEY")

        with mock.patch.dict(os.environ, {"TEST_KEY": "actual_value"}):
            assert TestObj.VAR == "actual_value"

    def test_int_casting(self):
        """Ensure values are cast to integer correctly."""

        class TestObj:
            VAR = EnvVar("TEST_INT", cast=int)

        with mock.patch.dict(os.environ, {"TEST_INT": "42"}):
            assert TestObj.VAR == 42

    def test_float_casting(self):
        """Ensure values are cast to float correctly."""

        class TestObj:
            VAR = EnvVar("TEST_FLOAT", cast=float)

        with mock.patch.dict(os.environ, {"TEST_FLOAT": "3.14"}):
            assert TestObj.VAR == 3.14

    def test_bool_casting(self):
        """Test boolean variations."""

        class TestObj:
            V1 = EnvVar("K1", cast=bool)
            V2 = EnvVar("K2", cast=bool)
            V3 = EnvVar("K3", cast=bool)

        env = {"K1": "true", "K2": "0", "K3": "yes"}
        with mock.patch.dict(os.environ, env):
            assert TestObj.V1 is True
            assert TestObj.V2 is False
            assert TestObj.V3 is True

    def test_required_validation(self):
        """Ensure validation error raised for missing required variables."""

        class TestObj:
            VAR = EnvVar("REQUIRED_KEY", required=True)

        with mock.patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                ValueError, match="Missing required environment variable"
            ):
                _ = TestObj.VAR

    def test_type_error_validation(self):
        """Ensure ValueError raised when casting fails."""

        class TestObj:
            VAR = EnvVar("NUM_KEY", cast=int)

        with mock.patch.dict(os.environ, {"NUM_KEY": "not_a_number"}):
            with pytest.raises(ValueError, match="Config 'NUM_KEY' must be int"):
                _ = TestObj.VAR


class TestConfigValidation:
    """Tests for Config.validate() specific business rules."""

    def test_validate_success_generic(self):
        """Test a minimal valid configuration."""
        env = {
            "BOT_TOKEN": "123:ABC",
            "PROVIDER_NAME": "other",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            Config.validate()
            assert Config.BOT_TOKEN == "123:ABC"

    def test_validate_missing_bot_token(self):
        """Test that BOT_TOKEN is strictly required."""
        with mock.patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                ValueError, match="Missing required environment variable: BOT_TOKEN"
            ):
                Config.validate()

    def test_validate_perplexity_requirements(self):
        """Test Perplexity specific requirements."""
        env_bad = {"BOT_TOKEN": "123", "PROVIDER_NAME": "perplexity"}
        with mock.patch.dict(os.environ, env_bad, clear=True):
            with pytest.raises(ValueError, match="PERPLEXITY_COOKIES required"):
                Config.validate()

    def test_validate_groq_requirements(self):
        """Test Groq specific requirements."""
        env_bad = {"BOT_TOKEN": "123", "PROVIDER_NAME": "groq"}
        with mock.patch.dict(os.environ, env_bad, clear=True):
            with pytest.raises(ValueError, match="GROQ_API_KEY required"):
                Config.validate()

    def test_config_defaults_integration(self):
        env = {
            "BOT_TOKEN": "123",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            # Resetting dict allows defaults to kick in
            assert Config.WEB_HOST == "127.0.0.1"
            assert Config.MAX_RETRIES == 5
