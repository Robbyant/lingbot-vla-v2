import pytest

from lingbotvla.utils.vla_prompt import validate_vla_prompt_type


def test_global_prompt_type_is_supported():
    assert validate_vla_prompt_type("global") == "global"


@pytest.mark.parametrize("prompt_type", ["subtask", "both"])
def test_unsupported_prompt_types_fail_fast(prompt_type):
    with pytest.raises(ValueError, match="only supports global task prompts"):
        validate_vla_prompt_type(prompt_type)
