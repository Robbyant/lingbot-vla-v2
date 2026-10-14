"""CPU regressions for action/state relative transforms, without dataset or model loading."""

import importlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
OMITTED = object()


@pytest.fixture(scope="module")
def feature_module():
    # Import the real modules without data/__init__.py's unrelated distributed,
    # LeRobot and model imports. No transform or tensor operation is mocked.
    prefix = "_relative_transform_test"
    paths = {
        prefix: ROOT / "lingbotvla",
        f"{prefix}.utils": ROOT / "lingbotvla/utils",
        f"{prefix}.data": ROOT / "lingbotvla/data",
        f"{prefix}.data.vla_data": ROOT / "lingbotvla/data/vla_data",
    }
    sys.path.insert(0, str(ROOT))
    for name, path in paths.items():
        package = ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package
    try:
        yield importlib.import_module(f"{prefix}.data.vla_data.utils")
    finally:
        sys.path.remove(str(ROOT))
        for name in list(sys.modules):
            if name == prefix or name.startswith(prefix + "."):
                del sys.modules[name]


def data_config():
    data = yaml.safe_load((ROOT / "configs/vla/real_robot/real_robot.yaml").read_text())["data"]
    return SimpleNamespace(
        joints=[repr(joint) for joint in data["joints"]], cameras=data["cameras"],
        norm_type=[repr(norm) for norm in data["norm_type"]],
    )


def make_transform(feature_module, config_path):
    transform = feature_module.FeatureTransform(
        str(config_path), data_config(), None, None,
        disabled_image_features=True, do_nomalize=False, chunk_size=2,
    )
    # The constructor performs normal training-config validation. Inspect the
    # numerical output before padding, as the norm-statistics path also does.
    transform.return_item_befor_padding = True
    return transform


def write_config(tmp_path, feature, relative_type=OMITTED, subtract_state=True):
    action = f"action.{feature}"
    state = f"observation.state.{feature}"
    options = {"origin_keys": action, "subtract_state": subtract_state}
    if relative_type is not OMITTED:
        options["relative_type"] = relative_type
    config = {
        "states": [{state: {"origin_keys": state}}],
        "actions": [{action: options}],
        "images": [], "norm_stats": "unused.json",
    }
    path = tmp_path / "robot.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def raw_item(feature, state, action):
    return {
        f"observation.state.{feature}": state.clone(),
        f"action.{feature}": action.clone(),
        f"action.{feature}_is_pad": torch.zeros(action.shape[0], dtype=torch.bool),
        "task": "move",
    }


def agilex_item():
    arm_state = torch.arange(14, dtype=torch.float64) / 10
    arm_action = torch.stack([arm_state + 0.25, arm_state - 0.5])
    item = raw_item("arm.position", arm_state, arm_action)
    item.update(raw_item("effector.position", torch.tensor([0.1, 0.9]), torch.tensor([[0.2, 0.8], [0.3, 0.7]])))
    return item


def test_official_agilex_apply_and_roundtrip(feature_module):
    transform = make_transform(feature_module, ROOT / "configs/robot_configs/agilex_cobot_magic.yaml")
    raw = agilex_item()
    relative = transform.apply({key: value.clone() if torch.is_tensor(value) else value for key, value in raw.items()})
    torch.testing.assert_close(relative["action.arm.position"], raw["action.arm.position"] - raw["observation.state.arm.position"])
    torch.testing.assert_close(relative["action.effector.position"], raw["action.effector.position"])
    absolute = transform.unapply(relative)
    for key in ("action.arm.position", "action.effector.position", "observation.state.arm.position", "observation.state.effector.position"):
        torch.testing.assert_close(absolute[key], raw[key])


def test_official_agilex_unapply(feature_module):
    transform = make_transform(feature_module, ROOT / "configs/robot_configs/agilex_cobot_magic.yaml")
    raw = agilex_item()
    relative = {key: value.clone() if torch.is_tensor(value) else value for key, value in raw.items()}
    relative["action.arm.position"] -= raw["observation.state.arm.position"]
    absolute = transform.unapply(relative)
    torch.testing.assert_close(absolute["action.arm.position"], raw["action.arm.position"])
    torch.testing.assert_close(absolute["action.effector.position"], raw["action.effector.position"])


@pytest.mark.parametrize("policy_eval", [False, True], ids=["training", "inference"])
def test_official_agilex_normalized_padded_roundtrip(feature_module, tmp_path, policy_eval):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    # A local, real tokenizer needs no pretrained checkpoint or network access.
    backend = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "move": 2}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]")
    stats = {}
    for kind in ("action", "observation.state"):
        for feature, dim in (("arm.position", 14), ("effector.position", 2)):
            stats[f"{kind}.{feature}"] = {"mean": [0.1] * dim, "std": [2.0] * dim}
    stats_path = tmp_path / "norm.json"
    stats_path.write_text(json.dumps({"norm_stats": stats}))
    transform = feature_module.FeatureTransform(
        str(ROOT / "configs/robot_configs/agilex_cobot_magic.yaml"), data_config(),
        SimpleNamespace(max_state_dim=55, max_action_dim=55, tokenizer_max_length=6),
        SimpleNamespace(tokenizer=tokenizer, image_processor=None),
        disabled_image_features=True, chunk_size=2, norm_stats_path=str(stats_path),
    )
    raw = agilex_item()
    inputs = {key: value.clone() if torch.is_tensor(value) else value for key, value in raw.items()}
    if policy_eval:
        inputs = {key: value for key, value in inputs.items() if not key.startswith("action.")}
    batch = transform.apply(inputs, policy_eval=policy_eval)
    expected = torch.zeros(2, 55)
    expected[:, :14] = (raw["action.arm.position"] - raw["observation.state.arm.position"] - 0.1) / (2.0 + 1e-6)
    expected[:, 28:30] = (raw["action.effector.position"] - 0.1) / (2.0 + 1e-6)
    if policy_eval:
        # Synthetic model output; the test covers inference processing and inverse mapping.
        batch["actions"] = expected.clone()
    torch.testing.assert_close(batch["actions"], expected)
    assert batch["state"].shape == (55,)
    assert batch["actions"].device.type == "cpu"
    assert batch["action_joint_mask"].nonzero().flatten().tolist() == list(range(14)) + [28, 29]
    absolute = transform.unapply(batch)
    for key in ("action.arm.position", "action.effector.position", "observation.state.arm.position", "observation.state.effector.position"):
        torch.testing.assert_close(absolute[key], raw[key], check_dtype=False, rtol=1e-5, atol=1e-6)


def test_state_derived_joint_actions_roundtrip(feature_module, tmp_path):
    path = write_config(tmp_path, "arm.position")
    config = yaml.safe_load(path.read_text())
    options = config["actions"][0]["action.arm.position"]
    options.pop("origin_keys")
    options["convert_from_state"] = True
    path.write_text(yaml.safe_dump(config))
    transform = make_transform(feature_module, path)
    states = torch.tensor([[1, 2, 3], [2, 4, 6], [3, 6, 9]], dtype=torch.float64)
    relative = transform.apply({
        "observation.state.arm.position": states.clone(),
        "observation.state.arm.position_is_pad": torch.zeros(3, dtype=torch.bool),
    })
    torch.testing.assert_close(relative["action.arm.position"], states[1:] - states[0])
    torch.testing.assert_close(transform.unapply(relative)["observation.state.arm.position"], states)


@pytest.mark.parametrize("feature,dim", [("arm.position", 3), ("waist.position", 4), ("head.position", 2), ("base.position", 3)])
def test_non_pose_defaults_are_componentwise(feature_module, tmp_path, feature, dim):
    transform = make_transform(feature_module, write_config(tmp_path, feature))
    state = torch.arange(dim, dtype=torch.float64) / 5
    action = torch.stack([state + 0.3, state - 0.7])
    relative = transform.apply(raw_item(feature, state, action))
    torch.testing.assert_close(relative[f"action.{feature}"], action - state)
    torch.testing.assert_close(transform.unapply(relative)[f"action.{feature}"], action)


@pytest.mark.parametrize("relative_type", [OMITTED, "quaternion", "quaternion_world", "quaternion_local"])
@pytest.mark.parametrize("arms", [1, 2])
def test_end_pose_semantics_and_roundtrip(feature_module, tmp_path, relative_type, arms):
    transform = make_transform(feature_module, write_config(tmp_path, "end.position", relative_type))
    s = 2 ** -0.5
    # State is rotated +90 degrees about z; absolute actions have identity rotation.
    state = torch.tensor([1, 2, 3, 0, 0, s, s], dtype=torch.float64).repeat(arms)
    action = torch.tensor([[3, 2, 3, 0, 0, 0, 1], [1, 4, 3, 0, 0, 0, 1]], dtype=torch.float64).repeat(1, arms)
    local = relative_type is OMITTED or relative_type == "quaternion_local"
    xyz = [[0, -2, 0], [2, 0, 0]] if local else [[2, 0, 0], [0, 2, 0]]
    expected = torch.tensor([row + [0, 0, -s, s] for row in xyz], dtype=torch.float64).repeat(1, arms)
    relative = transform.apply(raw_item("end.position", state, action))
    torch.testing.assert_close(relative["action.end.position"], expected)
    torch.testing.assert_close(transform.unapply(relative)["action.end.position"], action)


@pytest.mark.parametrize("feature", ["arm.position", "end.position"])
@pytest.mark.parametrize("relative_type", [None, "subtract"])
def test_explicit_subtraction_is_preserved(feature_module, tmp_path, feature, relative_type):
    transform = make_transform(feature_module, write_config(tmp_path, feature, relative_type))
    state = torch.arange(7, dtype=torch.float64) / 10
    action = torch.stack([state + 1, state - 1])
    relative = transform.apply(raw_item(feature, state, action))
    torch.testing.assert_close(relative[f"action.{feature}"], action - state)
    torch.testing.assert_close(transform.unapply(relative)[f"action.{feature}"], action)


@pytest.mark.parametrize("relative_type", ["quaternion", "quaternion_world", "quaternion_local"])
def test_quaternion_relative_non_pose_is_rejected_at_construction(feature_module, tmp_path, relative_type):
    with pytest.raises(ValueError, match=r"action\.arm\.position.*end\.position"):
        make_transform(feature_module, write_config(tmp_path, "arm.position", relative_type))


@pytest.mark.parametrize("feature", ["arm.position", "end.position", "effector.position"])
def test_absolute_actions_do_not_apply_relative_type(feature_module, tmp_path, feature):
    transform = make_transform(feature_module, write_config(tmp_path, feature, "quaternion_local", subtract_state=False))
    state = torch.arange(3, dtype=torch.float64)
    action = torch.stack([state + 1, state - 1])
    absolute = transform.apply(raw_item(feature, state, action))
    torch.testing.assert_close(absolute[f"action.{feature}"], action)
    torch.testing.assert_close(transform.unapply(absolute)[f"action.{feature}"], action)


def test_relative_effector_remains_rejected(feature_module, tmp_path):
    with pytest.raises(AssertionError, match="cannot be subtracted from state"):
        make_transform(feature_module, write_config(tmp_path, "effector.position"))
