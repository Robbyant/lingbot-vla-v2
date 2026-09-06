import pytest
import torch

from lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2 import (
    LingbotVlaV2Policy,
    _action_valid_mask,
)


class _Config:
    action_fp32 = False
    action_dim = 2
    loss_type = "fm"


class _LossModel:
    def __init__(self, losses):
        self.losses = losses

    def forward(self, *args, **kwargs):
        del args, kwargs
        zero = self.losses.new_zeros(())
        return (
            self.losses,
            zero,
            zero,
            zero,
            None,
            zero,
            zero,
            {},
            None,
            None,
            None,
        )


class _Policy:
    config = _Config()

    def __init__(self, losses):
        self.model = _LossModel(losses)


def _forward_with_losses(losses, *, action_is_pad, joint_mask=None):
    policy = _Policy(losses)
    return LingbotVlaV2Policy.forward(
        policy,
        images=None,
        img_masks=None,
        state=torch.zeros(1),
        lang_tokens=None,
        lang_masks=None,
        actions=torch.zeros(1),
        joint_mask=joint_mask,
        action_is_pad=action_is_pad,
    )


def test_action_valid_mask_excludes_padded_steps_and_trims_extra_steps():
    losses = torch.ones(2, 3, 4)
    action_is_pad = torch.tensor(
        [
            [False, True, False, True],
            [True, False, False, True],
        ]
    )

    mask = _action_valid_mask(losses, action_is_pad)

    expected = torch.tensor(
        [
            [[True], [False], [True]],
            [[False], [True], [True]],
        ]
    )
    assert torch.equal(mask, expected)


def test_action_valid_mask_repeats_for_a_paired_loss_batch():
    losses = torch.ones(4, 3, 2)
    action_is_pad = torch.tensor(
        [
            [False, True, False],
            [True, False, False],
        ]
    )

    mask = _action_valid_mask(losses, action_is_pad)

    expected = (~action_is_pad).repeat(2, 1).unsqueeze(-1)
    assert torch.equal(mask, expected)


def test_policy_excludes_padded_steps_from_loss_and_denominator():
    losses = torch.arange(1, 19, dtype=torch.float32).reshape(2, 3, 3)
    losses.requires_grad_()
    action_is_pad = torch.tensor(
        [
            [False, True, False],
            [True, True, False],
        ]
    )

    result = _forward_with_losses(losses, action_is_pad=action_is_pad)
    total_loss, loss_vla, loss_dict = result[0], result[1], result[6]

    assert torch.allclose(loss_dict["batch_mean_losses"], torch.tensor([4.5, 16.5]))
    assert torch.allclose(loss_vla, torch.tensor(8.5))

    total_loss.backward()
    expected_grad = torch.zeros_like(losses)
    expected_grad[0, (0, 2), :2] = 1 / 6
    expected_grad[1, 2, :2] = 1 / 6
    assert torch.allclose(losses.grad, expected_grad)


def test_policy_intersects_joint_and_episode_masks():
    losses = torch.arange(1, 10, dtype=torch.float32).reshape(1, 3, 3)
    joint_mask = torch.tensor(
        [
            [
                [True, False, True],
                [True, True, True],
                [False, True, True],
            ]
        ]
    )

    result = _forward_with_losses(
        losses,
        action_is_pad=torch.tensor([[False, True, False]]),
        joint_mask=joint_mask,
    )

    assert torch.allclose(result[1], torch.tensor(5.25))
    assert torch.allclose(result[6]["batch_mean_losses"], torch.tensor([5.25]))


@pytest.mark.parametrize(
    "loss_shape,pad_shape",
    [
        ((2, 3, 4), (3, 3)),
        ((2, 3, 4), (2, 2)),
        ((2, 3, 4), (2, 3, 1)),
    ],
)
def test_action_valid_mask_rejects_incompatible_shapes(loss_shape, pad_shape):
    with pytest.raises(ValueError):
        _action_valid_mask(torch.ones(loss_shape), torch.zeros(pad_shape))
