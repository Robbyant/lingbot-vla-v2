import logging
from pathlib import Path
import argparse
import importlib
import inspect
import os
import sys

import numpy as np
from matplotlib import pyplot as plt

import torch
import yaml

from lingbotvla.data.vla_data.base_dataset import LeRobotDataset

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    LEROBOT_DATASET_API = "v3"
except ImportError:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
    LEROBOT_DATASET_API = "v2"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

POLICY_MODULES = {
    "qwen2": "deploy.lingbot_vla_policy",
    "qwen3vl": "deploy.lingbot_vla_v2_policy",
}


def _get_training_config_path(model_path: str) -> Path:
    return Path(model_path).parent.parent.parent / "lingbotvla_cli.yaml"


def resolve_policy_module(policy: str, model_path: str) -> str:
    if policy != "auto":
        return POLICY_MODULES[policy]

    training_config_path = _get_training_config_path(model_path)
    with open(training_config_path, "r") as f:
        training_config = yaml.safe_load(f)

    model_config = training_config.get("model", {})
    config_key = model_config.get("config_key", "")
    tokenizer_path = str(model_config.get("tokenizer_path", "")).lower()
    if config_key == "LingbotVLAV2Config" or ("qwen3" in tokenizer_path and "vl" in tokenizer_path):
        return POLICY_MODULES["qwen3vl"]
    return POLICY_MODULES["qwen2"]


def model_uses_video(model_path: str) -> bool:
    training_config_path = _get_training_config_path(model_path)
    with open(training_config_path, "r") as f:
        training_config = yaml.safe_load(f)
    return bool(training_config.get("data", {}).get("video_enabled", False))


def load_policy_server(policy: str, model_path: str):
    policy_module = resolve_policy_module(policy, model_path)
    module = importlib.import_module(policy_module)
    print(f"Using policy module: {policy_module}")
    for class_name in ("LingbotVLAv2Server", "LingbotVLAServer"):
        if hasattr(module, class_name):
            return getattr(module, class_name)
    raise AttributeError(
        f"Policy module {policy_module} does not define LingbotVLAv2Server or LingbotVLAServer."
    )


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def prepare_eval_observation(policy, traj):
    traj = dict(traj)
    for image_key in policy.vla.feature_transform.org_features['images']:
        image = (traj[image_key]).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        traj[image_key] = image

    # if state or action is a 0-d tensor, convert it to 1-d tensor
    for action_feature in policy.vla.feature_transform.org_features['actions']:
        if len(traj[action_feature].shape) == 1:
            traj[action_feature] = traj[action_feature].unsqueeze(-1)

    for state_feature in policy.vla.feature_transform.org_features['states']:
        if len(traj[state_feature].shape) == 0:
            traj[state_feature] = traj[state_feature].unsqueeze(-1)

    org_traj = dict(traj)
    for k, v in org_traj.items():
        if isinstance(v, np.ndarray):
            org_traj[k] = torch.from_numpy(v)

    feature_transform = policy.vla.feature_transform
    original_disabled_image_features = feature_transform.disabled_image_features
    feature_transform.disabled_image_features = True
    try:
        org_traj = feature_transform.apply(org_traj)
        org_traj = feature_transform.unapply(org_traj)
    finally:
        feature_transform.disabled_image_features = original_disabled_image_features

    return traj, org_traj


def plot_trajectory_results(
    state_joints_across_time: np.ndarray,
    gt_action_across_time: np.ndarray,
    pred_action_across_time: np.ndarray,
    traj_id: int,
    action_keys: list[str],
    action_horizon: int,
    save_plot_path: str,
) -> None:
    """
    Plot and save trajectory results comparing ground truth and predicted actions.

    Args:
        state_joints_across_time: Array of state joints over time
        gt_action_across_time: Ground truth actions over time
        pred_action_across_time: Predicted actions over time
        traj_id: Trajectory ID
        action_keys: List of action modality keys
        action_horizon: Action horizon used for inference
        save_plot_path: Path to save the plot
    """
    actual_steps = len(gt_action_across_time)
    action_dim = gt_action_across_time.shape[1]

    indices_to_plot = list(range(action_dim))

    num_plots = len(indices_to_plot)
    if num_plots == 0:
        logging.warning("No valid indices to plot")
        return

    # Always plot and save
    fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(8, 4 * num_plots))

    # Handle case where there's only one subplot
    if num_plots == 1:
        axes = [axes]

    # Add a global title showing the modality keys
    fig.suptitle(
        f"Trajectory {traj_id}",
        fontsize=16,
        color="blue",
    )
 
    for plot_idx, action_idx in enumerate(indices_to_plot):
        ax = axes[plot_idx]
        # The dimensions of state_joints and action are the same
        # only when the robot uses actions directly as joint commands.
        # Therefore, do not plot them if this is not the case.
        if state_joints_across_time.shape == gt_action_across_time.shape:
            ax.plot(state_joints_across_time[:, action_idx], label="state joints")
        ax.plot(gt_action_across_time[:, action_idx], label="gt action")
        ax.plot(pred_action_across_time[:, action_idx], label="pred action")

        # put a dot every ACTION_HORIZON
        for j in range(0, actual_steps, action_horizon):
            if j == 0:
                ax.plot(j, gt_action_across_time[j, action_idx], "ro", label="inference point")
            else:
                ax.plot(j, gt_action_across_time[j, action_idx], "ro")

        ax.set_title(f"Action {action_idx}")
        ax.legend()

    plt.tight_layout()

    # Create filename with trajectory ID
    Path(save_plot_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_plot_path)

    plt.close()  # Close the figure to free memory



def evaluate_single_trajectory(
    policy,
    dataset,
    traj_id: int,
    modality_keys: list[str] | None = None,
    steps=300,
    action_horizon=16,
    save_plot_path=None,
    max_infer_time = 10
):
    # Ensure steps doesn't exceed trajectory length
    # episode_data_index is the stable accessor across lerobot v2.x/v3.x packages
    start_id, end_id = dataset.episode_data_index['from'][traj_id], dataset.episode_data_index['to'][traj_id]
    
    gt_action_across_time = []
    state_joints_across_time = []
    pred_action_across_time  = []

    action_features = policy.vla.feature_transform.org_features['actions']
    state_features = policy.vla.feature_transform.org_features['states']

    if policy.chunk_ret:
        count = 0
        for data_id in range(start_id, end_id, action_horizon):
            traj, org_traj = prepare_eval_observation(policy, dataset[data_id])
            count += 1

            gt_action_across_time += [
                np.concatenate(
                    [to_numpy(org_traj[action_feature])[:action_horizon] for action_feature in action_features],
                    axis=-1,
                )
            ]
            state_joints_across_time += [
                np.concatenate([to_numpy(org_traj[state_feature]).reshape(1, -1) for state_feature in state_features], axis=-1)
            ]

            preds = policy.infer(traj)
            pred_action_across_time += [
                np.concatenate([to_numpy(preds[action_feature]) for action_feature in action_features], axis=-1)
            ]

            if count >= max_infer_time:
                break
    else:
        max_eval_steps = max_infer_time * action_horizon
        step_end_id = min(end_id, start_id + max_eval_steps)
        for data_id in range(start_id, step_end_id):
            traj, org_traj = prepare_eval_observation(policy, dataset[data_id])

            gt_action_across_time += [
                np.concatenate(
                    [to_numpy(org_traj[action_feature])[:1] for action_feature in action_features],
                    axis=-1,
                )
            ]
            state_joints_across_time += [
                np.concatenate([to_numpy(org_traj[state_feature]).reshape(1, -1) for state_feature in state_features], axis=-1)
            ]

            preds = policy.infer(traj)
            pred_action_across_time += [
                np.concatenate(
                    [to_numpy(preds[action_feature]).reshape(1, -1) for action_feature in action_features],
                    axis=-1,
                )
            ]
    
    gt_action_across_time = np.concatenate(gt_action_across_time, axis=0)
    state_joints_across_time = np.concatenate(state_joints_across_time, axis=0)
    pred_action_across_time = np.concatenate(pred_action_across_time, axis=0)
    
    pred_action_across_time = np.array(pred_action_across_time)
    assert gt_action_across_time.shape == pred_action_across_time.shape, (
        f"gt_action: {gt_action_across_time.shape}, pred_action: {pred_action_across_time.shape}"
    )

    # calc MSE and MAE across time
    mse = np.mean((gt_action_across_time - pred_action_across_time) ** 2)
    mae = np.mean(np.abs(gt_action_across_time - pred_action_across_time))
    logging.info(f"Unnormalized Action MSE across single traj: {mse}")
    logging.info(f"Unnormalized Action MAE across single traj: {mae}")

    logging.info(f"gt_action_joints vs time {gt_action_across_time.shape}")
    logging.info(f"pred_action_joints vs time {pred_action_across_time.shape}")

    # Plot trajectory results
    plot_trajectory_results(
        state_joints_across_time=state_joints_across_time,
        gt_action_across_time=gt_action_across_time,
        pred_action_across_time=pred_action_across_time,
        traj_id=traj_id,
        action_keys=policy.vla.feature_transform.org_features['actions'],
        action_horizon=action_horizon,
        save_plot_path=save_plot_path or f"/tmp/open_loop_eval/traj_{traj_id}.jpeg",
    )

    return mse, mae


def main(policy, robo_name, data_root, traj_ids, chunk_size, save_plot_path, max_infer_time):

    policy.data_config.num_episode = None
    policy.data_config.chunk_size = policy.config.chunk_size

    policy.data_config.train_path = data_root
    policy.data_config.data_name = robo_name

    data_path = Path(data_root)
    if data_path.is_absolute() and data_path.exists():
        repo_id = data_path.name
        root = data_path
    else:
        repo_id = data_root
        root = None
    dataset_meta = LeRobotDatasetMetadata(repo_id, root=root)
    delta_timestamps = {}
    for action_feature in policy.vla.feature_transform.org_features['actions']:
        delta_timestamps[action_feature] = [t / dataset_meta.fps for t in range(policy.config.chunk_size)]
    dataset = LeRobotDataset(repo_id, root=root, delta_timestamps=delta_timestamps)
    print(f"Dataset length: {len(dataset)}")
    logging.info(f"Running evaluation on trajectories: {traj_ids}")

    all_mse = []
    all_mae = []

    for traj_id in traj_ids:
        episodes_meta = dataset.meta.episodes
        if isinstance(episodes_meta, dict) and "episode_index" in episodes_meta:
            valid_episode_ids = episodes_meta["episode_index"]  # lerobot v3.0 meta
        elif LEROBOT_DATASET_API == "v2" or isinstance(episodes_meta, dict):
            valid_episode_ids = episodes_meta.keys()
        else:
            valid_episode_ids = episodes_meta

        if traj_id not in valid_episode_ids:
            logging.warning(f"Trajectory ID {traj_id} is out of range. Skipping.")
            continue

        print(f"Running trajectory: {traj_id}")
        policy.reset(robo_name)
        mse, mae = evaluate_single_trajectory(
            policy,
            dataset,
            traj_id,
            save_plot_path=os.path.join(save_plot_path,f'{traj_id}.png'),
            action_horizon=chunk_size,
            max_infer_time=max_infer_time,
        )
        print(f"MSE for trajectory {traj_id}: {mse}, MAE: {mae}")
        all_mse.append(mse)
        all_mae.append(mae)

    if all_mse:
        avg_mse = np.mean(np.array(all_mse))
        avg_mae = np.mean(np.array(all_mae))
        print(f"Average MSE across all trajs: {avg_mse}")
        print(f"Average MAE across all trajs: {avg_mae}")
    else:
        logging.info("No valid trajectories were evaluated.")
    logging.info("Done")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="open loop test")

    parser.add_argument('--model_path',  type=str, required=True)

    parser.add_argument('--robo_name',   type=str, required=True, help='robot type')
    parser.add_argument('--norm_path',   type=str, default=None, help='norm file path of training data')
    parser.add_argument('--data_path',   type=str, default=None, help='path of validation data')
    
    parser.add_argument('--traj_ids',    type=int, nargs='+', default=[0])

    parser.add_argument('--use_length',  type=int, default=50, help='use length of action chunk')
    parser.add_argument('--chunk_ret', type=str2bool, default=None, help='return action chunk or one action per infer call')
    parser.add_argument('--max_infer_time', type=int, default=10, help='max chunk forwards; step mode evaluates max_infer_time * use_length steps')
    parser.add_argument("--num_denoising_step", type=int, default=10, help="num of denoising step")
    parser.add_argument("--use_compile", action='store_true', help="use torch compile or not")
    parser.add_argument('--video_debug_dir', type=str, default=None, help='save sampled video clips for deploy debug')
    parser.add_argument(
        "--policy",
        type=str,
        choices=["auto", "qwen2", "qwen3vl"],
        default="auto",
        help="policy implementation to use; auto reads lingbotvla_cli.yaml",
    )

    parser.add_argument('--save_plot_path', type=str, default='./open_loop_test/')
    parser.add_argument('--use_bf16', action='store_true', help='use bfloat16 to reduce GPU memory')
    args = parser.parse_args()

    os.makedirs(args.save_plot_path, exist_ok=True)
    traj_ids = args.traj_ids
    chunk_ret = args.chunk_ret
    if chunk_ret is None:
        chunk_ret = not model_uses_video(args.model_path)
        print(f"chunk_ret not set; using chunk_ret={chunk_ret} based on video_enabled")

    PolicyServer = load_policy_server(args.policy, args.model_path)
    model_kwargs = dict(
        path_to_pi_model=args.model_path,
        robot_norm_path=args.norm_path,
        use_length=args.use_length,
        use_bf16=args.use_bf16,
        use_fp32=not args.use_bf16,
        chunk_ret=chunk_ret,
        use_compile=args.use_compile,
    )
    if "video_debug_dir" in inspect.signature(PolicyServer).parameters:
        model_kwargs["video_debug_dir"] = args.video_debug_dir
    model = PolicyServer(**model_kwargs)
    data_path = args.data_path if args.data_path is not None else model.data_config.train_path
    
    model.reset(args.robo_name)
    main(model, args.robo_name, data_path, traj_ids, args.use_length, args.save_plot_path, args.max_infer_time)
