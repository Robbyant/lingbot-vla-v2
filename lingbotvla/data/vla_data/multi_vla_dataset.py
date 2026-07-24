from typing import Callable
import numpy as np
from lingbotvla.utils import helper
import torch
try:
    from lerobot.common.constants import HF_LEROBOT_HOME
except ImportError:
    from lerobot.utils.constants import HF_LEROBOT_HOME
from torchvision.transforms.v2 import Resize
import torch.nn.functional as F
from tqdm import tqdm

from torch.utils.data import Dataset
from .base_dataset import VLADataset

logger = helper.create_logger(__name__)

def get_all_tasks(task_files, sep=' '):
    task_files = task_files.split(',')
    data_names, task_list = [], []
    for task_file in task_files:
        assert task_file.lower().endswith('.txt')
        with open(task_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data_name, task = line.split(sep)
                data_names.append(data_name)
                task_list.append(task)
        f.close()
    return data_names, task_list


class MultiVLADataset(Dataset):
    """A dataset consisting of multiple underlying `LeRobotDataset`s.

    The underlying `LeRobotDataset`s are effectively concatenated, and this class adopts much of the API
    structure of `LeRobotDataset`.
    """

    def __init__(
        self,
        repo_file: str,
        dataset_config,
        robot_config_root,
        config=None,
        processor=None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerances_s: dict | None = None,
        video_backend: str = 'torchcodec',
        chunk_size: int = 50,
        image_size = (224, 224),
        do_nomalize = True,
        disabled_image_features = False,
        return_item = False,
        transform=None,
        prompt_type = 'global',
        image_augment = False,
        use_depth_align=False,
        use_future_image=False,
    ):
        
        self.config = config
        self.processor = processor
        self.return_item = return_item

        data_names, repo_ids  = get_all_tasks(repo_file)
        self.data_names, self.repo_ids = data_names, repo_ids
        # super().__init__()
        self.tolerances_s = tolerances_s if tolerances_s else dict.fromkeys(repo_ids, 0.0001)
        # Construct the underlying datasets passing everything but `transform` and `delta_timestamps` which
        # are handled by this class.
        
        self.feature_transforms = {}

        _datasets = []
        for i, repo_id in tqdm(enumerate(repo_ids), desc="Initializing datasets", total=len(repo_ids)):
            feature_transform = self.feature_transforms[self.data_names[i]] if self.data_names[i] in self.feature_transforms else None
            dataset = VLADataset(
                    repo_id,
                    self.data_names[i],
                    dataset_config,
                    robot_config_root,
                    config,
                    processor,
                    video_backend = video_backend,
                    chunk_size = chunk_size,
                    image_size = image_size,
                    do_nomalize = do_nomalize,
                    return_item = return_item,
                    disabled_image_features = disabled_image_features,
                    feature_transform = feature_transform,
                    transform=transform,
                    image_augment = image_augment,
                    use_depth_align = use_depth_align,
                    use_future_image=use_future_image,
                )
            if self.data_names[i] not in self.feature_transforms:
                self.feature_transforms[self.data_names[i]] = dataset.feature_transform
            _datasets.append(dataset)

        self._datasets = _datasets

        dataset_start_index = []
        start_index = 0
        for dataset in self._datasets:
            dataset_start_index.append(start_index)
            start_index += dataset.dataset.num_frames
        self.dataset_start_index = dataset_start_index


    @property
    def num_frames(self) -> int:
        """Number of samples/frames."""
        return sum(d.dataset.num_frames for d in self._datasets)

   
    @property
    def num_episodes(self) -> int:
        """Number of episodes."""
        return sum(d.dataset.num_episodes for d in self._datasets)

    def __len__(self):
        return self.num_frames

    def getdata(self, idx: int) -> dict[str, torch.Tensor]:
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds.")
        # Determine which dataset to get an item from based on the index.
        dataset_idx = np.searchsorted(self.dataset_start_index,  np.array([idx]), side='right') - 1
        dataset_idx = dataset_idx[0]
        dataset = self._datasets[dataset_idx]
        item = dataset.getitem(idx - self.dataset_start_index[dataset_idx])

        if isinstance(item, list):
            if len(item) != 1:
                raise ValueError("Expected a single item from the dataset.")
            item[0]['rep_id'] = dataset.data_name
        else:
            item['rep_id'] = dataset.data_name
        
        return item

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:

        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds.")
        max_retries = 200
        attempts = 0
        cur = idx
        last_err = None
        #return self.getdata(cur)
        while attempts < max_retries:
            try:
                return self.getdata(cur)
            except Exception as e:
                last_err = e
                attempts += 1
                dataset_idx = np.searchsorted(self.dataset_start_index,  np.array([cur]), side='right') - 1
                dataset_idx = dataset_idx[0]
                dataset = self._datasets[dataset_idx]
                logger.info(f"Last error: {repr(last_err)},\n"
                      f"Dataset: {dataset.dataset.repo_id}")
                cur = np.random.randint(0, len(self))
                if cur >= len(self):
                    cur = 0
                continue

        raise RuntimeError(
            f"Failed to fetch a valid item starting from idx={idx} after {attempts} attempts. "
            f"Last error: {repr(last_err)}"
        )
