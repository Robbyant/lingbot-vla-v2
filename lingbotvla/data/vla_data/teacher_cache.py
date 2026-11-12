"""Offline cache of the frozen teacher targets used for depth / video alignment.

The three teachers (MoGe2 + MoRGBD for depth, DINO video) are frozen and their inputs only
depend on the sample (image augmentation off, future frame at a fixed offset), so their
targets can be computed once and read back by the data loader instead of running the teachers
every step.

Layout of a cache directory, one row per sample index of the training dataset:

    depth.npy           [N, T, D]  current-frame depth target (bf16 bits stored as int16)
    video_current.npy   [N, T, D]  DINO patch target of the current frame
    video_future.npy    [N, T, D]  DINO patch target of the future frame
    future_index.npy    [N]        sample index whose current frame is this sample's future
                                   frame; the future depth target is depth[future_index]
    done.npy            [N]        1 once a row has been written
    meta.json                      dataset / teacher fingerprint, checked when reading

The future frame is clamped to the last frame of the episode exactly as LeRobot does for
delta timestamps, so the future depth target needs no row of its own.
"""

import json
import os
import time
from typing import Callable, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from lingbotvla.utils import helper

logger = helper.create_logger(__name__)

TARGET_KEYS = ("depth", "video_current", "video_future")
# Keys added to every sample by TeacherCacheDataset; the training loop pops them.
BATCH_KEYS = {
    "depth": "teacher_depth",
    "future_depth": "teacher_future_depth",
    "video_current": "teacher_video_current",
    "video_future": "teacher_video_future",
}


def pop_batch_targets(batch: Dict) -> Dict[str, torch.Tensor]:
    """Restore the online target layouts after collation and transfer to the training device.

    MoRGBD returns BCHW features, then get_depth_target permutes/flattens them to BTD
    with channels-first storage. Cache rows and DataLoader collation are contiguous BTD.
    Values alone are not enough: a different stride can change the compiled loss/reduction
    kernels and their gradients. DINO patch targets are already contiguous online.
    """
    targets = {key: batch.pop(key) for key in BATCH_KEYS.values()}
    for key in ("teacher_depth", "teacher_future_depth"):
        value = targets[key]
        targets[key] = value.transpose(1, 2).contiguous().transpose(1, 2)
    return targets


def future_frame_offset(multi_dataset) -> int:
    offsets = {round(d.chunk_size - 1) for d in multi_dataset._datasets}
    if len(offsets) != 1:
        raise ValueError(f"datasets disagree on the future frame offset: {offsets}")
    return offsets.pop()


def compute_future_index(multi_dataset) -> np.ndarray:
    """Sample index of each sample's future frame, clamped to its episode like LeRobot does."""
    offset = future_frame_offset(multi_dataset)
    out = np.empty(len(multi_dataset), dtype=np.int64)
    for start, sub in zip(multi_dataset.dataset_start_index, multi_dataset._datasets):
        episodes = sub.dataset.meta.episodes
        for ep_from, ep_to in zip(episodes["dataset_from_index"], episodes["dataset_to_index"]):
            local = np.arange(ep_from, ep_to)
            out[start + local] = start + np.minimum(local + offset, ep_to - 1)
    return out


def fingerprint(multi_dataset, align_params, image_augment: bool, cameras) -> Dict:
    return {
        "datasets": [[sub.dataset.repo_id, int(sub.dataset.num_frames)] for sub in multi_dataset._datasets],
        "future_frame_offset": future_frame_offset(multi_dataset),
        "image_augment": bool(image_augment),
        # The teachers only see the first camera, so the order matters.
        "cameras": list(cameras or []),
        "moge_path": align_params["depth"]["moge_path"],
        "morgbd_path": align_params["depth"]["morgbd_path"],
        "video_ckpt_path": align_params["video"]["ckpt_path"],
        "video_input_size": align_params["video"].get("input_size"),
        "video_use_warmup_frame": align_params["video"].get("use_warmup_frame"),
    }


def _to_int16(t: torch.Tensor) -> np.ndarray:
    if t.dtype != torch.bfloat16:
        raise TypeError(f"teacher cache requires bfloat16 targets, got {t.dtype}")
    return t.contiguous().view(torch.int16).cpu().numpy()


class TeacherCache:
    """Read-only view of a cache directory. Memory maps are opened lazily, once per process,
    so that each data loader worker gets its own file handles after fork."""

    def __init__(self, cache_dir: str, expected_meta: Optional[Dict] = None, batch_size: Optional[int] = None):
        self.cache_dir = cache_dir
        with open(os.path.join(cache_dir, "meta.json")) as f:
            self.meta = json.load(f)
        if expected_meta is not None and self.meta["fingerprint"] != expected_meta:
            raise ValueError(
                f"teacher cache {cache_dir} was built for different data or teachers:\n"
                f"  cache:    {self.meta['fingerprint']}\n  expected: {expected_meta}"
            )
        if batch_size is not None and self.meta.get("build_batch_size", batch_size) != batch_size:
            raise ValueError("teacher cache was built with a different micro batch size; rebuild or verify it")
        n = self.meta["num_samples"]
        done = np.load(os.path.join(cache_dir, "done.npy"))
        if done.shape != (n,) or done.dtype != np.uint8:
            raise ValueError(f"teacher cache {cache_dir} has an invalid completion map")
        if not done.all():
            raise ValueError(f"teacher cache {cache_dir} is incomplete: {int((done == 0).sum())} rows missing")
        for key in TARGET_KEYS:
            array = np.load(os.path.join(cache_dir, f"{key}.npy"), mmap_mode="r")
            if array.dtype != np.int16 or array.shape != (n, *self.meta["shapes"][key]):
                raise ValueError(f"teacher cache {cache_dir} has an invalid {key} array")
        future = np.load(os.path.join(cache_dir, "future_index.npy"), mmap_mode="r")
        if future.shape != (n,) or future.dtype != np.int64 or np.any((future < 0) | (future >= n)):
            raise ValueError(f"teacher cache {cache_dir} has invalid future-frame indices")
        self._arrays = None

    def __len__(self):
        return self.meta["num_samples"]

    def _open(self):
        if self._arrays is None:
            self._arrays = {
                key: np.load(os.path.join(self.cache_dir, f"{key}.npy"), mmap_mode="r")
                for key in (*TARGET_KEYS, "future_index")
            }
        return self._arrays

    def get(self, index: int) -> Dict[str, torch.Tensor]:
        arrays = self._open()
        rows = {
            "depth": arrays["depth"][index],
            "future_depth": arrays["depth"][int(arrays["future_index"][index])],
            "video_current": arrays["video_current"][index],
            "video_future": arrays["video_future"][index],
        }
        return {
            BATCH_KEYS[key]: torch.from_numpy(np.array(row)).view(torch.bfloat16) for key, row in rows.items()
        }


class TeacherCacheDataset(Dataset):
    """Adds the cached teacher targets to every sample of a MultiVLADataset and drops the
    teacher input images, which are then no longer needed."""

    def __init__(self, dataset, cache: TeacherCache, keep_teacher_images: bool = False):
        if len(cache) != len(dataset):
            raise ValueError(f"cache has {len(cache)} rows, dataset {len(dataset)}")
        self.dataset = dataset
        self.cache = cache
        self.keep_teacher_images = keep_teacher_images
        dataset.return_sample_index = True

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # The dataset may substitute another sample when one fails to load, so use the index it
        # reports rather than idx.
        item.update(self.cache.get(item.pop("sample_index")))
        if not self.keep_teacher_images:
            item.pop("pil_images", None)
            item.pop("future_pil_images", None)
        return item


class _IndexedSubset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        index = int(self.indices[i])
        item = self.dataset[index]
        if int(item["sample_index"]) != index:
            raise ValueError(f"cannot cache sample {index}: dataset substituted sample {item['sample_index']}")
        return item


def build_teacher_cache(
    dataset,
    compute_targets: Callable[[Dict], Dict[str, torch.Tensor]],
    cache_dir: str,
    meta: Dict,
    rank: int,
    world_size: int,
    batch_size: int,
    num_workers: int,
    collate_fn,
    barrier: Callable[[], None],
):
    """Run the teachers over every sample once and write the targets to cache_dir.

    Each rank takes a contiguous slice of sample indices. compute_targets maps a collated batch
    (on GPU) to {"depth", "video_current", "video_future"} bf16 tensors of shape [B, T, D].
    Rows already marked done are skipped, so an interrupted build can be resumed.
    Progress is published every 50 batches, after flushing all target arrays. A resumed
    build can use a different world size once all previous writers have stopped.

    Every batch has exactly batch_size samples: the compiled depth teacher picks kernels by batch
    size, so a smaller last batch gives bf16 results a little off from training, which always runs
    the teachers on micro batches of batch_size. The last batch is padded by repeating its final
    sample; which samples share a batch does not change the result.
    """
    n = len(dataset)
    if n == 0 or batch_size < 1:
        raise ValueError("teacher cache needs a nonempty dataset and a positive batch size")
    dataset.return_sample_index = True
    os.makedirs(cache_dir, exist_ok=True)
    done_path = os.path.join(cache_dir, "done.npy")

    # Rank 0 sizes the files from one batch, then every rank opens them for writing.
    if rank == 0 and not os.path.exists(done_path):
        probe = collate_fn([dataset[min(i, n - 1)] for i in range(batch_size)])
        probe = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in probe.items()}
        shapes = {k: tuple(v.shape[1:]) for k, v in compute_targets(probe).items()}
        for key in TARGET_KEYS:
            np.lib.format.open_memmap(
                os.path.join(cache_dir, f"{key}.npy"), mode="w+", dtype=np.int16, shape=(n, *shapes[key])
            )
        np.save(os.path.join(cache_dir, "future_index.npy"), compute_future_index(dataset))
        with open(os.path.join(cache_dir, "meta.json"), "w") as f:
            json.dump({"num_samples": n, "shapes": shapes, "fingerprint": meta,
                       "build_batch_size": batch_size}, f, indent=2)
        # Publish the completion map last: its presence means initialization finished.
        with open(done_path + ".tmp", "wb") as f:
            np.save(f, np.zeros(n, dtype=np.uint8))
        os.replace(done_path + ".tmp", done_path)
    barrier()
    with open(os.path.join(cache_dir, "meta.json")) as f:
        stored = json.load(f)
        if stored["fingerprint"] != meta:
            raise ValueError(f"{cache_dir} holds a cache of different data or teachers")
        if stored["num_samples"] != n or stored.get("build_batch_size", batch_size) != batch_size:
            raise ValueError(f"{cache_dir} was built with a different dataset size or micro batch size")

    arrays = {key: np.load(os.path.join(cache_dir, f"{key}.npy"), mmap_mode="r+") for key in TARGET_KEYS}
    for key, array in arrays.items():
        if array.dtype != np.int16 or array.shape != (n, *stored["shapes"][key]):
            raise ValueError(f"{cache_dir} has an invalid {key} array")
    done = np.load(done_path, mmap_mode="r+")
    if done.shape != (n,):
        raise ValueError(f"{cache_dir} has a completion map for a different dataset size")
    lo, hi = n * rank // world_size, n * (rank + 1) // world_size
    todo = np.arange(lo, hi)[done[lo:hi] == 0]
    if len(todo) % batch_size:
        todo = np.concatenate([todo, np.full(batch_size - len(todo) % batch_size, todo[-1])])
    logger.info(f"rank {rank}: samples [{lo}, {hi}), {len(todo)} to compute")

    loader = DataLoader(
        _IndexedSubset(dataset, todo),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    pending = []

    def checkpoint_progress():
        # Ranks write disjoint rows. Never mark a row complete before its targets are durable.
        for array in arrays.values():
            array.flush()
        if pending:
            done[np.concatenate(pending)] = 1
            done.flush()
            pending.clear()

    start, written = time.time(), 0
    for batch_number, batch in enumerate(loader, 1):
        index = batch.pop("sample_index").numpy()
        batch = {k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        targets = compute_targets(batch)
        for key in TARGET_KEYS:
            if tuple(targets[key].shape) != (len(index), *stored["shapes"][key]):
                raise ValueError(f"teacher returned an unexpected {key} shape: {tuple(targets[key].shape)}")
            arrays[key][index] = _to_int16(targets[key])
        pending.append(index)
        if batch_number % 50 == 0:
            checkpoint_progress()
        written += len(np.unique(index))
        if rank == 0 and written % (50 * batch_size) < len(index):
            rate = written / (time.time() - start)
            logger.info(f"rank 0: {written}/{len(todo)} samples, {rate:.1f} samples/s per rank")
    checkpoint_progress()
    barrier()
    if rank == 0:
        if not done.all():
            raise ValueError(f"teacher cache {cache_dir}: {int((done == 0).sum())} rows remain incomplete")
        logger.info(f"teacher cache {cache_dir}: {int(done.sum())}/{n} rows written in {time.time() - start:.0f}s")
    barrier()
