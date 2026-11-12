"""CPU coverage of cache indexing, bit-preserving IO, validation, and interrupted builds.

Run: python -m unittest discover -s tests -p test_teacher_cache.py -v
No models, training data, or GPU are required.
"""
import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, default_collate

from lingbotvla.data.vla_data.teacher_cache import (
    BATCH_KEYS, TARGET_KEYS, TeacherCache, TeacherCacheDataset, _to_int16,
    build_teacher_cache, compute_future_index, pop_batch_targets,
)


class SmallDataset:
    def __init__(self, n, substitute=False):
        self.n, self.substitute = n, substitute
        self.return_sample_index = False
        self._datasets = [SimpleNamespace(chunk_size=3, dataset=SimpleNamespace(
            meta=SimpleNamespace(episodes={'dataset_from_index': [0], 'dataset_to_index': [n]})))]
        self.dataset_start_index = [0]

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        if not 0 <= index < self.n:
            raise IndexError(index)
        actual = 1 if self.substitute and index == 0 else index
        item = dict(pil_images=torch.tensor([actual]), future_pil_images=torch.tensor([actual]))
        if self.return_sample_index:
            item['sample_index'] = actual
        return item


def targets(batch):
    values = batch['pil_images'].reshape(-1, 1, 1).expand(-1, 2, 3).to(torch.bfloat16)
    return {key: values + 10 * i for i, key in enumerate(TARGET_KEYS)}


class TeacherCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.meta = {'datasets': [['example', 6]], 'future_frame_offset': 2}
        self.cuda = patch.object(torch.Tensor, 'cuda', lambda tensor, *args, **kwargs: tensor)
        self.cuda.start()
        self.addCleanup(self.cuda.stop)

    def build(self, dataset, compute=targets, batch_size=2, rank=0, world_size=1, barrier=lambda: None):
        build_teacher_cache(dataset, compute, str(self.root), self.meta, rank, world_size,
                            batch_size, 0, default_collate, barrier)

    def test_bit_preserving_io_and_future_depth(self):
        self.build(SmallDataset(6))
        cache = TeacherCache(str(self.root), self.meta, batch_size=2)
        for i in range(6):
            row = cache.get(i)
            self.assertTrue(torch.equal(row['teacher_depth'], torch.full((2, 3), i, dtype=torch.bfloat16)))
            self.assertTrue(torch.equal(row['teacher_future_depth'], torch.full((2, 3), min(i+2, 5), dtype=torch.bfloat16)))
        bits = torch.tensor([0, 1, -1, 32767, -32768, 32704], dtype=torch.int16)
        self.assertTrue(np.array_equal(_to_int16(bits.view(torch.bfloat16)), bits.numpy()))

    def test_episode_boundaries_across_datasets(self):
        sub = lambda starts, ends: SimpleNamespace(chunk_size=3, dataset=SimpleNamespace(
            meta=SimpleNamespace(episodes={'dataset_from_index': starts, 'dataset_to_index': ends})))
        class LengthData(SimpleNamespace):
            def __len__(self):
                return 9
        data = LengthData(_datasets=[sub([0, 3], [3, 6]), sub([0], [3])], dataset_start_index=[0, 6])
        self.assertEqual(compute_future_index(data).tolist(), [2, 2, 2, 5, 5, 5, 8, 8, 8])

    def test_smaller_than_microbatch(self):
        batches = []
        def compute(batch):
            batches.append(len(batch['pil_images']))
            return targets(batch)
        self.build(SmallDataset(2), compute, batch_size=4)
        self.assertEqual(batches, [4, 4])
        self.assertEqual(len(TeacherCache(str(self.root))), 2)

    def test_empty_dataset_rejected(self):
        with self.assertRaisesRegex(ValueError, 'nonempty'):
            self.build(SmallDataset(0))

    def test_batched_targets_restore_online_depth_layout(self):
        online = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3).to(torch.bfloat16).transpose(1, 2)
        batch = {key: online.contiguous() for key in BATCH_KEYS.values()}
        video = batch['teacher_video_current']
        batch['state'] = torch.zeros(2, 1)
        restored = pop_batch_targets(batch)
        for key in ['teacher_depth', 'teacher_future_depth']:
            self.assertEqual(restored[key].stride(), online.stride())
            self.assertTrue(torch.equal(restored[key].view(torch.int16), online.view(torch.int16)))
        self.assertIs(restored['teacher_video_current'], video)
        self.assertEqual(set(batch), {'state'})

    def test_target_dtype_and_dataset_length_rejected(self):
        with self.assertRaisesRegex(TypeError, 'bfloat16'):
            _to_int16(torch.zeros(2, 3, dtype=torch.float16))
        self.build(SmallDataset(6))
        with self.assertRaisesRegex(ValueError, 'rows'):
            TeacherCacheDataset(SmallDataset(5), TeacherCache(str(self.root)))

    def test_incomplete_and_mismatched_cache_rejected(self):
        self.build(SmallDataset(6))
        with self.assertRaisesRegex(ValueError, 'different data'):
            TeacherCache(str(self.root), {'different': True})
        with self.assertRaisesRegex(ValueError, 'micro batch'):
            TeacherCache(str(self.root), batch_size=4)
        done = np.load(self.root/'done.npy');done[2] = 0;np.save(self.root/'done.npy', done)
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            TeacherCache(str(self.root))

    def test_invalid_array_and_future_index_rejected(self):
        self.build(SmallDataset(6))
        original = np.load(self.root/'depth.npy').copy()
        np.save(self.root/'depth.npy', original.astype(np.float32))
        with self.assertRaisesRegex(ValueError, 'invalid depth'):
            TeacherCache(str(self.root))
        np.save(self.root/'depth.npy', original)
        np.save(self.root/'future_index.npy', np.full(6, 6, dtype=np.int64))
        with self.assertRaisesRegex(ValueError, 'future-frame'):
            TeacherCache(str(self.root))

    def test_interruption_preserves_completed_batches(self):
        calls = 0
        def interrupted(batch):
            nonlocal calls
            calls += 1
            if calls == 52:  # probe + 50 durable batches, then interruption
                raise RuntimeError('simulated interruption')
            return targets(batch)
        with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
            self.build(SmallDataset(55), interrupted, batch_size=1)
        done = np.load(self.root/'done.npy')
        self.assertEqual(int(done.sum()), 50)
        batches = []
        def resumed(batch):
            batches.extend(batch['pil_images'].reshape(-1).tolist())
            return targets(batch)
        self.build(SmallDataset(55), resumed, batch_size=1)
        self.assertEqual(batches, list(range(50, 55)))
        self.assertTrue(np.load(self.root/'done.npy').all())
        self.assertEqual(len(TeacherCache(str(self.root))), 55)

    def test_disjoint_rank_writers(self):
        barrier = threading.Barrier(2)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(self.build, SmallDataset(7), targets, 2, rank, 2, barrier.wait)
                       for rank in range(2)]
            for future in futures:
                future.result(timeout=30)
        cache = TeacherCache(str(self.root))
        self.assertEqual(len(cache), 7)
        self.assertTrue(torch.equal(cache.get(6)['teacher_depth'], torch.full((2, 3), 6, dtype=torch.bfloat16)))

    def test_substitution_rejected_during_build(self):
        with self.assertRaisesRegex(ValueError, 'substituted sample'):
            self.build(SmallDataset(6, substitute=True))

    def test_reader_follows_actual_sample_and_workers(self):
        self.build(SmallDataset(6))
        wrapped = TeacherCacheDataset(SmallDataset(6, substitute=True), TeacherCache(str(self.root)))
        row = wrapped[0]
        self.assertNotIn('pil_images', row)
        self.assertNotIn('future_pil_images', row)
        self.assertTrue(torch.equal(row['teacher_depth'], torch.ones((2, 3), dtype=torch.bfloat16)))
        for workers in [0, 2]:
            batches = list(DataLoader(wrapped, batch_size=2, num_workers=workers))
            self.assertEqual(sum(len(b['teacher_depth']) for b in batches), 6)
        verify = TeacherCacheDataset(SmallDataset(6), TeacherCache(str(self.root)), keep_teacher_images=True)
        self.assertIn('pil_images', verify[0])


if __name__ == '__main__':
    unittest.main()
