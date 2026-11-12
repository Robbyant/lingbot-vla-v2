# Offline teacher targets for SFT

The depth teachers (MoGe2 → MoRGBD) and the DINO video teacher are frozen. With image augmentation disabled and fixed future-frame selection, their alignment targets depend on the sample, not on the student parameters. This feature precomputes those targets once and reads them during training. The student architecture, alignment losses, and online teacher implementations are unchanged.

The feature is **off by default**: leave `train.teacher_cache_dir` unset to run the online teachers as before. It is independent of the ViT, kernel-fusion, FSDP-tuning, and attention optimizations.

## Configuration and commands

Use the usual model/data/teacher paths in `configs/vla/robotwin/robotwin_dist_muon.yaml`. All ranks must see the same writable cache directory during construction, and the same read-only directory during training. The commands below use the tested single-node configuration: 8 GPUs, micro batch 32, gradient accumulation 1, global batch 256, and activation checkpointing.

```bash
# Run the teachers over the training dataset, write the cache, then exit.
torchrun --standalone --nproc-per-node 8 tasks/vla/train_lingbotvla.py \
  configs/vla/robotwin/robotwin_dist_muon.yaml \
  --train.enable_gradient_checkpointing true \
  --train.micro_batch_size 32 --train.global_batch_size 256 \
  --train.gradient_accumulation_steps 1 \
  --train.teacher_cache_dir /path/to/teacher-cache \
  --train.teacher_cache_mode build

# Train without loading or running the teacher models.
torchrun --standalone --nproc-per-node 8 tasks/vla/train_lingbotvla.py \
  configs/vla/robotwin/robotwin_dist_muon.yaml \
  --train.enable_gradient_checkpointing true \
  --train.micro_batch_size 32 --train.global_batch_size 256 \
  --train.gradient_accumulation_steps 1 \
  --train.teacher_cache_dir /path/to/teacher-cache

# Run the online teachers as well, report target differences across all ranks,
# and use the cached targets for training. This mode does not measure speedup.
torchrun --standalone --nproc-per-node 8 tasks/vla/train_lingbotvla.py \
  configs/vla/robotwin/robotwin_dist_muon.yaml \
  --train.enable_gradient_checkpointing true \
  --train.micro_batch_size 32 --train.global_batch_size 256 \
  --train.gradient_accumulation_steps 1 \
  --train.teacher_cache_dir /path/to/teacher-cache \
  --train.teacher_cache_mode verify
```

The corresponding YAML fields are:

```yaml
train:
  teacher_cache_dir: /path/to/teacher-cache  # null / omitted disables caching
  teacher_cache_mode: read                 # build | read | verify; default: read
```

Use an existing configuration with the desired micro/global batch settings; cache flags do not change those settings. `build` and `verify` require a directory. Supported configurations use `data_name: multi`, depth + future-depth + future-video alignment, DINO current/future **patch** targets, and `image_augment: false`. CLS-target alignment is not cached.

## Format, indexing, and resumption

The directory contains:

| File | Content |
|---|---|
| `depth.npy` | Current-frame depth targets |
| `video_current.npy`, `video_future.npy` | DINO current/future patch targets |
| `future_index.npy` | Global sample index of the future frame, clamped at the episode end |
| `done.npy` | One completion byte per sample |
| `meta.json` | Dataset/configuration identity, shapes, sample count, build micro batch size |

The three target arrays store the original **bf16 bit patterns as int16**, with no further quantization. Future-depth targets reuse the current-depth row at `future_index`, rather than storing another copy. Targets are attached to the actual sample index returned by the dataset, including when a training read substitutes another sample after a load failure. During construction, a substituted sample is rejected: every cache row must correspond to its requested frame.

Before computing the losses, cached depth targets are restored to the non-contiguous BTD layout returned by `get_depth_target` (BCHW features permuted and flattened). Preserving values alone is insufficient: changing target strides can select different compiled loss/reduction kernels. `verify` checks the strides as well as the values. Read mode also skips teacher-dependent image visualizations; use `verify` to retain the teacher inputs and visualizations.

Targets are memory-mapped and copied row-wise by DataLoader workers. The reader checks completion, target dtype/shape, future-index range, and configuration identity before training. In read mode the teacher-only images are removed from samples; the student's own image inputs remain present.

Re-run the same `build` command to resume. Each rank flushes targets before marking rows complete, every 50 batches and at completion. Interrupted work since the last progress flush may be recomputed. Ranks own disjoint index ranges; the world size may change on restart **after all old writers have stopped**. Do not run independent builder jobs against the same directory concurrently. A final partial micro batch is padded by repeating its last sample to preserve the teacher kernel shapes used in training.

Build and read with the same micro batch size: the compiled teacher can select different kernels for different batch shapes. New caches record this size and reject a mismatch. Older prototype caches without this metadata are accepted; verify them before reuse.

## Cache identity and limitations

Metadata compares the dataset identifiers/order/frame counts, camera order, future offset, teacher checkpoint **paths**, and recorded video-teacher settings. This is **not** a content hash of all data or teacher weights. Rebuild after changing data contents, preprocessing, teacher weights, or teacher configuration, even if paths and frame counts remain the same. Enabling random image augmentation is rejected.

A cache trades repeated GPU teacher work for preprocessing time and storage bandwidth. Our full RoboTwin clean50 cache has 549,787 frames and occupies approximately 806 GiB; cache size scales with the number and shapes of targets. Use local fast storage or validate your storage throughput with the end-to-end benchmark. Construction of a small complete-episode subset is a functional check, not a measurement of full-dataset build time.

## Compatibility with FSDP communication scheduling

When combined with [PR #67](https://github.com/Robbyant/lingbot-vla-v2/pull/67), use `train.fsdp_prologue_under=vit` (the default) in teacher-cache **read** mode. No online teacher computation remains to overlap the prefetch communication. Cache `build`/`read`/`verify` modes do not require that PR; the option is supplied by its FSDP communication scheduler. Combined performance is separate from this feature's standalone benchmark.

## Numerical validation

`verify` logs the maximum absolute difference and minimum exact-match fraction across all ranks for all four target types. A successful exact check reports `max|diff| 0` and `exact 1` for each target. It trains on the cached values while still executing the online teachers.

Skipping teacher initialization in `read` mode changes RNG consumption. The same training seed alone therefore does not ensure identical flow-matching noise/time or step-by-step VLA losses. For an isolated comparison, use identical student weights, data, explicit flow noise/time, and compare gradients and updated parameters against repeat-baseline variation. Target equality, short training-loss agreement, and downstream task accuracy are separate claims; this feature has not been validated for full convergence or downstream success rate.

CPU regression tests (no teacher checkpoints or GPU needed):

```bash
python -m unittest discover -s tests -p test_teacher_cache.py -v
```

The fresh PR-head validation and benchmark results are recorded in the PR description. Performance uses the full slowest-rank update, including data wait and logging; the trainer's `StepTime` alone excludes data loading. The added `Data_Wait` field makes cache IO visible.
