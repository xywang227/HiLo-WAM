# HiLo-WAM

**HiLo-WAM** is a hierarchical world action model built on Wan2.2-TI2V-5B. A high-level video branch models sparse keyframes, while a low-level video branch and an action expert predict local motion and robot actions through mixed attention.

[![English](https://img.shields.io/badge/README-English-111111.svg)](./README.md)
[![中文](https://img.shields.io/badge/README-%E4%B8%AD%E6%96%87-d14836.svg)](./README_zh.md)

[![Project Page](https://img.shields.io/badge/Project_Page-HiLo--WAM-2ea44f.svg)](https://xywang227.github.io/HiLo-WAM/)

This release includes the default configuration, training, and policy evaluation for **RoboTwin**, and the default configuration and training for **RoboDojo**. RoboDojo policy integration and evaluation belong in the official RoboDojo repository. Training uses **LeRobot v3.0** datasets.

## Repository layout

```text
configs/
  model/hilo_wam.yaml       # Shared HiLo-WAM architecture
  data/{robotwin,robodojo}.yaml
  task/{robotwin,robodojo}.yaml
  train.yaml
  sim_robotwin.yaml
src/hilo_wam/               # Model, trainer, LeRobot reader and processors
scripts/
  train.py
  train_zero1.sh            # Single node, 8 GPUs, DeepSpeed ZeRO-1
  preprocess_action_dit_backbone.py
  precompute_text_embeds.py
  convert_lerobot_v21_to_v30.py
experiments/robotwin/       # Evaluation manager and hilo_wam_policy
third_party/RoboTwin/      # Vendored simulator integration; assets installed separately
```

## Installation

Use Linux, Python 3.10, and an NVIDIA GPU with a driver supporting CUDA 12.8. Run commands from this repository's root. The training launcher targets one node with eight GPUs; select GPU IDs with `CUDA_VISIBLE_DEVICES`.

```bash
conda create -n hilo-wam python=3.10 -y
conda activate hilo-wam
conda install -c conda-forge 'ffmpeg>=4,<8' -y
bash install.sh
```

`install.sh` installs PyTorch 2.7.1 / torchvision 0.22.1 from the CUDA 12.8 wheel index, installs DeepSpeed after PyTorch without precompiling optional CUDA extensions, then installs this package as `hilo-wam` with the Python import name `hilo_wam`. It does not install either benchmark's simulator. TorchCodec 0.5 requires a compatible FFmpeg shared library; see the [TorchCodec installation guide](https://github.com/meta-pytorch/torchcodec#installing-torchcodec).

For RoboTwin evaluation, also follow the environment and asset instructions in [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin), using the vendored tree at `third_party/RoboTwin`. Its source revision is recorded in [README.vendor.md](third_party/RoboTwin/README.vendor.md).

The upstream simulator dependency list pins older PyTorch and Hugging Face versions. If its installer changes those packages in the model environment, rerun `bash install.sh` afterward to restore HiLo-WAM's dependencies. Asset downloading is available through `bash script/_download_assets.sh` from the RoboTwin directory.

## Pretrained backbone

Both training and evaluation require the Wan backbone files. The loader resolves the upstream model directories below `DIFFSYNTH_MODEL_BASE_PATH` and downloads missing components. Keep that directory consistent across preparation, training, and evaluation.

```bash
mkdir -p checkpoints
export DIFFSYNTH_MODEL_BASE_PATH="$PWD/checkpoints"
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface

python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/hilo_wam.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda --dtype bfloat16
```

This initializes the 1024-dimensional ActionDiT backbone by interpolation from the Wan video transformer. The default action and proprioception dimensions are both 14 for the two benchmarks. Training reads cached T5 embeddings, so it does not keep the text encoder in GPU memory. Evaluation loads the text encoder for online instructions.

## Data preparation

### LeRobot v2.1 to v3.0

The conversion entry point wraps the [LeRobot v0.4.0 converter](https://github.com/huggingface/lerobot/blob/v0.4.0/src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py), fixes parquet rollover indexing, and validates episode mappings, video paths, and timestamp bounds. Install its dependencies in a separate environment because upstream LeRobot and this training package have different dependency pins:

```bash
conda create -n hilo-wam-data python=3.10 -y
conda activate hilo-wam-data
conda install -c conda-forge 'ffmpeg>=4,<8' -y
python -m pip install 'lerobot==0.4.0'

# Input: ./data/robotwin2.0/robotwin2.0/{data,meta,videos}
python scripts/convert_lerobot_v21_to_v30.py \
  --repo-id robotwin2.0 --root ./data/robotwin2.0

# Input: ./data/robodojo/{data,meta,videos}
python scripts/convert_lerobot_v21_to_v30.py \
  --repo-id robodojo --root ./data

conda activate hilo-wam
```

`--root` is the **parent** of `--repo-id`. The converted dataset replaces the input directory; the original v2.1 tree is retained as its sibling `<dataset>_old`. Existing backup or staging directories are never overwritten. If conversion was interrupted, inspect the directories and use `--resume-staging` to finish an existing `<dataset>_v30` tree. To validate an already converted dataset, use the same arguments with `--validate-only`. A matching source checkout can alternatively be supplied with `--lerobot-src ./third_party/lerobot/src`.

The expected v3 layout is:

```text
<dataset>/
  meta/info.json                      # codebase_version: v3.0
  meta/tasks.parquet
  meta/episodes/chunk-*/file-*.parquet
  data/chunk-*/file-*.parquet
  videos/<camera-key>/chunk-*/file-*.mp4
```

### RoboTwin and RoboDojo inputs

The default dataset roots are `./data/robotwin2.0/robotwin2.0` and `./data/robodojo`. You can set `ROBOTWIN_DATASET_DIR` or `ROBODOJO_DATASET_DIR` to use another location. The [RoboTwin dataset prepared by FastWAM](https://huggingface.co/datasets/yuanty/robotwin2.0-fastwam) is one possible v2.1 input; convert it before training. Prepare RoboDojo data using the official benchmark's data pipeline, then convert it if necessary.

Both configurations expect these LeRobot feature keys:

| Feature | Shape / meaning |
| --- | --- |
| `observation.images.cam_high` | Head camera, RGB |
| `observation.images.cam_left_wrist` | Left wrist camera, RGB |
| `observation.images.cam_right_wrist` | Right wrist camera, RGB |
| `observation.state` | 14-dimensional proprioception |
| `action` | 14-dimensional actions |

The three views are packed into a 384 × 320 image. Each sample contains a 32-step action horizon and nine low-level video frames, together with eight past and eight future sparse keyframes at stride 16. The training split supplies normalization statistics, saved as `dataset_stats.json` in the run directory; validation uses those same statistics. To reuse existing statistics, set `data.train.pretrained_norm_stats=./data/dataset_stats.json`.

### Cache text embeddings

Generate the cache before training, using the same task and dataset overrides as the training command:

```bash
python scripts/precompute_text_embeds.py task=robotwin
python scripts/precompute_text_embeds.py task=robodojo
```

The defaults are `./data/text_embeds_cache/robotwin` and `./data/text_embeds_cache/robodojo`. Override them with `ROBOTWIN_TEXT_EMBED_CACHE_DIR` and `ROBODOJO_TEXT_EMBED_CACHE_DIR`. Caches are keyed by the formatted instruction text and use the same 128-token context as the model.

## Training

Run the retained default configurations on one node with eight GPUs:

```bash
bash scripts/train_zero1.sh 8 task=robotwin
bash scripts/train_zero1.sh 8 task=robodojo
```

Both use BF16, AdamW, cosine decay, learning rate `1e-4`, batch size 4 per GPU, and three epochs. Weights & Biases is disabled by default; enable it with `wandb.enabled=true`. Paths and training settings can be overridden with Hydra arguments:

```bash
bash scripts/train_zero1.sh 8 task=robotwin \
  data.train.dataset_dirs='[./data/my_robotwin_v3]' \
  data.train.text_embedding_cache_dir=./data/my_text_cache \
  output_dir=./runs/robotwin/my_run
```

A run contains:

```text
runs/<task>/<timestamp>/
  config.yaml
  dataset_stats.json
  checkpoints/weights/step_XXXXXX.pt   # Policy/model weights
  checkpoints/state/step_XXXXXX/      # Optimizer, scheduler, RNG and sampler progress
  eval/                              # Validation videos and metrics
```

Resume training from the state directory, keeping the original task and model settings:

```bash
bash scripts/train_zero1.sh 8 task=robotwin \
  resume=./runs/robotwin/my_run/checkpoints/state/step_010000
```

For a short launch check, set `max_steps=1 log_every=1 eval_every=0`. A successful check must reach the optimizer step and checkpoint save. Parsing configuration alone is not a training check. GPU memory requirements depend on batch size; `batch_size=1 model.mot_checkpoint_mixed_attn=true` reduces activation memory. The launcher also accepts a smaller local GPU count for debugging, while the released training recipe uses eight.

## RoboTwin inference and evaluation

After installing RoboTwin and its assets, evaluate a trained **weights** checkpoint together with its run's normalization statistics:

```bash
python experiments/robotwin/eval_robotwin_single.py \
  ckpt=./runs/robotwin/my_run/checkpoints/weights/step_010000.pt \
  EVALUATION.dataset_stats_path=./runs/robotwin/my_run/dataset_stats.json \
  EVALUATION.task_name=adjust_bottle
```

The adapter automatically links `experiments/robotwin/hilo_wam_policy` into `third_party/RoboTwin/policy/hilo_wam_policy`. It maintains sparse observation history, predicts 32-action chunks, executes each chunk, and resets history between episodes. Default policy inference uses synchronized high/low/action denoising with dynamic skipping. Results are written under `./evaluate_results/robotwin`.

To distribute benchmark tasks across eight GPUs on the same machine:

```bash
python experiments/robotwin/run_robotwin_manager.py \
  ckpt=./runs/robotwin/my_run/checkpoints/weights/step_010000.pt \
  EVALUATION.dataset_stats_path=./runs/robotwin/my_run/dataset_stats.json \
  MULTIRUN.num_gpus=8
```

Adjust `MULTIRUN.num_gpus` for the evaluation machine. `EVALUATION.skip_get_obs_within_replan=true` avoids rendering intermediate RGB observations while executing an action chunk. Set it to `false` for denser recorded videos. To use a separately installed RoboTwin tree, set `EVALUATION.robotwin_root=./external/RoboTwin`.

For RoboDojo, use the weights checkpoint, its `config.yaml`, and `dataset_stats.json` when adapting the policy in the official RoboDojo repository. This release contains no RoboDojo inference launcher.

## Acknowledgements and license

This repository builds on FastWAM, [Wan2.2](https://github.com/Wan-Video/Wan2.2), [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio), [LeRobot](https://github.com/huggingface/lerobot), and [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin). Original copyright and third-party license notices are retained; see [LICENSE](LICENSE), [NOTICE](NOTICE), and [RoboTwin's license](third_party/RoboTwin/LICENSE). Pretrained weights and datasets retain their upstream terms.
