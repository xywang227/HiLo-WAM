# HiLo-WAM

**HiLo-WAM** 是基于 Wan2.2-TI2V-5B 的分层世界动作模型。高层视频分支预测稀疏关键帧，低层视频分支和动作专家通过混合注意力预测局部运动及机器人动作。

[English](README.md)

本仓库提供 **RoboTwin 默认配置、训练和策略评测**，以及 **RoboDojo 默认配置和训练**。RoboDojo 推理与评测适配放在官方 RoboDojo 仓库中。本仓库训练使用 **LeRobot v3.0** 格式数据。

## 目录结构

```text
configs/
  model/hilo_wam.yaml       # 共用模型结构
  data/{robotwin,robodojo}.yaml
  task/{robotwin,robodojo}.yaml
  train.yaml
  sim_robotwin.yaml
src/hilo_wam/               # 模型、训练器、LeRobot 数据读取及处理
scripts/
  train.py
  train_zero1.sh            # 单节点 8 卡 DeepSpeed ZeRO-1
  preprocess_action_dit_backbone.py
  precompute_text_embeds.py
  convert_lerobot_v21_to_v30.py
experiments/robotwin/       # RoboTwin 评测入口及 hilo_wam_policy
third_party/RoboTwin/      # 仿真集成源码，需另外安装环境及下载资源
```

## 环境安装

使用 Linux、Python 3.10 和支持 CUDA 12.8 的 NVIDIA 驱动。从仓库根目录执行以下命令。正式训练配置面向单节点 8 卡，可通过 `CUDA_VISIBLE_DEVICES` 选择 GPU。

```bash
conda create -n hilo-wam python=3.10 -y
conda activate hilo-wam
conda install -c conda-forge 'ffmpeg>=4,<8' -y
bash install.sh
```

`install.sh` 先从 CUDA 12.8 wheel 源安装 PyTorch 2.7.1 和 torchvision 0.22.1，接着安装 DeepSpeed（不预编译可选 CUDA 扩展），最后安装当前仓库。安装包名为 `hilo-wam`，Python 导入名为 `hilo_wam`。TorchCodec 0.5 需要兼容的 FFmpeg 动态库，详见 [TorchCodec 安装说明](https://github.com/meta-pytorch/torchcodec#installing-torchcodec)。

模型环境安装不包含仿真器。运行 RoboTwin 评测前，请参照 [RoboTwin 官方仓库](https://github.com/RoboTwin-Platform/RoboTwin) 完成 `third_party/RoboTwin` 的环境安装和资源下载；所使用的源码版本记录在 [README.vendor.md](third_party/RoboTwin/README.vendor.md)。

上游仿真环境的依赖列表固定了较旧的 PyTorch 和 Hugging Face 版本。如果在模型环境中运行其安装器后改变了这些包，请再次执行 `bash install.sh` 恢复本项目依赖。资源下载入口为 RoboTwin 目录下的 `bash script/_download_assets.sh`。

## 预训练模型准备

训练和评测都需要 Wan 模型组件。加载器从 `DIFFSYNTH_MODEL_BASE_PATH` 下的上游模型目录解析权重，并下载缺失组件。预处理、训练及推理应使用同一个目录。

```bash
mkdir -p checkpoints
export DIFFSYNTH_MODEL_BASE_PATH="$PWD/checkpoints"
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface

python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/hilo_wam.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda --dtype bfloat16
```

该步骤将 Wan 视频 Transformer 的权重插值为隐藏维度 1024 的 ActionDiT 初始化权重。两个 benchmark 的默认动作和本体状态维度均为 14。训练使用预先缓存的 T5 文本嵌入，不在 GPU 中保留文本编码器；推理时加载文本编码器处理在线指令。

## 数据准备

### LeRobot v2.1 转换为 v3.0

转换脚本封装了 [LeRobot v0.4.0 的官方转换器](https://github.com/huggingface/lerobot/blob/v0.4.0/src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py)，修正 parquet 文件切分时的索引记录，并检查 episode 映射、视频路径及时间戳范围。官方 LeRobot 与本项目的依赖版本不同，因此使用独立的数据转换环境：

```bash
conda create -n hilo-wam-data python=3.10 -y
conda activate hilo-wam-data
conda install -c conda-forge 'ffmpeg>=4,<8' -y
python -m pip install 'lerobot==0.4.0'

# 输入：./data/robotwin2.0/robotwin2.0/{data,meta,videos}
python scripts/convert_lerobot_v21_to_v30.py \
  --repo-id robotwin2.0 --root ./data/robotwin2.0

# 输入：./data/robodojo/{data,meta,videos}
python scripts/convert_lerobot_v21_to_v30.py \
  --repo-id robodojo --root ./data

conda activate hilo-wam
```

`--root` 是 `--repo-id` 对应目录的**父目录**。转换成功后，原路径保存 v3 数据，原始 v2.1 数据保留在同级的 `<数据集名>_old` 目录。已有的备份或临时转换目录不会被覆盖。转换中断时，检查目录后可使用 `--resume-staging` 完成已有的 `<数据集名>_v30` 转换目录。对已有 v3 数据运行同样的命令并加上 `--validate-only` 即可检查数据；也可以使用 `--lerobot-src ./third_party/lerobot/src` 指定匹配版本的源码。

v3 数据布局如下：

```text
<dataset>/
  meta/info.json                      # codebase_version 为 v3.0
  meta/tasks.parquet
  meta/episodes/chunk-*/file-*.parquet
  data/chunk-*/file-*.parquet
  videos/<camera-key>/chunk-*/file-*.mp4
```

### RoboTwin 与 RoboDojo 数据配置

默认数据目录分别为 `./data/robotwin2.0/robotwin2.0` 和 `./data/robodojo`。可以通过环境变量 `ROBOTWIN_DATASET_DIR`、`ROBODOJO_DATASET_DIR` 指定其他位置。[FastWAM 预处理的 RoboTwin 数据](https://huggingface.co/datasets/yuanty/robotwin2.0-fastwam) 可作为 v2.1 输入，训练前需完成转换。RoboDojo 数据通过官方 benchmark 数据处理流程准备，必要时再转换为 v3。

两个配置均读取以下 LeRobot 特征：

| 特征 | 内容 |
| --- | --- |
| `observation.images.cam_high` | 头部 RGB 相机 |
| `observation.images.cam_left_wrist` | 左腕 RGB 相机 |
| `observation.images.cam_right_wrist` | 右腕 RGB 相机 |
| `observation.state` | 14 维本体状态 |
| `action` | 14 维动作 |

三路相机图像拼接为 384 × 320。默认样本包含 32 步动作、9 帧低层视频，以及间隔 16 的 8 帧历史和 8 帧未来稀疏关键帧。归一化统计由训练集计算并保存为运行目录中的 `dataset_stats.json`，验证集共用该统计。复用已有统计时设置 `data.train.pretrained_norm_stats=./data/dataset_stats.json`。

### 文本嵌入缓存

训练前使用相同的任务和数据路径覆盖参数生成缓存：

```bash
python scripts/precompute_text_embeds.py task=robotwin
python scripts/precompute_text_embeds.py task=robodojo
```

默认缓存目录为 `./data/text_embeds_cache/robotwin` 和 `./data/text_embeds_cache/robodojo`，可通过 `ROBOTWIN_TEXT_EMBED_CACHE_DIR`、`ROBODOJO_TEXT_EMBED_CACHE_DIR` 修改。缓存根据格式化后的指令文本生成文件名，与模型统一使用 128 个 token 的上下文长度。

## 单节点 8 卡训练

```bash
bash scripts/train_zero1.sh 8 task=robotwin
bash scripts/train_zero1.sh 8 task=robodojo
```

默认使用 BF16、AdamW、余弦学习率、`1e-4` 学习率、每卡 batch size 4 和 3 个 epoch。Weights & Biases 默认关闭，使用 `wandb.enabled=true` 开启。通过 Hydra 参数修改路径及训练设置：

```bash
bash scripts/train_zero1.sh 8 task=robotwin \
  data.train.dataset_dirs='[./data/my_robotwin_v3]' \
  data.train.text_embedding_cache_dir=./data/my_text_cache \
  output_dir=./runs/robotwin/my_run
```

运行结果目录：

```text
runs/<task>/<timestamp>/
  config.yaml
  dataset_stats.json
  checkpoints/weights/step_XXXXXX.pt   # 模型权重，用于策略推理
  checkpoints/state/step_XXXXXX/      # 优化器、调度器、随机数及采样进度
  eval/                              # 验证视频和指标
```

断点续训需传入训练状态目录，并保持原任务及模型设置：

```bash
bash scripts/train_zero1.sh 8 task=robotwin \
  resume=./runs/robotwin/my_run/checkpoints/state/step_010000
```

短程启动检查可加 `max_steps=1 log_every=1 eval_every=0`，应执行到参数更新和 checkpoint 保存，不能仅以配置解析成功作为训练跑通的依据。显存不足时，可用 `batch_size=1 model.mot_checkpoint_mixed_attn=true` 降低激活显存。脚本支持较少的本机 GPU 数量用于调试，正式默认训练方案使用 8 卡。

## RoboTwin 推理与评测

安装 RoboTwin 环境和资源后，使用训练产生的**模型权重**及对应运行的归一化统计：

```bash
python experiments/robotwin/eval_robotwin_single.py \
  ckpt=./runs/robotwin/my_run/checkpoints/weights/step_010000.pt \
  EVALUATION.dataset_stats_path=./runs/robotwin/my_run/dataset_stats.json \
  EVALUATION.task_name=adjust_bottle
```

评测入口自动将 `experiments/robotwin/hilo_wam_policy` 链接到 `third_party/RoboTwin/policy/hilo_wam_policy`。策略维护稀疏观测历史，每次预测并执行 32 步动作，在 episode 之间清空历史。默认采用高层、低层及动作的同步去噪，并启用动态跳步。评测结果保存在 `./evaluate_results/robotwin`。

在同一台机器上将 benchmark 任务分配到 8 张 GPU：

```bash
python experiments/robotwin/run_robotwin_manager.py \
  ckpt=./runs/robotwin/my_run/checkpoints/weights/step_010000.pt \
  EVALUATION.dataset_stats_path=./runs/robotwin/my_run/dataset_stats.json \
  MULTIRUN.num_gpus=8
```

根据机器调整 `MULTIRUN.num_gpus`。默认的 `EVALUATION.skip_get_obs_within_replan=true` 在连续执行动作块时跳过中间 RGB 渲染；需要更密集的录制视频时设为 `false`。使用外部 RoboTwin 安装目录时，可设置 `EVALUATION.robotwin_root=./external/RoboTwin`。

RoboDojo 的推理适配在官方 RoboDojo 仓库完成，使用本仓库输出的模型权重、`config.yaml` 和 `dataset_stats.json`。本发布不包含 RoboDojo 推理启动脚本。

## 致谢及许可证

本项目基于 FastWAM、[Wan2.2](https://github.com/Wan-Video/Wan2.2)、[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)、[LeRobot](https://github.com/huggingface/lerobot) 和 [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin)。保留了原有版权和第三方许可证声明，详见 [LICENSE](LICENSE)、[NOTICE](NOTICE) 及 [RoboTwin 许可证](third_party/RoboTwin/LICENSE)。预训练权重和数据集遵循各自的上游使用条款。
