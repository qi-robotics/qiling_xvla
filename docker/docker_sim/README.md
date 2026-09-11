# Docker 部署（客户）

工作数据写在宿主机 **`~/X-VLA`**（可用 `QILING_ROOT` 改），容器内是 `/workspace/X-VLA`。

```text
~/X-VLA/
├── datasets/
├── outputs/
├── configs/                  # 可选，同名 yaml 覆盖仓库默认配置
├── reports/
└── .cache/huggingface/       # 第一次训练/rollout 自动从 hf-mirror 下载
```

本目录按角色分子目录，客户只跑 `scripts/`：

```text
docker/docker_sim/
├── README.md
├── Dockerfile.isaac / Dockerfile.xvla
├── docker-compose.yml
├── scripts/          # 客户入口：up / fetch_modelscope / record / train / rollout
├── container/        # 容器 ENTRYPOINT 与策略 IPC，不要手跑
└── tools/            # fetch / prefetch / patch 的 Python，不要手跑
```

**先选一条路，不要两条都走。** 每条路按顺序各执行列出的那几条，每一步只有一条主命令。

**路径 A — 不采集、不训练，直接看 rollout（推荐先走这条）**

```bash
./docker/docker_sim/scripts/up.sh
./docker/docker_sim/scripts/fetch_modelscope.sh
./docker/docker_sim/scripts/rollout.sh --seed 40
```

**路径 B — 自己录数据再训练**

```bash
./docker/docker_sim/scripts/up.sh
./docker/docker_sim/scripts/record.sh --count 3
./docker/docker_sim/scripts/train.sh
./docker/docker_sim/scripts/rollout.sh --seed 40
```

下面各节解释这些主命令是什么意思，以及**不要当成步骤去跑**的可选参数。

---

## 0. 机器要求

| 项 | 要求 |
|---|---|
| 系统 | Linux x86_64 |
| GPU | NVIDIA 独立显卡，建议 24GB 显存 |
| 驱动 | ≥ 570.169 |
| 软件 | Docker、[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) |
| 显示 | 录制默认无窗口；rollout 默认开 GUI |

宿主机不用装 CUDA Toolkit / Isaac / LeRobot。`nvidia-smi` 能看到 GPU 即可。

磁盘大约：Isaac 一族 ~24GB + xvla 镜像 ~20GB，另加数据和 ckpt。走路径 A 再加约 **4GB**（数据集 ~2.1GB + 权重 ~1.8GB）。

脚本已写死国内源（**不用你再配代理**）：

- Hugging Face → `https://hf-mirror.com`
- PyPI（编 Isaac 薄层）→ `https://pypi.tuna.tsinghua.edu.cn/simple`
- 路径 A 的数据 / ckpt → 魔搭 `modelscope.cn`

Isaac 官方底包只能从 NVIDIA NGC `nvcr.io` 拉（许可证不允许转到阿里云）。这一步若很慢，给 **Docker 守护进程** 配代理；与 Hugging Face 无关。

启动 Isaac 即视为接受 NVIDIA Omniverse EULA。

全程用 **当前用户** 跑 `docker`，不要 `sudo docker`。用 sudo 登录的话凭证写在 `/root/.docker/`，后面普通用户 `pull` 仍会失败。

Isaac 容器 UID **1234**。`./docker/docker_sim/scripts/up.sh` 会把 `~/X-VLA` 设成可写。只有目录变成 root 才能写、脚本报权限错误时，才执行：

```bash
sudo chown -R 1234:1234 "$HOME/X-VLA"
chmod -R a+rwX "$HOME/X-VLA"
```

改工作根时，每条命令前面加同样的前缀，例如 `QILING_ROOT=/data/X-VLA ./docker/docker_sim/scripts/up.sh`。

---

## 1. 准备镜像（两条路都要，只做一次）

**就执行这一条：**

```bash
./docker/docker_sim/scripts/up.sh
```

它会依次：登录阿里云 → 拉 Isaac 底包 → 拉 xvla 镜像 → 本机编译薄层 `qiling-isaac:5.1.0`。做完即可，不必再单独 `docker login`。

---

## 2. 路径 A：下载现成数据 + ckpt

魔搭仓库 [keno123/qi-studio_embodied_edu](https://modelscope.cn/datasets/keno123/qi-studio_embodied_edu) 是课程资料合集，里面还有 `bottleInBowl`、`smolVLA` 等 **不属于本产品** 的目录。整仓大约二十多 GB，**不要整仓下载**。

本任务只要 `xvla/` 下这两棵子树：

- 数据集：[slender_pin_lerobot_v3_xvla_datasets](https://modelscope.cn/datasets/keno123/qi-studio_embodied_edu/tree/master/xvla/slender_pin_lerobot_v3_xvla_datasets)（~2.1GB）
- 200k 权重：[xvla_slender_pin_full_from60k_plus200k_v1](https://modelscope.cn/datasets/keno123/qi-studio_embodied_edu/tree/master/xvla/xvla_slender_pin_full_from60k_plus200k_v1) 里的 `checkpoints/200000/pretrained_model`（~1.8GB）

**就执行这一条：**

```bash
./docker/docker_sim/scripts/fetch_modelscope.sh
```

它只拉上面两棵子树（合计约 3.6GB），写到 `rollout.sh` 会自动找到的位置，并补上 `checkpoints/last` → `200000`：

| 魔搭路径 | 放到 |
|---|---|
| `xvla/slender_pin_lerobot_v3_xvla_datasets/` | `~/X-VLA/datasets/slender_pin_lerobot_v3_xvla_v1/` |
| `.../checkpoints/200000/pretrained_model/` | `~/X-VLA/outputs/xvla_slender_pin_full_from60k_plus200k_v1/checkpoints/200000/pretrained_model/` |

默认 **不拉** `training_state`（优化器状态 ~3.5GB）。

下完后接着执行 `./docker/docker_sim/scripts/rollout.sh --seed 40`（见第 4 节）。第一次 rollout 会从 hf-mirror 拉 `facebook/bart-large`（tokenizer，进 `~/X-VLA/.cache/huggingface`，不是镜像里）。

**不要当步骤执行（只有这些情况才用）：**

| 命令 | 什么时候用 |
|---|---|
| `./docker/docker_sim/scripts/fetch_modelscope.sh --dry-run` | 只想看会下哪些文件、不下 |
| `./docker/docker_sim/scripts/fetch_modelscope.sh --with-optimizer` | 要接着这份 200k 任务继续训练，才需要优化器状态 |
| 自己用 `modelscope download --include ...` | 不想用本脚本、已会魔搭 CLI 时的等价做法；**不要和 `fetch_modelscope.sh` 重复下两遍** |

---

## 3. 路径 B：自己录数据 → 训练

走路径 A 的人跳过本节。

### 3.1 录数据

**就执行这一条：**

```bash
./docker/docker_sim/scripts/record.sh --count 3
```

一次做完：生成采集计划 + 仿真 Auto-IK 录制 + 校验。默认无 GUI。

`--count` 是录几条专家轨迹。不写也是 3。3 条只够把链路跑通；要自己训一个能用的模型，把数字改大后再执行一次，例如 `--count 300`。不要 3 条跑完再跑一条 300，除非你就是想加录。

写出：

| 目录 | 内容 |
|---|---|
| `~/X-VLA/datasets/raw_slender_pin_v1` | 计划 |
| `~/X-VLA/datasets/recorded_slender_pin_v1` | 专家数据（npz + 三相机 mp4） |

中断后再次执行同一条命令，默认从断点续录。

**不要当步骤执行：**

| 参数 | 什么时候用 |
|---|---|
| `--gui` | 想看 Isaac 窗口 |
| `--overwrite` | 不要已有数据，整份重录 |
| `--no-resume` | 不续录，按现有计划从头再跑 |

改任务：把 yaml 放到 `~/X-VLA/configs/`，文件名与仓库 `configs/` 相同。

### 3.2 训练

**就执行这一条：**

```bash
./docker/docker_sim/scripts/train.sh
```

含义：用你录好的数据，**从头训一个 XVLA**（不是从官方 xvla-base 微调）。默认 20 万 step、batch 4。没有 LeRobot 数据集时会先从 `recorded_slender_pin_v1` 自动转换。第一次会从 hf-mirror 下载 `facebook/bart-large`。

ckpt 写在 `~/X-VLA/outputs/xvla_slender_pin_full/`。

**下面不是第二、第三步，不要接着跑。** 它们和上面那条是三种不同用法，三选一：

| 命令 | 含义 | 什么时候用 |
|---|---|---|
| `./docker/docker_sim/scripts/train.sh` | 从头训 | **默认，客户走路径 B 用这条** |
| `./docker/docker_sim/scripts/train.sh --from-base` | 从官方 `lerobot/xvla-base` 微调，会多下一份基础权重 | 只有你明确要从公开基座接着训，才用；**不要和第一条连着跑** |
| `./docker/docker_sim/scripts/train.sh --pretrained-path outputs/xvla_slender_pin_full/checkpoints/last/pretrained_model --resume` | 从上次同一个输出目录接着训（恢复优化器） | 只有上次 `./docker/docker_sim/scripts/train.sh` 中断了，才用 |

---

## 4. Rollout（两条路最后一步都是它）

**就执行这一条：**

```bash
./docker/docker_sim/scripts/rollout.sh --seed 40
```

默认开 Isaac GUI，用脚本自动找到的 ckpt 和数据集跑一局，`--seed 40` 与 README 演示动图同一局。视频在 `~/X-VLA/outputs/slender_pin_xvla_rollout/seed40/videos/`。

不要手动开 `serve_xvla_policy.py`。

自动查找顺序（不用你填路径）：

- ckpt：先魔搭那份 `..._from60k_plus200k_v1`（`last` 或 `200000`），再找路径 B 训出来的 `outputs/xvla_slender_pin_full/`
- 数据集：`datasets/slender_pin_lerobot_v3_xvla_v1`，否则魔搭原名，再否则旧名 `..._pi05_v1`

**不要当步骤执行：**

| 参数 | 什么时候用 |
|---|---|
| `--headless` | 这台机器没有显示器 / 不想开窗口 |
| `--checkpoint PATH` / `--dataset PATH` | 自动找到的不是你想用的那份，才手动指定（路径相对 `~/X-VLA`） |

例如无窗口、并且你刚用路径 B 训完、想指定这份 ckpt：

```bash
./docker/docker_sim/scripts/rollout.sh --seed 40 --headless \
  --checkpoint outputs/xvla_slender_pin_full/checkpoints/last/pretrained_model \
  --dataset datasets/slender_pin_lerobot_v3_xvla_v1
```

---

## 5. 客户只用这些脚本

| 脚本 | 作用 | 要不要单独跑 |
|---|---|---|
| `./docker/docker_sim/scripts/up.sh` | 登录阿里云、拉镜像、编 Isaac 薄层 | 要，两条路第一步 |
| `./docker/docker_sim/scripts/fetch_modelscope.sh` | 只下载魔搭上的 XVLA 数据集 + 200k 权重 | 只要走路径 A |
| `./docker/docker_sim/scripts/record.sh` | 录数据 | 只要走路径 B |
| `./docker/docker_sim/scripts/train.sh` | 正式训练 | 只要走路径 B |
| `./docker/docker_sim/scripts/rollout.sh` | 仿真 + 录像 | 要，两条路最后一步 |
| `./docker/docker_sim/scripts/prefetch_hf.sh` | 预下载 tokenizer | **不用单独跑**；`train.sh` / `rollout.sh` 缺 tokenizer 时会自己下 |

不要当入口：`scripts/common.sh`、`container/`、`tools/`。

---

## 6. 镜像

| 镜像 | 谁准备 | 用途 |
|---|---|---|
| `nvcr.io/nvidia/isaac-sim:5.1.0` | 客户从 NGC pull（`up.sh` 里做） | Isaac 底包 |
| `qiling-isaac:5.1.0` | 客户本机 `up.sh` 编译 | 录制、rollout 仿真 |
| `registry.cn-hangzhou.aliyuncs.com/keno/qi-xvla:lerobot050` | 交付方推到阿里云，客户 pull | 训练、rollout 策略 |

只训练只要 xvla 镜像。要录或 rollout 必须有 `qiling-isaac:5.1.0`。路径 A 不训练 **仍然需要 Isaac**（仿真在 Isaac 里跑）。
