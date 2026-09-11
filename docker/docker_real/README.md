# Docker 部署（真机）

该目录是真机侧的训练和 rollout：一个镜像 `qi-reasoning:local`，Compose 起三个角色。

- `control`：`topic_convertor` + rollout bridge（ROS 2）
- `worker`：XVLA 实时推理
- `trainer`：XVLA 训练

**不要和** `docker/docker_sim/` **混用。** 仿真走 Isaac + 另一套镜像；真机 SDK、RealSense、图像压缩节点也不在本目录，它们跑在机器人 PC 上。

脚本按自己所在目录找配置，可在仓库根直接跑 `./docker/docker_real/scripts/...`。下文主命令都按这个写。

数据和产物写在本目录挂载里，**不是**仿真用的 `~/X-VLA`：

```text
docker/docker_real/
├── config/qiling.yaml     # 唯一要改的配置
├── models/                # 从魔搭拉下来的权重
├── datasets/              # 训练用 LeRobot 数据集
├── outputs/               # 训练结果
└── runtime/               # 自动生成的 yaml / 缓存 / 日志（不要手改）
```

---

## 0. 先选一条路

**路径 A — 不训练，用现成 200k ckpt 做真机 rollout（推荐先走这条）**

```bash
./docker/docker_real/scripts/build.sh
./docker/docker_real/scripts/download_models.sh all
# 机器人 PC 上先开 SDK + 三路相机（见第 4 节）
./docker/docker_real/scripts/start_rollout.sh
```

**路径 B — 自己用真机数据训练，再 rollout**

```bash
./docker/docker_real/scripts/build.sh
./docker/docker_real/scripts/download_models.sh base
# 把 LeRobot 数据集放到 docker/docker_real/datasets/qiling_training_dataset/
./docker/docker_real/scripts/train.sh
# 改 config/qiling.yaml 指向训出来的 ckpt 后，再 start_rollout.sh
```

下面各节解释这些主命令；可选参数不要当成步骤连着跑。

---

## 1. 机器要求

| 项 | 要求 |
|---|---|
| 系统 | Linux x86_64 |
| 软件 | Docker Engine、Docker Compose v2、[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) |
| GPU | NVIDIA 独立显卡 + 兼容驱动（镜像里有 CUDA 用户态，主机仍要有驱动） |
| 网络 | 本机 ROS 2 DDS 能和机器人 PC 互通，`ROS_DOMAIN_ID` 一致 |

主机不用装 ROS、Conda、LeRobot、CUDA Toolkit。全程用**当前用户**跑 `docker`，不要 `sudo docker`。

镜像里有两套 Python，脚本已写死路径，不要自己 `conda activate`：

| 用途 | 解释器 |
|---|---|
| ROS 2 Humble | `/usr/bin/python3`（3.10） |
| XVLA 训练 / 推理 | `/opt/qi-reasoning/bin/python`（3.12） |

构建默认走国内源（清华 APT / PyPI / Anaconda，ROS 底包走 DaoCloud，权重走魔搭）。改地址只动 `dependencies/build.env`，不必改 Dockerfile。版本细节见 [dependencies/README.md](dependencies/README.md)。

---

## 2. 准备镜像（两条路都要，只做一次）

**就执行这一条：**

```bash
./docker/docker_real/scripts/build.sh
```

它会：拉 ROS 底包 → 编最小 ROS 2 工作空间 → 装固定版本 Miniconda / PyTorch / LeRobot → 生成 `qi-reasoning:local` → 渲染 `config/qiling.yaml` → 有 GPU 时做一次 CUDA 检查。

**不要当步骤执行：**

| 命令 | 什么时候用 |
|---|---|
| `./docker/docker_real/scripts/setup.sh` | 旧名字，内部就是 `build.sh`，不必再跑 |
| `QI_BUILD_WITHOUT_PROXY=1 ./docker/docker_real/scripts/build.sh` | 交付验收：确认不靠代理也能编过 |
| `./docker/docker_real/scripts/test_build_without_proxy.sh` | 同上，会临时动 Docker systemd 并重启 Docker；**不要在有别的容器在跑时用** |

---

## 3. 下载权重

默认从魔搭 [keno123/qi-studio_embodied_edu](https://modelscope.cn/datasets/keno123/qi-studio_embodied_edu) 只拉 `xvla/` 下两棵子树，不要整仓下：

| 资产 | 魔搭路径 | 放到 |
|---|---|---|
| 训练基座 | `xvla/xvla_base` | `docker/docker_real/models/qiling_xvla_base/` |
| 真机 200k ckpt | `xvla/real_200k_checkpoints` | `docker/docker_real/models/qiling_xvla_real_200k/` |

**路径 A 就执行这一条：**

```bash
./docker/docker_real/scripts/download_models.sh all
```

**不要当步骤执行（三选一，不要连跑）：**

| 命令 | 什么时候用 |
|---|---|
| `./docker/docker_real/scripts/download_models.sh all` | **默认**，基座 + 200k 都要 |
| `./docker/docker_real/scripts/download_models.sh base` | 只要自己训，暂时不 rollout |
| `./docker/docker_real/scripts/download_models.sh rollout` | 只要现成 ckpt 做 rollout，不训 |

仓库若是私有的，先在当前 shell 导出 `MS_TOKEN`，再跑上面那条；不要把 token 写进 yaml、脚本或镜像。

训练数据集**不会**随 `download_models.sh` 下来。自备数据放到 `docker/docker_real/datasets/qiling_training_dataset/`。只有在 `config/qiling.yaml` 的 `assets.dataset` 填好仓库后，才用 `./docker/docker_real/scripts/download_dataset.sh`。现在默认是空的，直接跑会失败。

---

## 4. 统一配置

训练和 rollout **只改这一个文件**：

```text
docker/docker_real/config/qiling.yaml
```

每次 `train.sh` / `start_rollout.sh` 会重新生成 `runtime/` 下的 yaml 和 `cyclonedds.xml`。那些是产物，不要手改。

首次真机必须：

```yaml
deployment:
  execution_mode: shadow
```

`shadow` 只推理和记日志，不向真机发命令。验证通过后再改成 `armed`；`armed` 启动时还要在终端输入 `ARM` 才会继续。

`ros_domain_id` 必须和机器人 PC 上 SDK、相机节点相同（默认 `16`）。跨机器发现异常时，把 `dds_interface: auto` 改成实际网卡名。

---

## 5. 路径 B：训练

先把已转换的 LeRobot 数据集放到：

```text
docker/docker_real/datasets/qiling_training_dataset/
```

**就执行这一条：**

```bash
./docker/docker_real/scripts/train.sh
```

默认从 `xvla_base` 全量微调，结果在 `docker/docker_real/outputs/xvla_full_finetune/`。步数、batch 等只改 `config/qiling.yaml`。

**不要当步骤执行：**

```bash
./docker/docker_real/scripts/train.sh --dry-run
```

只检查命令、不真正开训。

训完若要用这份 ckpt 做真机 rollout，在 `config/qiling.yaml` 把 `rollout.model.checkpoint_path` 指到容器内路径，例如：

```text
/opt/qiling/outputs/xvla_full_finetune/checkpoints/200000/pretrained_model
```

不填则继续用魔搭那份 `rollout_checkpoint`。

---

## 6. 路径 A / 训完之后：真机 rollout

### 6.1 机器人 PC（本目录之外）

先启动机器人 SDK。再在机器人 PC 的完整工程里开三路相机。下面路径以现场机器为例，按实际安装位置改：

```bash
cd /home/coral/liujun/qiling_television
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=16
export ROS_LOCALHOST_ONLY=0
bash src/scripts/start_rollout_cameras.sh
```

如果相机和压缩包还没编过，先：

```bash
cd /home/coral/liujun/qiling_television
source /opt/ros/humble/setup.bash
colcon build --packages-select qiling_recording_real qiling_rollout_ros
source install/setup.bash
```

`start_rollout_cameras.sh` 会开三路 RealSense `640×480 @ 30 Hz`，等 5 秒，再开 `15 Hz`、JPEG quality=80 的压缩。脚本保持前台；`Ctrl+C` 会一起停相机和压缩。

开跑前确认：

- SDK 在发状态、能收命令
- 三路压缩图像话题正常
- Domain ID 与 `config/qiling.yaml` 相同
- 遥操和其它控制发布者都已停掉

### 6.2 本机启动

**就执行这一条：**

```bash
./docker/docker_real/scripts/start_rollout.sh
```

固定顺序：`topic_convertor` → 等 `/human_lower_state` → rollout bridge → 当前姿态过渡到 home → 稳定再延时 10 秒 → worker 建 IPC → `ROLLOUT`。

任务正常结束（观察机器人回到 home）：

```bash
./docker/docker_real/scripts/finish_rollout.sh
./docker/docker_real/scripts/stop_rollout.sh
```

出现异常动作立刻：

```bash
./docker/docker_real/scripts/abort_rollout.sh
```

`abort` 会卡住当前测得的手臂位姿并拆容器；正常收工用 `finish` 回 home，再用 `stop`。

---

## 7. 客户只用这些脚本

| 脚本 | 作用 | 要不要单独跑 |
|---|---|---|
| `./docker/docker_real/scripts/build.sh` | 编 `qi-reasoning:local` | 要，第一步 |
| `./docker/docker_real/scripts/download_models.sh` | 从魔搭拉基座 / 200k ckpt | 要 |
| `./docker/docker_real/scripts/train.sh` | 训练 | 只要走路径 B |
| `./docker/docker_real/scripts/start_rollout.sh` | 起 control + worker | 要，最后一步 |
| `./docker/docker_real/scripts/finish_rollout.sh` | 正常结束、回 home | 任务结束时 |
| `./docker/docker_real/scripts/stop_rollout.sh` | 停容器 | `finish` 之后，或收工 |
| `./docker/docker_real/scripts/abort_rollout.sh` | 紧急停 | 只有异常时 |
| `./docker/docker_real/scripts/download_dataset.sh` | 从魔搭拉训练集 | 只有 yaml 里填了 dataset 仓库 |
| `./docker/docker_real/scripts/shell.sh control` / `reasoning` | 进容器排错 | 不用当步骤 |
| `./docker/docker_real/scripts/render_config.sh` | 只重新渲染 runtime yaml | 不用单独跑，train/rollout 会做 |

不要当入口：`common.sh`、`container/*.sh`、`app/`、`tools/`、`Dockerfile`、`compose.yaml`。

---

## 8. 镜像

一个 Dockerfile，一个镜像，三个 Compose 服务共用：

| 镜像 | 谁准备 | 用途 |
|---|---|---|
| `qi-reasoning:local` | 客户本机 `build.sh` 编译 | control / worker / trainer |

权重、数据集、输出、日志都不进镜像。源码是发布快照，改主工程后这里不会自动变；版本见 [SOURCE_SNAPSHOT.md](SOURCE_SNAPSHOT.md)。
