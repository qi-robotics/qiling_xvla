# Qiling XVLA 训练与真机 Rollout

`qiling_release` 是可独立复制和交付的 Docker 目录，包含 XVLA 训练、推理以及
真机 rollout 主机侧控制链路。用户不需要在主机安装 ROS 2、Conda、LeRobot
或 CUDA Toolkit。

交付镜像统一命名为：

~~~text
qi-reasoning:local
~~~

同一个镜像由 Docker Compose 以三种角色运行：

- `control`：依次启动 `topic_convertor` 和 rollout bridge；
- `worker`：加载 XVLA checkpoint，通过本机 IPC 向 bridge 返回 action chunk；
- `trainer`：使用外置 LeRobot 数据集训练 XVLA。

真机 SDK、RealSense 驱动、三路相机和真机侧图像压缩节点不包含在本目录中。

## 1. 目录结构

~~~text
qiling_release/
├── Dockerfile
├── compose.yaml
├── config/qiling.yaml                 # 训练和 rollout 唯一用户配置
├── dependencies/
│   ├── build.domestic.env             # 国内直连构建源
│   ├── build.proxy.env                # 代理构建源
│   ├── proxy.env.example              # 本机代理配置示例
│   ├── python-constraints.txt
│   └── condarc
├── ros_ws/src/                        # 主机侧最小 ROS 2 源码
├── app/                               # XVLA worker、训练和运行验证
├── container/                         # 容器入口
├── third_party/                       # LeRobot 0.5.0 与离线 tokenizer
├── scripts/                           # 用户入口脚本
├── models/                            # 外置模型目录
├── datasets/                          # 外置 LeRobot 数据集目录
├── outputs/                           # 外置训练结果目录
└── runtime/                           # 生成配置、缓存和运行日志
~~~

模型、数据集、训练结果和运行日志不会复制进镜像。

## 2. 主机要求

- Linux x86_64；
- Docker Engine；
- Docker Compose v2；
- NVIDIA GPU 和兼容驱动；
- NVIDIA Container Toolkit；
- 训练或 rollout 时，GPU 应支持当前使用的 PyTorch 2.7/CUDA 12.6；
- rollout 主机能够通过 ROS 2 DDS 与机器人 PC 通信。

检查基础环境：

~~~bash
docker --version
docker compose version
nvidia-smi
docker run --rm --gpus all ubuntu:22.04 nvidia-smi
~~~

主机上的 NVIDIA 驱动由用户安装；CUDA 12.6 用户态库由 Python 依赖带入镜像。

## 3. 第一次准备

~~~bash
cd qiling_release
chmod +x scripts/*.sh container/*.sh
~~~

然后根据网络环境选择下面一种构建方式，不需要两种都执行。

## 4. 无代理构建（国内网络推荐）

~~~bash
./scripts/build_no_proxy.sh
~~~

该模式使用：

- DaoCloud 代理的 ROS Humble 镜像；
- 清华 Ubuntu 22.04 APT 镜像；
- 清华 ROS 2 APT 镜像；
- 清华 Miniconda 镜像；
- 清华 PyPI 镜像。

脚本会清除本次构建进程继承的代理变量。如果检测到 Docker daemon 配置了代理，
脚本会在 `/run/systemd/system/docker.service.d/` 创建临时覆盖并重启 Docker。
构建成功、失败或收到 `Ctrl+C` 后，都会删除临时覆盖并恢复原来的 Docker daemon
代理设置。

只有检测到 daemon 代理时才会使用 `sudo` 和重启 Docker。重启期间会短暂影响主机
上的其他容器，建议在没有其他 Docker 任务时构建。

兼容入口：

~~~bash
./scripts/test_build_without_proxy.sh
~~~

它会直接转发到 `build_no_proxy.sh`。

如果进程被 `kill -9`，退出钩子可能无法恢复临时配置，可手动执行：

~~~bash
sudo rm -f /run/systemd/system/docker.service.d/zz-qiling-no-proxy.conf
sudo systemctl daemon-reload
sudo systemctl restart docker
~~~

## 5. 代理构建

### 5.1 使用当前 shell 的代理

如果当前 shell 已设置 `HTTP_PROXY` 或 `HTTPS_PROXY`：

~~~bash
./scripts/build_with_proxy.sh
~~~

### 5.2 仅为本次构建指定代理

~~~bash
QI_PROXY_URL=http://127.0.0.1:7890 \
  ./scripts/build_with_proxy.sh
~~~

### 5.3 使用本地配置文件

~~~bash
cp dependencies/proxy.env.example dependencies/proxy.env
nano dependencies/proxy.env
./scripts/build_with_proxy.sh
~~~

`dependencies/proxy.env` 已被 `.gitignore` 排除，不应提交真实代理地址或认证信息。

代理模式使用官方 ROS、Ubuntu、ROS 2、Anaconda、PyPI 和 PyTorch CUDA 12.6
索引。代理参数只传给 Dockerfile 的构建过程，不会写进最终镜像。

需要区分两条网络链路：

1. `docker pull` 由 Docker daemon 发起，使用 daemon 自己的代理配置；
2. Dockerfile 中的 `apt`、`curl`、`pip` 使用脚本临时传入的代理。

如果主机不能直连 Docker Hub，还需要提前为 Docker daemon 配置代理。
Compose 构建使用 host 网络，因此 `QI_PROXY_URL=http://127.0.0.1:7890`
可以访问主机上的代理程序。

## 6. 构建缓存与速度

Dockerfile 使用 Python 3.12，与当前 LeRobot 0.5.0 的 `Python >=3.12` 要求一致。
没有直接改用 `pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime`，因为该版本官方
镜像使用 Python 3.11，无法满足本交付包的 LeRobot 版本要求。

耗时依赖被拆成独立缓存层：

~~~text
Miniconda Python 3.12
  -> pip/setuptools/wheel
  -> PyTorch 2.7 + torchvision 0.22
  -> evdev 原生编译
  -> XVLA 通用依赖
  -> LeRobot + ModelScope
  -> 运行验证
~~~

`build-essential` 只存在于被丢弃的 builder 阶段，不会进入最终镜像。
PyTorch、通用依赖和 LeRobot 不再位于同一个 `RUN` 中：即使后面的包安装失败，
已经成功的 PyTorch 层仍可复用。

pip 下载目录使用 BuildKit cache mount：

- 构建失败后保留已下载文件；
- 后续构建复用下载缓存；
- 缓存不会增加最终镜像体积。

默认不会强制刷新基础镜像。需要主动检查新基础镜像时：

~~~bash
QI_PULL_BASE=1 ./scripts/build_no_proxy.sh
~~~

或：

~~~bash
QI_PULL_BASE=1 QI_PROXY_URL=http://127.0.0.1:7890 \
  ./scripts/build_with_proxy.sh
~~~

第一次构建仍需下载数 GB 的 PyTorch/CUDA 包；优化重点是提高代理选择能力，并避免
安装后期失败时全部重新下载。不要在普通重试前执行 `docker builder prune`，否则会
清除这部分构建缓存。

高级入口为：

~~~bash
./scripts/build.sh domestic
./scripts/build.sh proxy
~~~

这两个命令只选择软件源，不负责临时启用或关闭代理，普通用户优先使用前面的包装脚本。

构建完成后会：

1. 生成 `qi-reasoning:local`；
2. 渲染 `config/qiling.yaml`；
3. 检测到 NVIDIA GPU 时验证容器内 PyTorch/CUDA；
4. 保留 ROS 编译和 Python 下载缓存供后续增量构建。

## 7. 统一配置

训练和 rollout 参数统一修改：

~~~text
config/qiling.yaml
~~~

每次启动前会生成：

~~~text
runtime/rollout_host.yaml
runtime/xvla_rollout.yaml
runtime/xvla_training.yaml
runtime/cyclonedds.xml
runtime/release.env
~~~

这些是运行产物，不要手动编辑。

## 8. 下载模型

默认从 ModelScope dataset 仓库下载：

~~~text
keno123/qi-studio_embodied_edu
├── xvla/xvla_base
└── xvla/real_200k_checkpoints
~~~

下载训练基础权重和 rollout 权重：

~~~bash
./scripts/download_models.sh all
~~~

只下载其中一项：

~~~bash
./scripts/download_models.sh base
./scripts/download_models.sh rollout
~~~

私有仓库使用：

~~~bash
export MS_TOKEN="ms-your-token"
./scripts/download_models.sh all
unset MS_TOKEN
~~~

Token 不会写入镜像或 YAML。

## 9. 训练

把已经转换好的 LeRobot 数据集放到：

~~~text
datasets/qiling_training_dataset/
~~~

如果数据集已上传至 ModelScope，先填写 `config/qiling.yaml` 中
`assets.dataset`，然后执行：

~~~bash
./scripts/download_dataset.sh
~~~

检查训练命令和配置但不启动：

~~~bash
./scripts/train.sh --dry-run
~~~

正式训练：

~~~bash
./scripts/train.sh
~~~

默认输出到：

~~~text
outputs/xvla_full_finetune/
~~~

训练数据路径、基础权重、batch size、steps、保存频率及输出目录都在
`config/qiling.yaml` 中修改。

## 10. 真机侧准备

先启动机器人 SDK。然后在机器人 PC 的完整 Qiling 工程中执行：

~~~bash
cd /home/coral/liujun/qiling_television
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=16
export ROS_LOCALHOST_ONLY=0
bash src/scripts/start_rollout_cameras.sh
~~~

如果机器人 PC 还没有编译相机和压缩包：

~~~bash
cd /home/coral/liujun/qiling_television
source /opt/ros/humble/setup.bash
colcon build --packages-select qiling_recording_real qiling_rollout_ros
source install/setup.bash
~~~

`start_rollout_cameras.sh` 会：

1. 启动三路 RealSense，分辨率为 `640x480 @ 30 Hz`；
2. 等待 5 秒；
3. 启动 `15 Hz、JPEG quality=80` 的压缩传输节点。

等价命令为：

~~~bash
ros2 launch qiling_recording_real tri_camera.launch.py

ros2 launch qiling_rollout_ros rollout_image_transport.launch.py \
  output_rate_hz:=15.0 \
  jpeg_quality:=80
~~~

真机侧需要满足：

- SDK 已发布机器人状态并接收控制命令；
- 三路压缩图像话题持续发布；
- ROS Domain ID 与 `config/qiling.yaml` 一致；
- 遥操和其他控制命令发布者已经停止。

## 11. 主机侧 Rollout

首次部署必须设置：

~~~yaml
deployment:
  execution_mode: shadow
~~~

然后运行：

~~~bash
./scripts/start_rollout.sh
~~~

脚本会同时启动 `control` 和 `worker`，内部顺序为：

~~~text
topic_convertor
  -> 等待 /human_lower_state
  -> rollout bridge
  -> 当前姿态到过渡点
  -> 双臂到 home
  -> 稳定并等待 10 秒
  -> bridge 开放本机 IPC
  -> worker 开始持续推理
  -> ROLLOUT
~~~

shadow 验证状态、图像、模型和 IPC 都正常后，再把
`deployment.execution_mode` 改为 `armed`。armed 启动时必须输入 `ARM`
进行二次确认。

任务正常完成：

~~~bash
./scripts/finish_rollout.sh
~~~

等待机器人回到 home 后停止全部容器：

~~~bash
./scripts/stop_rollout.sh
~~~

异常动作、碰撞风险或需要立即停止时：

~~~bash
./scripts/abort_rollout.sh
~~~

`abort` 不执行自动回 home；操作人员应保持急停可用。

## 12. 常用排查

查看服务状态：

~~~bash
docker compose ps
docker compose logs -f control
docker compose logs -f worker
~~~

进入容器：

~~~bash
./scripts/shell.sh control
./scripts/shell.sh reasoning
~~~

检查镜像：

~~~bash
docker image inspect qi-reasoning:local
docker run --rm --gpus all qi-reasoning:local
~~~

查看 BuildKit 缓存：

~~~bash
docker buildx du
~~~

依赖版本说明见 [dependencies/README.md](dependencies/README.md)，源码快照版本见
[SOURCE_SNAPSHOT.md](SOURCE_SNAPSHOT.md)。
