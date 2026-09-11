# Docker 部署

按环节分目录，**不要混用**。仿真和真机不是同一套镜像、也不是同一套命令。

| 目录 | 环节 | 客户说明 |
|---|---|---|
| [docker_sim/](docker_sim/) | 仿真采集 / 训练 / Isaac rollout | [docker_sim/README.md](docker_sim/README.md) |
| [docker_real/](docker_real/) | 真机训练 / 真机 rollout | [docker_real/README.md](docker_real/README.md) |

Quest 遥操**不用 Docker**，源码安装见 [teleop/README.md](../teleop/README.md)。代码在公开仓 [qiling_television](https://github.com/qi-robotics/qiling_television)，直接 `git clone https://github.com/qi-robotics/qiling_television.git` 即可。

都在**仓库根目录**执行。

**仿真（Isaac，不碰真机）：**

```bash
./docker/docker_sim/scripts/up.sh
./docker/docker_sim/scripts/fetch_modelscope.sh
./docker/docker_sim/scripts/rollout.sh --seed 40
```

**真机（本机 Docker + 机器人 PC 的 SDK/相机）：**

```bash
./docker/docker_real/scripts/build.sh
./docker/docker_real/scripts/download_models.sh all
./docker/docker_real/scripts/start_rollout.sh
```

首次真机必须保持 `config/qiling.yaml` 里 `execution_mode: shadow`。完整步骤、训练、停机见 [docker_real/README.md](docker_real/README.md)。
