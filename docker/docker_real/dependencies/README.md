# 镜像依赖说明

## 基础镜像

- 构建阶段：`ros:humble-ros-base-jammy`；
- 最终阶段：`ros:humble-ros-core-jammy`；
- 默认通过 `m.daocloud.io/docker.io/library/ros:*` 拉取；
- 只支持 Linux x86_64。

使用较小的 `ros-core` 作为最终层，ROS 编译工具只留在中间阶段。

## APT

Dockerfile 在第一次 `apt-get update` 前统一替换：

- Ubuntu 22.04：`https://mirrors.tuna.tsinghua.edu.cn/ubuntu`；
- ROS 2：`https://mirrors.tuna.tsinghua.edu.cn/ros2/ubuntu`。

构建阶段安装 colcon、pip 和 ROSIDL 生成器。最终镜像只安装 rollout 需要的
CycloneDDS、Pinocchio、cv_bridge、ROS launch、消息包及 Python 运行库。

## Conda 与 Python

- Miniconda：`Miniconda3-py312_25.9.1-3-Linux-x86_64.sh`；
- 安装位置：`/opt/qi-reasoning`；
- Python：3.12；
- 安装包通过官方 SHA256 校验；
- Conda channel 配置见 `condarc`；
- pip 默认源：`https://pypi.tuna.tsinghua.edu.cn/simple`。

ROS 2 使用镜像自带的系统 Python 3.10。XVLA 训练和推理固定使用：

```text
/opt/qi-reasoning/bin/python
```

## XVLA 关键版本

完整直接约束见 `python-constraints.txt`，关键版本为：

- LeRobot 0.5.0（本地源码快照）；
- PyTorch 2.7.0；
- torchvision 0.22.0；
- Transformers 5.3.0；
- torchcodec 0.10.0；
- PyAV 15.1.0；
- ModelScope 1.39.1。

LeRobot 的其余直接依赖由随镜像交付的 `third_party/lerobot/pyproject.toml`
声明，并受 `python-constraints.txt` 中的已验证版本约束。

## 不进入镜像的内容

- XVLA Base 和微调 checkpoint；
- LeRobot 数据集；
- 训练输出；
- rollout 日志和运行时缓存；
- 原始项目中与训练/rollout 无关的包。

这些内容均通过 Compose 目录挂载提供。
