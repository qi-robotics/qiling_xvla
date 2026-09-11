# 镜像依赖与构建源

## 构建配置

- `build.domestic.env`：国内直连模式；
- `build.proxy.env`：代理模式；
- `build.env`：旧入口兼容配置，内容等同国内模式；
- `proxy.env.example`：代理地址示例；
- `python-constraints.txt`：已经验证的 Python 关键版本；
- `condarc`：Conda 国内 channel 配置。

代理地址不要写入 Dockerfile 或提交到仓库。可将 `proxy.env.example` 复制为
`proxy.env`，该文件已被 `.gitignore` 排除。

## Python 与 PyTorch

ROS 2 Humble 使用 Ubuntu 22.04 的系统 Python 3.10。XVLA 使用独立的
Miniconda Python 3.12：

~~~text
/opt/qi-reasoning/bin/python
~~~

没有直接使用 `pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime` 作为最终基础
镜像，因为该标签使用 Python 3.11，而本目录中的 LeRobot 0.5.0 要求
Python >=3.12。

关键版本：

- LeRobot 0.5.0，本地源码快照；
- Python 3.12；
- PyTorch 2.7.0；
- torchvision 0.22.0；
- Transformers 5.3.0；
- torchcodec 0.10.0；
- PyAV 15.1.0；
- ModelScope 1.39.1；
- evdev 2.0.0。

`evdev` 需要 GCC 编译。编译器只安装在 `reasoning-builder` 中间阶段，
不会复制进最终镜像。

## 缓存边界

Dockerfile 将依赖拆成：

1. pip 构建工具；
2. PyTorch/torchvision；
3. evdev；
4. XVLA 通用依赖；
5. LeRobot/ModelScope；
6. 最终导入验证。

所有 pip 安装都使用 BuildKit cache mount。下载缓存不会进入最终镜像，
但能被失败重试和后续构建复用。

## 不进入镜像的内容

- XVLA Base 和微调 checkpoint；
- LeRobot 数据集；
- 训练输出；
- rollout 日志；
- pip 下载缓存；
- 原始工程中与训练/rollout 无关的代码。

这些内容通过 Compose 目录挂载或 ModelScope 下载提供。
