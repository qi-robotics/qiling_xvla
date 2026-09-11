# 遥操（源代码安装，不用 Docker）

Quest 3 双臂 + O6 遥操**不在本仓库、也不用 Docker**。源码在 GitHub 公开仓：

[https://github.com/qi-robotics/qiling_television](https://github.com/qi-robotics/qiling_television)

本目录只说明：怎么 clone、和仿真 / 真机 Docker 怎么分开。装依赖、编、启动以 **`qiling_television` 仓库根目录的 README** 为准。

不要和 `docker/docker_sim/`、`docker/docker_real/` 混用。真机 Docker rollout 开始前，必须先停掉遥操发布者。

---

## 克隆

仓库属于组织 `qi-robotics`，可见性是 **Public**。clone **不需要** GitHub 账号，也 **不需要** 加入组织。

**就执行这一条：**

```bash
git clone https://github.com/qi-robotics/qiling_television.git
cd qiling_television
```

默认分支是 `main`。不要 clone 个人 fork，也不要按对方 README 里可能残留的 `--branch xrtele` / `liujun0808/...` 去拉。

往这个仓 **push** 才需要该仓的 Write，和 public 无关。

---

## 和 Docker 交付的关系

| 环节 | 怎么部署 | 仓库 |
|---|---|---|
| 仿真采集 / 训练 / Isaac rollout | Docker | 本仓库 `docker/docker_sim/` |
| 真机训练 / 策略 rollout | Docker | 本仓库 `docker/docker_real/` |
| Quest 遥操 | **源码安装 ROS 2** | [`qiling_television`](https://github.com/qi-robotics/qiling_television) |

遥操控制程序是 ROS 2 / C++，**不要在 conda 里编译或启动**。

---

## 拿到代码之后（默认顺序）

详细步骤、Quest APK、排错见 [qiling_television 的 README](https://github.com/qi-robotics/qiling_television/blob/main/README.md)。客户侧默认按这个顺序，每步一条主命令：

1. Ubuntu 22.04 上安装 [ROS 2 Humble](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html)（系统 apt，不是 conda）
2. `./src/scripts/install_xrtele_dependencies.sh`
3. 安装 [XRoboToolkit PC Service v1.0.0](https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/tag/v1.0.0)（仿真机和真机 PC 都要）
4. Quest 3 侧载该仓附带的 APK（见对方 README 第 6 节）
5. `./src/scripts/build_xrtele.sh`
6. `./src/scripts/start_xrobotoolkit_real.sh`，头显里打开 Controller Tracking 和 Send
7. **先仿真：** `./src/scripts/start_sim_teleop.sh`
8. **再真机：** SDK 已启动后 `./src/scripts/start_real_teleop.sh`（`ROS_DOMAIN_ID` 默认 16，须与真机 Docker 配置一致）

真机首次必须清空双臂周围、准备急停，并停掉其它 `/human_lower_command`、`/handscmd` 发布者。
