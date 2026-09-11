# qiling_rollout_ros

ROS 2 Humble / Python 3.10 包。主机侧 bridge 负责真机状态与相机接入、归位/rollout/abort 状态机、
MIT 命令安全边界、重力前馈，以及与独立 XVLA worker 的本机 IPC；真机侧 compressor 负责三路
rollout 图像的限频和 JPEG 压缩。完整操作步骤见根目录
[`ROLLOUT_WORKFLOW.md`](../../ROLLOUT_WORKFLOW.md)。

## 包结构

```text
qiling_rollout_ros/
├── config/rollout_host.yaml        # ROS 参数与机器人安全参数
├── launch/rollout_host.launch.py   # bridge launch 文件
├── launch/rollout_image_transport.launch.py # 真机侧三相机压缩传输
├── qiling_rollout_ros/
│   ├── async_action_scheduler.py  # 无 ROS 的滚动 action 队列
│   ├── reference_limiter.py        # 30Hz 插值与 50Hz 位置/速度/加速度限制
│   ├── rollout_image_compressor.py # raw RGB → 限频 JPEG CompressedImage
│   └── rollout_ros_bridge.py       # 运行节点与状态机
├── test/test_async_action_scheduler.py # action 队列离线测试
├── package.xml                     # ROS 运行依赖
└── setup.py                        # ament_python 安装定义
```

## 节点与接口

节点名：`/qiling_rollout_ros_bridge`

订阅：

- `/human_lower_state`：26 维机器人关节状态；启动归位和 abort hold 都依赖它。
- 三路 `sensor_msgs/CompressedImage`：仅在 `ROLLOUT` 状态创建订阅；由真机侧 compressor 发布。

发布：

- `/human_lower_command`：26 维 `mit_msgs/MITJointCommands`，仅 armed 状态下发布。
- `/handscmd`：`qi/msg/HandsCmd`，用于 O6。

服务：

- `/rollout/finish`，`std_srvs/srv/Trigger`：人工确认任务完成，直接回 home 后结束。
- `/rollout/abort`，`std_srvs/srv/Trigger`：停止推理并进入 `ABORT_HOLD`。

## 状态机

Armed：

```text
WAITING_FOR_STATE → MOVE_TO_TRANSITION → SETTLE_AT_TRANSITION
→ MOVE_TO_HOME → SETTLE_AT_HOME → WAIT_BEFORE_ROLLOUT → ROLLOUT
```

完成：

```text
ROLLOUT → RETURN_DIRECT_TO_HOME → SETTLE_RETURN_HOME → FINISHED
```

中断：

```text
ROLLOUT / timeout → ABORT_HOLD
ABORT_HOLD + state stale → ABORT_FAULT
```

## 配置要点

`config/rollout_host.yaml` 定义 26 电机索引（腿 `0..11`、左臂 `12..18`、右臂 `19..25`）、双臂
home/过渡点、关节限位、MIT `kp/kd`、Pinocchio 重力前馈、O6 位置、超时和 IPC 参数。
其中 action 调度参数默认维持 `12` 个未来 30 Hz action、剩余 `8` 步时预取下一次推理。新 chunk 到达时
仅保护最近 `3` 步旧动作，删除更远的旧预测，再以新观测对应的动作补足队列；替换边界前 `4` 步只对
7 个右臂关节做渐进融合，O6 保持离散滞回控制。超过 `12` 个 policy step 的过期推理结果会被拒绝。
每一对相邻、已排程的 policy action 会线性插值到 50 Hz，
随后由 `right_max_velocity_rad_s` 与 `right_max_acceleration_rad_s2` 限制连续的右臂 `q_ref`；没有
下一条验证 action 或 action 超时就冻结当前参考，不会外推。它们是无真机时使用的保守初值，真机恢复后
应依据 P95 端到端时延重新标定。

真机侧相机节点仍以 `640×480 @ 30 Hz` 发布 raw 图像，`rollout_image_compressor` 默认只向主机发送
`15 Hz、JPEG quality=80` 的三路 `CompressedImage`。这样不改变相机采集配置，也避免三路未压缩 RGB
占满 Wi-Fi。raw 接收和压缩图像传输均使用 `RELIABLE + KEEP_LAST(depth=1)`；三路编码由多线程 executor
处理，既与 RealSense 的 reliable 发布端一致，也避免丢失一个 DDS 分片就丢掉整幅 JPEG。raw 和压缩
话题名称、压缩质量及输出频率均可通过 launch 参数或节点参数调整。

该包不启动 `topic_convertor`，也不运行 Quest、XR bridge 或 differential IK；这些命令源必须与 rollout
互斥，避免多个发布者同时写入 `/human_lower_command`。
