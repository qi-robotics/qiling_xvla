# 构建源码快照

该发布目录是可独立复制、构建和运行的源码快照，不在容器运行时挂载原项目源码。

- LeRobot：`0.5.0`
- LeRobot Git commit：`00b662de02734a6972ec674b8792696ecd1cb28e`
- Python：`3.12`
- PyTorch：`2.7.0`
- torchvision：`0.22.0`
- Transformers：`5.3.0`
- ModelScope：`1.39.1`
- ModelScope Hub：`0.2.0`

ROS 控制部分的源码快照来自：

- `src/common_msgs/mit_msgs`
- `src/qi_ros2/cyclonedds_ws/src/qi/qi`
- `src/topic_convertor/src/topic_convertor`
- `src/qiling_rollout_ros`
- `src/qi_robot_description/urdf/s4_dual_arm.urdf`

XVLA 推理训练部分的源码快照来自：

- `/home/ub/code/lerobot-v0.5.0/src`
- `rollout/worker/xvla_worker.py`
- `training/scripts/run_xvla_train.py` 的容器化版本

更新主工程代码后，本目录中的快照不会自动变化。发布新版本时需要显式同步并重新执行验证。
