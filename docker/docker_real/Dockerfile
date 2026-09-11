ARG ROS_BUILD_IMAGE=m.daocloud.io/docker.io/library/ros:humble-ros-base-jammy
ARG ROS_RUNTIME_IMAGE=m.daocloud.io/docker.io/library/ros:humble-ros-core-jammy

FROM ${ROS_RUNTIME_IMAGE} AS apt-base

ARG UBUNTU_APT_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/ubuntu
ARG ROS2_APT_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/ros2/ubuntu

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

# Ubuntu 22.04 uses /etc/apt/sources.list. The official ROS image also ships a
# ros2.list entry. Both are rewritten before any apt package is downloaded.
RUN sed -ri \
      "s#https?://(archive|security)\.ubuntu\.com/ubuntu/?#${UBUNTU_APT_MIRROR}/#g" \
      /etc/apt/sources.list \
    && find /etc/apt/sources.list.d -type f -name '*.list' -exec \
      sed -ri "s#https?://packages\.ros\.org/ros2/ubuntu/?#${ROS2_APT_MIRROR}/#g" '{}' +


FROM ${ROS_BUILD_IMAGE} AS ros-builder

ARG UBUNTU_APT_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/ubuntu
ARG ROS2_APT_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/ros2/ubuntu

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

RUN sed -ri \
      "s#https?://(archive|security)\.ubuntu\.com/ubuntu/?#${UBUNTU_APT_MIRROR}/#g" \
      /etc/apt/sources.list \
    && find /etc/apt/sources.list.d -type f -name '*.list' -exec \
      sed -ri "s#https?://packages\.ros\.org/ros2/ubuntu/?#${ROS2_APT_MIRROR}/#g" '{}' + \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
      python3-colcon-common-extensions \
      python3-pip \
      ros-humble-rosidl-generator-dds-idl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/qiling/control_ws
COPY ros_ws/src ./src
RUN source /opt/ros/humble/setup.bash \
    && colcon build --merge-install --cmake-args -DCMAKE_BUILD_TYPE=Release \
    && rm -rf build log


FROM apt-base AS reasoning-builder

ARG MINICONDA_URL=https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py312_25.9.1-3-Linux-x86_64.sh
ARG MINICONDA_SHA256=07fc00c146a481f7ae2657a31eb7ec0d5d77a107b981d58ab5d7cd8bc99be778
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

ENV CONDA_ROOT=/opt/qi-reasoning \
    CONDARC=/opt/qiling/dependencies/condarc \
    PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY dependencies/condarc /opt/qiling/dependencies/condarc

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && curl --fail --location --retry 5 --retry-delay 2 \
      "${MINICONDA_URL}" -o /tmp/miniconda.sh \
    && echo "${MINICONDA_SHA256}  /tmp/miniconda.sh" | sha256sum --check --strict \
    && bash /tmp/miniconda.sh -b -p "${CONDA_ROOT}" \
    && rm -f /tmp/miniconda.sh \
    && rm -rf /var/lib/apt/lists/*

COPY dependencies/python-constraints.txt /opt/qiling/dependencies/python-constraints.txt
COPY third_party/lerobot /tmp/lerobot

# PyTorch 2.7.0 and its NVIDIA CUDA dependencies are available from TUNA's
# PyPI mirror. LeRobot is installed from the vendored 0.5.0 source snapshot.
RUN "${CONDA_ROOT}/bin/python" -m pip install --upgrade pip setuptools wheel \
    && "${CONDA_ROOT}/bin/python" -m pip install \
      -c /opt/qiling/dependencies/python-constraints.txt \
      torch==2.7.0 torchvision==0.22.0 \
    && "${CONDA_ROOT}/bin/python" -m pip install \
      -c /opt/qiling/dependencies/python-constraints.txt \
      "/tmp/lerobot[xvla]" \
      modelscope==1.39.1 \
      modelscope-hub==0.2.0 \
    && "${CONDA_ROOT}/bin/conda" clean --all --yes \
    && find "${CONDA_ROOT}" -type d -name __pycache__ -prune -exec rm -rf '{}' + \
    && find "${CONDA_ROOT}" -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete \
    && rm -rf /tmp/lerobot /root/.cache

RUN "${CONDA_ROOT}/bin/python" -c \
      "import av, lerobot, modelscope, torch, transformers; print(torch.__version__, transformers.__version__)"


FROM apt-base AS runtime

ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

ENV CONDA_ROOT=/opt/qi-reasoning \
    PATH=/opt/qi-reasoning/bin:${PATH} \
    CONDARC=/opt/qiling/dependencies/condarc \
    PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HOME=/opt/qiling/runtime/hf_cache \
    XDG_CACHE_HOME=/opt/qiling/runtime/cache

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      procps \
      python3-numpy \
      python3-opencv \
      python3-yaml \
      ros-humble-cv-bridge \
      ros-humble-launch-ros \
      ros-humble-pinocchio \
      ros-humble-rmw-cyclonedds-cpp \
      ros-humble-ros2launch \
      ros-humble-rosidl-generator-dds-idl \
      ros-humble-sensor-msgs \
      ros-humble-std-srvs \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ros-builder /opt/qiling/control_ws/install /opt/qiling/control_ws/install
COPY --from=reasoning-builder /opt/qi-reasoning /opt/qi-reasoning
COPY dependencies/condarc /opt/qiling/dependencies/condarc
COPY ros_ws/s4_dual_arm.urdf /opt/qiling/assets/s4_dual_arm.urdf
COPY third_party/tokenizer_cache /opt/qiling/hf_seed
COPY app /opt/qiling/app
COPY tools /opt/qiling/release/tools
COPY container /opt/qiling/container

RUN chmod 0755 /opt/qiling/container/*.sh

WORKDIR /opt/qiling
ENTRYPOINT ["/opt/qiling/container/reasoning_entrypoint.sh"]
CMD ["/opt/qi-reasoning/bin/python", "/opt/qiling/app/verify_runtime.py"]
