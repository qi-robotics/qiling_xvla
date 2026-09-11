# syntax=m.daocloud.io/docker.io/docker/dockerfile:1.7

ARG ROS_BUILD_IMAGE=m.daocloud.io/docker.io/library/ros:humble-ros-base-jammy
ARG ROS_RUNTIME_IMAGE=m.daocloud.io/docker.io/library/ros:humble-ros-core-jammy

FROM ${ROS_RUNTIME_IMAGE} AS apt-base

ARG UBUNTU_APT_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/ubuntu
ARG UBUNTU_SECURITY_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/ubuntu
ARG ROS2_APT_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/ros2/ubuntu

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

# Keep archive and security mirrors separately configurable. This lets the
# proxy profile use upstream endpoints while the domestic profile uses TUNA.
RUN sed -ri \
      "s#https?://archive\\.ubuntu\\.com/ubuntu/?#${UBUNTU_APT_MIRROR}/#g; \
       s#https?://security\\.ubuntu\\.com/ubuntu/?#${UBUNTU_SECURITY_MIRROR}/#g" \
      /etc/apt/sources.list \
    && find /etc/apt/sources.list.d -type f -name '*.list' -exec \
      sed -ri "s#https?://packages\\.ros\\.org/ros2/ubuntu/?#${ROS2_APT_MIRROR}/#g" '{}' +


FROM ${ROS_BUILD_IMAGE} AS ros-builder

ARG UBUNTU_APT_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/ubuntu
ARG UBUNTU_SECURITY_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/ubuntu
ARG ROS2_APT_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/ros2/ubuntu

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

RUN sed -ri \
      "s#https?://archive\\.ubuntu\\.com/ubuntu/?#${UBUNTU_APT_MIRROR}/#g; \
       s#https?://security\\.ubuntu\\.com/ubuntu/?#${UBUNTU_SECURITY_MIRROR}/#g" \
      /etc/apt/sources.list \
    && find /etc/apt/sources.list.d -type f -name '*.list' -exec \
      sed -ri "s#https?://packages\\.ros\\.org/ros2/ubuntu/?#${ROS2_APT_MIRROR}/#g" '{}' + \
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
ARG PYTORCH_INDEX_URL=

ENV CONDA_ROOT=/opt/qi-reasoning \
    CONDARC=/opt/qiling/dependencies/condarc \
    PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY dependencies/condarc /opt/qiling/dependencies/condarc

# build-essential exists only in this discarded builder stage. It is required
# because evdev is distributed as a source archive for this platform.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      curl \
    && curl --fail --location --retry 5 --retry-delay 2 \
      "${MINICONDA_URL}" -o /tmp/miniconda.sh \
    && echo "${MINICONDA_SHA256}  /tmp/miniconda.sh" | sha256sum --check --strict \
    && bash /tmp/miniconda.sh -b -p "${CONDA_ROOT}" \
    && rm -f /tmp/miniconda.sh \
    && rm -rf /var/lib/apt/lists/*

COPY dependencies/python-constraints.txt /opt/qiling/dependencies/python-constraints.txt

# Each expensive or failure-prone dependency group has its own cache boundary.
# BuildKit's pip cache survives a failed RUN but is not copied into the image.
RUN --mount=type=cache,id=qiling-pip-cache,target=/root/.cache/pip,sharing=locked \
    "${CONDA_ROOT}/bin/python" -m pip install --upgrade pip setuptools wheel

RUN --mount=type=cache,id=qiling-pip-cache,target=/root/.cache/pip,sharing=locked \
    pytorch_args=(); \
    if [[ -n "${PYTORCH_INDEX_URL}" ]]; then \
      pytorch_args+=(--index-url "${PYTORCH_INDEX_URL}"); \
    fi; \
    "${CONDA_ROOT}/bin/python" -m pip install \
      "${pytorch_args[@]}" \
      -c /opt/qiling/dependencies/python-constraints.txt \
      torch==2.7.0 torchvision==0.22.0

# Install the native package before LeRobot so a later resolver/package failure
# never invalidates the large PyTorch layer.
RUN --mount=type=cache,id=qiling-pip-cache,target=/root/.cache/pip,sharing=locked \
    "${CONDA_ROOT}/bin/python" -m pip install \
      -c /opt/qiling/dependencies/python-constraints.txt \
      evdev==2.0.0

# Preinstall the large, stable XVLA dependency set. Local LeRobot changes
# therefore do not force these packages to be downloaded again.
RUN --mount=type=cache,id=qiling-pip-cache,target=/root/.cache/pip,sharing=locked \
    "${CONDA_ROOT}/bin/python" -m pip install \
      -c /opt/qiling/dependencies/python-constraints.txt \
      accelerate==1.14.0 \
      av==15.1.0 \
      datasets==4.8.5 \
      diffusers==0.35.2 \
      huggingface-hub==1.23.0 \
      numpy==2.2.6 \
      opencv-python-headless==4.12.0.88 \
      safetensors==0.8.0 \
      tokenizers==0.22.2 \
      torchcodec==0.10.0 \
      transformers==5.3.0

COPY third_party/lerobot /tmp/lerobot

RUN --mount=type=cache,id=qiling-pip-cache,target=/root/.cache/pip,sharing=locked \
    "${CONDA_ROOT}/bin/python" -m pip install \
      -c /opt/qiling/dependencies/python-constraints.txt \
      "/tmp/lerobot[xvla]" \
      modelscope==1.39.1 \
      modelscope-hub==0.2.0 \
    && "${CONDA_ROOT}/bin/conda" clean --all --yes \
    && find "${CONDA_ROOT}" -type d -name __pycache__ -prune -exec rm -rf '{}' + \
    && find "${CONDA_ROOT}" -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete \
    && rm -rf /tmp/lerobot

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
