#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_DROPIN_DIR=/run/systemd/system/docker.service.d
RUNTIME_DROPIN="$RUNTIME_DROPIN_DIR/zz-qiling-no-proxy.conf"
OVERRIDE_INSTALLED=0

restore_docker_proxy() {
  local status=$?
  trap - EXIT INT TERM

  if [[ "$OVERRIDE_INSTALLED" == "1" ]]; then
    echo "[no-proxy-build] restoring the original Docker daemon proxy settings"
    sudo rm -f "$RUNTIME_DROPIN"
    sudo systemctl daemon-reload
    sudo systemctl restart docker
  fi

  exit "$status"
}

wait_for_docker() {
  local attempt
  for attempt in {1..30}; do
    if docker info >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "[no-proxy-build] Docker did not become ready in time." >&2
  return 1
}

trap restore_docker_proxy EXIT INT TERM

docker_info="$(docker info 2>&1)" || {
  echo "[no-proxy-build] Unable to inspect Docker daemon:" >&2
  echo "$docker_info" >&2
  exit 3
}

if grep -Eq '^[[:space:]]*(HTTP|HTTPS) Proxy:[[:space:]]*[^[:space:]]' <<<"$docker_info"; then
  echo "[no-proxy-build] temporarily disabling the Docker daemon proxy"
  sudo install -d -m 0755 "$RUNTIME_DROPIN_DIR"
  printf '%s\n' \
    '[Service]' \
    'Environment="HTTP_PROXY="' \
    'Environment="HTTPS_PROXY="' \
    'Environment="ALL_PROXY="' \
    'Environment="NO_PROXY="' \
    | sudo tee "$RUNTIME_DROPIN" >/dev/null
  OVERRIDE_INSTALLED=1

  sudo systemctl daemon-reload
  sudo systemctl restart docker
  wait_for_docker
fi

# These changes affect only this script and its child build process.
unset http_proxy https_proxy all_proxy no_proxy
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY
unset QI_BUILD_HTTP_PROXY QI_BUILD_HTTPS_PROXY QI_BUILD_ALL_PROXY QI_BUILD_NO_PROXY

echo "[no-proxy-build] profile=domestic; all shell build proxies are disabled"
"$SCRIPT_DIR/build.sh" domestic
