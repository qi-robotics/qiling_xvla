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
    echo "[no-proxy-test] restoring the original Docker daemon proxy settings"
    sudo rm -f "$RUNTIME_DROPIN"
    sudo systemctl daemon-reload
    sudo systemctl restart docker
  fi

  exit "$status"
}

trap restore_docker_proxy EXIT INT TERM

echo "[no-proxy-test] installing a temporary runtime-only Docker proxy override"
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

# These changes affect only this script and its child build process.
unset http_proxy https_proxy all_proxy no_proxy
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY

echo "[no-proxy-test] running the delivery build with domestic mirrors only"
QI_BUILD_WITHOUT_PROXY=1 "$SCRIPT_DIR/build.sh"
