#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROXY_FILE="$RELEASE_ROOT/dependencies/proxy.env"

if [[ -f "$PROXY_FILE" ]]; then
  set -a
  source "$PROXY_FILE"
  set +a
fi

proxy_url="${QI_PROXY_URL:-${HTTPS_PROXY:-${https_proxy:-${HTTP_PROXY:-${http_proxy:-}}}}}"
if [[ -z "$proxy_url" ]]; then
  cat >&2 <<'EOF'
[proxy-build] No proxy address was provided.
Set it for this command, for example:
  QI_PROXY_URL=http://127.0.0.1:7890 ./scripts/build_with_proxy.sh
or copy dependencies/proxy.env.example to dependencies/proxy.env.
EOF
  exit 2
fi

no_proxy_value="${QI_NO_PROXY:-${NO_PROXY:-${no_proxy:-localhost,127.0.0.1,::1}}}"
export QI_BUILD_HTTP_PROXY="$proxy_url"
export QI_BUILD_HTTPS_PROXY="$proxy_url"
export QI_BUILD_ALL_PROXY="$proxy_url"
export QI_BUILD_NO_PROXY="$no_proxy_value"

echo "[proxy-build] proxy is enabled for Dockerfile RUN instructions"
echo "[proxy-build] base-image pulls use the Docker daemon network/proxy configuration"
echo "[proxy-build] profile=proxy"

exec "$SCRIPT_DIR/build.sh" proxy
