#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "test_build_without_proxy.sh is kept for compatibility; forwarding to build_no_proxy.sh"
exec "$SCRIPT_DIR/build_no_proxy.sh" "$@"
