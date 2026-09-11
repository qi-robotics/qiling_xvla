#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "setup.sh is kept for compatibility; forwarding to build.sh"
exec "$script_dir/build.sh" domestic "$@"
