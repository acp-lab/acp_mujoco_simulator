#!/usr/bin/env bash
# Install a separate, reproducible PX4 checkout; no system package changes.
set -euo pipefail
workspace_dir="${1:?Usage: bash setup_px4_sitl.sh /absolute/path/to/workspace}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
runtime_dir="${workspace_dir}/.px4_runtime"
px4_dir="${runtime_dir}/PX4-Autopilot"
python_dir="${runtime_dir}/python"
mkdir -p "$runtime_dir"
if [[ ! -d "$px4_dir/.git" ]]; then
  git clone --depth 1 --branch v1.16.0 https://github.com/PX4/PX4-Autopilot.git "$px4_dir"
fi
if [[ "$(git -C "$px4_dir" rev-parse HEAD)" != "6ea3539157ca358c70a515878b77077af7d4611d" ]]; then
  echo "Expected PX4 v1.16.0; refusing to change an existing checkout." >&2
  exit 1
fi
git -C "$px4_dir" submodule update --init --recursive --depth 1 \
  src/modules/mavlink/mavlink src/drivers/gps/devices src/lib/events/libevents \
  src/lib/heatshrink/heatshrink src/modules/uxrce_dds_client/Micro-XRCE-DDS-Client \
  src/lib/cdrstream/cyclonedds src/lib/cdrstream/rosidl
if [[ ! -x "$python_dir/bin/python" ]]; then
  if command -v uv >/dev/null; then
    uv venv --python /usr/bin/python3 --system-site-packages "$python_dir"
  else
    python3 -m venv --system-site-packages "$python_dir"
  fi
fi
if command -v uv >/dev/null; then
  uv pip install --python "$python_dir/bin/python" -r "$script_dir/../config/requirements-px4.txt"
else
  "$python_dir/bin/python" -m pip install -r "$script_dir/../config/requirements-px4.txt"
fi
export PATH="$python_dir/bin:$PATH"
make -C "$px4_dir" px4_sitl_default -j"${PX4_BUILD_JOBS:-6}"
