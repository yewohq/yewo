#!/usr/bin/env bash
set -euo pipefail

benchmark_dir="$(cd "$(dirname "$0")" && pwd)"
android_world_dir="$benchmark_dir/vendor/android_world"
android_world_revision="${ANDROID_WORLD_REVISION:-main}"
android_env_version="${ANDROID_ENV_VERSION:-}"
overrides="$benchmark_dir/upstream-overrides.txt"
python="$benchmark_dir/.venv/bin/python"

if [[ ! -d "$android_world_dir/.git" ]]; then
  git clone https://github.com/google-research/android_world.git "$android_world_dir"
fi

git -C "$android_world_dir" fetch --depth 1 origin "$android_world_revision"
git -C "$android_world_dir" checkout --detach FETCH_HEAD
resolved_android_world_revision="$(git -C "$android_world_dir" rev-parse HEAD)"

uv venv --python 3.12 --clear "$benchmark_dir/.venv"
uv pip install --python "$python" --override "$overrides" -e "$benchmark_dir/../../..[jev]"

# Follow upstream's AndroidEnv requirement by default. A caller can still pin
# it explicitly when reproducing a historical run.
requirements=(-r "$android_world_dir/requirements.txt")
if [[ -n "$android_env_version" ]]; then
  requirements+=("android-env==$android_env_version")
fi
uv pip install --python "$python" --override "$overrides" "${requirements[@]}"

# AndroidWorld's setup.py imports pkg_resources and generates its protobuf
# bindings with grpcio-tools, so it is built against the environment installed
# above instead of an isolated one that has neither.
uv pip install --python "$python" "setuptools<81" wheel
uv pip install --python "$python" --override "$overrides" --no-build-isolation \
  -e "$android_world_dir"

"$python" -W ignore -c "
from android_world import registry, suite_utils
from android_world.env import env_launcher
from android_world.agents import m3a, seeact, t3a
from android_world.task_evals.information_retrieval.proto import state_pb2
import yewo
"

echo "AndroidWorld $resolved_android_world_revision and Yewo are installed in $benchmark_dir/.venv"
