"""Run the upstream AndroidWorld suite with Yewo as its agent."""

import atexit
import importlib.util
import os
import sys
from pathlib import Path

from launcher import CHILD_ENV, launch

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]

# The parent owns SDK discovery, checkpoint defaults, redaction, and teeing.
# The child imports AndroidWorld only after those run conditions are in place.
if __name__ == "__main__" and os.environ.get(CHILD_ENV) != "1":
    raise SystemExit(launch(Path(__file__).resolve(), sys.argv[1:], BENCHMARK_ROOT))

from absl import app, flags  # noqa: E402 - child-only imports follow the parent handoff
from evaluator_diagnostics import install_clock_diagnostics  # noqa: E402
from yewo_agent import YewoAndroidWorldAgent  # noqa: E402

UPSTREAM_ROOT = Path(
    os.environ.get("ANDROID_WORLD_ROOT", BENCHMARK_ROOT / "vendor" / "android_world")
).resolve()


def _load_upstream_runner():
    runner_path = UPSTREAM_ROOT / "run.py"
    if not runner_path.is_file():
        raise RuntimeError(
            f"AndroidWorld was not found at {UPSTREAM_ROOT}. Run ./setup.sh first."
        )
    spec = importlib.util.spec_from_file_location("android_world_upstream_runner", runner_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load AndroidWorld runner at {runner_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


UPSTREAM = _load_upstream_runner()
install_clock_diagnostics()
YEWO_MODEL = flags.DEFINE_string(
    "yewo_model",
    "openrouter/google/gemini-2.5-flash",
    "LiteLLM model used for Yewo planning.",
)
YEWO_FAST_MODEL = flags.DEFINE_string(
    "yewo_fast_model", None, "Optional LiteLLM model used for Yewo steps."
)
YEWO_GROUNDING = flags.DEFINE_enum(
    "yewo_grounding",
    "coordinates",
    ["coordinates", "describe", "marks"],
    "Yewo target-grounding mode.",
)
YEWO_MAX_STEPS = flags.DEFINE_integer(
    "yewo_max_steps", 40, "Yewo's own safety ceiling for one task."
)
YEWO_SETTLE_TIME = flags.DEFINE_float(
    "yewo_settle_time", 2.0, "Maximum seconds Yewo waits for an action to settle."
)
YEWO_OUTPUT_DIR = flags.DEFINE_string(
    "yewo_output_dir",
    str(BENCHMARK_ROOT.parents[1] / "runs" / "android_world" / "yewo"),
    "Directory for Yewo screenshots, prompts, logs, and usage.",
)


def _build_yewo_agent(env, family=None):
    if family != "android_world":
        raise ValueError("The Yewo adapter currently supports the android_world family only.")
    device_name = f"emulator-{UPSTREAM._DEVICE_CONSOLE_PORT.value}"
    agent_instance = YewoAndroidWorldAgent(
        env=env,
        device_name=device_name,
        output_dir=Path(YEWO_OUTPUT_DIR.value),
        model=YEWO_MODEL.value,
        fast_model=YEWO_FAST_MODEL.value,
        grounding=YEWO_GROUNDING.value,
        max_steps=YEWO_MAX_STEPS.value,
        settle_time=YEWO_SETTLE_TIME.value,
    )
    atexit.register(agent_instance.close)
    return agent_instance


UPSTREAM._get_agent = _build_yewo_agent


if __name__ == "__main__":
    app.run(UPSTREAM.main)
