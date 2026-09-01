#!/usr/bin/env python3
"""Print a read-only PX4 SITL compatibility report."""

import argparse

from multi_drone_mujoco.backends.px4_environment import (
    PX4_COMPATIBLE_COMMIT,
    PX4_COMPATIBLE_TAG,
    inspect_px4_environment,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("offline", "local", "remote"),
        help="deployment mode (otherwise PX4_MODE, then offline)",
    )
    args = parser.parse_args()
    report = inspect_px4_environment(mode=args.mode)
    print(f"verified PX4: {PX4_COMPATIBLE_TAG} ({PX4_COMPATIBLE_COMMIT})")
    print(f"mode: {report.mode.value}")
    print(f"PX4_AUTOPILOT_DIR: {report.autopilot_dir or '<unset>'}")
    print(f"PX4_BINARY: {report.binary or '<unset>'}")
    print(f"PX4_BIND_HOST: {report.bind_host or '<unset>'}")
    print(f"PX4_BIND_PORT: {report.bind_port if report.bind_port is not None else '<unset>'}")
    print(f"PX4_SIM_HOSTNAME: {report.remote_simulator_host or '<externally configured>'}")
    print(f"pymavlink: {'available' if report.pymavlink_available else 'missing'}")
    print(f"checkout tag: {report.git_tag or '<unknown>'}")
    print(f"checkout commit: {report.git_commit or '<unknown>'}")
    print(f"verified checkout: {'yes' if report.compatible else 'no'}")
    for issue in report.issues:
        print(f"issue: {issue}")
    return 0 if not report.issues else 2


if __name__ == "__main__":
    raise SystemExit(main())
