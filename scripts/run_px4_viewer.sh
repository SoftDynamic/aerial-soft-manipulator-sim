#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/run_px4_viewer.sh MODE [--dry-run] [-- EXTRA_ARGS...]

MODE: mouse | tendon | keyboard-force | joint

Examples:
  ./scripts/run_px4_viewer.sh mouse
  ./scripts/run_px4_viewer.sh joint --dry-run
  ./scripts/run_px4_viewer.sh mouse -- --duration 600

Environment overrides:
  MUJOCO_UAV_PYTHON, PX4_AUTOPILOT_DIR, PX4_BIND_HOST, PX4_BIND_PORT,
  PX4_SIM_HOSTNAME, PX4_GCS_BIND_HOST, PX4_GCS_BIND_PORT,
  PX4_VIEWER_DURATION, PX4_VIEWER_LOG_DIR
EOF
}

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd -- "$script_dir/.." && pwd -P)
python_bin=${MUJOCO_UAV_PYTHON:-/home/ustc/miniconda3/envs/mujoco_uav/bin/python}
autopilot_dir=${PX4_AUTOPILOT_DIR:-/home/ustc/PX4-Autopilot}
bind_host=${PX4_BIND_HOST:-127.0.0.1}
bind_port=${PX4_BIND_PORT:-4560}
simulator_host=${PX4_SIM_HOSTNAME:-127.0.0.1}
gcs_bind_host=${PX4_GCS_BIND_HOST:-127.0.0.1}
gcs_bind_port=${PX4_GCS_BIND_PORT:-14550}
duration=${PX4_VIEWER_DURATION:-300}
log_dir=${PX4_VIEWER_LOG_DIR:-$repo_root/results/px4/viewer}
parameter_file=$repo_root/configs/px4/cf2x_scaled_stablesoft_passive.params

if (($# == 0)) || [[ ${1:-} == --help || ${1:-} == -h ]]; then
  usage
  exit 0
fi

mode=$1
shift
dry_run=false
extra=()
while (($#)); do
  case $1 in
    --dry-run) dry_run=true; shift ;;
    --) shift; extra=("$@"); break ;;
    *) printf 'error: unexpected launcher argument: %s (use -- before runner arguments)\n' "$1" >&2; exit 2 ;;
  esac
done

case $mode in
  mouse|tendon|keyboard-force|joint) ;;
  *) printf 'error: unknown mode: %s\n' "$mode" >&2; usage >&2; exit 2 ;;
esac

[[ -x $python_bin ]] || { printf 'error: Python is not executable: %s\n' "$python_bin" >&2; exit 2; }
[[ -d $autopilot_dir ]] || { printf 'error: PX4 checkout is not a directory: %s\n' "$autopilot_dir" >&2; exit 2; }
[[ -f $parameter_file ]] || { printf 'error: PX4 parameter file is missing: %s\n' "$parameter_file" >&2; exit 2; }
for port_name in bind_port gcs_bind_port; do
  port_value=${!port_name}
  [[ $port_value =~ ^[0-9]+$ ]] && ((port_value >= 1 && port_value <= 65535)) || {
    printf 'error: %s must be an integer in [1, 65535]: %s\n' "$port_name" "$port_value" >&2
    exit 2
  }
done

common=(
  "$python_bin" "$repo_root/scripts/run_px4_sitl.py"
  --mode local --bind-host "$bind_host" --bind-port "$bind_port"
  --launch-owned-px4 --autopilot-dir "$autopilot_dir"
  --simulator-host "$simulator_host"
  --parameter-file "$parameter_file"
  --gcs-bind-host "$gcs_bind_host" --gcs-bind-port "$gcs_bind_port"
  --auto-flight --viewer --plant active_arm
  --plant-step-freq 1200 --sim-freq 2400 --control-freq 200
  --initial-position 0,0,1 --target-position 0,0,1
  --duration "$duration" --log-dir "$log_dir"
  --interaction-target-policy arm_segments
)

preset=()
case $mode in
  mouse)
    preset=(--interaction-mode mouse_force --force-step-mn 0.5 --max-force-mn 1.0)
    ;;
  tendon)
    preset=(
      --interaction-mode tendon_modal
      --tendon-acceleration-step 0.005 --tendon-acceleration-limit 0.04
    )
    ;;
  keyboard-force)
    preset=(--interaction-mode keyboard_force --force-step-mn 0.5 --max-force-mn 1.0)
    ;;
  joint)
    preset=(
      --interaction-mode joint_torque
      --joint-torque-step-unm 10 --joint-torque-limit-unm 50
    )
    ;;
esac

command=("${common[@]}" "${preset[@]}" "${extra[@]}")
if $dry_run; then
  printf 'cd %q && ' "$repo_root"
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

cd -- "$repo_root"
exec "${command[@]}"
