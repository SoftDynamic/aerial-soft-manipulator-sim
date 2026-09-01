# PX4 SITL operations and compatibility

## Verified compatibility target

The real-integration target is PX4 tag `v1.17.0`, commit
`d6f12ad1c4f70ad3230afd7d86e971421e02fef4`. The commit, rather than a moving
branch or container tag, is the compatibility authority for reproducible
tests.

Compatibility evidence:

- PX4 release: <https://github.com/PX4/PX4-Autopilot/releases/tag/v1.17.0>
- exact source: <https://github.com/PX4/PX4-Autopilot/tree/v1.17.0>
- official simulation topology: <https://docs.px4.io/main/en/simulation/>
- MAVLink `HIL_ACTUATOR_CONTROLS`: <https://mavlink.io/en/messages/common.html#HIL_ACTUATOR_CONTROLS>
- v1.17 simulator implementation: <https://github.com/PX4/PX4-Autopilot/blob/v1.17.0/src/modules/simulation/simulator_mavlink/SimulatorMavlink.cpp>

The official topology says PX4's simulation module connects to the
simulator's TCP listener, conventionally port 4560. Therefore set the MuJoCo
transport `tcp_role="listen"` for standard SITL. `tcp_role="connect"` exists
for a deliberate reverse proxy/custom peer, not as the standard-PX4 default.
PX4 and the simulator may be on different hosts. Configure PX4's simulator
host to the reachable MuJoCo host and arrange routing/firewall separately.

The MAVLink common definition specifies 16 normalized controls in `[-1, 1]`,
with assignment depending on simulated hardware, and arming in `mode`. PX4
v1.17's simulator source fills `HIL_ACTUATOR_CONTROLS.controls` from actuator
outputs only while armed and sets the arming mode bit. Neither source describes
these values as microsecond PWM. This is the basis for the explicit
normalized→PWM calibration layer.

## Environment check

Nothing is downloaded or installed by the checker:

```bash
export PX4_AUTOPILOT_DIR=/absolute/path/to/PX4-Autopilot
export PX4_BINARY=/absolute/path/to/px4
export PX4_BIND_HOST=127.0.0.1
export PX4_BIND_PORT=4560
export PX4_SIM_HOSTNAME=127.0.0.1
/home/ustc/miniconda3/envs/mujoco_uav/bin/python scripts/check_px4_environment.py --mode local
```

`PX4_BIND_HOST` and `PX4_BIND_PORT` are where this simulator listens.
`PX4_SIM_HOSTNAME` is the MuJoCo host address configured on the PX4 side; for a
remote PX4 it is usually not the bind wildcard. Use a specific bind interface
where possible; exposing `0.0.0.0` requires an explicit network-security
decision. The checker has separate `offline`, `local`, and `remote` modes.
Remote mode never inspects or requires a local PX4 checkout or binary.

Install the Python transport dependency only in the chosen environment:

```bash
/home/ustc/miniconda3/envs/mujoco_uav/bin/python -m pip install -e '.[px4]'
```

This command is intentionally not run by the project or test suite.

## Standalone runner

The runner never starts PX4 unless `--launch-owned-px4` is present. A complete
local rigid flight is:

```bash
/home/ustc/miniconda3/envs/mujoco_uav/bin/python scripts/run_px4_sitl.py \
  --mode local --bind-host 127.0.0.1 --bind-port 4560 \
  --launch-owned-px4 --autopilot-dir /home/ustc/PX4-Autopilot \
  --simulator-host 127.0.0.1 \
  --parameter-file configs/px4/cf2x_27g.params \
  --gcs-bind-host 127.0.0.1 --gcs-bind-port 14550 \
  --auto-flight --plant rigid --sim-freq 2400 --control-freq 200 \
  --initial-position 0,0,1 --target-position 0,0,1 \
  --duration 30 --log-dir results/px4
```

For an externally managed remote PX4, omit `--launch-owned-px4` and
`--autopilot-dir`, select `--mode remote`, and bind to the explicitly chosen
reachable interface. Active-arm runs add `--plant active_arm`, use the passive
arm parameter file, and retain `--plant-step-freq 1200`; tendon profiles are
`disabled`, `x_pos`, `x_neg`, `diagonal`, and `reverse`.

`--disconnect-time SECONDS` is a deliberate fault injection and is accepted
only with an explicitly owned local PX4. It closes that child at the specified
simulation time, expects a transport disconnect, applies safe rotors, and
stops before another physics step. It is not a general process-kill feature.

## Local source launch

After independently preparing the official PX4 checkout at the verified tag,
the official custom-simulator target is:

```bash
cd "$PX4_AUTOPILOT_DIR"
make px4_sitl none_iris
```

Start the MuJoCo TCP listener before PX4. The helper
`px4_source_launch_command()` only constructs this command. The optional
`Px4OwnedProcessLauncher` terminates only the process object it created; the
backend itself never starts, discovers, or kills PX4. An externally started or
remote PX4 remains entirely externally owned.

## Remote and container topology

For a remote PX4 machine, bind MuJoCo to an explicitly chosen reachable
interface, set PX4's simulation host to that MuJoCo machine, and use the same
port at both ends. Latency changes wall-clock throughput, never simulation
`dt`. Do not open a firewall from project code.

For Docker, pin the image/tag or build from the verified source. The current
official prebuilt PX4 images focus on SIH/Gazebo and document normal UDP ports;
they do not by themselves configure this custom simulator TCP flow. Use host
networking or an explicit TCP route only after checking which side owns the
listener. Never use `latest` in a recorded validation.

## Frozen real-flight gates (P8-P10)

These criteria are frozen before the first real PX4 experiment. A run is
invalid if the PX4 tag/commit, parameter set, model hash, simulator frequency,
or actuator calibration is unrecorded.

P8, rigid 27 g CF2X, no arm, requires: takeoff, 30 s position hold, landing,
and disconnect stop; 30 s RMS position error ≤0.10 m, peak error ≤0.25 m,
RMS roll/pitch ≤5 deg, peak tilt ≤15 deg, RPM and PWM always within configured
bounds, nominal saturation ≤5% of samples, zero nominal transport timeouts,
no NaN/Inf or MuJoCo warnings, healthy estimator/arming checks, and identical
physics-step traces on deterministic Fake replay. A 2400/4800 Hz comparison
must agree within 2 cm position and 2 deg attitude over the fixed evaluation
window. PX4 parameters must describe this plant; MuJoCo `KF`, `KM`, mass, and
inertia are not tuning knobs for passing P8.

P9 adds the passive arm only after P8. It requires the same safety bounds,
bounded natural arm motion, visible UAV reaction and recovery, no contact,
zero actuator discontinuities, and 2400/4800 Hz endpoint agreement within
2 cm UAV position and 0.5 mm arm-tip position. Energy/work diagnostics and all
saturation/timeout counts are recorded.

P10 then enables the already accepted two-dimensional tendon-modal controller.
Zero tendon input must reproduce P9. Single-axis, diagonal, and reversed small
bends must keep the tendon-modal reference within 300 µm, tension within the
existing 0.01 N limit, PWM/RPM within calibration, and UAV peak tilt ≤15 deg.
The 2400/4800 Hz endpoint agreement is ≤2 cm UAV position and ≤20 µm tendon
modal coordinate. Disconnect/reset must command safe rotors and zero/accepted
tendon lifecycle behavior. A 30 minute wall-clock soak must show no increasing
thread, socket, process, or file-descriptor count.

Real P8-P10 tests are marked integration and must skip with the exact missing
environment reason. They must never be part of ordinary offline test startup.

## Recorded real-PX4 evidence

The following runs used the exact compatibility commit, loopback TCP, ideal
sensors, the repository parameter files, and zero motor lag. JSON logs contain
the full configuration, command/environment allowlist, bounded PX4 log tail,
50 Hz traces, estimator/link health, mechanical energy, rotor/tendon power and
work, landing state, and exit reason.

| Gate/run | Main evidence |
| --- | --- |
| P8 rigid, 30 s, 2400 Hz | position RMSE 0.0520 m; peak 0.1672 m; peak tilt 1.04 deg; zero saturation/timeouts; landed and disarmed |
| P8 rigid, 30 s, 4800 Hz | position RMSE 0.0573 m; peak 0.1676 m; peak tilt 1.10 deg; zero saturation/timeouts; landed and disarmed |
| P8 +X disturbance | 0.2 m/s velocity impulse; recovery 1.84 s; peak post-impulse error 0.0795 m; estimator flags 831 |
| P9 passive, +X/2400 Hz | 30 s RMSE 0.0524 m; recovery 1.94 s; arm-tip motion 3.91 mm; no contact or saturation |
| P9 passive, -X/4800 Hz | 30 s RMSE 0.0520 m; recovery 1.86 s; arm-tip motion 4.03 mm; no contact or saturation |
| P10 zero tendon | exact zero reference, actual and tension; normal flight/landing |
| P10 +X/-X/diagonal | 100 µm commands; final scalar errors 0.03/0.01 µm and diagonal error 0.06 µm; peak tension at most 3.12 mN |
| P10 direction reversal, 30 s | +100 to -100 µm; final actual -99.97 µm; RMSE 0.0495 m; no link/contact/saturation fault |
| P10 2400/4800 Hz | flight endpoint difference 6.7 mm; tendon endpoint difference 0.19 µm |
| P10 active soak | 1951.54 s wall / 2100 s simulation; 2,520,000 plant steps; RMSE 0.0111 m; peak yaw 4.12 deg; zero link/contact/saturation faults; landed/disarmed |
| Owned disconnect | injected at 6.0 s; outcome `disconnected`; one disconnect; physics stopped at exactly 7200 1200-Hz steps |

The P9 energy run recorded initial/final flight mechanical energy
0.29268/0.29212 J, peak absolute rotor rigid-body power 0.0977 W, and positive
rotor work 0.0904 J. The P10 30 s run recorded initial/final energy
0.29268/0.29328 J, peak rotor/tendon power 0.0977 W/11.6 µW, positive rotor
work 0.0924 J, and positive tendon work 1.23 µJ. These are signed mechanical
plant ledgers, not electrical battery models.

Within the frozen 300 µm calibrated reference domain, the real controller's
largest observed tension was 8.10 mN, so the 10 mN allocator bound did not
activate in nominal flight. Dedicated offline allocator/controller fault tests
exercise and release the exact 10 mN saturation; a nominal run is not mislabeled
as a saturation event.

The active soak held one owned lifecycle for 32 min 31 s. At startup, 8 min,
17 min, 25 min, and 31 min, the runner remained at 21 threads/8 file
descriptors and PX4 at 29 threads/14 file descriptors; the same single TCP
connection remained established. Natural completion removed both processes
and released port 4560. The 50 Hz in-memory trace is intentionally detailed
but grew linearly to 105,063 samples and a 122.5 MB JSON file. This is not a
thread/socket/FD leak, but streaming or downsampling is recommended for
substantially longer campaigns.

One host-specific visualization limitation remains: the available `DISPLAY=:1`
completed a viewer flight and landing but its X server raised `GLXBadContext`
while destroying the context, and the same failure reproduces with
`BaseAviary` alone. Headless exits are clean. Treat viewer natural-exit
validation on this display as not passed until the GLX environment is repaired.

## P11 interactive arm operation

The normal entry point is:

```bash
./scripts/run_px4_viewer.sh mouse
```

The launcher resolves the repository from its own path, starts an owned local
PX4, uses the active arm at 2400/1200/200 Hz, holds `(0,0,1)`, enables the
overlay, and lands/disarms on normal exit. It is a Bash-array wrapper around
the full Python CLI; it does not install software, alter firewall rules, or
hide the printed command from `--dry-run`.

```bash
./scripts/run_px4_viewer.sh tendon
./scripts/run_px4_viewer.sh keyboard-force
./scripts/run_px4_viewer.sh joint
./scripts/run_px4_viewer.sh mouse --dry-run
./scripts/run_px4_viewer.sh mouse -- --duration 600
```

| Preset | Interaction | Default bound |
| --- | --- | --- |
| `mouse` | pick and drag any manifest arm segment | 1 mN, unverified all-segment debug envelope |
| `keyboard-force` | XY force pulses on the currently selected segment | 1 mN |
| `tendon` | accepted Gate C2 two-mode controller | 0.04 m/s²; 300 µm L1; <10 mN tension |
| `joint` | internal ball-joint `qfrc_applied` diagnostic | 10 µN·m step; 50 µN·m limit |

The earlier **terminal-tip-only** 5 mN/0.2 s envelope remains valid only for
its tested target and directions. It is not automatically generalized to all
segments. The launcher therefore starts arbitrary-segment force modes at a
conservative 1 mN. Values above 5 mN still require
`--allow-unverified-force`; joint values above 50 µN·m separately require
`--allow-unverified-joint-torque`. Both opt-ins are boundary experiments, not
declarations of safety. Release is always immediate zero.

### Launcher environment

| Variable | Default |
| --- | --- |
| `MUJOCO_UAV_PYTHON` | `/home/ustc/miniconda3/envs/mujoco_uav/bin/python` |
| `PX4_AUTOPILOT_DIR` | `/home/ustc/PX4-Autopilot` |
| `PX4_BIND_HOST` / `PX4_BIND_PORT` | `127.0.0.1` / `4560` |
| `PX4_SIM_HOSTNAME` | `127.0.0.1` |
| `PX4_GCS_BIND_HOST` / `PX4_GCS_BIND_PORT` | `127.0.0.1` / `14550` |
| `PX4_VIEWER_DURATION` | `300` seconds |
| `PX4_VIEWER_LOG_DIR` | `results/px4/viewer` |

The wrapper checks the Python executable, PX4 directory, parameter file, and
port ranges before launch. Extra runner arguments after `--` are appended last
and can intentionally override a preset.

### Local owned PX4 viewer

```bash
/home/ustc/miniconda3/envs/mujoco_uav/bin/python scripts/run_px4_sitl.py \
  --mode local --bind-host 127.0.0.1 --bind-port 4560 \
  --launch-owned-px4 --autopilot-dir /home/ustc/PX4-Autopilot \
  --simulator-host 127.0.0.1 \
  --parameter-file configs/px4/cf2x_scaled_stablesoft_passive.params \
  --gcs-bind-host 127.0.0.1 --gcs-bind-port 14550 --auto-flight \
  --plant active_arm --plant-step-freq 1200 \
  --sim-freq 2400 --control-freq 200 \
  --initial-position 0,0,1 --target-position 0,0,1 --duration 60 \
  --viewer --interaction-mode mouse_force \
  --force-step-mn 2 --max-force-mn 5 --force-release immediate \
  --log-dir results/px4/p11
```

Use `--interaction-target-policy arm_segments` (the viewer default) for all
manifest segments. Passing `--interaction-target-body` or
`--interaction-target-site`, or choosing policy `single`, retains the original
single-target compatibility mode. `joint_torque` is supported by either arm
plant; tendon mode requires the active arm.

### Externally managed local PX4

Start the compatible PX4 separately with `PX4_SIM_HOSTNAME=127.0.0.1`, then
run the same command without `--launch-owned-px4` and `--autopilot-dir`:

```bash
/home/ustc/miniconda3/envs/mujoco_uav/bin/python scripts/run_px4_sitl.py \
  --mode local --bind-host 127.0.0.1 --bind-port 4560 \
  --parameter-file configs/px4/cf2x_scaled_stablesoft_passive.params \
  --gcs-bind-host 127.0.0.1 --gcs-bind-port 14550 --auto-flight \
  --plant passive_arm --sim-freq 2400 --control-freq 200 \
  --duration 60 --viewer --interaction-mode keyboard_force \
  --force-step-mn 2 --max-force-mn 5 --log-dir results/px4/p11
```

### Remote PX4

Replace `192.0.2.10` with the MuJoCo host interface reachable by PX4 and set
the remote PX4 simulator hostname to that address. Network/firewall changes
remain external to this project.

```bash
/home/ustc/miniconda3/envs/mujoco_uav/bin/python scripts/run_px4_sitl.py \
  --mode remote --bind-host 192.0.2.10 --bind-port 4560 \
  --gcs-bind-host 192.0.2.10 --gcs-bind-port 14550 --auto-flight \
  --plant active_arm --plant-step-freq 1200 \
  --sim-freq 2400 --control-freq 200 --duration 60 \
  --viewer --interaction-mode tendon_modal --log-dir results/px4/p11
```

### Viewer controls

| Input | Effect |
| --- | --- |
| double-click any arm segment | select it; in joint mode select its parent joint |
| double-click background | cancel body selection after the current drag releases |
| double-click UAV/world/other body | reject it and show the reason in the overlay |
| Ctrl + right drag | lock the segment and exact pick point; apply bounded point force until release |
| W / S, force modes | one-step `+X / -X` pulse |
| A / D, force modes | one-step `+Y / -Y` pulse |
| 1 / 2 / 3, joint mode | choose ball-joint rotational DOF 0 / 1 / 2 |
| W / S, joint mode | one-step positive / negative diagnostic torque |
| A / D, joint mode | previous / next internal arm joint |
| Up / Down | change force or diagnostic-torque magnitude |
| Space | pause/resume without advancing simulation time or sending frames |
| `.` | exactly one plant step while paused |
| F8 / F9 | immediately zero force/tendon/joint input |
| Backspace | latch interaction emergency stop |
| Esc or L | zero interaction, request normal landing/disarm, then exit |

The passive viewer callback reports key presses but not releases. Therefore a
keyboard direction is deliberately a single plant-step pulse; operating-system
key repeat refreshes pulses while the key is held. Stopping repeat cannot leave
a persistent wrench. Mouse release is read directly from the native perturb
state and zeros on the next accepted plant step. The drag target cannot switch
until release. The overlay shows target body ID/name, associated site/joint,
exact world point, body COM, force and generated moment, selection rejection,
joint DOF/velocity/torque, PX4 state, tendon state, actuator channels,
saturation, clipping, and scheduler frequencies. Joint rows are labelled
`DIAGNOSTIC JOINT TORQUE (qfrc_applied)`; this mode is not tendon drive and must not be cited as real
soft-arm actuator performance evidence.

### Deterministic headless replay

An event script uses simulation time only:

```bash
/home/ustc/miniconda3/envs/mujoco_uav/bin/python scripts/run_px4_sitl.py \
  --mode local --bind-host 127.0.0.1 --bind-port 4560 \
  --launch-owned-px4 --autopilot-dir /home/ustc/PX4-Autopilot \
  --simulator-host 127.0.0.1 \
  --parameter-file configs/px4/cf2x_scaled_stablesoft_passive.params \
  --gcs-bind-host 127.0.0.1 --gcs-bind-port 14550 --auto-flight \
  --plant passive_arm --sim-freq 2400 --control-freq 200 --duration 12 \
  --interaction-mode replay \
  --interaction-script configs/px4/interaction_force_5mn_x.json \
  --log-dir results/px4/p11
```

Schema-3 output JSON contains a top-level `events` list reconstructed from the
accepted per-step commands. Pass that output log itself to
`--interaction-script` to replay the exact interaction command stream. The log
also contains parameter/event/model hashes, actuator calibration, frequencies,
50 Hz state trace, metrics, PX4 log tail, and exit state.

### Verified P11 evidence (2026-09-01)

All runs below used PX4 commit
`d6f12ad1c4f70ad3230afd7d86e971421e02fef4`, loopback transport, ideal
sensors, `cf2x_scaled_stablesoft_passive.params`, 200 Hz PX4 control, immediate
release, automatic landing, and zero motor lag.

| Plant / event | Hz | Peak tilt | Min flight z | Tip motion | Recovery entry | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| passive, +X 2 mN, 0.2 s | 2400 | 1.20° | 0.832 m | 10.06 mm | 0.00 s | pass |
| passive, ±X/±Y 5 mN, 0.2 s each | 2400 | 3.16° | 0.835 m | 24.63 mm | 0.00 s after final release | pass |
| passive, +X 10 mN, 0.2 s | 2400 | 5.78° | 0.833 m | 44.45 mm | 0.235 s | boundary probe passed |
| passive, +X 20 mN, 0.2 s | 2400 | 9.86° | 0.833 m | 70.02 mm | 1.29 s | boundary probe passed |
| active, +X 5 mN, 0.2 s | 2400 | 3.03° | 0.833 m | 24.40 mm | 0.00 s | pass |
| active, diagonal tendon accel | 2400 | 0.11° | 0.834 m | 0.90 mm | 0.00 s | pass |
| passive, +X 5 mN, 0.2 s | 4800 | 3.11° | 0.835 m | 24.55 mm | 0.00 s | pass |

Every row completed landing/disarm with zero timeout, disconnect, contact,
rotor saturation, tendon fault, NaN, or MuJoCo warning. The tendon run ended at
reference `(50, 50) µm`, actual approximately `(49.85, 49.93) µm`, with
2.84 mN peak tension. The frozen default manual envelope is **5 mN in the XY
plane for 0.2 s**: all four directions were exercised. The 10/20 mN results
are only single-direction boundary probes and do not enlarge that envelope.
No 0.5 s real-PX4 force matrix has yet been executed.

For the fixed 5 mN/+X run, 2400/4800 Hz endpoint differences were 1.92 mm UAV
position, 0.046° attitude, and 0.082 mm relative tip position. Maximum sampled
trajectory differences were 20.058 mm, 0.114°, and 0.198 mm respectively. The
UAV maximum is 0.058 mm above the suggested 2 cm whole-trajectory threshold,
so that optional strict trajectory sub-gate remains open even though the
frozen P9 endpoint criteria pass.

### P11.1 all-segment and joint evidence (2026-09-01)

The active-arm real-PX4 matrix in
`configs/px4/interaction_arm_segments_1mn_matrix.json` waits six seconds for
flight settling, then applies 1 mN for 0.1 s at each segment body origin. The
root (`B00`), middle (`B04`), and tip (`B08`) additionally receive `-X`, `+Y`,
and `-Y`. The table shows each segment's `+X` event window; points are world
ENU at event start. Every application produced a 4.444 µN·m moment about its
body COM.

| Segment | Point ENU m | Peak tilt | Min z | Max position error | recovery entry |
| --- | --- | ---: | ---: | ---: | ---: |
| B00 | `(0.0066,-0.0030,0.9410)` | 0.123° | 0.9560 m | 0.0446 m | 0 s |
| B01 | `(0.0054,-0.0027,0.9371)` | 0.144° | 0.9611 m | 0.0394 m | 0 s |
| B02 | `(0.0038,-0.0036,0.9326)` | 0.105° | 0.9654 m | 0.0350 m | 0 s |
| B03 | `(0.0018,-0.0051,0.9274)` | 0.069° | 0.9691 m | 0.0314 m | 0 s |
| B04 | `(-0.0001,-0.0070,0.9216)` | 0.044° | 0.9722 m | 0.0287 m | 0 s |
| B05 | `(-0.0014,-0.0089,0.9155)` | 0.121° | 0.9749 m | 0.0266 m | 0 s |
| B06 | `(-0.0016,-0.0101,0.9090)` | 0.153° | 0.9773 m | 0.0248 m | 0 s |
| B07 | `(-0.0010,-0.0099,0.9022)` | 0.167° | 0.9794 m | 0.0228 m | 0 s |
| B08 | `(0.0001,-0.0086,0.8951)` | 0.179° | 0.9812 m | 0.0206 m | 0 s |

Across the nine extra root/middle/tip direction events, peak event-window tilt
was 0.250°, minimum altitude 0.9827 m, maximum position error 0.0186 m, and
every release was already inside the 0.10 m recovery entry bound. The complete
14 s run recorded 18 releases, no clipping, timeout, disconnect, contact,
rotor saturation, tendon fault, NaN, or MuJoCo warning, then landed, disarmed,
and exited 0. Evidence log:
`results/px4/p11_1/px4-run-20260901T033951.567669Z.json`.

This freezes the **all-segment debug probe envelope at 1 mN for 0.1 s**. The
mouse preset uses the same 1 mN magnitude cap, but a human remains responsible
for releasing promptly; no result here expands the older 5 mN tip-only
envelope or validates a continuous 1 mN hold.

For `joint_torque`, the compiled joint stiffness is 0.003086 N·m/rad. A fixed
root 10 µN·m, 0.1 s B04-axis-0 probe produced 0.125 mm maximum tip motion and
0.196 rad/s maximum joint speed, settled to 0.00057 mm tip error, and ended
with zero `qfrc_applied`, contact, and warnings. The free-UAV real-PX4 probe
then applied ±10 µN·m for 0.2 s on B04 DOF 15. The signed release displacement
reversed from approximately `(-34,+77,+1)` µm to `(+49,-99,+0)` µm; UAV RPY
increments also reversed in roll/pitch. Peak relative tip motion was 0.152 mm,
peak tilt 0.086°, recovery entry 0 s, with zero clipping, timeout, disconnect,
contact, or saturation, followed by landing/disarm and exit 0. Evidence log:
`results/px4/p11_1/px4-run-20260901T034014.983863Z.json`. These are diagnostic
structural responses, not tendon-controller or real-actuator evidence.

### GLX revalidation

The project does not catch or suppress X11/GLX errors. First run a minimal
viewer in the intended local graphical login, then the owned-PX4 viewer command
above, interact, press Esc, and verify landing plus natural process exit:

```bash
DISPLAY=:1 /home/ustc/miniconda3/envs/mujoco_uav/bin/python \
  -m multi_drone_mujoco.examples.scaled_stablesoft_passive_demo
```

`GLXBadContext/GLXBadDrawable` was previously reproduced during context
destruction on `DISPLAY=:1` without PX4. On 2026-09-01 the P11 owned-PX4
`mouse_force` viewer command above was rerun for a bounded 3 s flight: it
created the window, flew, landed, disarmed, closed the owned process, and
exited with code 0. The earlier BadContext/BadDrawable did not recur; Xlib only
reported that the `NV-GLX` extension was absent. The viewer lifecycle sub-gate
therefore passes on that run. A human mouse selection/drag/release was not
performed in this non-interactive validation session, so the final manual
input/overlay observation remains open and must not be replaced by headless
results. P11.1 repeated owned-PX4 lifecycle smoke tests through the short
launcher for both `mouse` (2 s) and `joint` (1 s); both created the viewer,
landed/disarmed, cleaned the owned PX4 process, and exited 0. They do not close
the manual gate: a person must still select/drag/release root, middle, and tip,
then select two joints and visually confirm the overlay and zeroing.
