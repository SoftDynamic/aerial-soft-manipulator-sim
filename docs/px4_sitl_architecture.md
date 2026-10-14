# PX4 SITL integration architecture

## Status and boundary

PX4 is an optional controller backend outside `BaseAviary`. The existing
RPM API remains available for automatic callers; the unified runner caches
RPM-derived thrust then calls `env.step()` while MuJoCo owns
vehicle and arm physics. Importing or running the core environments must not
import `pymavlink`, locate a PX4 checkout, open a socket, or start a process.

The integration is split into these replaceable layers:

```text
MuJoCo truth -> ideal sensor adapter -> SimulatorTransport -> PX4 SITL
       ^                                      |
       |                                      v
 env.step() <- named site-actuator thrust cache <- RPM <- PWM <- normalized output

independent path: tendon-modal controller -> ideal tendon tension actuators
```

`multi_drone_mujoco.backends` owns typed PX4 contracts and lifecycle. The
unified CLI exposes UAV `manual|px4` and arm `manual|controller` independently.
Core model/environment imports do not start or require a PX4 process. The
retained PID is an internal dependency, not an additional CLI source.

## Data and actuator semantics

All real-time values cross a boundary in immutable records with explicit
units, simulation timestamps in microseconds, and monotonically increasing
sequence numbers. Bare four-element arrays are used only at the legacy
`env.step(rpm)` boundary.

The initial PX4 actuator chain is deliberately two separate calibration
models:

1. PX4 normalized motor controls (dimensionless, configured domain) are
   reordered and linearly scaled to PWM pulse-width semantics in microseconds.
2. PWM is mapped linearly to steady-state revolutions per minute. An optional
   first-order lag uses an exact exponential discrete update.
3. Cache thrust `KF * RPM**2` at the PX4 update boundary.
4. Before every MuJoCo advance, restore automatic thrust at named actuator
   addresses. Site gear produces axial thrust plus `KM * RPM**2` reaction
   torque; rotor positions produce roll/pitch torque. Manual groups use the
   same actuators directly through native sliders and are never restored.

Standard PX4 SITL `HIL_ACTUATOR_CONTROLS` values are normalized actuator
controls; they are **not physical microsecond PWM values**. PWM in this
integration is a configured actuator-model semantic, not a claim about the
MAVLink packet or a hardware output protocol. Hardware may instead expose PWM,
DShot, or another driver-specific representation. No hidden square-root map is
used between normalized output, PWM, and RPM.

Arming, disarming, armed idle, clipping policy, channel order, CW/CCW metadata,
PWM endpoints, RPM endpoints, and motor time constant are configuration. The
CF2X default keeps simulator rotor order 0..3 (front-left, rear-left,
rear-right, front-right) and directions `(-1, +1, -1, +1)`, matching the yaw
signs preserved from the Commit 1 propulsion model.

## Frames and sensor truth

MuJoCo state is treated as an ENU world with an FLU body (`x` forward, `y`
left, `z` up). PX4-facing values use NED world and FRD body. The fixed proper
rotation matrices are:

```text
v_ned = [[0,1,0], [1,0,0], [0,0,-1]] v_enu
v_frd = [[1,0,0], [0,-1,0], [0,0,-1]] v_flu
R_ned_frd = R_ned_enu R_enu_flu R_flu_frd
```

Quaternions are scalar-first `(w, x, y, z)` and represent the active rotation
from PX4 FRD body coordinates to NED world coordinates. Angular rate and
specific force are expressed in FRD. Specific force is acceleration minus
gravity, rotated into the body: a level stationary vehicle reads about
`(0, 0, -9.81) m/s^2`, while ideal free fall reads zero. GPS is an ideal local
tangent-plane approximation about a configured WGS84 origin. Barometric
altitude is metres above the configured origin. The ideal magnetic field is a
configured NED vector. IMU, GPS, barometer, and magnetometer update rates are
configured independently; noise, bias, delay, dropout, and failure injection
remain separate future sensor decorators.

MuJoCo truth, simulated sensor values, and any PX4 state estimate are distinct
data products. The first integration sends only ideal simulated sensors and
does not report a PX4 estimate as truth.

## Time and lockstep

`SimulatorClock` is simulation time, never wall-clock time. One lockstep cycle
is:

```text
sample truth at t -> send SensorFrame(t) -> await matching actuator output(t)
-> convert normalized/PWM/RPM -> advance MuJoCo once -> sample t + dt
```

The wall-clock timeout only bounds I/O. Network latency may slow wall time but
does not alter simulation time or motor dynamics `dt`. Duplicate and stale
packets never advance physics, future packets are rejected diagnostically, and
a timeout/disconnect returns a disarmed safety command with `advance=False`.
Pause sends and advances nothing; a requested single step permits exactly one
matched cycle. Reset clears packet history and motor state.

## Transport and deployment topology

`SimulatorTransport` contains a bounded TCP peer handshake, sensor send,
bounded actuator receive, health, and close operations. The simulator sends a
MAVLink heartbeat after TCP acceptance but does not require a peer heartbeat
on the simulation link before the first sensor frame. `FakeTransport` supplies deterministic
offline tests. `SimulatorMavlinkTransport` lazily imports optional `pymavlink`
and can connect to a configured host/port. Neither the backend nor transport
owns an external PX4 process.

Supported topologies use the same backend interface:

- local: MuJoCo and externally launched PX4 on one host;
- container: PX4 in Docker, with an explicitly published/reachable simulator
  endpoint;
- remote: PX4 on another computer, with routing/firewall managed outside this
  package;
- future HITL: replace transport and actuator decoder while retaining sensor,
  PWM/RPM, and safety contracts as applicable.

Local process launch support is a separate utility with explicit ownership.
Only a process created by that launcher may be terminated by it. Configuration
uses local-only `PX4_AUTOPILOT_DIR`/`PX4_BINARY`, listener-side
`PX4_BIND_HOST`/`PX4_BIND_PORT`, and PX4-side `PX4_SIM_HOSTNAME`; no user
directory, host, port, PWM, or RPM value is embedded in business logic.

## Rigid CF2X motor mapping

PX4 and MuJoCo use different body frames and channel orders. PX4's 10016
geometry is FRD and orders motors FR, RL, FL, RR. The repository MJCF sites
and propulsion equations are FLU and order rotors FL, RL, RR, FR. Therefore
the adapter permutation (simulator rotor → PX4 zero-based channel) is
`(2, 1, 3, 0)`. Signs below are moments in the MuJoCo FLU body frame; `+1`
direction means clockwise and `-1` counter-clockwise when viewed from above.

| PX4 channel | MuJoCo rotor | Position | Direction | roll | pitch | yaw |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| Motor 1 | 3 | front-right | CW | - | - | + |
| Motor 2 | 1 | rear-left | CW | + | + | + |
| Motor 3 | 0 | front-left | CCW | + | - | - |
| Motor 4 | 2 | rear-right | CCW | - | + | - |

This follows directly from PX4 v1.17's effectiveness equation
`moment = ct * position.cross(axis) - ct * km * axis`, with upward FRD axis
`(0, 0, -1)`, and from the Commit 1 propulsion equations. The
single-channel unit probe checks the same six signs and permutation. The table
reports MuJoCo FLU moments. PX4's standard 10016 rotor labels cannot be copied
directly into this adapter: after the FRD/FLU moment conversion, the repository
parameter files use channel `KM` signs `(-,-,+,+)`. A real single-axis A/B
probe against PX4 v1.17 confirmed these signs; the opposite set produces
positive yaw feedback and divergence.

For the rigid 27 g plant, `MAX_RPM` is derived from the frozen 2.25
thrust-to-weight ratio and `HOVER_RPM` from weight balance. With the recorded
linear `1000..2000 us → 0..MAX_RPM` model, hover motor output is exactly
normalized `2/3` or `1666.6667 us`; no square root is applied in the PX4
adapter. PX4's internal thrust fraction is separately `1/2.25 = 4/9`.
`THR_MDL_FAC=1` explicitly linearizes the quadratic plant inside PX4, mapping
that internal hover thrust to `sqrt(4/9)=2/3` motor output. The repository
parameter file is `configs/px4/cf2x_27g.params`.

## Coupled scheduling and ownership

The standalone runner keeps the three controller clocks explicit. Rigid and
passive runs use a 200 Hz PX4/plant boundary. Active-arm runs use an exact
1200 Hz common plant scheduler, update PX4 at 200 Hz (every sixth tick), and
update the frozen tendon-modal controller at 240 Hz (every fifth tick) when
`arm_source=controller`; manual arm groups are left untouched.
MuJoCo may use 2400 or 4800 Hz substeps without changing either controller's
accepted timestep. PX4 owns only UAV flight stabilization; the tendon-modal
controller owns only the two bending modes, and MuJoCo remains the sole
coupled plant.

An optional local launcher is owned by the runner only when explicitly
requested. It terminates only that exact child process during cleanup;
it cannot discover or kill any external PX4. Timeout or disconnect returns
safe zero rotor output, clears selected automatic caches, and does not advance
another MuJoCo plant step; manual arm controls remain unchanged.

## Known first-stage limitations

- The PWM and steady RPM mappings are static approximations unless the
  configured first-order motor lag is enabled; they require calibration for a
  physical ESC/motor/propeller.
- The initial sensor model is ideal and uses a local GPS approximation.
- MAVLink dialect/message availability is checked at connection/use time.
- Packet timestamp matching assumes the simulator peer preserves simulation
  timestamps. Firmware-specific behavior must be verified by integration tests.
- PX4 airframe parameters must be created for the 27 g CF2X plant; physics
  constants and accepted Gate C2 tendon tuning must not be altered to fit a
  stock PX4 airframe.
- Real PX4 flight gates require an external compatible PX4 installation and
  optional `pymavlink`; ordinary unit tests require neither.

## P11 interaction layer

P11 is an optional layer above the runner, not a PX4 backend feature:

```text
viewer key/native perturb OR deterministic JSON event
                         |
                         v
              immutable InteractionCommand
          /              |                 \
 world-ENU point   diagnostic ball-      2-D modal acceleration
 force on arm body joint qfrc_applied     into frozen Gate C2 controller
          \              |                 /
                  MuJoCo coupled plant
                         |
               ideal sensors -> real PX4
```

`multi_drone_mujoco.backends.interaction` defines the GUI-independent command,
keyboard dead-man state, bounded spring/damper calculation, deterministic
simulation-time timeline, and MuJoCo point-force adapter. Every command records
simulation timestamp, monotonic sequence, mode, body/site/joint IDs and names,
DOF axis/index, frame, force or modal acceleration or diagnostic torque,
application point, body COM, generated moment, active/released state, source,
and clipping state.
Replay event selection depends only on simulation time. Wall time is used only
for socket bounds and paused viewer refresh.

The active and passive plants expose the same scaled StableSoft manifest. At
runtime the interaction catalog resolves every manifest segment and ball joint
against the compiled model, verifies that the bodies form one descendant
chain, associates sites by `site_bodyid`, verifies each joint's owning body and
ball type, and maps its three DOFs from `jnt_dofadr`. This produces stable
allowlists without putting `B0`--`B8` strings in viewer logic. The UAV free
joint, mount, rotors, world, ground, decorations, and all other bodies/joints
are outside those lists. Explicit `--interaction-target-body/site` arguments
retain the P11 single-target behavior; otherwise `arm_segments` permits every
manifest segment.

The viewer converts `MjvPerturb.localpos` through the selected body's current
pose to the exact world pick point. A drag locks both body and local point until
release. The point force becomes the world body wrench
`torque = (point - body_COM) × force`. Diagnostic joint effort is written only
to the selected internal ball-joint DOF in `qfrc_applied`; it never writes
`qpos` or `qvel`. Both arm `xfrc_applied` rows and all arm diagnostic DOFs are
cleared before each command and in `finally`, reset, e-stop, landing, viewer
close, timeout/disconnect, and runner close paths.

The optional `px4_interactive_viewer` module is imported only for interactive
viewer modes. It uses MuJoCo's native body selection and perturb intent, clears
all native viewer wrenches, filters through the catalog, clamps the force norm,
and returns the same command used by headless replay. Illegal selection is
reported but cannot become a command target. Joint mode maps segment selection
to its parent joint (the root segment intentionally has none) and also supports
keyboard cycling. The adapter never advances physics. The runner remains the
sole physics loop and owns viewer closure before environment/model closure.
Headless imports never require GLFW or a GLX context.

Keyboard tendon input calls only
`TendonCompatibleAccelerationController.update_bending_acceleration()` at the
existing 240 Hz schedule. The controller's 0.04 m/s² acceleration, 0.002 m/s
velocity, 300 µm L1 reference, and 10 mN allocator limits remain frozen.
PX4 owns the UAV; this controller owns the two bending modes; MuJoCo is the
only bidirectional coupling point.

`joint_torque` is explicitly a structural diagnostic. Its 10 µN·m initial
step and 50 µN·m hard default limit are conservative relative to the compiled
3.086 mN·m/rad joint stiffness; exceeding the limit requires the separate
`--allow-unverified-joint-torque` opt-in. It is neither tendon actuation nor
evidence of real soft-arm actuator performance.

Logs use schema 4 and include the command line, PX4 expected/actual commit,
parameter/event/model SHA-256 values, actuator calibration, all scheduler
frequencies, latest command/state snapshot, link/estimator state, recovery metrics, force and
tendon maxima, saturation, contact, landing, and exit state.

Commit 1 retired experimental pulse/impulse/disconnect injection and energy/work
reporting; manual/PX4 entry points and lockstep/control behavior remain. See
[the current debugging index](simulation_architecture.md).
