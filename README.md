# UAV + soft-arm MuJoCo simulation

CF2X UAV physics, a scaled StableSoft arm with four ideal tendon actuators,
native MuJoCo interaction, and an optional PX4 SITL backend.

Run from the repository root in the existing `mujoco_uav` environment:

```bash
python scripts/run_sim.py --uav-source manual --arm-source manual
python scripts/run_sim.py --uav-source manual --arm-source controller
python scripts/run_sim.py --uav-source px4 --arm-source manual
python scripts/run_sim.py --uav-source px4 --arm-source controller
```

Manual/manual is the default. Open the native **Control** panel on the right:
four `r#_thrust_N` sliders command rotor thrust and four `t#_tension_N`
sliders command tendon tension. Manual UAV control has no stabilization.
Space pauses/resumes the manual loop; `.` steps, Backspace resets, Q closes.
Automatic groups hold their latest valid command before every physics substep;
manual groups retain the native slider values.

Use `--headless --duration 0.025 --export-xml /tmp/uav_arm.xml` for a short
run and XML export. Inspect other options with `python scripts/run_sim.py --help`.
Daily settings live in [configs/sim.yaml](configs/sim.yaml): robot physics,
simulation, owned PX4 endpoints/profile, and optional diagnostics. Explicit CLI
values override YAML, which overrides verified source defaults. Every run saves
its effective values and their origins under `simulation.log_dir`.

GUI duration defaults to unlimited, ending when its native window closes.
Headless runs default to five simulated seconds. `--duration` bounds either mode.
`python scripts/run_sim.py --uav-source px4` launches the configured local owned
SITL and uses the existing bootstrap/Offboard/arm/ramp/landing lifecycle. External
peers require explicit configuration with `launch_owned_px4: false` and
`auto_flight: false`; they are never automatically armed.

The redundant viewer wrapper and its low-level CLI flags have been removed.
The frozen stage validation recorded **未在规定窗口内满足悬停判据** in the
2 s undisturbed baseline, so it did not launch a pulse trial. See the
[measurement summary](docs/validation/px4_stage_summary.json) and
[raw Trial 1 log](docs/validation/px4_stage_trial1.json).
See [simulation architecture and debugging index](docs/simulation_architecture.md)
for parameters, call chains, commands and actual verification;
see [PX4 operations](docs/px4_sitl_operations.md) for the supported SITL workflow.

The arm attachment remains fixed with eight internal ball joints. The root attachment is unchanged in this stage. Historical RL task environments,
training demos, GPU batching, scans and report generators have been retired.
Packaging remains in `pyproject.toml` and `setup.py`.

## Citation

If you use this work, please cite:

```bibtex
@misc{tayal2026mujocodronesgym,
  title={MuJoCo-Drones-Gym: A GPU-Accelerated Multi-Drone Simulator for Control and Reinforcement Learning}, 
      author={Manan Tayal},
      year={2026},
      eprint={2606.08039},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2606.08039}, 
}
```

## Acknowledgements

- [gym-pybullet-drones](https://github.com/learnsyslab/gym-pybullet-drones) — inspiration for the environment API and task design
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) — Bitcraze Crazyflie 2.x MJCF model
- [Bitcraze](https://www.bitcraze.io/) — Crazyflie 2.x hardware platform and firmware parameters

## License

MIT
