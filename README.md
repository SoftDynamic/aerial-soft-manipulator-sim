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
PX4 selections connect to a prepared peer; `--launch-owned-px4` explicitly
starts an installed compatible PX4. `scripts/run_px4_viewer.sh` remains a
small convenience wrapper: `mouse`, `keyboard-force`, and `joint` select
px4/manual; `tendon` selects px4/controller.

See [simulation architecture and debugging index](docs/simulation_architecture.md)
for parameters, call chains, commands and actual verification;
see [PX4 operations](docs/px4_sitl_operations.md) for the supported SITL workflow.

The arm attachment remains fixed with eight internal ball joints. A passive
two-axis root attachment belongs to Commit 3. Historical RL task environments,
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
