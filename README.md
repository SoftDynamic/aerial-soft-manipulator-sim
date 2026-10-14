# UAV + soft-arm MuJoCo simulation

CF2X UAV physics, a scaled StableSoft arm with four ideal tendon actuators,
native MuJoCo interaction, and an optional PX4 SITL backend.

The retained manual entry point is:

```bash
python -m multi_drone_mujoco.examples.manual_aerial_soft_arm_demo \
  --rpm-step 100 --tension-step 0.001 --initial-rpm hover
```

PX4 entry points remain `scripts/run_px4_sitl.py` and
`scripts/run_px4_viewer.sh`. Run from the repository root in the existing
simulation environment; use `PYTHONPATH=. python scripts/run_px4_sitl.py --help`
to inspect PX4 options. The shell launcher accepts `mouse`, `tendon`,
`keyboard-force`, and `joint`; PX4 must be prepared independently.

See [simulation architecture and debugging index](docs/simulation_architecture.md)
for model parameters, XML export, current call chains, and cleanup scope;
see [PX4 operations](docs/px4_sitl_operations.md) for the supported SITL workflow.

Commit 1 retains fixed arm attachment and RPM/external-force rotor physics.
Unified startup, independent control sources, native slider ownership, and a
two-DOF root joint remain future work. Historical RL task environments,
training demos, GPU batching, parameter scans, and report generators have been
retired. Packaging remains in `pyproject.toml` and `setup.py`.

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
