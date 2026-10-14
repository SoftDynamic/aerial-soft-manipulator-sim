# 1. 当前结构、调用链与启动

Commit 1 在 `/home/ustc/mujoco_UAV-refactor` 的 `refactor/uav-arm-sim` 上清理，基点为 `c6a2acbda58fd7095c88c4c7ccbfbe7c0475fd45`。原 `/home/ustc/mujoco_UAV` 的 X650 工作没有迁移。下文包内路径相对于 `multi_drone_mujoco/`。

```text
scripts/run_px4_sitl.py、scripts/run_px4_viewer.sh  # 保留 PX4 入口
multi_drone_mujoco/
  examples/manual_aerial_soft_arm_demo.py          # 保留手动循环、按键、鼠标、关闭流程
  models/parameters.py、mjcf_builder.py
  models/components/{scaled_stablesoft_arm,tendon_drive,passive_arm}.py
  envs/base_aviary.py
  control/ideal_tendon_tension.py、scaled_stablesoft_bending_control.py
  control/tendon_{compatible_acceleration_controller,bending_reference,
                 modal_coordinates,modal_pd,modal_static_feedforward}.py
  control/bounded_tendon_allocator.py、pid_control.py
  backends/{contracts,actuation,cf2x_px4,sensors,transport,lockstep,px4_sitl}.py
  backends/{px4_environment,px4_control,px4_runner,interaction,px4_interactive_viewer}.py
  wrappers/wind.py、utils/enums.py、utils/logger.py
  assets/cf2/、assets/stablesoft/、tests/
configs/px4/                                     # 原有参数与交互 replay
```

```text
手动 _make_env / PX4 create_default
  → scaled_stablesoft_baseline_spec → BaseAviary.__init__
  → build_aviary_mjcf → build_scaled_stablesoft_arm_mjcf
                      → build_ideal_tendon_tension_actuator_xml
  → MjModel.from_xml_string → resolve_ideal_tendon_actuator_bindings（编译后一次）

手动 run_viewer → _run_manual_loop → set_scaled_stablesoft_tension_command
                                 → BaseAviary.step → _physics → mj_step
                                 → viewer.sync；鼠标力 snapshot/追加/clear

PX4 main → Px4MujocoRunner.create_default → run
  → Px4OwnedProcessLauncher.start / transport listener、accept_peer
  → MujocoTruthSource.sample → IdealSensorAdapter.sample
  → PX4SITLBackend.step → LockstepCoordinator.exchange
  → send_sensor_frame → HIL_SENSOR/GPS/STATE_QUATERNION
  → receive_actuator → normalized→PWM→RPM → BaseAviary.step
  → Px4MavlinkFlightControl（Offboard、解锁、位置设定、降落）

软臂 _setup_tendon_controller（复用环境 bindings）
  → make_scaled_stablesoft_bending_controller → update_bending_acceleration
  → reference / modal PD / static feedforward / bounded allocation
  → set_scaled_stablesoft_tension_command → write_ideal_tendon_tension_command
```

在本 worktree 根目录、已有 `mujoco_uav` 环境下启动。手动默认暂停，Space 开始/暂停；两项 step 参数必填：

```bash
conda activate mujoco_uav
cd /home/ustc/mujoco_UAV-refactor
python -m multi_drone_mujoco.examples.manual_aerial_soft_arm_demo \
  --rpm-step 100 --tension-step 0.001 --initial-rpm hover
```

PX4 原生鼠标交互入口（需要已准备的 PX4 checkout、SITL 二进制及图形登录）：

```bash
MUJOCO_UAV_PYTHON=/home/ustc/miniconda3/envs/mujoco_uav/bin/python \
PX4_AUTOPILOT_DIR=/home/ustc/PX4-Autopilot \
  ./scripts/run_px4_viewer.sh mouse
```

直接 CLI 为 `PYTHONPATH=. python scripts/run_px4_sitl.py --help`；当前 UAV 是 CF2X，`--plant` 选择 `rigid|passive_arm|active_arm`。shell 保留 `mouse|tendon|keyboard-force|joint`，以及 `--dry-run`。`tendon` 使用原有反馈控制器，尚未提供独立 UAV/软臂来源选择或 native slider 所有权。

# 2. 调试与修改位置索引

| 功能 | 文件 | 关键函数/数据 | 修改位置或方法 |
|---|---|---|---|
| UAV 质量、惯量 | `models/parameters.py` | `DRONE_PARAMS[DroneModel.CF2X]` | `mass`、`ixx/iyy/izz`；builder 与环境仍共用同一参数表。 |
| 旋翼参数与方向 | `models/parameters.py`、`envs/base_aviary.py` | `kf/km/arm_length`、`_physics` | RPM²→推力/反扭矩和 world-frame `xfrc_applied` 路径未改变。 |
| 软臂长度、质量、刚度、阻尼 | `models/components/scaled_stablesoft_arm.py` | `ScaledStableSoftArmSpec`、`scaled_stablesoft_baseline_spec` | 默认 9 段、0.08 m、0.003 kg；共用基线为 stiffness=0.003086、target damping ratio=0.3、显式 damping=None。不要把基础 Spec 的零刚度默认值改成该基线。 |
| 挂载位置、内部关节 | `models/mjcf_builder.py`、`models/components/scaled_stablesoft_arm.py` | `mount_z`、`build_scaled_stablesoft_arm_mjcf` | 当前根段固定，后续段 ball joint；根部万向节尚未实现。 |
| 四条 tendon、理想张力 | `models/components/{scaled_stablesoft_arm,tendon_drive}.py` | `build_scaled_stablesoft_arm_mjcf`、`IdealTendonTensionSpec` | 4 条 spatial tendon；motor gear=-1，输入 N，现用上限 0.01 N。 |
| 执行器映射与写入 | `control/ideal_tendon_tension.py`、`envs/base_aviary.py` | `resolve_ideal_tendon_actuator_bindings`、`write_ideal_tendon_tension_command`、`set_scaled_stablesoft_tension_command` | 编译后绑定名称/顺序/传动/控制地址；动态命令继续验证维数、有限值和范围。 |
| 反馈控制 | `control/scaled_stablesoft_bending_control.py` 及模态控制模块 | `make_scaled_stablesoft_bending_controller`、`update_bending_acceleration` | 保留 240 Hz reference、PD、静态前馈、有界分配与数值故障行为。 |
| MJCF 构造/加载/导出 | `models/mjcf_builder.py`、`envs/base_aviary.py` | `build_aviary_mjcf(...).xml`、`__init__`、`save_generated_xml` | 模型组合只在 builder 检查；环境检查 physics/action/frequency。加载后 `env.save_generated_xml("debug_xml/uav_arm.xml")` 导出当前模型。 |
| 手动 viewer 与鼠标 | `examples/manual_aerial_soft_arm_demo.py` | `run_viewer`、`_run_manual_loop`、`_ManualPerturbationAviary`、`_closing_viewer` | 原循环保留；先计算旋翼外力再加鼠标 wrench，每个控制周期后清除缓存；保留关闭等待。 |
| PX4 viewer 与交互 | `backends/px4_interactive_viewer.py`、`interaction.py` | `Px4InteractionViewer`、`MujocoArmInteractionApplier` | 保留 body/site 选择、力脉冲、立即释放、暂停/单步；不让手动入口依赖 PX4 配置。 |
| PX4 sensor 输入 | `backends/sensors.py`、`transport.py` | `MujocoTruthSource.sample`、`IdealSensorAdapter.sample`、`send_sensor_frame` | ENU/FLU→NED/FRD、HIL 消息、时间戳/频率保持。 |
| PX4 motor 输出 | `backends/transport.py`、`actuation.py`、`px4_sitl.py` | `receive_actuator`、`Px4NormalizedOutputToPwm.convert`、`PwmToRotorSpeedModel.convert` | HIL 输出解码、通道顺序、解锁/未解锁和 PWM/RPM 换算保持。 |
| PX4 飞行/故障 | `backends/px4_control.py`、`px4_runner.py`、`lockstep.py` | `request_arm`、`_prepare_auto_flight`、`_land_and_disarm`、`exchange` | 保留启动/放行/降落/关闭，超时或断联时安全未解锁输出且不推进物理。 |
| 日志 | `backends/px4_runner.py` | `Px4RunMetrics`、`_write_log` | schema 4：运行/故障指标、交互事件、最后状态、最终 RPM/张力；移除无限增长的 50 Hz trace 与实验能量/功率报告。 |

XML 导出复用现有入口中的 `env`；单独调试可用：

```python
from multi_drone_mujoco.examples.manual_aerial_soft_arm_demo import _make_env
env = _make_env(sim_freq=2400, ctrl_freq=240, visual_mode="mesh")
try:
    env.save_generated_xml("debug_xml/uav_arm.xml")
finally:
    env.close()
```

# 3. 已完成清理、验证与后续边界

- **KEEP**：CF2X 参数、旋翼外力及物理步进、固定挂载/内部关节、四 tendon/理想张力、模态控制/静态前馈/有界分配、手动/PX4 入口、原生交互和必要网格。既有 passive_arm、3 g payload、wind、PID 实现仍有保留依赖；不为删文件而更改其物理行为。
- **MOVE/共享**：唯一 `scaled_stablesoft_baseline_spec()` 取代手动/被动示例与 PX4 的重复配置；被动示例删除前，将手动入口所需的关闭函数原样移入手动文件，没有新模块、转发私有 `_spec` 或 PX4 耦合。
- **DELETE**：RL/训练/竞速/编队环境、旧示例和 render 脚本、随机化/课程/障碍 wrappers、GPU vectorized、原始 StableSoft 副本、演示媒体/benchmark CSV；旧四通道控制器、未被生产调用的 watchdog；历史标定/扫描/响应拟合及批量报告 tests/helpers。保留 actuation 符号/功率恒等式测试，改用生产 binding/writer；固定臂 helper 只留这些测试实际调用的 fixture/姿态函数。
- **SIMPLIFY**：删除 `build_aviary_xml`，调用者直接取 builder 的 `.xml`；模型组合校验集中于 builder，并先于 pose reshape；runner 复用环境 bindings 和 `contracts.finite_tuple`；replay 解析及是否含 tendon 事件只处理一次。删除旧 pulse、速度冲量/主动断联注入参数与报告；保留 replay/鼠标/按键扰动和真实通信失败处理。悬停读环境 `HOVER_RPM`，不再回落到 CF2X 常量；缺模型不再生成空 XML hash；PID 要求显式环境。CLI 保留异常 traceback，不把失败包装成有效结果。
- **验证**：聚焦测试覆盖模型编译、执行器映射/符号、扰动叠加/释放、数值控制、sensor/motor conversion、fake transport/锁步。修改前后比较 manual、PX4 rigid/passive/active 四配置：XML hash、body/joint 和 actuator 名称/顺序/ID/控制地址/gear/range 完全一致；RPM `[12700,13900,15100,16300]` 的旋翼 wrench、控制器输出及四步 headless 状态按 `rtol=1e-12, atol=1e-14` 对比。CLI help 和 shell dry-run 不代表真实 GUI 或 live SITL 验证；本次没有执行这两项，也没有运行全量测试。
- **刻意延后**：手动循环仍在 example；PX4 viewer 不改为 generic viewer；所需生命周期/API 处理和 legacy 单目标交互仍有选定入口/测试依赖；`setup.py`/`pyproject.toml` 不迁移、不改安装依赖。X650 始终留在原 worktree，后续只能选择性迁移；近似 CAD/推进参数不能当实测值，全局 `ctrl[0::2]`/`ctrl[1::2]` 也不能直接混入 tendon 模型。

**Commit 2**：增加 `scripts/run_sim.py`、`uav_source=manual|px4` 与 `arm_source=manual|controller` 四种组合、原生 actuator sliders 和单一写入归属；新增旋翼 actuator 时防止与 Python 外力双重施力，再合并入口/viewer。

**Commit 3**：根部改为同点正交的两个被动 hinge，不提供独立轴向 twist；root joints 与内部 ball joints 分开命名/绑定。以上功能均未进入 Commit 1。
