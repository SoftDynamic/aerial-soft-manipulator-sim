# 1. 统一入口、控制来源与原生调试

Commit 2 在 `/home/ustc/mujoco_UAV-refactor` 的 `refactor/uav-arm-sim` 上实现，基点为 Commit 1 `9fd5bbe47f7e9ecc2d1adc6055e1ec110d4d7957`。原 `/home/ustc/mujoco_UAV` 的 X650 工作未迁移。以下包内路径相对于 `multi_drone_mujoco/`。

在已有 `mujoco_uav` 环境、本 worktree 根目录启动：

```bash
conda activate mujoco_uav
cd /home/ustc/mujoco_UAV-refactor
python scripts/run_sim.py --uav-source manual --arm-source manual
python scripts/run_sim.py --uav-source manual --arm-source controller
python scripts/run_sim.py --uav-source px4 --arm-source manual
python scripts/run_sim.py --uav-source px4 --arm-source controller
```

默认 manual/manual、active_arm、mesh、native viewer。manual UAV 不启动 PX4，也不自动稳定；四旋翼初始输入为零，需自己设置推力。手动循环初始暂停，Space 开始/暂停，`.` 单步，Backspace 重置模型和选中的反馈控制器，Q 关闭。右侧点击 **Control** 展开八个原生 actuator 滑块；暂停时也可修改。双击选择 UAV/软臂 body，Ctrl+右键拖动施加平移扰动，Ctrl+左键拖动施加转动扰动，普通拖动调整相机。

| 控件名称 | 含义/单位 | CF2X 基线范围 | 本次编译地址（实现按名称解析） |
|---|---|---|---|
| `r0_thrust_N`、`r1_thrust_N`、`r2_thrust_N`、`r3_thrust_N` | 单旋翼推力 N | 0–0.148989375 N | 0、1、2、3；对应原 prop0–3 |
| `t0_tension_N`、`t1_tension_N`、`t2_tension_N`、`t3_tension_N` | tendon 张力 N | 0–0.01 N | 4、5、6、7；对应 xp_yp、xp_yn、xn_yp、xn_yn |

旋翼输入是推力，不是 RPM。上限来自 `kf * MAX_RPM² = thrust2weight_ratio * mass * 9.81 / 4`。CF2X 的 site actuator gear 为 `[0,0,1,0,0,sign*km/kf]`，反扭矩符号依次为 `- + - +`；site 位置产生 roll/pitch 力矩。多 UAV 的 rotor 名称继续按 `r4_thrust_N` 等递增。软臂 motor gear=-1，沿原四条 spatial tendon 收缩；力学模型和固定挂载不变。

每组只由所选来源拥有。manual 值直接从 native `data.ctrl` 进入物理，循环不缓存或重写该组。PX4 在 200 Hz 计算最终 RPM，再转为推力；软臂在 240 Hz 使用现有模态反馈计算张力。两组自动命令只在各自更新时验证/缓存，**每一次 `mj_step` 前**按名称绑定的控制地址重新写入，包括没有控制器更新的中间子步。解除武装、失联、超时和关闭先清零自动缓存；不会清零手动组。viewer 后续同步改变自动滑块也无法覆盖下一物理步的缓存命令。软臂反馈独立于 mouse/tendon 等交互模式启用，它是已标定的模态弯曲控制，不是世界坐标末端位置控制。

PX4 两条命令连接已有本地兼容 peer（默认监听 127.0.0.1:4560）；需要自启进程和自动飞行时显式增加：

```bash
python scripts/run_sim.py --uav-source px4 --arm-source controller \
  --launch-owned-px4 --autopilot-dir /home/ustc/PX4-Autopilot \
  --simulator-host 127.0.0.1 \
  --parameter-file configs/px4/cf2x_scaled_stablesoft_passive.params \
  --gcs-bind-host 127.0.0.1 --gcs-bind-port 14550 --auto-flight --duration 5
```

`scripts/run_px4_viewer.sh` 仅保留 Bash convenience wrapper：mouse、keyboard-force、joint 明确选 px4/manual；tendon 明确选 px4/controller，原交互含义保留。默认 PX4 active-arm GUI 使用现有 mouse_force adapter，Space 暂停，`.` 单步，Esc/L 降落退出，Backspace e-stop。更多网络/生命周期参数见 `python scripts/run_sim.py --help` 和 [PX4 操作说明](px4_sitl_operations.md)。`--headless` 关闭 viewer；`--plant rigid|passive_arm|active_arm` 保留既有模型选择，controller 要求 active_arm。`--visual-mode mesh|cylinder`、`--sim-freq`、`--plant-step-freq`、`--control-freq` 可显式设置；自动臂模式的 plant frequency 须整除 240 Hz。`--arm-acceleration AX AY` 提供已有模态加速度参考。

导出实际构造的 XML 并短跑：

```bash
python scripts/run_sim.py --uav-source manual --arm-source manual \
  --headless --duration 0.025 --export-xml /tmp/uav_arm.xml
```

# 2. 生产调用链与修改索引

```text
scripts/run_sim.py
  manual → create_manual_env → BaseAviary → run_manual（一个 passive viewer）
  px4    → Px4MujocoRunner.create_default → run（沿用原生命周期/一个 viewer）

共用模型：scaled_stablesoft_baseline_spec + IdealTendonTensionSpec
  → build_aviary_mjcf → 旋翼 site actuator + 原软臂构造/四 tendon motor
  → MjModel.from_xml_string → 编译后按名称绑定 rotor/tendon 地址

原生 manual 滑块 → 当前 ctrl
PX4 sensor exchange → normalized → PWM → RPM → set_rotor_rpm_command（缓存 N）
arm controller → initialize/update_scaled_stablesoft_feedback
  → 原 reference / modal PD / static feedforward / bounded allocation
  → set_scaled_stablesoft_tension_command（缓存 N）

viewer.sync（读输入）→ 按控制频率更新自动命令 → BaseAviary.step
  → 逐物理子步：鼠标 wrench snapshot + 环境力
  → apply_automatic_commands（只写自动组）→ mj_step
  → 在物理区间末 clear xfrc_applied → viewer.sync（显示结果）
```

| 功能 | 文件/关键位置 | 修改或调试方式 |
|---|---|---|
| UAV 质量、惯量/旋翼参数 | `models/parameters.py`：`DRONE_PARAMS[DroneModel.CF2X]` | `mass`、`ixx/iyy/izz`、`kf/km/arm_length`、`thrust2weight_ratio`；环境和 builder 共用。 |
| 旋翼 MJCF/映射 | `models/mjcf_builder.py`：`build_aviary_mjcf`、`AviaryMjcfBuild.rotor_actuator_names` | 使用既有 prop site，6D gear 同时表达轴向力和反扭矩；编译后 `BaseAviary.ROTOR_CONTROL_ADDRESSES` 解析实际地址，不能假设 ctrl 切片。 |
| 软臂几何、质量、刚度、阻尼 | `models/components/scaled_stablesoft_arm.py`：`ScaledStableSoftArmSpec`、`scaled_stablesoft_baseline_spec` | 9 段、0.08 m、0.003 kg；stiffness=0.003086、target damping ratio=0.3、显式 damping=None；内部 8 ball joints。 |
| UAV-arm 挂载 | `models/mjcf_builder.py`：`mount_z`；原 arm builder | 根段仍固定，未增加 root hinge，也未改 tendon routing 或网格。 |
| tendon actuator 和绑定 | `models/components/tendon_drive.py`；`control/ideal_tendon_tension.py` | 0.01 N 上限、motor gear=-1；`resolve_ideal_tendon_actuator_bindings`、`write_ideal_tendon_tension_command` 验证传动、顺序、有限值和范围。 |
| 自定义软臂控制连接 | `control/scaled_stablesoft_bending_control.py`：`initialize_scaled_stablesoft_feedback`、`update_scaled_stablesoft_feedback` | 在这两个小函数中接入；初始化读取实际 tendon length/velocity，update 复用既有 factory/数值控制器，再调用 env tension setter 缓存，不另建 manager。 |
| 物理步进/缓存 | `envs/base_aviary.py`：`set_rotor_rpm_command`、`apply_automatic_commands`、`clear_automatic_commands`、`step` | 排查命令先看缓存和源选择，再看名称地址；无 Python 旋翼外力路径。ground/drag/downwash/wind 仍独立叠加。 |
| 鼠标扰动 | `scripts/run_sim.py`：`run_manual`；`backends/px4_interactive_viewer.py` | manual 使用原生完整 wrench；PX4 保留 arm target/point force 过滤，并允许 UAV 完整 wrench 和 arm 原生转动扰动；runner 跳过重复施力。观测函数不清外力，Base 每个子步从同一快照开始并在区间结束清除。 |
| PX4 sensor 输入 | `backends/sensors.py`、`transport.py`：`MujocoTruthSource.sample`、`IdealSensorAdapter.sample`、`send_sensor_frame` | ENU/FLU→NED/FRD、HIL 消息、时间戳/频率保持。 |
| PX4 motor 输出 | `backends/transport.py`、`actuation.py`、`px4_sitl.py` | `receive_actuator`、`Px4NormalizedOutputToPwm.convert`、`PwmToRotorSpeedModel.convert`；原 channel permutation、解锁/PWM/RPM 语义保持。 |
| PX4 飞行/故障 | `backends/px4_runner.py`、`px4_control.py`、`lockstep.py` | 原启动、handshake、bootstrap、Offboard/arm、spool、landing/disarm、owned process cleanup；超时/断联停止推进并清零自动缓存。 |
| 操作日志 | `backends/px4_runner.py`：`_write_log` | schema 4，保留状态/通信/同步指标；最终 tendon 值按 named bindings 读取，另记最终 armed。 |

旧 `examples/manual_aerial_soft_arm_demo.py`、`scripts/run_px4_sitl.py` 及只服务旧手动按键逻辑的测试已删除。旧 `_ManualPerturbationAviary` 被 Base 的明确 wrench 边界替代。物理不在 startup script 内实现；未创建新 runner/controller 框架。

# 3. 本次验证与 Commit 3 边界

- **自动化/无 GUI**：252 项聚焦测试通过；四种来源均通过真实模型短步进 + fake PX4；验证 240 Hz feedback、200 Hz PX4、1200 Hz plant 和每个 2400 Hz 子步的缓存重写。打乱八个 actuator 的 XML 排列后仍正确按名称写入；手动组不缓存、不被其他组/归零路径覆盖。解除武装（含非零 payload）、超时、断联、控制器 numerical fault、重置均有回归检查。
- **旋翼等价性**：从 Commit 1 提取原 `_physics`，在旋转姿态、非零不等 RPM、单旋翼和非零 yaw 情形，将旧 world wrench 转为广义力后，与实际 `qfrc_actuator` 比较：`rtol=1e-12, atol=1e-14`。外力路径没有重复旋翼力。
- **扰动/后端**：检查 UAV/arm 完整 wrench 的子步持有、释放、转动扰动、与 rotor actuator 共存；既有 sensor conversion、motor mapping、PWM/RPM、lockstep、safe disarmed 输出的聚焦测试通过。机械 body/joint、质量和惯量与 Commit 1 快照一致，root 仍固定。
- **真实 GUI**：2026-10-14 在 `DISPLAY=:1` 实际启动 manual/manual passive viewer。截图只采集本次 MuJoCo 窗口，八个单位名称可见；实际点选 rotor/tendon 滑块并记录对应动力学变化，两组没有互相覆盖。Space 暂停/恢复，双击选中 `drone0`，Ctrl+右键拖动在物理边界产生非零力/力矩，释放后归零，Backspace 回到初始状态，Q 自然关闭退出 0。此验证独立于 headless 测试；没有把 fake viewer 当真实 GUI。
- **真实 SITL**：同日使用已安装的兼容 PX4，从统一入口执行 headless px4/controller，飞行窗口 0.5 s / 600 plant steps；原 preflight、arm/Offboard、landing/disarm 和 owned cleanup 全部完成，退出 0、timeout=0、disconnect=0、最终 rotor/tendon=0。全部运行文件/log/XML 放 `/tmp`，未改 PX4 源码。日志 `/tmp/uav-arm-commit2-live/px4-run-20261014T081650.260447Z.json`。这只是短生命周期验证，不是长期飞行稳定性结论。

原 X650 worktree 的 branch、HEAD、status、index 和文件内容在本次提交前后比对。没有运行全量测试、长参数扫描，也没有 push。

**Commit 3 尚未实现**：把根部固定挂载改为同点正交的两个被动 hinge，不提供独立轴向 twist；root joints 与内部 ball joints 分开命名/绑定。不得在本次 actuator/control ownership 提交中提前改变机械拓扑。X650 仍在原 worktree，需后续独立迁移。
