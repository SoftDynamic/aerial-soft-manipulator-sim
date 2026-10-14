# 统一入口、配置和物理参数

工作目录 `/home/ustc/mujoco_UAV-refactor`，分支 `refactor/uav-arm-sim`；本阶段从 `7faf780` 实施。原 X650 worktree 未迁移。默认仍是 CF2X + 九段 active StableSoft + mesh，根段刚性挂在 UAV 上，内部八个 ball joints 和四条 tendon routing 保持原状。

```bash
conda activate mujoco_uav
python scripts/run_sim.py
python scripts/run_sim.py --uav-source px4
python scripts/run_sim.py --uav-source px4 --arm-source controller
python scripts/run_sim.py --uav-source manual --arm-source controller
python scripts/run_sim.py --config configs/sim.yaml
python scripts/run_sim.py --headless --duration 0.5
python scripts/run_sim.py --headless --duration 0.05 --export-xml outputs/uav_arm.xml
```

## 日常 CLI 和配置优先级

公开选项只有 `--uav-source manual|px4`、`--arm-source manual|controller`、`--config PATH`、`--initial-position X,Y,Z`、`--target-position X,Y,Z`、`--arm-acceleration AX AY`、`--duration SECONDS`、`--headless`、`--export-xml PATH`、`--visual-mode mesh|cylinder`、`--help`。

`configs/sim.yaml` 是主配置，分为 `robot.uav`、`robot.soft_arm`、`simulation`、`px4`、`diagnostics`。现有 PX4 `.params` 继续作为兼容性证据和控制器参数输入，没有新增参数生成框架。底层 PX4 连接、bootstrap、超时、ramp、landing 等设置使用既有 `Px4RunnerConfig` 默认值，必要时在 YAML 的 `px4` 段显式设置。冗余 `scripts/run_px4_viewer.sh` 已删除。

启动只解析一次，明确优先级为 **显式 CLI > 显式 YAML > 已验证源码默认值**。未提供的 CLI 值为 `None`，不会覆盖 YAML。解析结果含 `values`、逐字段 `origins`（`default/yaml/cli`）和配置文件路径；写入 `effective-config-*.json`。PX4 运行日志还保存实际 backend defaults 的 `runner_config`、同一份 effective snapshot、模型/参数文件 SHA256 和校准值。YAML 中的相对路径以 YAML 所在目录为基准；显式 CLI 路径以当前工作目录为基准。未知字段和无效模型值在启动/构造边界拒绝。

## 物理参数贯通

```text
run_sim.resolve_config
  DRONE_PARAMS → immutable ResolvedUavParameters
  scaled_stablesoft_baseline_spec → resolved ScaledStableSoftArmSpec
  robot.soft_arm.max_tension_n → IdealTendonTensionSpec

manual / PX4 → BaseAviary → build_aviary_mjcf → compiled named bindings
  UAV snapshot → body inertial, rotor geometry/gear/ranges,
                 mass/torque limits, MAX_RPM/HOVER_RPM
  arm spec → geometry, mass, stiffness, damping, friction
  tendon spec → actuator ctrlrange / native slider / command validation
              → existing BoundedTendonAllocatorConfig
PX4 snapshot → cf2x calibration → normalized/PWM/RPM adapters
             → physical PX4 parameter-file validation
```

UAV 裸质量为 0.027 kg，软臂为 0.003 kg，总质量 0.030 kg；软臂只加一次。最大旋翼能力仍从裸机 `thrust2weight_ratio * mass * 9.81` 得出，载荷不增加最大推力。默认总推力上限 0.5959575 N、MAX_RPM=21713.713729，含臂 HOVER_RPM=15258.842632。惯量仍是 UAV 裸机惯量，软臂惯量由原组件生成。

`kf=3.16e-10 N/RPM²`、`km=7.94e-12 Nm/RPM²` 对应运行时的数值 RPM 平方，修正了原先误写的 rad/s 注释，没有改变系数或换算单位。旋翼 FLU 顺序仍为 FL、RL、RR、FR，反扭矩符号 `- + - +`，PX4 channel permutation `(2,1,3,0)`。

PX4 校准接收同一 physical snapshot 和载荷质量，PWM/RPM 两个转换器共享该校准结果。现有电机标定只支持默认裸质量、kf、km 和推重比；修改这些值的 PX4 模型明确拒绝，manual 模型仍可支持有效值。几何或载荷修改必须提供匹配参数文件；这不代表原 PX4 增益或软臂反馈对新模型已验证。

参数文件校验检查 FRD 旋翼坐标、KM 符号/比例、CA_ROTOR_COUNT、THR_MDL_FAC 和含臂 MPC_THR_HOVER；可选轴向/PZ 设置不能偏离支持的对齐模型。已验证文件坐标与 `L/√2` 相差约 1.011 µm，显式使用 1.1 µm 几何容差；其他值独立严查。保留所有原 `.params` 数值与控制器增益。

软臂配置支持九段约束下的长度、质量、半径、刚度、显式阻尼或目标阻尼比、tendon frictionloss、最大张力和显示模式。显式阻尼与目标阻尼比不能同时非空；选择一种时将另一项设为 `null`。没有改变 attachment 或 routing。

运行期张力上限唯一来自 `robot.soft_arm.max_tension_n`，默认 0.01 N。它传给 actuator/滑块、runtime command validation、反馈 allocator 和饱和统计；初始化反馈时检查 compiled actuator 的范围完全一致。删除了兼容加速度控制器中“只能是 0.01 N”的数值限制，保留分配算法、零 preload 和数值故障处理。偏离默认软臂参数/张力会打印未验证提示，不能视为新模型稳定性保证。

## 控制来源和 GUI 生命周期

manual/manual、manual/controller、px4/manual、px4/controller 四种组合保留。手动组始终由 native Control 滑块拥有；Python 不向该组写控制器零值。自动组按原频率更新，最新有效命令在每个 MuJoCo 子步前按名称地址写入。PX4 为 200 Hz，软臂反馈为 240 Hz，plant 为 1200 Hz，物理为 2400 Hz。没有重复旋翼外力路径。

默认 `simulation.duration_s: null` 时 GUI 由 viewer 关闭决定结束，不用巨大的假时长。显式时长按模拟时间计。Headless 使用显式时长或有限 `headless_duration_s`（默认 5 s）。Manual 初始暂停，Space 开始/暂停，`.` 单步，Backspace 重置模型/反馈，Q 或关闭窗口退出。原生 Control 滑块及鼠标 body 选择/力和力矩扰动保留。

PX4 默认是 YAML 指定的 local owned hover profile，仍沿用原 TCP/MAVLink、estimator bootstrap、Offboard、arm、bounded ramp、plant release、position target、landing/disarm、owned cleanup。外部 peer 必须显式禁用 owned launch 和 auto flight；自动 arming 被限制在 local owned SITL。

所有保留的 arm GUI 路径共享已有 viewer adapter；rigid native viewer 也检测关闭和 Space/`.`。暂停不推进 PX4 序号/模拟时间；暂停时清空自动命令，恢复后保持安全零值至下一次原定 PX4 交换收到新输出，不重播暂停前电机命令。关闭窗口启动已有降落流程，先释放 viewer；landing/headless cleanup 不要求已关闭的窗口仍可用。锁步/通信失败清空自动组并执行有界 owned process cleanup，不退回 manual rotor。

## 长时间内存与诊断

Lockstep 去重为 deque/set，最多 512 个 packet identity；移出历史的 stale 包仍按时间戳拒绝。Future、duplicate、sensor sequence、timeout/disconnect 语义保持。10000 个有效包和过期 stale/duplicate/future 的合成测试验证了上限。Owned PX4 日志仍使用既有 64 KiB 上限，GCS warnings 仍为末 100 项。

普通 GUI 不保存详细 interaction event trace；显式 replay/诊断事件 deque 上限 256，`_last_state` 保持覆盖快照。只有显式 headless `diagnostics.trial: baseline|pulse` 才保存张力诊断时序，默认 20 Hz、最多 256 个样本；峰值来自原 1200 Hz plant 测量节拍，独立于低频保存数据。没有新增日志框架、控制来源或生产实验脚本。

直接张力诊断仅允许 owned PX4 SITL + headless + manual arm + 四个 tendon actuators。基线四条张力为零；独立基线通过后 `pulse` 将 tendon 0 写入 0.001 N，0.30 s 后四条归零；只临时拥有 tendon 组，rotor 继续由 PX4 拥有。异常退出也清零诊断 tendon 命令。记录 simulation timestamp/phase、PX4 armed/mode/status、位置/速度/RPY/target/error、实际 rotor ctrl/RPM、实际 actuator force、tendon lengths/velocities、arm-tip displacement、pulse/release 标志及 estimator telemetry。命令变为零不证明机械状态返回基线。

既有恢复计算提取为一个复用函数；每次 release 重置 candidate 和旧 recovery result。阈值需连续满足完整窗口，瞬时过阈值和 NaN 不能建立恢复。没有将 zero modal acceleration 当作 zero tendon tension。

## 本阶段验证和冻结真实试验

- 聚焦自动化：最终 **234 passed**，选择既有 13 个测试文件，未运行全仓库测试。包含配置优先级/路径/log、编译惯量/质量/rotor ranges、默认模型与 `7faf780` 比较、四来源按名称写入/手动值保留、tendon limit=0.02、真实 ideal actuator force 脉冲与归零、完整测量节拍峰值、baseline gate、连续 recovery 和每次 release 重置、301/302 s mocked GUI、关闭后的 landing、pause 时间顺序/安全零值、transport/PWM/RPM/armed/timeout/cleanup、有界历史。
- Headless MuJoCo：额外运行 manual 0.5 s 和 controller 0.05 s，导出真实 XML。自动测试也使用真实 compiled model 与 physics substeps。
- 真实 GUI：2026-10-14，`DISPLAY=:1`，manual/manual，无默认时长。实际 rotor/tendon 滑块输入分别为 0.01638883125 N / 0.0011 N，状态记录显示各自保留。Space 暂停/恢复、原生 body 选择/鼠标扰动（321 个非零 wrench physics intervals）、Backspace 重置到位置 `[0,0,1]` 和零 ctrl、WM_DELETE_WINDOW 关闭后退出 0。截图/状态在 `/tmp/uav-stage-gui-*.png` 与 `/tmp/uav-arm-stage-gui-*.json`。未运行真实 300 s GUI，也未运行真实 PX4 GUI；其生命周期结论来自 mocks。

[冻结协议](validation/px4_stage_protocol.json) 在真实启动前保存（2026-10-14 19:28:16，北京时间）。默认 CF2X/StableSoft、刚性根挂载、initial/target `[0,0,1]`，无调参。Trial 1 为独立无扰动基线；只有其通过才允许独立重启 Trial 2，并重新通过自己的基线。每次真实 launch attempt 消耗名额，上限两次，startup/restart/cleanup 合计预算 600 s。

冻结 provisional criteria：每次 release 后观察 2 s，末 1 s 连续 error≤0.10 m 且 speed≤0.10 m/s；允许的 pulse 为 `xp_yp` tendon（index 0）0.001 N × 0.30 s，之后四张力零；恢复观察 5 s，要求 1 s 连续满足同一阈值。没有在看结果后改变标准。

**Trial 1 结果：未在规定窗口内满足悬停判据。** 真实启动仅一次，总 wall time 2.869 s（含 startup/cleanup），baseline 2 s / 2400 plant intervals，保存 40 个样本。mean error=0.0817053 m，max error=0.1434491 m，max speed=0.3313485 m/s；baseline 末 error=0.1434491 m、speed=0.1096046 m/s。最大绝对 RPY=(0.021883°, 0.030248°, 0.013416°)，rotor saturation=0，max RPM=16806.4155。样本中四条实际 tendon force 和 command 全零，armed=true，PX4 mode=393216（Offboard）。这段是 startup/release 后的无扰动响应，不是 tendon pulse 的影响。

按冻结 gate 未启动 Trial 2、未施加实际 pulse，因此扰动峰值、pulse 时间和 recovery time 均为**未测量**。不存在恢复成功结论。基线末 arm-tip relative displacement=6.276e-5 m，tendon velocities 最大约 4.99e-6 m/s；零张力不等于软臂完全静止。

降落完成（2.790833 s 模拟时间），最终 disarmed、rotor/tendon ctrl 零，没有残留 PX4 进程。transport timeout/disconnect 均零；tail log 保留 `vehicle_command lost`、`vehicle_command_ack lost` 和 `Disarming denied: not landed`。Estimator health flags=831，timeout flags 未收到（null），不把缺失 telemetry 当健康证明。物理总质量/最大推力/悬停参数检查一致，未发现这些配置值冲突；有限窗口内的位置/速度 criterion 未满足，不能推断 PX4 永远不能 hover，也不归因给未施加的 tendon 扰动。

[原始 Trial 1 日志](validation/px4_stage_trial1.json) 和 [摘要](validation/px4_stage_summary.json) 随提交保存，负结果没有删除。冻结时实施 diff fingerprint 与最终实现可能因后续无调参的 cleanup/NaN/GUI 时序修正不同；真实测量文件保持原样，模型和 `.params` 的 SHA256 保持一致。


## 8 秒无扰动 PX4 hover 诊断（2026-10-14）

本轮仅扩展既有 bounded diagnostic recorder；在原 send 调用捕获实际 local ENU setpoint，并加 configured initial-position offset 得到 world ENU。记录 bootstrap、preflight、Offboard warmup、spool、release、height transition、fixed hover、landing/disarm 的原传感器时钟/序号。没有额外 sensor frame、actuator writer 或控制器改动。高度极值与末窗口统计来自原 1200 Hz 测量节拍，20 Hz 保存 160 样本，上限仍为 256。

[新冻结协议](validation/px4_hover8_protocol.json)：默认 CF2X + StableSoft、px4/manual、headless、四张力零；release offset=1 m、transition=1 s、release RPM fraction=1.1 均未改。独立观察 release 后 8 s，末 7–8 s 连续 error≤0.10 m 且 speed≤0.10 m/s；最多一次真实启动，wall budget 180 s。之前的 2 s 协议及负结果保持原样。

真实 SITL 一次完成（含启动/降落/清理/日志收集 wall 4.828 s）。Release sensor timestamp=14,520,000 µs，seq=2704；首次 flight send 在 plant t=0，stamp=14,525,000 µs。固定目标首次发送于 plant t=1.000 s，stamp=15,525,000 µs、seq=2905；这是原 clock convention 的 5 ms 偏移，不是时间重置。保存样本 t=0.25/0.50/0.75/1.00 s 的实际最后发送 world height 分别为 1.754167/1.504167/1.254167/1.004167 m；t=1.05 s 起的保存值为固定 1.0 m。

初始 vertical velocity=0，release RPM=16806.4155（hover 的 1.101421 倍），实际收到 RPM 对应总 commanded thrust=0.357024 N，重力约 0.2943 N。全测量节拍 height max=1.138586 m@0.7475 s、min=0.834221 m@2.4700 s，正/负高度误差 +0.138586/−0.165779 m；全程 max position error=0.165889 m、max speed=0.330529 m/s。保存样本在 2.5–8 s vertical velocity 均为正，高度从下方趋近目标，未见再次换向；不根据低频曲线计算短时峰值。

末 1 s 的 1200 个测量点：max error=0.031681 m、max speed=0.011957 m/s、mean error=0.027402 m，**通过冻结判据**。8 s 末 height=0.979524 m、vertical velocity=+0.006690 m/s、3D error=0.023617 m、speed=0.011462 m/s。残余高度误差仍在减小，不能声称零偏差、无限时间稳定或重复性已验证。这支持 startup/release transient 解释，并说明旧 2 s 窗口不足以建立稳态表现；单次无消融试验不能证明唯一因果。

Rotor saturation=0、timeout/disconnect=0；全部样本 armed/Offboard、tendon command/force 零，没有 pulse。最终降落并 disarmed，无残留 PX4 进程。估计高度与 ground truth 的保存时刻最大差约 0.010191 m，末差 +0.002882 m；GCS telemetry age 未记录，不能据此认定 estimator lag 为原因。Health flags=831、timeout flags=null；尾日志保留 vehicle command/ACK lost 和未落地时 disarm denied，缺失 telemetry 不等于健康证明。

[摘要](validation/px4_hover8_summary.json) 与 [完整有界试验记录](validation/px4_hover8_trial1.json) 随提交保存。聚焦测试 60 passed（既有四个文件，含实际 headless MuJoCo 短步进与 fake PX4 回归），真实物理结论仅来自本次真实 SITL。日常配置、物理参数、PX4 gain 文件、挂载及原 X650 worktree 均未改。
