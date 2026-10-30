# 单指高度任务：本机验收阶段

目标：指尖世界坐标 Z 为 0.16 m，允许误差 ±0.005 m；指尖三维速度不超过
0.01 m/s，连续满足 1 秒仿真时间才算当前达标。训练环境当前没有推力扰动；
本机单次推力实验见本文末尾，不改变默认训练任务。
`check_task.py` 仅做本机验收；`train_ppo.py` 是服务器上的 PPO 训练入口。

`finger_env.py` 是普通 MuJoCo + NumPy 的任务原型，不依赖 Gymnasium/SB3。
`finger_env_mjx.py` 是新增的 JAX/MJX 版，使用 Aero/Playground 的 State 接口；
后端运行及物理一致性仍需在训练服务器验证。
模型现已增加 Aero 风格的肌腱传感器；已有 `view_hand.py` 不变。

## 运行

无窗口验收，运行两个 8 秒仿真回合，然后运行释放指令作为奖励对照：

```bash
/home/zhili/Microduck_project/microduck_rl/.venv/bin/python /home/zhili/aerofuke/najiushi/rl/check_task.py
```

打开窗口观察同样的验收流程，按真实时间播放：

```bash
/home/zhili/Microduck_project/microduck_rl/.venv/bin/python /home/zhili/aerofuke/najiushi/rl/check_task.py --render
```

固定请求指令是 0.100 m，由脚本自动执行，不需要拖动滑块。每回合重新伸直并释放肌腱；
窗口关闭会中断验收。输出默认写入 `runs/acceptance.csv`，再次运行会覆盖；可用 `--csv` 指定新路径。

## 五项任务分别在哪里实现

- `finger_env.py` 的 `reset()`：角度和速度归零、肌腱释放、计时和历史成功状态清零。
- `step()`：动作 `[a]` 范围为 [-1, 1]；+1 收紧至 0.063 m，-1 释放至 0.115 m。
  指令变化速度最多 0.02 m/s，即每个 20 ms 控制周期最多变化 0.0004 m。
- `_observation()`：16 维 float32 数值，包括角度、角速度、高度、指尖三维速度、
  高度误差、实际执行的长度指令、保持进度、剩余时间，以及主动腱实际长度和速度。
  加入保持进度和上次实际指令，方便后续策略判断计时与动作限速状态。
  末尾腱长现由命名传感器读取并叠加 Aero 式观测噪声，腱速由命名速度传感器读取。
- `step()` 中的奖励：跟踪奖励、速度惩罚、指令变化惩罚、合格保持奖励和达标保持奖励。
  每项都在 `info["reward_terms"]` 和 CSV 中单独记录。
- `step()` 中的成功计时：每个 1 ms 物理步都检查高度和速度，超限立即清零。
  达标不结束回合，之后离开目标会失去当前达标状态；`success_ever` 记录本回合是否曾达标。

### 奖励形式

每个物理步积分：

```text
tracking_weight * exp(-(高度误差 / height_reward_scale)^2)
- tip_speed_weight * (指尖速度 / tip_speed_scale)^2
- joint_speed_weight * mean((关节角速度 / joint_speed_scale)^2)
+ holding_weight * 当前高度和速度均合格
+ success_weight * 当前连续保持已满1秒
```

速度惩罚的平方项上限为 100。每个控制周期另外扣除：

```text
command_change_weight * (本次指令变化 / 最大允许单步变化)^2
```

所有上述奖励率都乘以对应仿真时间，避免更换控制频率后奖励总量无意改变。
发生数值异常或 MuJoCo 警告则终止并扣除失败惩罚；正常运行到 8 秒标记超时截断，之后必须 reset。

### 主要配置

`config.json` 集中设置目标、容差、限速、回合时间和奖励权重。
当前每个回合 400 个控制步，每个控制步推进 20 个 1 ms 物理步。
这些奖励权重是便于调试的初值，尚未通过 PPO 训练调优。

## 验收和测试

固定指令验收同时检查：初态可重复、动作不超过限速、奖励等于各项之和、成功后继续运行、
正常回合只在时间到达时截断、释放对照不能获得成功且总奖励低于目标指令。

```bash
/home/zhili/Microduck_project/microduck_rl/.venv/bin/python -m unittest discover -s /home/zhili/aerofuke/najiushi/rl -p test_finger_env.py
```

本机此次验收：0.100 m 指令在约 1.78 秒首次达标，最终高度约 15.843 cm，保持到 8 秒结束，
两个回合结果相同。这验证了任务与基线控制逻辑，不代表策略已经学会控制，也不代表 MJX 已通过验证。

## MJX 版任务环境

本次新增两个文件：

- `finger_env_mjx.py`：函数式环境 `FingerHeightMjxEnv`，以及可单独测试的纯 JAX 任务函数。
- `test_finger_env_mjx.py`：JAX 数学与原版任务对照、JIT/vmap 状态流转测试。

继续共用 `config.json` 和原始 `hand.xml`，没有更改本机环境、模型或物理参数。
本机仍用 `check_task.py --render` 查看；不要用 `check_task.py` 启动 MJX 环境。

### 与本机版本的对应关系

| 本机版本 | MJX 版本 |
|---|---|
| `reset(seed=...)` | `reset(jax.random.PRNGKey(...))` |
| 环境对象保存可变 data/计时 | 全部可变量保存在返回的 State 中 |
| `step(action)` | `step(state, action)` |
| 返回 obs/reward/terminated/truncated/info | 返回 State，含 obs/reward/done/metrics/info |
| 20 次 Python 物理循环 | `jax.lax.scan` 执行 20 个物理步 |
| 每步 `mj_step` 后 `mj_forward` | 每步 `mjx.step` 后 `mjx.forward` |
| 浮点累计保持时间 | 整数累计物理步，避免 float32 的 1 秒边界误差 |
| 首次成功时间未设置时为 None | 首次成功时间未设置时为 -1，保证 JAX 树形状固定 |

动作映射、0.02 m/s 限速、16 维观测顺序/归一化、各项奖励、成功后继续运行和 8 秒超时保持一致。
成功判定仍检查指尖三维线速度，而非仅检查竖直速度。数值异常/非有限动作会产生失败终止；
MJX 没有普通 MuJoCo 的 warning 计数器，无法完全复制原版的 warning 终止检测。
MJX 版保留最后一个有限状态供诊断，done 后 step 不再推进，必须 reset 或使用训练包装器自动重置。

### 服务器用法

在原 Aero 训练 Python 环境中，把本目录代码、配置及 `descrption/hand.xml` 一起放到服务器：

```bash
python check_mjx.py --task --xml ../descrption/hand.xml --batch-size 4
```

这会实际测试 MJX reset/step、完整 8 秒任务、原版与 MJX 的最终姿态/奖励、重复 reset 及批量运行。
首次编译可能需要数分钟。`--task` 模式不使用旧模型检查模式的 `--steps` 参数，
任务步数由 `config.json` 的回合时长决定。模型不支持时直接报错，不偷偷替换圆柱或积分器。
对照检查允许最终高度差 5 mm、最终关节差 5 度、总奖励差约 10%（另加 0.1 绝对容差）；
这是初步验收门槛，不是后端完全相同的证明。

Python 接口示例（从 rl 目录执行）：

```python
import jax
from finger_env_mjx import FingerHeightMjxEnv

env = FingerHeightMjxEnv()
reset = jax.jit(env.reset)
step = jax.jit(env.step)
state = reset(jax.random.PRNGKey(0))
state = step(state, env.action_for_length(0.100))
# state.obs / state.reward / state.done
# state.info['hold_seconds'], state.info['is_success']
# state.metrics['height'], state.metrics['reward/tracking']
```

接 Brax 时将 `env.for_brax_training()` 传给 Playground 的 `wrap_for_brax_training`，
设置 `episode_length=env.episode_length`、`action_repeat=1`、`full_reset=True`。
适配器让 Brax 自己标记超时截断，避免把 8 秒超时误当作真正终止、影响价值函数的 bootstrap。
full_reset 可完整重置任务计时；环境也处理了缓存自动重置只替换 data/obs 的情况。
没有修改 Aero 项目的环境注册表。PPO 入口使用下面的独立训练脚本。

### 已完成和未完成的验证

本机可执行：

```bash
/home/zhili/Microduck_project/microduck_rl/.venv/bin/python -m unittest discover -s /home/zhili/aerofuke/najiushi/rl -p 'test_finger_env*.py'
```

新测试覆盖：动作限速的 JIT/vmap、指尖世界速度与 `mj_objectVelocity` 的对照、
恰好 1000 个物理步的保持计时、离开目标后计时清零、完整回合采样上的观测/奖励对照。
环境状态流转测试使用明确标记的简化物理替身，检查 JIT/vmap、超时、缓存自动重置和无效动作，
它不验证 MJX 引擎本身。Microduck 原环境缺少 mujoco-mjx/Playground；
服务器准备阶段已另建 `.venv-server-check` 验证真实后端，最新结果见 `server/VALIDATION.md`。

## PPO 训练入口：Playground / Brax

新文件各司其职：

- `train_ppo.py`：创建 MJX 训练/评估环境、调用 Brax PPO、打印进度、保存日志和模型快照。
- `ppo_config.json`：PPO 超参数和网络大小；任务目标/奖励仍在原来的 `config.json`。
- `test_train_ppo.py`：配置和启动流程测试，使用明确的模拟后端，不运行真正的 PPO。

关系为：`train_ppo.py` → `finger_env_mjx.py` → `hand.xml` + `config.json`，
同时读取 `ppo_config.json`，将训练参数交给 Brax 的 `ppo.train()`。
训练入口会用现有 `for_brax_training()` 处理超时，并通过 Playground 包装器完成批量和自动重置。
不重复实现 PPO，不引入 Gymnasium/SB3，也不修改 Aero 注册表或自动安装/升级依赖。

### 推荐执行顺序

本机先检查配置（不运行 MJX、不启动训练、不创建训练输出目录）：

```bash
/home/zhili/Microduck_project/microduck_rl/.venv/bin/python /home/zhili/aerofuke/najiushi/rl/train_ppo.py --dry-run
```

以下命令在服务器原 Aero 训练 Python 环境、项目根目录下执行：

```bash
# 先确认模型和任务能在实际服务器后端运行
python najiushi/rl/check_mjx.py --task

# 4 环境、1024 总环境步的小规模真实训练，只检查流程
python najiushi/rl/train_ppo.py --smoke

# 小规模流程通过后，启动默认正式配置
python najiushi/rl/train_ppo.py
```

`--smoke` 不是 dry-run，会调用真正的 MJX/Brax 并保存 checkpoint，但步数很少，不预期学会任务。
默认正式配置：64 并行训练环境、16 评估环境、100 万总环境步、策略和价值网络各两层 64 单元。
单回合仍为 400 个控制步（8 秒），每个控制步内部有 20 个物理步；环境步数不能等同于物理步数。
本入口先限制为单主机单计算设备，避免初版多 GPU 的采样分片和批量整除问题。

常用覆盖参数：

```bash
python najiushi/rl/train_ppo.py --num-timesteps 2000000 --seed 2
python najiushi/rl/train_ppo.py --smoke --dry-run
python najiushi/rl/train_ppo.py --output-dir /path/to/logs --run-name finger-first-run
```

配置优先级为：`ppo_config.json` → `--smoke` 覆盖 → 命令行覆盖。
改变 `num_envs` 时必须满足 Brax 的 `batch_size * num_minibatches` 能被 `num_envs` 整除。
`episode_length` 根据任务配置自动推导，`action_repeat` 固定为 1，避免重复放大控制周期。

### 每次训练保存什么

默认新建 `rl/runs/finger-height-时间-随机后缀/`，不会覆盖之前的验收 CSV 或已有训练目录：

```text
task_config.json          实际任务参数快照
ppo_config.json           解析后的训练计划，包括种子、网络和 PPO 参数
hand.xml                  这次训练的模型快照
source/                   训练入口、MJX 环境和配置类代码快照
metadata.json             依赖版本、设备、模型哈希、运行状态和最终结果
progress.jsonl            每次进度回调的步数、用时和评估/训练指标
checkpoints/数字目录/      Brax 原生 checkpoint 与网络重建配置
latest_checkpoint.txt     完成后最新 checkpoint 的相对路径
```

checkpoint 通常包含策略、价值网络和观测归一化参数，可用于后续推理；这里不承诺保存优化器的
完整训练状态，也没有实现完整断点续训。本机 checkpoint 回放已提供 `play_ppo.py`；回放使用该次运行的模型、
任务配置、观测构造和动作限速，不能只拷贝神经网络权重。
中断或失败会在 metadata 记录状态；已完成保存的 checkpoint 保留。
Brax 可能因采样批次/评估周期取整而略超出请求总步数，metadata 记录实际 checkpoint 步数。

### 当前验证范围

配置检查、参数覆盖/整除约束、训练/评估环境分离、full_reset 包装器参数、进度记录、
配置/模型/代码快照、checkpoint 路径、失败记录和禁止覆盖已有目录均有本机测试。
保存流程测试使用模拟 Brax 后端，不代表真实 checkpoint 序列化或 PPO 更新已经在本机运行。
Microduck 原环境缺少 Brax/MJX；独立检查环境的真实运行结果见 `server/VALIDATION.md`。
V100 GPU 上的依赖和运行仍须执行服务器验收。
入口会检查安装版本的函数签名，遇到不支持的参数明确报错，不会静默忽略设置。

```bash
/home/zhili/Microduck_project/microduck_rl/.venv/bin/python -m unittest discover -s /home/zhili/aerofuke/najiushi/rl -p test_train_ppo.py
```

## 本机单次小推力与恢复曲线

`check_push.py` 使用普通 MuJoCo，在现有固定目标高度 0.16 m 下，保持肌腱长度指令
0.100 m。它先运行无推力对照，再运行一次推力实验；不运行 PPO，不修改 MJX 环境、
默认任务配置或模型。`PushProbeEnv` 仅覆盖普通环境的单物理步钩子。

推力配置独立保存在 `push_config.json`：第 4.0 秒开始，在指尖 `index_tip` 沿世界
-Z 方向施加 0.1 N，持续 0.1 秒。力通过 `mj_applyFT` 在指尖位置投影到关节，
不是直接修改关节角度/速度，也不是把力作用在远节质心。每个 1 ms 物理步检查施力区间，
结束即清零；同样配置重复运行可以复现。配置时间必须对齐物理步长。

```bash
# 无窗口，生成数据和图像
/home/zhili/Microduck_project/microduck_rl/.venv/bin/python /home/zhili/aerofuke/najiushi/rl/check_push.py

# 先生成无推力对照，再实时观看有推力的 8 秒回合
/home/zhili/Microduck_project/microduck_rl/.venv/bin/python /home/zhili/aerofuke/najiushi/rl/check_push.py --render

# 验证施力时序、指尖力矩投影、重复性和恢复
/home/zhili/Microduck_project/microduck_rl/.venv/bin/python -m unittest discover -s /home/zhili/aerofuke/najiushi/rl -p test_check_push.py
```

每次创建独立的 `runs/push-时间/`，包含：

- `push.csv`、`no_push.csv`：逐物理步数据，高度、三维速度、关节角、肌腱长度和拉力。
- `metrics.json`：相对无推力对照的最大偏移、速度峰值、恢复时间和实际推力冲量。
- `config.json`、`hand.xml`：任务/推力配置和模型快照。
- `recovery.png/svg`：全回合六幅曲线；`recovery_detail.png/svg`：推力附近的放大曲线。

图中橙色区域是施力区间。黄/青弹簧的力只绘制弹性分量，不包含阻尼。
肌腱长度指令虽然固定，但其位置执行器会随实际长度和速度改变拉力，恢复也包含弹簧与阻尼的作用，
不能把这个结果称为 PPO 抗扰动能力。

恢复判据仍要求高度误差和三维速度同时合格：
`sustained_recovery_started_after_release_s` 是撤力后进入持续合格区间的起始时间，
`full_hold_reacquired_after_release_s` 是随后完整保持 1 秒的确认时间；未恢复时为 null。
`first_qualifying_sample_after_release_s` 仅表示第一次合格采样，之后可能再次超限，不等于稳定恢复。
`lost_height_or_speed_qualification` 表示推力是否实际打断过合格状态。

本机默认实验：相对无推力对照，最大高度偏移约 0.109 mm，速度峰值约 0.01584 m/s，
超过 0.01 m/s 限制。撤力后约 0.023 秒进入持续合格区间，约 1.023 秒确认重新保持达标。
最终高度约 0.15843 m。这个推力主要通过速度指标打断达标，尚未将高度推出 ±5 mm 容差。

## 肌腱观测与 Aero 对齐

参考本机 Aero `right_hand.xml` 的 `len_if`，以及 `rotate_z.py` 的 `_get_obs()`：

```text
hand.xml: tendonpos len_if → sensordata → 均匀测量噪声 → 观测第 14 项（从 0 起）
hand.xml: tendonvel vel_if → sensordata → 原有腱速观测第 15 项
```

只给主动腱 `index_flexor` 添加传感器，与 Aero 针对执行器所控肌腱的做法一致。
其他关节/指尖观测、16 维观测布局及现有归一化保留。腱长噪声公式为
`sensor_length + U(-1, 1) * tendon_noise_level * tendon_length_noise_m`。
默认分别为 1.0 和 0.005 m，沿用 Aero 的默认幅度 ±5 mm；这是参考设置，不是实机标定值。
设 `tendon_noise_level=0` 可以进行无噪声观测对照。
腱速传感器保留了原先的速度观测，Aero 参考任务本身只使用腱长，不额外对腱速加噪声。

普通 MuJoCo 与 MJX 环境都通过传感器名解析 `sensor_adr`/`sensor_dim`，从
`data.sensordata` 读取。每次 reset/策略步生成一次噪声，固定 seed 可重复，
MJX 的随机 key 和观测读数保存在 State.info 中，以兼容 JIT/vmap 和自动重置。
NumPy 与 JAX 不保证不同后端生成相同噪声序列，后端对照仅对无噪声部分比较数值，
腱长噪声单独验证边界和种子重复性。

物理引擎仍用真实腱长算力；测量噪声只影响策略输入，不改腱长、执行器或奖励真值。
`info`、MJX metrics 和新的本机验收 CSV 同时区分：

- `tendon_length`：物理腱长，供诊断。
- `tendon_length_sensor`：`len_if` 的无噪声虚拟传感器读数。
- `tendon_length_observed`：叠加噪声后实际提供给策略的腱长，归一化前单位为米。

本机测试覆盖命名传感器与物理腱长/腱速一致、噪声边界、seed 重复、JIT/vmap，
并验证加入观测噪声不改变固定动作下的物理轨迹、执行器力和奖励。
MJX 真引擎和 Brax 训练的最新本机 CPU 检查记录见 `server/VALIDATION.md`；服务器 GPU 验收仍需执行。

## 下载服务器训练成果并在本机回放

下载完整的 `runs/finger-height-.../`，保留 `hand.xml`、`task_config.json`、`metadata.json`、
`source/`、`latest_checkpoint.txt` 和 `checkpoints/` 的相对目录结构。
不要只下载最后的 checkpoint：策略还需要当次模型、环境观测构造和动作限速。

本机使用服务器准备阶段创建的 `.venv-server-check`（或安装了同版核心依赖的 Python）：

```bash
/home/zhili/aerofuke/.venv-server-check/bin/python /home/zhili/aerofuke/najiushi/rl/play_ppo.py \
  --run-dir /下载到本机的完整训练目录 --render --episodes 3
```

`play_ppo.py` 强制 CPU 策略推理，恢复 Brax 保存的策略及观测归一化参数，
通过普通 MuJoCo 打开本机窗口；不需要 V100 或 CUDA 动态库设置。
优先读取运行目录的 `source/finger_env.py`、配置和模型快照，并验证模型 SHA256。
当前工作区后来改过模型或观测，不会影响该次回放。
默认加载 `latest_checkpoint.txt` 指向的权重，可用 `--checkpoint 000001032192` 指定其他保存点。

不带 `--render` 可无窗口评估。结果保存为运行目录中新建的 `replay-时间/trajectory.csv` 和
`summary.json`，统计总奖励、首次连续保持成功时间、是否曾达标和最终是否达标。
策略输出采用确定性模式，但传感器噪声仍按训练配置启用。
普通 MuJoCo 与 MJX 的物理和随机数实现存在细小差异，本机回放是直观检查和补充评估，
不能保证奖励与服务器逐位一致。

### 已训练 PPO 的小推力回放测试

`check_policy_push.py` 加载下载运行的模型、环境快照、配置及 PPO 权重，先运行无推力对照，
再使用相同随机种子运行有推力回放。两次均由策略持续控制肌腱，不使用固定腱长控制。
默认回放 12 秒，第 4 秒在 `index_tip` 世界位置施加 `[0, 0, -0.1] N`，持续 2 秒（第 6 秒卸载）；
每个物理子步通过 `mj_applyFT` 施力，脉冲结束后清除外力。

```bash
/home/zhili/aerofuke/.venv-server-check/bin/python \
  /home/zhili/aerofuke/najiushi/rl/check_policy_push.py \
  --run-dir /home/zhili/aerofuke/downloads/finger-height-20261029-193528-0f1992 \
  --render
```

可用 `--force 0 0 -0.5`、`--push-start 4`、`--push-duration 2` 修改扰动，
用 `--episode-seconds 12` 设置回放时长。延长仅作用于本地回放；策略输入的时间观测
仍按原训练的 8 秒倒计时，超过后保持零，8 秒之后的策略表现需单独看待。
结果保存在运行目录的 `policy-push-时间/`，包含 1 ms 物理轨迹、20 ms 策略动作、
`summary.json` 和 `recovery.png`。本实验只用高度误差及连续高度达标时间评价，
仍记录速度，不更改原训练的奖励、成功条件或 checkpoint，也不执行 PPO 更新。
`height_hold_completed_after_release_s` 是卸载后完成连续高度保持的时间；
若扰动未使高度越界，这个值不能解读为一次从失败状态恢复所需的时间。

带窗口时会显示时间、`PUSH ON` 状态及红色施力方向箭头；箭头固定长 4 cm，仅供显示，
不是位移或力大小的比例尺。默认按正常速度播放；可选 `--slowdown 5` 将 12 秒物理过程慢放到约 60 秒，
不改变物理步长、策略控制周期或推力持续时间。`--keep-open` 在结束时冻结最终状态，
手动关闭窗口后保存结果并退出。例如在上一条命令后追加：
`--force 0 0 -0.5 --slowdown 5 --keep-open`。

施力点通过 `--point` 指定：`tip` 指尖（默认），`distal` 远节圆柱中心，
`middle` 中节圆柱中心，`proximal` 近节圆柱中心。施力点随指节移动，
箭头起点跟随选中的点；CSV 保存每个物理步施力前的世界坐标和施力点名称。
`--force FX FY FZ` 单位为 N，始终是世界方向：`0 0 -1` 向下，`0 0 1` 向上，
`0 -1 0` 沿 -Y，`0 1 0` 沿 +Y。当前手指关节只绕 X 轴转动，
纯 X 方向的力没有绕该轴的力矩；模型没有侧摆自由度，因而不适合用纯 X 力观察侧向摆动。
例：`--render --point middle --force 0 -1 0 --push-duration 2 --episode-seconds 12`。

### 随机施力点和方向的 PPO 训练

`finger_env.py` 的 `PUSH_POINTS` 定义 10 个随指节移动的点：近、中、远节各自长度的
25%、50%、75% 处，以及指尖。位置由指节坐标变换每个物理步更新，不修改质量、惯量或碰撞形状。
`PUSH_DIRECTIONS` 是世界 YZ 平面每隔 45° 的 8 个单位方向；不采样无侧摆自由度对应的纯 X 力。
每回合 reset 独立选一个点、一个方向、一个力值以及一次扰动的开始和持续时长。
施力期间点跟随指节，世界方向及幅值保持不变，不在每个控制步重新抽方向。
扰动参数不加入策略的 16 维观测；策略根据已有状态和腱长观测作出反应。

- `config_push.json`：80% 回合有一次扰动，20% 无扰动；幅值均匀采样 1～4 N。
- `config_push_warmup.json`：相同点与方向，幅值 0.25～1 N，供初期练习。
- 每回合 12 秒；第 3～5 秒间随机开始，持续 0.5～2 秒。
- 新配置 `require_low_speed=false`，速度与关节速度惩罚权重为零，先只考核高度。
  `config.json` 仍是原来的无扰动、要求低速的任务。
- 有扰动回合只从卸载后的第一个物理步开始累计保持及成功奖励，需在 16 cm ±5 mm
  内连续保持 1 秒。受力期间仍有逐物理步的高度跟踪奖励。
- `push_recovery_rate` 用真正抽到扰动的回合作分母，避免无扰动回合冲高恢复率。
  这是撤力后曾连续达标的比例，不代表受力期间始终在误差带内。

先在本机验证（不训练）：

```bash
/home/zhili/aerofuke/.venv-server-check/bin/python /home/zhili/aerofuke/najiushi/rl/check_random_push.py
/home/zhili/aerofuke/.venv-server-check/bin/python /home/zhili/aerofuke/najiushi/rl/train_ppo.py \
  --task-config /home/zhili/aerofuke/najiushi/rl/config_push.json --dry-run
```

同步代码到服务器后，可先从现有权重开始小力训练：

```bash
cd /root/finger-v100
bash server/run.sh smoke --task-config najiushi/rl/config_push_warmup.json
bash server/run.sh train --task-config najiushi/rl/config_push_warmup.json \
  --restore-checkpoint najiushi/rl/runs/finger-height-20261029-193528-0f1992/checkpoints/000001032192
```

`--restore-checkpoint` 恢复策略及观测归一化参数，重新初始化价值网络、优化器和训练步数，
属于新任务上的继续学习，不是完整恢复旧训练进程。强扰动阶段将 `--task-config` 换成
`config_push.json`，并将 `--restore-checkpoint` 指向小力阶段的 checkpoint。
两个强度配置是手动分阶段使用，当前未实现自动课程升级；不要分别从零训练并当成连续课程。

1～4 N 的所有组合不保证物理上都可控：单输入腱驱动、5 N 执行器拉力上限、关节限位
和弹簧都会影响能力。训练后需按点、方向、力值分组评估，并与固定腱长及无扰动控制比较。
当前的 0.5～2 秒矩形力用于恢复训练；未来若要模拟短暂碰撞应另设更短的脉冲时长。
