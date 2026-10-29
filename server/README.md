# V100 32GB 服务器训练

这套准备针对 Linux x86_64、Python 3.12、单张 V100 32GB，使用
JAX 0.8.2 + MJX-JAX 3.10.0 + 本项目随附的 Playground + Brax PPO 0.14.2。
安装到解压目录的 `.venv-server`，不修改其他项目环境。

当前训练任务是指尖到达 0.16 m 高度并保持，含 Aero 式肌腱观测噪声；
尚未给训练环境加入随机推力。`check_push.py` 是独立的本机对照实验。
第一轮训练用于验证无外部推力的学习和完整保存流程，不代表抗扰动训练已完成。

## 1. 上传

本机运行以下命令生成 `dist/finger-v100-时间.tar.gz`：

```bash
python3 server/build_bundle.py
```

将压缩包上传服务器并解压；压缩包包含模型、任务、测试、固定依赖清单和
Playground 源码，不包含旧实验结果、本机虚拟环境或 Git 数据。

```bash
tar -xzf finger-v100-时间.tar.gz
cd finger-v100
sha256sum -c MANIFEST.sha256
```

`时间` 替换为实际文件名。后面的命令都从解压后的 `finger-v100` 目录执行。

## 2. 安装到独立 Python 环境

先在服务器镜像中选好 Python 3.12。不要选只提供 CUDA 13 JAX 的镜像。
服务器需要 NVIDIA 驱动；Python CUDA 12 运行库由 pip 清单安装，无需手动混装 CUDA Toolkit。

```bash
nvidia-smi
python3.12 --version
bash server/setup_v100.sh
```

如果 Python 3.12 的命令名就是 `python`：

```bash
SERVER_PYTHON=python bash server/setup_v100.sh
```

`requirements-core.lock` 和 `requirements-v100.lock` 固定了直接及间接依赖。
GPU 清单固定 CUDA 12 插件及 cuDNN 9.8.0.87；较新 cuDNN 即使使用 CUDA 12，
也可能已不支持 V100。NCCL 固定为 2.27.7。**不要再执行无版本约束的依赖升级。**
安装会下载数 GB 的 GPU 运行库，请预留磁盘空间和安装时间。
若 Python 缺少 venv/pip，需要先让服务器镜像提供这些组件；本脚本不改系统包。

兼容依据：

- [JAX 安装要求](https://docs.jax.dev/en/latest/installation.html)
- [cuDNN 9.8 支持 Volta](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.8.0/reference/support-matrix.html)
- [较新 cuDNN 的支持矩阵](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.18.0/reference/support-matrix.html)

## 3. 依次验收

```bash
# 检查真实 GPU、矩阵/卷积内核、版本、模型、传感器、JIT 步进和训练 API
bash server/run.sh preflight

# 完整 8 秒任务；与普通 MuJoCo 对照，并执行批量环境
bash server/run.sh check

# 真正执行 4 环境、约 1024 步的 PPO 试训，确认能更新和保存 checkpoint
bash server/run.sh smoke
```

每步成功后再进行下一步。第一次 JAX 编译可能较慢；日志开头出现编译等待不代表卡死。
`smoke` 的目的只是验收，不预期学会任务。

预检查会拒绝无 GPU 的静默 CPU 回退。物理模型不支持时直接报错，不替换圆柱、肌腱或积分器。
如果报 `libEGL.so` 等图形运行库缺失，需要服务器镜像提供 EGL/OpenGL 运行库；
训练本身不开实时窗口。不通过关闭传感器或改物理参数绕过检查。

## 4. 正式训练

```bash
bash server/run.sh train
```

默认 64 个训练环境、16 个评估环境、100 万总环境步。每个环境步是 20 ms，
内部推进 20 个物理步。具体配置见 `najiushi/rl/ppo_config.json`。

可覆盖训练步数和种子：

```bash
bash server/run.sh train --num-timesteps 2000000 --seed 2
```

训练较长时在服务器已有的 tmux 会话里执行，避免断开 SSH 后进程退出：

```bash
tmux new -s finger
bash server/run.sh train
# Ctrl+B 然后 D：离开会话，训练继续
# 下次连接：tmux attach -t finger
```

如果没有 tmux，可使用服务商的持久终端。脚本不会后台偷偷启动训练。

## 5. 保存结果

终端日志：`server/logs/`。

每次训练在 `najiushi/rl/runs/finger-height-时间-随机后缀/` 创建新目录，包括：

- `metadata.json`：版本、设备、状态、最终指标及 checkpoint 位置。
- `progress.jsonl`：训练进度和评估指标。
- `checkpoints/`、`latest_checkpoint.txt`：模型和最新 checkpoint 路径。
- 模型、任务配置、PPO 配置和代码快照。

运行完成后，把整个训练目录下载回本机；不要只下载权重。
尚未实现完整优化器状态续训，也没有完成实机部署；终止后不承诺能无损接着训练。

出现错误时保留并提供该阶段 `server/logs/` 中的完整日志。
预检查、真实 MJX 验收和 smoke 是否在本机通过，另见 `VALIDATION.md`；
本机 CPU 验证不能替代服务器 V100 的实际验收。
