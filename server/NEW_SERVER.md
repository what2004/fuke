# 新服务器：下载代码、安装环境、恢复 checkpoint、继续训练

适用：本项目的 Ubuntu 22.04、Python 3.12 镜像；单张 RTX 3090 / T4 / V100，使用锁定 CUDA 12 依赖。
目录统一为 `/root/finger-v100`。安装脚本中的 v100 是沿用文件名，不限制只能使用 V100。

## 1. 本机终端：先上传最新代码

```bash
(
  set -e
  cd /home/zhili/aerofuke
  git add .gitignore najiushi/rl server
  if ! git diff --cached --quiet; then
    git commit -m "Prepare randomized push training on a new GPU server"
  fi
  git push origin main
)
```

必须成功后再继续。代码和旧训练结果分开传输；结果已位于仓库的 finger-ppo-v1 Release。

## 2. 新服务器终端：安装工具并登录 GitHub

```bash
apt update && apt install -y git gh tmux libegl1 libgl1
gh auth login --hostname github.com --web
```

按 Enter 后，在本机浏览器访问 https://github.com/login/device，输入服务器显示的一次性验证码，
使用 what2004 登录授权。服务器没有浏览器导致 xdg-open 报错不影响手工授权。
如询问 Git 协议，选 HTTPS。等待 Logged in as what2004，再执行后续步骤。

## 3. 新服务器终端：克隆代码

```bash
git clone https://github.com/what2004/fuke.git /root/finger-v100
```

成功后：

```bash
cd /root/finger-v100
test -f server/start_push_warmup.sh && echo "Latest training entry found"
nvidia-smi
python --version
```

Python 必须是 3.12.x；若 `python` 不是 3.12，不要直接运行安装步骤。
如镜像的 3.12 命令为 python3.12，将下一步的 SERVER_PYTHON=python 改为 SERVER_PYTHON=python3.12。

若 git clone 出现以前的 gnutls_handshake 错误，先确认 /root/finger-v100 中尚无完整 checkout。
可以用下面的 GitHub API 压缩包路径作为替代，保留现有目录，不执行 rm/reset：

```bash
(
  set -e
  test ! -e /root/finger-v100/server/start_push_warmup.sh
  gh api repos/what2004/fuke/tarball/main > /tmp/fuke-source.tar.gz
  gzip -t /tmp/fuke-source.tar.gz
  mkdir -p /root/finger-v100
  tar -xzf /tmp/fuke-source.tar.gz --strip-components=1 -C /root/finger-v100
  test -f /root/finger-v100/server/start_push_warmup.sh
)
```

替代方法下载的是源码快照，不带 .git。以后更新用同样的快照方式，不能直接 git pull。

## 4. 新服务器终端：安装独立训练环境

```bash
(
  set -e
  cd /root/finger-v100
  SERVER_PYTHON=python bash server/setup_v100.sh
)
```

使用镜像中的 Python 3.12 创建 `.venv-server`，安装 MuJoCo、MJX、JAX CUDA 12、Brax 和随附 Playground。
不需要另外安装 PyTorch、CUDA Toolkit 或覆盖驱动。下载量数 GB，等到 Setup complete 再继续。
不要另行无版本升级 jax、mujoco 或 cuDNN。

## 5. 新服务器终端：下载并恢复旧训练结果

```bash
(
  set -e
  cd /root/finger-v100
  mkdir -p downloads najiushi/rl/runs
  gh release download finger-ppo-v1 \
    --repo what2004/fuke \
    --pattern finger-result.tar.gz \
    --dir /root/finger-v100/downloads
  gzip -t downloads/finger-result.tar.gz
  tar -xzf downloads/finger-result.tar.gz -C najiushi/rl/runs
  test -f najiushi/rl/runs/finger-height-20261029-193528-0f1992/checkpoints/000001032192/ppo_network_config.json
  echo "Checkpoint restored successfully"
)
```

## 6. 新服务器终端：进入持久终端并开始训练

先执行：

```bash
tmux new -s finger
```

进入新终端后再执行：

```bash
bash /root/finger-v100/server/start_push_warmup.sh
```

脚本自动设置 NVIDIA 库路径，并依次执行 GPU/MJX 检查、配置检查、带 checkpoint 的小规模试跑、正式训练。
任何一步失败即停止。使用 0.25～1 N 随机力，从原有策略继续训练，输出到新目录。
按 Ctrl+B，松开后按 D，可离开 tmux 而继续训练。重新查看：`tmux attach -t finger`。

完成标志：`Training finished. Checkpoint: ...`。记录这个新的完整运行目录。
当前单次回放可用，240 组固定组合的自动验收工具尚未提供；不能仅凭一次日志恢复率认定课程通过。
