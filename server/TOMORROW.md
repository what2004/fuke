# 从现有 checkpoint 开始小力扰动训练

前提：服务器仍保留 `/root/finger-v100/.venv-server`，以及原运行目录
`/root/finger-v100/najiushi/rl/runs/finger-height-20261029-193528-0f1992`。
若释放实例导致环境丢失，先按 server/README.md 安装环境并还原完整训练结果。

## 1. 本机 Ubuntu 终端：提交并上传代码

整段复制执行。括号内任何一步失败都会停止；不会上传 downloads、环境和训练结果。

```bash
(
  set -e
  cd /home/zhili/aerofuke
  test "$(git branch --show-current)" = main
  git add .gitignore najiushi/rl server
  if ! git diff --cached --quiet; then
    git commit -m "Add randomized finger push training and checkpoint warm start"
  fi
  git push origin main
)
```

看到 push 成功（或 Everything up-to-date）后再执行服务器部分。

## 2. 服务器终端：同步代码

通过租用平台的终端或已能登录的终端连接服务器，整段执行：

```bash
(
  set -e
  cd /root/finger-v100
  git pull --ff-only origin main
  test -f server/start_push_warmup.sh
  test -f najiushi/rl/config_push_warmup.json
  echo "Code synchronized. Ready for small-force training."
)
```

只有看到最后的 Ready 提示才继续。若出现 TLS、网络、冲突错误，同步没有完成，
先解决该错误，不要直接用旧代码训练。不要执行 git reset --hard 或删除原训练目录。

## 3. 服务器终端：检查、试跑、训练

```bash
bash /root/finger-v100/server/start_push_warmup.sh
```

脚本依次执行 GPU/MJX 检查、配置检查、小规模 PPO 试跑、正式训练；任一步失败立即停止。
`server/run.sh` 自动设置虚拟环境中 NVIDIA 库的 `LD_LIBRARY_PATH`，无需额外 export。

试跑与正式训练都从原始 `000001032192` checkpoint 加载策略与观测归一化器，
重新初始化 critic 和 optimizer；试跑仅验证新流程，不作为正式训练的起点。
两次运行均创建独立运行目录，不覆盖原 checkpoint。

训练内容：10 个点、8 个世界 YZ 平面方向、0.25～1 N 随机力，12 秒一回合；
80% 回合一次扰动，20% 无扰动。先只考核撤力后高度恢复及连续保持，不考核速度。
日志保存在 server/logs，训练结果在 najiushi/rl/runs。

终端需保持连接；如要断开连接仍训练，可在服务器的 tmux 会话中执行第 3 步。
训练完成看到 `Training finished. Checkpoint: ...` 后，记录新的运行目录；
后续强扰动阶段从这个新 checkpoint 接着训练。
