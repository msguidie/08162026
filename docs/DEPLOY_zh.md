# 部署指南（交钥匙版）

本文覆盖四件事：A 回放存到 GitHub；B 在 Render 上启用 AI 席位；C 本地 Windows + RTX 3060 推理 worker；D 在 NSCC 上训练。
C、D 两节在训练系统与 worker 完成后补全（见 docs/PROGRESS.md）。

## A. 历史回放：写入单独的 GitHub 仓库

1. 在 GitHub 新建一个仓库，例如 `msguidie/splendor-replays`（Private 或 Public 都可以，勾选 "Add a README" 以便有 `main` 分支）。
2. 生成 Fine-grained Personal Access Token：GitHub 右上角头像 → Settings → Developer settings →
   Personal access tokens → Fine-grained tokens → Generate new token。
   - Repository access：Only select repositories → 选 `splendor-replays`。
   - Permissions → Repository permissions → **Contents: Read and write**（Metadata 会自动带上）。
   - Expiration 建议 1 年，到期后重新生成并更新 Render 的环境变量。
3. Render 控制台 → 你的后端服务（`zero8162026`）→ Environment → 添加：

   | Key | Value |
   |---|---|
   | `REPLAY_GITHUB_TOKEN` | 上一步生成的 token |
   | `REPLAY_GITHUB_REPO` | `msguidie/splendor-replays` |
   | `REPLAY_GITHUB_BRANCH` | `main` |
   | `REPLAY_GITHUB_DIR` | `replays`（可省略，默认 replays） |

   保存后 Render 会自动重新部署（或手动 Manual Deploy）。
4. 验证：浏览器打开 `https://zero8162026.onrender.com/api/replays/status`，应看到 `{"github":true,...}`。
   打完一局后，仓库里会出现 `replays/2026/09/<gameId>.json` 与 `replays/index.json`；
   游戏首页的 **Replays** 按钮即可浏览与回放。
5. 说明：
   - 没有配置 token 时功能照常可用，只是回放只保存在服务器内存（最近 100 局，重启即失）。
   - 每局只在 **结束（GAME_OVER）** 时写入一次（一次文件 PUT + 一次 index 更新）；中途 quit / 闲置回收的对局不保存。
   - 每局 JSON 约 1.5–3 KB；`index.json` 每局约 150 字节。

## B. 在 Render 上启用 AI 席位

Render → Environment 添加 `AI_WORKER_SECRET=<一串随机长字符串>`（例如用密码管理器生成 32 位以上）。
未设置时，大厅不会显示任何 AI 相关按钮；设置后大厅会显示 **Add AI**，但只有本地 worker 连上后按钮才可用
（`/api/ai/status` 返回 `available: true`）。同一个 secret 要填到本地 worker 的 `.env`。

## C. 本地 Windows 10/11 + RTX 3060 推理 worker

worker 是一个 socket.io **客户端**：它主动出站连接 Render，不需要端口映射、内网穿透或防火墙规则。
详细英文说明见 `splendor_ai/README.md` 的 "Deployment worker" 一节；下面是中文步骤。

1. 安装 **Python 3.11**（python.org，安装时勾选 *Add python.exe to PATH*）。
2. 把整个仓库（含 `splendor_ai/`）放到本机，例如 `C:\splendor`。在该目录打开 PowerShell：

   ```powershell
   py -3.11 -m venv .venv
   .venv\Scripts\activate
   pip install torch --index-url https://download.pytorch.org/whl/cu126
   pip install -r splendor_ai\requirements-worker.txt
   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   ```

   **先装 CUDA 版 torch，再装其余依赖**（否则会装成 CPU 版）。最后一行打印 `True` 说明 3060 可用。
3. 配置：

   ```powershell
   copy splendor_ai\.env.example splendor_ai\.env
   notepad splendor_ai\.env
   ```

   至少改两项：`SERVER_URL=https://zero8162026.onrender.com`，`AI_WORKER_SECRET=<与 Render 上完全相同的值>`。
   其它键（模型目录、时间预算、确定化数量 K、日志）在文件里有中英文说明。
4. 放模型：把训练导出的 `shared.pt`（或 `ind2.pt / ovt.pt / team.pt` 等按模式的文件）放进 `models\`。
   **还没有模型也可以先跑**：worker 会退化为贪心策略并打印一条警告，正好用来验证连线。
5. 启动：

   ```powershell
   splendor_ai\run_worker.bat
   ```

   脚本会激活 `.venv`、读取 `splendor_ai\.env`，并在异常退出时自动重启（Ctrl-C 退出）。
   离线自检：`splendor_ai\run_worker.bat --once` 会对一个内置局面算一步并打印动作。
6. 验证：
   - 浏览器打开 `https://zero8162026.onrender.com/api/ai/status`，应为 `{"enabled":true,"available":true,...}`；
   - 大厅出现 **Add AI** 按钮，点一下即加入 `Bot Alpha`；团队模式下用座位上的 **AI** 小按钮给机器人选座；
   - `logs\moves.jsonl` 每步一行（`level` 为 `search` 表示走的是神经网络 + 搜索）。
7. 长期开机：把 `run_worker.bat` 的快捷方式放进 `shell:startup`（Win+R 输入）即可开机自启；
   Render 免费实例休眠时 worker 会一直重试（1 s → 30 s 退避），服务醒来后自动重新注册。

资源占用：显存约 1 GB（含 CUDA 上下文），内存约 1 GB，单核 CPU；默认每步思考 1.5 s（硬上限 2.5 s，服务器允许 15 s），
在 3060 上约 2000–3000 次模拟/步。想更强就调大 `TIME_BUDGET_MS`。

常见问题：`Invalid worker secret` = 两边 secret 不一致（注意空格/引号）；`/api/ai/status` 显示 `enabled:false` = Render 没设
`AI_WORKER_SECRET`；`torch.cuda.is_available()` 为 `False` = 装成了 CPU 版 torch，`pip uninstall torch` 后按第 2 步重装。

## D. NSCC ASPIRE 2A 训练

（待训练系统完成后补全：conda 环境、`requirements.txt`、`qsub scripts/nscc_train.pbs`、续训、评估、导出。）
