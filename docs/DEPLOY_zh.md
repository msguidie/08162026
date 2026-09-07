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
同一个 secret 要填到本地 worker 的 `.env`。

**按钮什么时候出现**：大厅的 **Add AI** 按钮只在服务器报告 `aiAvailable: true` 时才**渲染**（`WaitingRoom.tsx`），
也就是「设了 secret **并且** 此刻有 worker 连着」；只设了 secret、worker 没连上时，大厅里根本看不到这个按钮，
不是显示成灰色的禁用状态。判断链路：

- `GET /api/ai/status` → `{"enabled":false,"available":false}`：Render 上没设 `AI_WORKER_SECRET`；
- → `{"enabled":true,"available":false}`：secret 设了，但 worker 还没连上（按钮不出现）；
- → `{"enabled":true,"available":true,"name":...}`：worker 已注册，按钮出现，可以点。

worker 连上/断开时服务器会立刻把大厅状态重新广播一次，所以已经坐在大厅里的人不用刷新页面：
按钮会自己出现或消失。（按钮出现之后仍可能是禁用态，那是另一回事：席位满了，或者已经加满 4 个机器人。）

## C. 本地 Windows 10/11 + RTX 3060 推理 worker

worker 是一个 socket.io **客户端**：它主动出站连接 Render，不需要端口映射、内网穿透或防火墙规则。
详细英文说明见 `splendor_ai/README.md` 的 "Deployment worker" 一节；下面是中文步骤。

1. 安装 **Python 3.11**（python.org，安装时勾选 *Add python.exe to PATH*）。
2. 把整个仓库（含 `splendor_ai/`）放到本机，例如 `C:\splendor`。在该目录打开 PowerShell：

   ```powershell
   # 只需一次：允许运行本机自己生成的脚本，否则下一行的 Activate.ps1 会被
   # 拦下（报 "cannot be loaded because running scripts is disabled"）。
   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned      # 提示时输入 Y
   py -3.11 -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install torch --index-url https://download.pytorch.org/whl/cu126
   pip install -r splendor_ai\requirements-worker.txt
   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   ```

   不想改执行策略也行：改用 `cmd`（不是 PowerShell）执行 `.venv\Scripts\activate.bat`，
   后面的命令一样跑。第 5 步的 `run_worker.bat` 本来就是 `cmd` 批处理，不受执行策略影响。
   激活成功的标志是提示符前面多了 `(.venv)`。

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

## D. NSCC ASPIRE 2A 训练（4×A100-40G，PBS `ai` 队列）

假设代码已经在 NSCC 的工作目录里（例如 `~/splendor`），以下都在**登录节点**执行；脚本里每一行都有英文注释。

**先记下你的项目代码（project code）**：ASPIRE 2A 的每个作业都要记账到一个项目，`qsub` 不带 `-P` 会被直接拒绝。
个人配额一般是 `personal-<nusnetid>`，以 NSCC 开通邮件为准（也可以 `qstat -Qf` / 问 helpdesk 确认）。
下面所有 `qsub` 都写成 `-P <项目代码>`，请替换成你自己的；训练脚本会自己把这个代码传给续跑的下一段作业
（所以训练作业要同时写 `-P <项目代码>` 和 `-v PROJECT=<项目代码>`：前者记账当前这段，后者传给下一段）。
忘了传会在作业一开始就打印 `ERROR: no NSCC project code` 并退出，不会白排队一整天。

1. 装环境（一次性，登录节点有外网）：

   ```bash
   cd ~/splendor
   bash splendor_ai/scripts/nscc_setup.sh
   ```

   它会 `module load anaconda` → 建 conda 环境 `splendor`（Python 3.11）→ `pip install -r splendor_ai/requirements.txt`
   （Linux 上 PyPI 的 torch 自带 CUDA）→ 打印版本 → 做一次导入自检。登录节点上 `torch.cuda.is_available()` 为 False 是正常的。

   **新开的登录 shell 里 `conda activate` 用不了**（报 "Your shell has not been properly configured"）：
   `module load anaconda` 只把 conda 放进 PATH，激活功能要先初始化一次。两种做法：

   ```bash
   # 做法 1（一次性，写进 ~/.bashrc，以后每次登录都可用）
   module load anaconda
   conda init bash
   source ~/.bashrc                       # 或者重新登录一次
   conda activate splendor

   # 做法 2（不改 ~/.bashrc，每个 shell 里临时启用）
   module load anaconda
   eval "$(conda shell.bash hook)"
   conda activate splendor
   ```

   PBS 脚本里不需要这一步：`nscc_train.pbs` / `nscc_eval.pbs` 自己 `source` 了
   `$(conda info --base)/etc/profile.d/conda.sh`。

   建议再跑一遍单元测试：
   `conda activate splendor && python -m pytest splendor_ai/tests -q -x --ignore=splendor_ai/tests/test_worker_e2e.py`。

2. 先做一次短的吞吐/连通性作业（也是 G4 门槛）：

   ```bash
   qsub -P <项目代码> -l walltime=00:30:00 \
        -v PROJECT=<项目代码>,TRAIN_TIMEOUT=25m,MAX_CHAIN=0 \
        splendor_ai/scripts/nscc_train.pbs
   qstat -u $USER            # 看排队/运行
   tail -f runs/nscc/logs/train.*.log
   ```

   `-l walltime=00:30:00` 把这段短作业的墙钟压到 30 分钟（脚本里的 `#PBS -l walltime=23:59:00` 是给正式训练用的；
   `#PBS` 指令在提交时就被 PBS 读走，读不到环境变量，所以只能在命令行上覆盖）。半小时的作业排队通常比一天的快得多。
   `TRAIN_TIMEOUT=25m` 让训练器提前 5 分钟自己收尾存盘，`MAX_CHAIN=0` 表示跑完不续下一段。

   日志里每一代会打印 sims/s、moves/s、games/s、各模式的对局长度、stuck 率、loss，以及对固定基准（random / greedy / mcts）的胜率。
   目标：整节点 ≥ 40 万 sims/s，每张推理 GPU ≥ 25 万 evals/s；否则先按 `splendor_ai/README.md` 的 "Training" 一节调
   `--set` 参数（actors、games_per_actor、sims）再继续。

3. 正式训练（自动续跑）：

   ```bash
   qsub -P <项目代码> -v PROJECT=<项目代码> splendor_ai/scripts/nscc_train.pbs
   ```

   作业申请 `select=1:ngpus=4:ncpus=64:mem=440GB`，walltime 23:59；内部用 `timeout --signal=INT 23h` 让训练器干净地保存并退出，
   然后**自动再次 `qsub` 自己**（最多 `MAX_CHAIN=30` 段）。`runs/nscc/` 里有 `weights/latest.pt`（最新权重）、
   `checkpoints/gen_XXXX.pt`（每代）、`metrics.jsonl`、`replay.npz`（回放缓冲）——再次提交时会自动 `--resume`。
   想停：`touch ~/splendor/STOP`（当前段跑完后不再接续），或 `qdel <jobid>`。
   GPU 布局：GPU0 学习器，GPU1–3 批量推理服务器，56 个 CPU 采样进程；这是评审给出的最优布局（13M 参数的网络单卡学习器已经过剩 15 倍以上，
   DDP 只会增加故障面），学习器代码保留了 `WORLD_SIZE>1` 的 DDP 分支以备将来把网络放大到 4000 万参数以上。
   课程：前 30 万局只练 2 人，之后切到五种配置混合（ind2 25% / ind3 10% / ind4 15% / 1v2 25% / 2v2 25%），25% 的对局混入历史检查点或贪心对手。
   预期：第一天内超过公开项目的总算力；第 2–3 天的检查点通常就是可部署版本；总共 5–7 天。

4. 评估（单 GPU，随时可跑，不影响训练）：

   ```bash
   qsub -P <项目代码> splendor_ai/scripts/nscc_eval.pbs          # 评 runs/nscc/weights/latest.pt
   qsub -P <项目代码> -v CKPT=runs/nscc/checkpoints/gen_0040.pt,GAMES=200 \
        splendor_ai/scripts/nscc_eval.pbs
   # 想先小跑一次看通不通（20 局、只跑 2 人模式、半小时墙钟）：
   qsub -P <项目代码> -l walltime=00:30:00 -v GAMES=20,MODES=ind2 \
        splendor_ai/scripts/nscc_eval.pbs
   ```

   评估作业不会自己续跑，所以只要 `-P` 就够了（`-v PROJECT=` 可省）。

   产出 `reports/arena_<ckpt>_<job>.md`：对固定锚点（random=0 Elo、greedy、mcts40/160/640）的 Bradley–Terry Elo、95% 置信区间、
   各模式胜率、按座位拆分（1v2 会分 solo/duo）、卡死/截断比例。G5 门槛：2 人模式对 mcts 锚点 ≥ 85%，且 Elo 随代数单调上升。

5. 导出给本地 worker：

   ```bash
   conda activate splendor
   python -m splendor_ai.export --ckpt runs/nscc/weights/latest.pt --out dist/model
   ```

   得到 `dist/model/shared.pt`（以及可选的 `ind2.pt … team.pt`、`manifest.json`）。把 `shared.pt` 复制到 Windows 机器的
   `models\` 目录（scp / OneDrive 都行），重启 `run_worker.bat` 即可上线；worker 会检查 `obs_version`，旧模型会被明确拒绝而不是乱下。

6. 只想训练某一种模式（专精模型）：

   **先看清配置是怎么决定模式比例的。** `nscc_4xa100.yaml` 里同时有 `selfplay.mode_mixture` 和
   `selfplay.phases`，而**只要 `phases` 非空，`mode_mixture` 就完全不起作用**——训练器每次开局都用
   `phase_for(已完成局数)` 选出当前阶段，再按该阶段自己的 `mixture` 抽模式（`selfplay/config.py::phase_for`）。
   现有的课程表是：

   | 阶段（`until_games`） | 模式比例 | 备注 |
   |---|---|---|
   | < 50 000 | `ind2: 1.0` | 热身，搜索也调小（`sims_full: 150`、`sims_fast: 40`） |
   | < 300 000 | `ind2: 1.0` | 满搜索预算，仍然只打 2 人 |
   | < 1 000 000 | ind2 0.40 / ind3 0.15 / ind4 0.15 / ovt 0.15 / team_adj 0.10 / team_opp 0.05 | 逐步过渡 |
   | 之后（`until_games: null`） | ind2 0.25 / ind3 0.15 / ind4 0.15 / ovt 0.20 / team_adj 0.15 / team_opp 0.10 | 稳态 |

   所以「只练 1v2」要改的是 `phases`，改 `mode_mixture` 没有任何效果。推荐做法是复制一份配置再改：

   ```bash
   cp splendor_ai/configs/nscc_4xa100.yaml splendor_ai/configs/nscc_ovt.yaml
   # 编辑 nscc_ovt.yaml：把 phases 换成一句
   #   phases:
   #     - until_games: null
   #       mixture: {ovt: 1.0}
   # （想保留便宜的热身阶段，就把每个阶段的 mixture 都改成 {ovt: 1.0}）
   qsub -P <项目代码> \
        -v PROJECT=<项目代码>,CONFIG=splendor_ai/configs/nscc_ovt.yaml,RUN_DIR=runs/ovt_only \
        splendor_ai/scripts/nscc_train.pbs
   ```

   等价的命令行写法（自己在交互节点上跑训练器时用；`qsub -v` 的值里不能带逗号，
   而这段 YAML 里全是逗号，所以它不适合从 `-v` 传进去）：

   ```bash
   python -m splendor_ai.selfplay.train \
       --config splendor_ai/configs/nscc_4xa100.yaml \
       --set run_dir=runs/ovt_only \
       --set 'selfplay.phases=[{until_games: null, mixture: {ovt: 1.0}}]'
   ```

   `--set` 的键必须是配置树里真实存在的字段，写错会立刻报错并退出（例如 `--set job_id=...` 会告诉你
   `RunConfig has no field 'job_id'`）；先用 `--print-config` 确认解析结果再提交作业。
   训练完导出为 `ovt.pt` 放进 `models\`，worker 会优先加载按模式命名的文件。
