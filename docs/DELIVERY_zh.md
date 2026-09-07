# 交付说明（中文）

分支：`claude/splendor-rl-training-system-v05e9l`。所有改动都是**附加式**的：现有登录 / 大厅 / 对局 / 计时 / 认输 / 重连 / 计分路径未改语义
（`server/index.js` 的 `game_action` 与 `resign` 处理体被原样抽成函数供 AI 复用，服务器 137 个测试覆盖）。
三份操作手册：`docs/DEPLOY_zh.md`（Render / GitHub / Windows / NSCC 全部步骤）、`splendor_ai/README.md`（英文技术手册）、
`docs/KNOWN_ISSUES.md`（发现的既有问题，未擅自修改）。

## 1. 历史回放（已完成）

- 每局结束后服务器把**最小化 JSON**（初始牌序 + 动作列表 + 结果，约 1.5–3 KB）写入你指定的 GitHub 仓库
  `replays/YYYY/MM/<id>.json`，并维护 `replays/index.json`；没配 token 时退化为内存保存最近 100 局。格式见 `docs/REPLAY_FORMAT.md`。
- 回放重建**复用服务器自己的规则引擎**（`server/replayEngine.js` 逐条重跑 `processAction`），零逻辑重复，不会与规则漂移。
- 前端：开始界面的 **Replays** 按钮 → 列表（日期、模式、玩家、胜者★、回合数）→ 回放器。回放器复用对局的全部组件和布局，
  随机视角（可切换），播放/暂停、单步前后、0.5×–4× 倍速、拖动进度、退出；动画沿用对局机制（市场翻牌、宝石增减、+1 卡、贵族提示、动作字幕）；
  手机与桌面均适配（截图在 `docs/screenshots/`）。
- 验证：服务器 137 个测试（含真实 socket 对局：2 人 / 1v2 / 2v2 / 认输 / 退出），真实服务器整局 → 列表 → 重建 → 渲染截图闭环。
- Render 接线：新建私有仓库 + Fine-grained PAT（仅该仓库 Contents 读写）→ Render 环境变量
  `REPLAY_GITHUB_TOKEN / REPLAY_GITHUB_REPO / REPLAY_GITHUB_BRANCH` → 重新部署 → `/api/replays/status` 显示 `github:true`。详见 `docs/DEPLOY_zh.md` A。

## 2. AI 研究结论（决定采用什么算法）

读了 15 个公开项目/论文（含你给的 chlligence PPO 仓库、Splendor-Zero、alpha-zero-general、seal256、BreckEmert、Rinascimento、
multiplayer-alphazero、LightZero 等，证据存于 `docs/research/`），两名独立"裁判"给出一致结论（`docs/AI_DESIGN.md`）：

- **不选纯 PPO**：你给的仓库自己的检查点记录显示 5180 万步后已平台化（对上一代胜率在 0.36–0.64 之间抖动），
  且它需要 400 行手工奖励塑形，五种胜负规则各要重调；它的 10 枚超出自动弃币与我们"禁止选取"相反。
- **采用 多人 AlphaZero**：策略网络 + **每个座位一个价值分量**（4 维向量，队友共享同一分量 → 2v2 自然协作，1v2 的不对称阈值一个函数解决），
  PUCT 自博弈；隐藏信息（牌堆顺序、对手暗预留）用**逐次模拟确定化（PIMC）**；KataGo 技巧（playout cap randomization、
  forced playouts + 目标剪枝、Dirichlet α=10/合法动作数）；部署时也搜索（同权重下搜索约 +600 Elo 是文献里最大的单项收益）。
- **一个网络覆盖所有模式**（2 人 / 3 人 / 4 人个人、1v2、2v2），用模式/座位/角色/阈值特征作条件；最后可按模式微调，只有胜出才上线；
  也可用 `--set` 只训单一模式的专精模型（1v2 的 solo 与 duo 由同一网络的角色特征区分，自博弈天然两侧都学）。
- **不用 DDP**：1300 万参数的学习器单卡已过剩 15 倍以上；采用 GPU0 学习器 + GPU1–3 批量推理服务器 + 56 个 CPU 采样进程。
  学习器保留了 DDP 分支以备未来放大网络。

## 3. `splendor_ai/`（已完成并验证）

| 模块 | 内容 | 验证 |
|---|---|---|
| `rules/` | 与 `server/gameLogic.js` **逐位一致**的 Python 引擎，65 个动作（30 取宝石含被迫少拿、15 预留、15 购买、5 选贵族，无 pass） | 与 Node 引擎随机对局交叉比对 **16,000 局 / 127 万步 / 0 差异**，双向合法动作集相等；73k 步/秒/核 |
| `encode.py` `symmetry.py` `values.py` | 860 维信息集观测（绝不读隐藏信息）；5 种颜色旋转对称性在 10 万局面上**精确**成立（免费 5 倍数据增广）；各模式终局价值向量 | 274+ 测试 |
| `model.py` | 12.5M 参数残差 MLP，掩码在图内，价值/得分/卡死辅助头，带版本门禁的检查点 | |
| `search/` | 开环 PIMC PUCT/Gumbel MCTS、每座位价值回传、同座位子决策（选贵族）、反"透视"惩罚、批量叶评估调度器 | 贪心 ≥96% 胜随机（全模式）；MCTS@400 **81%** 胜贪心（配对换座） |
| `selfplay/` | 采样进程（PCR、跨局批量叶评估、课程、对手池）、推理服务器、代际回放窗、学习器、可续跑的编排器、bootstrap、PPO 备用学习器（未完成，明确标注） | **G3**：本沙箱 4 核 CPU 21 分钟自博弈后，无搜索策略 0.81–0.83 胜随机，48 次模拟搜索 0.80–0.92 胜贪心 |
| `arena.py` `anchors.py` `export.py` | 配对种子 + 全座位轮换 + Bradley–Terry Elo（固定锚点 random/greedy/mcts40/160/640，单调）、导出 `shared.pt` | 54 测试 |
| `worker/` | Windows/Linux 推理 worker：从服务器负载水合状态并校验守恒律、时限内随时可答的搜索、1 步防卡死过滤、发送前重验、四级兜底、断线重连 | 真实服务器 **30 局全模式**，每步都由 worker 回答，0 拒绝，p99 延迟 104 ms |
| `scripts/` | `nscc_setup.sh`、`nscc_train.pbs`（自动接续）、`nscc_eval.pbs`、`smoke_cpu.sh` | |

服务器侧 AI 接口（`server/aiBridge.js`、`server/aiFallback.js`、大厅 **Add AI**）：worker 主动出站连接 Render，轮到机器人时服务器推送局面，
worker 返回动作，服务器走**与人类完全相同**的落子路径；worker 掉线/超时/非法动作时用内置贪心兜底，对局永不挂起；机器人对局同样进入回放。

## 4. 你要做的事（按顺序）

1. **Render**：设置 `REPLAY_GITHUB_*`（回放）和 `AI_WORKER_SECRET`（AI 席位）→ 部署。→ `docs/DEPLOY_zh.md` A/B
2. **NSCC**：`bash splendor_ai/scripts/nscc_setup.sh` → 30 分钟吞吐作业 → `qsub -P <项目> splendor_ai/scripts/nscc_train.pbs` → 定期 `nscc_eval.pbs` → `python -m splendor_ai.export`。→ D
3. **Windows 3060**：装 Python + CUDA torch + `requirements-worker.txt` → `.env` → 把 `shared.pt` 放进 `models\` → `run_worker.bat`。→ C
   （模型没训好之前也可以先跑 worker 验证连线，它会用贪心策略。）

## 5. 诚实的预期与风险

- 训练系统只在 CPU 沙箱上验证过小规模 2 人循环与五模式 3 分钟混合冒烟；A100 上的吞吐数字是推算的（README "Training" 有量表与 G4 门槛），
  第一次上节点请先跑 30 分钟作业看 sims/s。
- 唯一可查的 AlphaZero-on-Splendor 前例价值头训练失败过，因此我加了辅助得分头、混合价值目标、短阈值冒烟矩阵和 PPO 备用学习器；
  若 2 人模式在第 10 代仍打不过 mcts 锚点（G5），先按 README 的排查清单处理，不要直接烧一周算力。
- 1v2 duo 与 2v2 的队友协作是自博弈约定，换搭档（人类）可能变弱；评估表单独列出 solo/duo 与"搭配历史检查点"的胜率。
- 既有规则问题见 `docs/KNOWN_ISSUES.md`（贵族选择孤立态、认输者计分、无合法动作时无 pass）；引擎与 AI 都忠实模拟当前服务器行为。
