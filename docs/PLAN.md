# Splendor（自定义规则版）— 历史回放 + 强化学习 AI + 本地 GPU 推理部署 · 总体计划

> 状态：草稿（等待研究 workflow 的裁决后补全「算法决策」一节）

## 0. Context（为什么做、做成什么样）

现有项目是一个基于《璀璨宝石》改编的在线对战网站：React 19 + Vite + zustand 前端（Vercel），Node/Express + socket.io 后端（Render，`server/index.js`），**规则的唯一真源是 `server/gameLogic.js`**。本版规则与官方不同（已完整读过并确认）：

- 卡牌：40/30/20，由循环模板生成，分值与官方不同（T1 有 1 分卡，T2 最高 3 分，T3 为 3–5 分），id 固定 0–89；贵族 10 张全为 3 分，翻开 n+1 张，id 0–9。
- **10 枚硬上限、不可弃币**：超出即禁止选取；持币 10 时不能拿任何宝石；预留只在持币 <10 且有金币时给金币。
- 拿 2 同色需该堆 ≥4；供应/上限不足时允许被迫少拿（1 或 2 枚）。
- 预留先拿金币、不可取消；从牌堆预留是隐藏信息；对手预留牌在客户端隐藏。
- 贵族在**任何动作**结束后检查：恰好 1 张自动获得，多张则选择。
- 模式：2/3/4 人个人（15 分触发末轮，回到首位玩家结束，同分少卡者胜）；**1v2**（solo 坐 0 号位先手，solo≥15、duo 合计≥34，超额多者胜，末轮不可撤销）；**2v2**（队伍总分 >30 且第二高分不低于对方第二高分，可撤销）。
- 边界：持币 10 + 预留 3 + 买不起任何卡 → 服务器没有 pass 动作，玩家无合法动作（罕见，属既有行为，本计划不改规则，只在 AI 侧兜底）。

三项交付（全部**附加式**，不改动现有玩法/接口语义）：

1. **历史回放**：每局结束后把最小化 JSON 写入一个**单独的 GitHub 仓库**；开始界面可浏览历史对局并回放（自动播放/倍速/暂停/单步/退出，随机视角，桌面与手机布局与对局一致）。
2. **可真实训练的 AI**（`splendor_ai/` 子目录）：完全按本版规则的 Python 引擎 + 分布式训练（NSCC ASPIRE 2A：1 节点 4×A100‑40G，PBS `ai` 队列，每 GPU 配 16 核）+ 评估 + 导出；覆盖 **2 人、1v2（AI 当 solo / AI 当 duo）、2v2**（3/4 人个人模式作为同一管线的可选配置）。
3. **部署**：本地 Windows 10/11 + RTX 3060 上运行推理 worker，**主动出站**连接 Render；大厅可添加 AI 席位；轮到 AI 时服务器把局面推给 worker，worker 返回动作，服务器按原有 `processAction` 落子。全免费方案。

流程要求：本文件获批后进入执行；在仓库维护 `docs/PLAN.md`（本计划）与 `docs/PROGRESS.md`（逐项状态），由多个 sub‑agent（opus，max effort）并行实现、独立评审、对抗验证。

---

## 1. 前期研究结论（已完成部分）

### 1.1 用户指定仓库 chlligence/PPO-splendorReinforcementLearning（已逐文件精读）
- SB3 `MaskablePPO`，2 人、**无贵族**、官方卡表；203 维观测、51 动作（含 pass）；PBRS 势函数 + 手工启发奖励；ELO 加权对手池（最新 50% / 均匀 10% / 随机 5% / ELO‑softmax），对手权重热插拔；座位交替；同种子换座评估；22 进程 ×50 万步 ×100 代。
- 与本版差异：超过 10 枚用**自动弃币启发式**（我们是禁止选取）；缺少「拿 2 不同色」；预留一律给金币（我们受 10 上限约束）；只支持 2 人；单 GPU、无 DDP；奖励塑形依赖手工规则，迁移性存疑。
- 可借鉴：动作掩码 PPO 骨架、对手池与热插拔、座位交替、γ 一致的 PBRS、配对种子换座评估、Elo 记账。

### 1.2 其他候选（已定位，正由 workflow 深读）
| 项目 | 方法 | 备注 |
|---|---|---|
| inhabae/Splendor-Zero | AlphaZero 式 + IS‑MCTS，C++ 引擎 | 声称 Spendee 榜首 2068 Elo（最强证据） |
| cestpasphoto/alpha-zero-general | Numba 加速 AlphaZero，内置 Splendor | 成熟、快 |
| seal256/splendor | MCTS + 策略网络 | 机会节点（翻牌）处理经验 |
| BreckEmert/Splendor-AI | Dueling DDQN（+MCTS 分支） | 自称超人类 |
| Nikola995/Splendor-AI | PettingZoo 多智能体环境 | 环境 API 设计参考 |
| roeey777 / filipmlynarski | 课程/DRL 项目 | 启发式基线 |
| davidADSP/SIMPLE | 多人回合制自博弈 PPO 框架 | 3–4 人自博弈组织方式 |
| Rinascimento (IEEE CoG) | 统计前向规划 / 事件价值函数 | Splendor 博弈结构分析 |
| DiVA 硕士论文 | Splendor 超人类 AI | 方法与教训 |

两名独立裁判（「最大化棋力」/「最大化一次实现即可用」）的结构化结论将写入 §4。

---

## 2. 交付一：历史回放（必做，先做）

### 2.1 存储格式（最小化，约 1.5–3 KB/局）
`replays/YYYY/MM/<gameId>.json`
```json
{"v":1,"id":"game-…","t":1725280000000,"mode":"INDIVIDUAL|TEAM|ONE_V_TWO","layout":null,
 "n":3,"players":[{"u":"alice","a":7,"team":0,"ai":false},…],"first":1,
 "setup":{"board":[[ids×4],[…],[…]],"decks":[[ids…],[…],[…]],"tiles":[ids…],"clock":true},
 "actions":[[0,"G",[0,1,2]],[1,"R",37],[2,"RD",2],[0,"B",12,"b"],[0,"N",4],[1,"X"],[2,"T"]],
 "result":{"phase":"GAME_OVER","scores":[…],"cards":[…],"winners":[…],"rating":[…]}}
```
动作码：`G` 拿宝石(颜色列表) / `R` 预留明牌(cardId) / `RD` 预留牌堆(tier) / `B` 买(cardId, `b`|`r`) / `N` 选贵族(tileId) / `X` 认输 / `T` 超时。金币是否获得、贵族自动获得等**由重放引擎确定性重算**，不存。
`replays/index.json`：`[{id,t,mode,n,players:[u…],winners,turns}]` 倒序追加（GitHub Contents API，读 sha→写，409 冲突重试 3 次）。

### 2.2 服务器（`server/`，全部新增文件 + 少量挂钩）
- 新 `server/replayRecorder.js`：`begin(room)` 抓初始 setup（牌堆顺序取自 `room.gameState.decks`）；`onAction(room, result)` 只在**回合完成**时追加（`SELECT_GEM` 仅 `completed===true`；`ENTER_RESERVE`/`CANCEL_GEMS` 不记；`RESERVE_*`/`BUY_CARD`/`CHOOSE_TILE`/`RESIGN`/`TIMEOUT` 记）；`finish(room)` 在 `GAME_OVER` 后生成 JSON → 内存 LRU（最近 100 局，无 GitHub 配置时也能用）→ 异步推 GitHub。
- 新 `server/replayGithub.js`：`fetch` 调 Contents API；env：`REPLAY_GITHUB_TOKEN`、`REPLAY_GITHUB_REPO`(owner/name)、`REPLAY_GITHUB_BRANCH`(默认 main)、`REPLAY_GITHUB_DIR`(默认 replays)。未配置 → 仅内存。
- 新 `server/replayEngine.js`：`reconstruct(replayJson)` = `createInitialGameState(players, {gameMode, teamLayout, unlimitedTime:true, firstPlayerIndex})` 后覆盖 board/decks/bonusTiles/currentPlayerIndex/roundStartPlayer，再逐条用**同一个** `processAction`/`processResign` 重放，产出 `frames[]`（每帧：`state`(clientView，含所有预留牌)、`action`、`actionResult`(含 `selected/gemsReturned/tileClaimed` 供动画)）。零逻辑重复，永不与服务器规则漂移。
- 新 REST：`GET /api/replays?limit&offset`（index 缓存，60s 刷新）、`GET /api/replays/:id`（取 JSON→重建→LRU 缓存 20 局）。
- `server/index.js` 挂钩（各 1 行）：`startGame()` 后 `begin`；`game_action` 成功后 `onAction`；`resign`/`eliminateTimedOutPlayer` 后记 `X`/`T`；任何一处进入 `GAME_OVER` 后 `finish`（在 `broadcastProcessedAction` 之后以带上 `ratingChanges`）。`quit_room`/闲置回收 → 丢弃不存。

### 2.3 前端（`src/replay/` 新目录 + 少量挂钩）
- `types.ts`：新增 `AppPhase` 值 `'REPLAY_BROWSER' | 'REPLAY_VIEWER'` 与回放类型（附加）。
- `gameStore.ts`：新增 `openReplayBrowser()/closeReplays()/openReplay(id)`（附加字段与动作）。
- `LoginScreen.tsx`：主视图与「已登录」视图各加一个 **Replays** 按钮（英文 UI）。
- `ReplayBrowser.tsx`：列表（日期、模式、玩家头像/名字、胜者、回合数），点击进入。手机适配（与登录卡片同风格）。
- `ReplayViewer.tsx`：**复用** `CardView/DeckView/NobleTiles/GemSupply/PlayerInfo`，布局类名与 `GameBoard` 一致（`game-shell`、`mobile-market-width`、左右/顶部对手面板、底部自面板），不接任何操作回调（只读）。视角默认随机，控制栏可切换。
  - 控制栏（桌面右上、手机底部固定，44px 触控目标）：Play/Pause、◀ 单步、▶ 单步、倍速循环 0.5×/1×/2×/4×、进度「Turn 12/58」+ 拖动条、视角、Exit。快捷键 Space/←/→。
  - 动画：沿用 GameBoard 的机制——市场卡 `layout` 进出场、贵族消失、宝石供应 ±数字（拿/还/金币）、对手「+1」卡、视角玩家获得贵族的提示；顶部左侧显示「Alice · took Indigo, Jade, Amber」动作字幕。
  - 末帧显示与对局相同的 Game Over 面板（`TeamGameOver` 从 GameBoard.tsx 导出复用，一行附加改动），按钮为 Exit Replay。
- `useReplayPlayer.ts`：帧索引/播放/倍速/定时器状态机，基准 2000ms/帧 ÷ 倍速。

### 2.4 Render 接线（交付时在聊天里给出）
新建仓库（如 `msguidie/splendor-replays`，可私有）→ 生成 fine‑grained PAT（仅该仓库 Contents: Read/Write）→ Render 服务 Environment 添加 `REPLAY_GITHUB_TOKEN / REPLAY_GITHUB_REPO / REPLAY_GITHUB_BRANCH` → 重新部署。

---

## 3. 交付三：AI 席位与本地 GPU worker 接口（服务器侧附加）

- 新 `server/aiBridge.js`：worker 用 socket.io 出站连接并 `ai_worker_auth {secret}`（env `AI_WORKER_SECRET`，未设则整套 AI 功能隐藏）；维护 `aiAvailable` 状态并广播到大厅；`requestMove(room, playerIndex, kind:'MOVE'|'TILE') → Promise`（超时 20s）。
- 大厅：新事件 `lobby_add_ai` / `lobby_remove_ai`（任何大厅成员可加/删，AI 自动 ready，名字 `Bot Alpha/Beta/…`，账号按普通账号建立并计分）；团队模式下允许成员为 AI 选座（`select_team_seat` 附加 `forUsername`）。`WaitingRoom.tsx` 加「Add AI」按钮与机器人徽标（英文）。
- 回合驱动：每次广播后若 `phase==='PLAYING'` 且当前玩家是 AI → 600ms 后向 worker 请求；worker 返回 `{type:'TAKE_GEMS',colors}|{type:'RESERVE',cardId}|{type:'RESERVE_DECK',tier}|{type:'BUY',cardId,source}|{type:'TILE',tileId}|{type:'RESIGN'}`；服务器翻译为既有协议（预留为 `ENTER_RESERVE`+`RESERVE_*` 两步）并走**与人类完全相同**的 `applyGameAction` 路径（把现有 `game_action` 处理体抽成函数供两处调用，行为不变）。
- 发送给 worker 的观测：`clientView` + 公开的预留信息（从回放记录器的动作日志推得：从明牌预留的卡对所有人可见，从牌堆预留的只知层级）+ `pendingTileChoice`。
- 兜底：worker 掉线/超时 → 内置极简贪心（买最高分可负担卡 > 拿最有用宝石 > 预留），避免局面挂死；无合法动作 → 认输。

---

## 4. 交付二：`splendor_ai/`（算法决策见 §4.2，待裁判结论填入）

### 4.1 目录
```
splendor_ai/
  README.md  requirements.txt  requirements-worker.txt  run_worker.bat  .env.example
  splendor_ai/
    rules/  cards.py engine.py actions.py encode.py      # 与 gameLogic.js 逐条对齐的纯 Python 引擎
    env/    game_batch.py workers.py                      # 多局批处理 + 多进程采样，按座位分发观测
    model/  network.py                                    # 策略-价值网络（结构见 §4.2）
    algo/   ppo.py league.py mcts.py                      # 训练算法 / 对手池 / 推理搜索
    train/  config.py configs/{ind2,ovt,team,ind3,ind4}.yaml train.py eval.py export.py
    bots/   random_bot.py greedy_bot.py onestep_bot.py    # 评估基线
    validation/ gen_trajectories.js replay_check.py       # 与 Node 引擎逐步比对（万局随机 + 真实回放）
    worker/ worker.py obs_adapter.py                      # 3060 推理 worker（python-socketio 出站）
  scripts/ nscc_train_ddp.pbs nscc_train_4modes.pbs smoke_cpu.sh
  tests/   test_rules.py test_actions.py test_encode.py
```

### 4.2 算法决策（负责人先验，待两名裁判结论后定稿）
候选：(1) 掩码 PPO 自博弈 + 联赛；(2) AlphaZero 式策略-价值网络 + MCTS 自博弈；(3) 混合：PPO 训练、推理时加搜索（含专家迭代微调）。

**先验推荐 = (3) 分两级：**
- **Level‑1（必须成功）**：多智能体掩码 PPO 自博弈 + 联赛（最新 50% / 均匀历史 / Elo‑softmax / 少量随机），每个由当前策略控制的座位都贡献样本，回报为**终局胜负**（个人：胜 +1 / 负 −1，多人按名次分配；团队：队伍胜负），可选 γ 一致的轻量 PBRS（分差势函数，随训练退火到 0）。价值头输出**每位玩家的胜率向量**（Multiplayer‑AlphaZero 式，同一网络可服务 MCTS）。DDP：每 rank = 1×A100 + 16 CPU 采样进程 + 批量 GPU 推理。此路线在用户指定仓库与 Big‑2 / Generals.io 等 4 人自博弈工作中被反复验证，纯 Python 引擎的吞吐足够（估计 ≥10 万步/秒/节点）。
- **Level‑2（提升上限，可开关）**：推理时 PUCT 搜索（对隐藏信息做确定化采样，机会节点限 1–2 个子节点），用 Level‑1 的策略/价值作先验；若时间允许，再做**专家迭代**（MCTS 改进后的策略作目标）微调——Splendor‑Zero 正是以此登顶 Spendee。3060 推理预算 1–2 s/步，4 GB 显存绰绰有余。

**多模式（回答用户「是否该一个网络」）**：一套代码、**按模式分别训练权重**（`ind2`、`ovt`、`team`；`ind3/ind4` 可选），互不干扰、可分别迭代；2p 权重作 1v2/2v2 的初始化以加速。1v2 内部用**同一个网络 + 角色特征**同时学 solo 与 duo（自博弈天然要求两侧都由策略下），部署时按 AI 所在席位使用；如需极致，可冻结对手池分别微调 solo/duo 专精。2v2 队友共享策略、队伍胜负作回报（共享回报下协作自然涌现；无需通信）。

**执行期深读清单（补充研究，用 workflow 完成）**：petosa/multiplayer-alphazero（多人价值向量与回传）、AlphaZe**（隐藏信息下的 AlphaZero 基线：PIMC/IS‑MCTS）、Deep Catan（多人随机博弈的专家迭代）、Big 2（4 人不完全信息 PPO 自博弈）、Generals.io 超人类自博弈（2026，工程配方）、Gumbel AlphaZero（少模拟次数的策略改进）、Rinascimento 三篇（Splendor 前向模型/分支因子/事件价值函数；框架代码 ivanbravi/RinascimentoFramework）。

### 4.3 环境与动作空间（与服务器完全一致）
- 动作 65 个：拿宝石 30（3 不同 10 + 2 不同 10 + 1 色 5 + 2 同色 5，合法性严格按 `canSelectGem`+`isGemTakeComplete`，含 8/9 枚时的被迫短拿）+ 预留 15（明牌 12 + 牌堆 3）+ 购买 15（明牌 12 + 预留 3）+ 选贵族 5。无 pass；无合法动作 → 训练中跳过并记惩罚，部署中认输。
- 观测：以行动者为视角、座位相对编码；每张卡 = 费用 5 + 颜色 one‑hot 5 + 分 + 层 one‑hot 3 + 我方可负担/缺口 + **卡牌 id embedding**；玩家 = 宝石 6、折扣 5、分、预留数、已知预留牌（自己全知；他人明牌预留可见，牌堆预留只知层级）、贵族数、同队/敌队、角色(solo/duo)、相对座位；全局 = 供应 6、牌堆余量 3、贵族 ≤5（需求 + 我方距离）、模式 one‑hot、回合、末轮标志、阈值进度。
- 验证：`gen_trajectories.js` 用 `server/gameLogic.js` 生成含所有模式的随机合法对局（≥1 万局，覆盖认输/超时/贵族选择），Python 引擎逐步重放并断言状态完全相等；真实回放 JSON 亦可喂入。**这是训练开始前的硬性 go/no‑go。**

### 4.4 训练/评估/导出
- DDP：`torchrun --nproc_per_node=4`，每 rank 自带 CPU 采样进程（16 核）+ GPU 批量推理 + 同步更新；也提供「4 个 GPU 并行跑 4 个模式」的 PBS 脚本。
- 评估：固定种子配对换座；对随机/贪心/一步搜索基线；检查点间 Elo 阶梯；按模式输出报告。
- 导出：`export.py` → `checkpoints/<mode>.pt`（state_dict + 配置），worker 直接加载。

### 4.5 NSCC 与 Windows 3060 说明（交付时在聊天里给出）
- NSCC：登录节点 `module load anaconda` → 建 env → `pip install -r requirements.txt`；`qsub scripts/nscc_train_ddp.pbs`（`-q ai -l select=1:ngpus=4:ncpus=64:mem=440GB`）；日志/检查点路径；如何续训与评估。
- Windows：安装 Python 3.11、CUDA 版 PyTorch（`--index-url https://download.pytorch.org/whl/cu126`）、`pip install -r requirements-worker.txt`；填 `.env`（`SERVER_URL`、`AI_WORKER_SECRET`）；`run_worker.bat` 常驻；Render 上设置 `AI_WORKER_SECRET`。

---

## 5. 执行方式与里程碑
1. `docs/PLAN.md`、`docs/PROGRESS.md` 建立并提交。
2. 研究深读（workflow：逐仓库精读 → 裁判 → 写入 §4.2）。
3. 回放：服务器 → 前端 → 手机/桌面截图验证 → 提交。
4. Python 引擎 + 跨语言验证（硬 gate）。
5. 训练系统（网络/算法/联赛/DDP）+ CPU 冒烟训练（数分钟内胜率高于随机）。
6. 评估/导出 + worker + 服务器 AI 桥 + 大厅 UI → 本地端到端联调（本地起 Node 服务器与 worker）。
7. 文档、requirements、PBS 脚本；全量评审（独立 reviewer + 对抗验证）；提交推送到 `claude/splendor-rl-training-system-v05e9l`。
8. 交付说明（中文，聊天中）。

## 6. 验证清单
- `npm run build`（tsc 严格）通过；`node server/index.js` 启动；脚本用 socket.io-client 打完整局（各模式）→ 内存回放可列出/重建；`/api/replays/:id` 帧数 = 动作数 + 1。
- Playwright（已装 Chromium）对回放浏览器与回放器截图：375×812 与 1280×800。
- `pytest`；跨语言重放 ≥1 万局零差异；`smoke_cpu.sh` 10 分钟内对随机胜率 >80%（2p）。
- worker 与本地服务器联调：AI 在 2p / 1v2 / 2v2 各完成一局；worker 断线时兜底生效。
- 现有功能回归：登录、大厅、开局、断线重连、计时、认输、2v2/1v2 结算路径未变（脚本对局 + 手动截图）。
