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

（待 worker 完成后补全：Python 安装、CUDA 版 PyTorch、`pip install -r splendor_ai/requirements-worker.txt`、`.env`、`run_worker.bat`、开机自启。）

## D. NSCC ASPIRE 2A 训练

（待训练系统完成后补全：conda 环境、`requirements.txt`、`qsub scripts/nscc_train.pbs`、续训、评估、导出。）
