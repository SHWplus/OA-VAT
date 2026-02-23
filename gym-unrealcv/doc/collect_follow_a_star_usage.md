### 使用说明：基于 UnrealCV 的无人机避障数据采集

本说明文档介绍如何使用脚本 `tools/collect_follow_a_star.py` 采集用于后续 Diffusion Policy 训练的数据（相机图片 + bbox + 无人机三维轨迹）。脚本支持二维/三维占用图、A* 路径规划（含 3D 平滑）、随机场景布局与合理的起始视角设置。

### 一、前置准备

- **场景二进制**
  - 使用 `load_env.py` 下载相应场景到 `gym_unrealcv/envs/UnrealEnv/`。
  - 确认对应 `setting/*.json` 中的 `env_bin` 指向你下载的可执行文件（Linux 注意大小写）。
- **依赖环境**
  - Python 包：`unrealcv`（随项目）、`opencv-python`（建议，用于图像写盘/距离变换）、`numpy`
  - 可选：`ultralytics`（如要启用 YOLO 检测；默认关闭）
  - 建议在虚拟环境下运行。

### 二、输出数据格式

采集脚本仅保存每帧三项核心字段（每行一条 JSON，文件 `labels.jsonl`）：
- **frame**: 图片文件名（无人机相机视角保存为 `frame_*.jpg`）
- **uav_pose**: 无人机位姿（六元：位置 + 旋转）
- **bbox_gt_xywh**: 目标（人）真值框，[x, y, w, h] 像素坐标

每回合还会保存用于质检/可视化的辅助文件（不进入训练）：
- `occupancy2d.npz`: 二维占用图和 meta
- `occupancy3d.npz`: 三维体素占用图和 meta（启用 `--use-occ3d` 时）
- `topdown.png`: 俯视拼接图（可选）

### 三、快速开始

- **二维占用与二维 A***（默认高度飞行，安全距离约束）：
```bash
python tools/collect_follow_a_star.py \
  --setting "tracking/general/FlexibleRoom.json" \
  --out "data/FlexibleRoom/avoid_2d" \
  --episodes 3 --steps 400 --fps 10 \
  --randomize-layout \
  --cell 50 --inflate 80 \
  --clearance 150 \
  --follow-dist 300 --cam-step 80 --uav-height 260
```

- **启用三维体素占用与三维 A***（含 3D 路径平滑）：
```bash
python tools/collect_follow_a_star.py \
  --setting "tracking/general/FlexibleRoom.json" \
  --out "data/FlexibleRoom/avoid_3d" \
  --episodes 3 --steps 400 --fps 10 \
  --randomize-layout \
  --use-occ3d \
  --cell 50 --inflate 80 \
  --cell-z 100 --inflate-z 120 \
  --clearance 150 --clearance-z 120 \
  --follow-dist 300 --cam-step 80 --uav-height 260
```

说明：
- `--randomize-layout` 会在每回合开始时随机摆放障碍（基于 `setting["env"]["objects"]`），提升数据多样性。
- 二维路径用于“人”的移动（稳定、快速）；无人机优先用 3D A* 路径规划，失败时回退 2D。

### 四、关键参数说明（常用）

- **基础**
  - `--setting`: 场景配置 JSON（相对 `gym_unrealcv/envs/setting/`）
  - `--out`: 输出目录（每回合一个子目录）
  - `--episodes / --steps / --fps`: 回合数、每回合步数、采样帧率
  - `--randomize-layout`: 开启随机障碍布局（推荐）
- **占用图（2D）**
  - `--cell`: XY 栅格分辨率（世界单位），50 常用
  - `--inflate`: XY 占用膨胀（世界单位）
  - `--clearance`: 无人机与障碍最小安全距离（世界单位），会将“离障碍小于 clearance 的空格”标为不可行
- **占用图（3D）**
  - `--use-occ3d`: 启用 3D 体素/3D A*
  - `--cell-z / --inflate-z / --clearance-z`: Z 分辨率、Z 膨胀、Z 安全距离
  - `--z-min / --z-max`: 可控 Z 规划范围（不设则用 `reset_area`）
- **无人机与目标**
  - `--uav-height`: 无人机飞行高度（固定高度，不随地形变化）
  - `--follow-dist`: 无人机相对目标的期望距离（在目标前进方向后方）
  - `--cam-step`: 每步水平步长（合理设为 ≈ `cell`）

### 五、采集流程概述

- 每回合：
  1) （可选）随机布局；生成 2D 占用与安全 2D 占用；（可选）生成 3D 体素占用与 3D 安全占用；
  2) 规划“人”的路线：2D A* 分段 + shortcut 平滑 + 等距下采样；
  3) 设定可见的初始无人机位姿（相机能看到人、距离合理）；
  4) 每步：
     - 人按路径推进（朝向路径方向）；
     - 无人机优先在 3D 安全占用上 A*，失败回退 2D；3D 路径经 `shortcut_path3d` 平滑后执行（水平 + 高度）；
     - 采集相机图片、GT bbox、无人机位姿，写入 `labels.jsonl`。

### 六、数据样例（labels.jsonl 每行）

```json
{"frame": "frame_000123.jpg", "uav_pose": [x,y,z,roll,yaw,pitch], "bbox_gt_xywh": [x,y,w,h]}
```

### 七、推荐配置（FlexibleRoom）

- 室内多障碍：`--cell 50 --inflate 80 --clearance 150`
- 3D：`--cell-z 100 --inflate-z 120 --clearance-z 120`
- 无人机高度：`--uav-height 240~280`（视场景高度与遮挡而定）
- 跟随距离：`--follow-dist 250~350`
- 步长：`--cam-step 60~100`（≈ 一个 XY 栅格）

### 八、常见问题与建议

- **UE4 启动正常但无图/卡住**：
  - 确保 `env_bin` 指向正确的可执行文件（Linux 区分大小写）；
  - 若报端口占用，脚本会自动换端口；仍异常时请关闭多余 UE4 进程。
- **占用图“过黑/过白”**：
  - 2D：适当调小 `--inflate`，或仅用 `setting["env"]["objects"]` 白名单；确保忽略地面/壳体/天空球等；
  - 3D：适度调节 `--inflate-z` 与 `--clearance-z`。
- **路径“锯齿/机械”**：
  - 已加入 3D `shortcut_path3d` 与 2D shortcut；仍不平滑可增大迭代、提高分辨率或缩小步长。
- **相机开局看不到目标**：
  - 采样初始位姿时会用 `object_mask` 检查 bbox；如仍失败，可增大 `--follow-dist` 或选更稀疏布局。
- **性能与速度**：
  - 降低 `--fps` 或 `--episodes/--steps`；关闭 3D 模式或提高 `--cell/--cell-z`；减少随机化频率。

### 九、可选质检

- 叠加俯视图核对占用图是否与场景一致：
```bash
python tools/preview_topdown.py \
  --setting "tracking/general/FlexibleRoom.json" \
  --out "data/FlexibleRoom/preview/topdown_occ.png" \
  --occ "data/FlexibleRoom/avoid_2d/ep000/occupancy2d.npz" \
  --res-w 640 --res-h 480 --fov 40 --overlap 0.15
```

### 十、训练建议

- **训练输入**：连续帧的图像 + bbox_gt（或后续替换为检测框）+ 无人机上一帧/多帧位姿；
- **清洗**：剔除 bbox 太小/缺失的样本；评估路径平滑度与与障碍距离分布，剔除异常样本。 