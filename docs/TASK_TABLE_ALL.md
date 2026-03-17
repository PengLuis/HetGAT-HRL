# HetGAT-HRL 全量实现任务表（执行版）

更新时间：2026-03-15

## A. 已完成

| ID | 任务 | 状态 | 代码位置 | 完成判据 |
|---|---|---|---|---|
| A1 | 统一 `Δt=20s` 离散时间步 | 已完成 | `core/mdp_spec.py`, `envs/base_env.py` | `EnvConfig` 强制 `dt=dt_seconds=20.0`，环境非20秒直接报错 |
| A2 | 风场改为无散度流函数 + 降雨IDW平滑 | 已完成 | `envs/hazards.py` | 存在 `wind_vector_at()` 与 `rainfall_at()`，场值连续且可查询 |
| A3 | UAV能耗接入逆风分量与降雨项 | 已完成 | `envs/base_env.py` | 飞行能耗乘子包含 headwind/rain 归一化项 |
| A4 | OSM+DEM路网入口 + 平均度约束参数 | 已完成（入口） | `core/topology.py`, `core/mdp_spec.py` | 支持 `map_source=osm_dem` 与 `avg_degree_min/max` |

## B. 执行中（本轮必须完成）

| ID | 任务 | 状态 | 目标文件 | 完成判据 |
|---|---|---|---|---|
| B1 | 2.3 Logistic 宏观破坏概率 `P_macro` | 已完成 | `envs/hazards.py` | `Z -> sigmoid(Z)` 边级宏观破坏 + 坡雨震交叉项 |
| B2 | 双阶段 `lambda` 渗流锁（30%阈值） | 已完成 | `envs/hazards.py` | `<30%:0.012`, `>=30%:0.0005`，并输出 `percolation_phase/lambda` |
| B3 | UGV多因子速度衰减（坡度/粗糙度/载重项） | 已完成 | `envs/base_env.py`, `core/topology.py` | `v=vmax/((1+0.015*slope)*(1+0.55*rough)*(1+0.00035*payload))` |
| B4 | UAV单次极限航程3000m硬约束 | 已完成 | `core/mdp_spec.py`, `envs/base_env.py` | `sortie_distance_m` 超 `uav_max_sortie_m` 直接坠毁 |
| B5 | 通信黑障闭环：全局观测遮蔽+目标冻结 | 已完成 | `envs/base_env.py` | `observe` 遮蔽全局特征；`set_recommended_goals` 对 blocked agent 冻结目标 |
| B6 | HetGAT指数风险掩码模块 | 已完成 | `agents/hetgat_risk.py` + `hrl/planner.py` | `e' = e - beta*exp(clamp(risk)*2)` 已接入高层评分 |
| B7 | SMDP-HRL触发器一致化（间隔/到达/risk_spike） | 已完成 | `hrl/planner.py`, `envs/base_env.py` | 触发包含 interval/risk/resolution/arrival |
| B8 | 奖励系统量级对齐与指标台账 | 已完成 | `core/mdp_spec.py`, `envs/base_env.py` | step/invalid/idle/delivery/timeout/pbrs/crash 全部日志化 |
| B9 | 90组矩阵与消融脚本 | 已完成 | `tools/run_experiment_matrix.py`, `tools/run_ablation_suite.py` | 支持 scale/scenario/seed 网格和消融运行 |
| B10 | Welch t-test + 95%CI统计脚本 | 已完成 | `eval/stats_welch.py`, `tools/analyze_significance.py` | 输出 `t,dof,pvalue,ci95` |
| B11 | L规模零样本泛化评测脚本 | 已完成 | `tools/eval_zero_shot_L.py` | 从M结果选top-k seed并在L-B/C输出评测CSV |

## C. 验收标准

1. `validate_step1.py` 与 `run_smoke_train.py` 能无错误运行。  
2. `step info` 中包含：`dt_seconds`, `avg_degree`, `blocked_ratio`, `percolation_phase`, `risk_spike`。  
3. 实验脚本可在不改代码前提下生成完整矩阵配置和统计输出模板。  
