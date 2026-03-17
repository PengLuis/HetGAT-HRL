from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import imageio.v2 as imageio
import matplotlib
import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

from hetgat_hrl.agents.actor_critic import RuleBasedLowLevelPolicy
from hetgat_hrl.core.mdp_spec import AgentKind, EnvConfig, TaskKind
from hetgat_hrl.envs.base_env import BaseHeteroDisasterEnv
from hetgat_hrl.hrl.planner import RiskTriggeredHRLPlanner
from tools.run_30iter_and_make_step_viewer import (
    adapt_trace_to_command_center_data,
    build_viewer_html_with_template,
    snapshot,
)

matplotlib.use("Agg")


@dataclass
class EvalCase:
    name: str
    config_path: Path
    checkpoint_path: Path
    seed: int = 0
    decision_interval: int = 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--results-root",
        type=str,
        default=r"E:\HetGAT\HetGAT-HRL\training_results",
    )
    p.add_argument(
        "--panel-template",
        type=str,
        default=r"E:\HetGAT\HetGAT-HRL\code_and_models\tools\templates\hetgat_command_center_panel.html",
    )
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--fps", type=int, default=5)
    return p.parse_args()


def _flatten_cfg(cfg_yaml: Dict[str, Any], seed: int) -> EnvConfig:
    env = dict(cfg_yaml.get("env", {}))
    phy = dict(cfg_yaml.get("physics", {}))
    rew = dict(cfg_yaml.get("reward", {}))
    dst = dict(cfg_yaml.get("disturbance", {}))
    merged: Dict[str, Any] = {}
    merged.update(env)
    merged.update(phy)
    merged.update(rew)
    merged.update(dst)
    merged["seed"] = int(seed)
    return EnvConfig(**merged)


def _ordered_agents(env: BaseHeteroDisasterEnv) -> List[str]:
    return sorted(
        env.state.agents.keys(),
        key=lambda aid: 0 if env.state.agents[aid].kind == AgentKind.UAV else 1,
    )


def _choose_goals_greedy(
    planner: RiskTriggeredHRLPlanner,
    env: BaseHeteroDisasterEnv,
) -> Dict[str, Optional[str]]:
    goals: Dict[str, Optional[str]] = {}
    used_tasks = set()

    for aid in _ordered_agents(env):
        cands = planner._candidate_tasks(env, aid)  # noqa: SLF001
        tids: List[str] = []
        feats: List[List[float]] = []
        for t in cands:
            tid = str(t.task_id)
            if tid in used_tasks:
                continue
            dist_norm = float(env._agent_distance_to_task(aid, t) / 3000.0)  # noqa: SLF001
            emer = 1.0 if t.kind == TaskKind.EMERGENCY else 0.0
            tids.append(tid)
            feats.append([dist_norm, emer])
        if not tids:
            goals[aid] = None
            continue

        x = torch.tensor(feats, dtype=torch.float32)
        logits = planner.allocator(x).view(-1)
        idx = int(torch.argmax(logits).item())
        chosen_tid = tids[idx]
        goals[aid] = chosen_tid
        used_tasks.add(chosen_tid)
    return goals


def _resolve_agent_xy(
    aid: str,
    a: Dict[str, Any],
    node_xy: Dict[int, Tuple[float, float]],
    transit_total_cache: Optional[Dict[str, Tuple[int, int, float]]] = None,
    dt_seconds: float = 1.0,
) -> Tuple[float, float]:
    transit = a.get("transit")
    if isinstance(transit, list) and len(transit) >= 3:
        n0, n1, raw = int(transit[0]), int(transit[1]), float(transit[2])
        p0 = node_xy.get(n0)
        p1 = node_xy.get(n1)
        if p0 is not None and p1 is not None:
            prog = 0.0
            if transit_total_cache is not None:
                prev = transit_total_cache.get(str(aid))
                if prev is None or int(prev[0]) != n0 or int(prev[1]) != n1:
                    total = max(float(raw), float(dt_seconds), 1e-6)
                    transit_total_cache[str(aid)] = (n0, n1, float(total))
                else:
                    total = max(float(prev[2]), float(raw), 1e-6)
                    transit_total_cache[str(aid)] = (n0, n1, float(total))
                prog = 1.0 - float(raw / max(total, 1e-6))
            else:
                prog = 0.0
            prog = max(0.0, min(1.0, float(prog)))
            return (
                p0[0] + (p1[0] - p0[0]) * prog,
                p0[1] + (p1[1] - p0[1]) * prog,
            )
    else:
        if transit_total_cache is not None:
            transit_total_cache.pop(str(aid), None)
    pos = a.get("pos_xy")
    if isinstance(pos, list) and len(pos) >= 2:
        return float(pos[0]), float(pos[1])
    node = a.get("node")
    if node is not None and int(node) in node_xy:
        return node_xy[int(node)]
    return 0.0, 0.0


def _collect_case_best_episode(
    case: EvalCase,
    episodes: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    cfg_yaml = yaml.safe_load(case.config_path.read_text(encoding="utf-8"))
    cfg = _flatten_cfg(cfg_yaml, seed=case.seed)

    env = BaseHeteroDisasterEnv(cfg)
    planner = RiskTriggeredHRLPlanner(
        decision_interval=max(int(case.decision_interval), 1),
        seed=case.seed,
    )
    low = RuleBasedLowLevelPolicy(seed=case.seed)

    payload = torch.load(case.checkpoint_path, map_location="cpu")
    state = payload.get("allocator_state_dict", payload)
    planner.allocator.load_state_dict(state, strict=True)
    planner.allocator.eval()

    best_key = (-1.0, -1e18)
    best_trace: Optional[List[Dict[str, Any]]] = None
    best_seed = case.seed
    rows: List[Dict[str, Any]] = []

    for ep in range(int(episodes)):
        ep_seed = int(case.seed + ep)
        state_ep = env.reset(seed=ep_seed)
        trace = [
            snapshot(
                env,
                info={"episode_start": True},
                rewards={aid: 0.0 for aid in state_ep.agents},
            )
        ]

        reward_sum = 0.0
        last_info: Dict[str, Any] = {}
        while not state_ep.done:
            goals = _choose_goals_greedy(planner, env)
            env.set_recommended_goals(goals)
            actions = low.act(env, high_goals=goals)
            out = env.step(actions)
            state_ep = out.state
            last_info = out.info
            reward_sum += float(sum(out.rewards.values()) / max(len(out.rewards), 1))
            trace.append(snapshot(env, info=out.info, rewards=out.rewards))

        completion = float(last_info.get("task_completion_rate", 0.0))
        reward_mean = float(reward_sum / max(state_ep.step_index, 1))
        row = {
            "episode": ep,
            "seed": ep_seed,
            "episode_reward_mean": reward_mean,
            "task_completion_rate": completion,
            "steps": int(state_ep.step_index),
        }
        rows.append(row)

        key = (completion, reward_mean)
        if key > best_key:
            best_key = key
            best_trace = trace
            best_seed = ep_seed

    if best_trace is None:
        raise RuntimeError(f"No episode trace collected for case={case.name}")

    nodes = [
        {"id": int(i), "x": float(n.x), "y": float(n.y)}
        for i, n in sorted(env.topology.nodes.items(), key=lambda kv: kv[0])
    ]
    edges: List[List[int]] = []
    for src, nbs in env.topology.adjacency.items():
        for dst in nbs:
            if int(src) < int(dst):
                edges.append([int(src), int(dst)])

    raw_payload = {
        "meta": {
            "case": case.name,
            "episodes": int(episodes),
            "best_seed": int(best_seed),
            "best_key": {
                "task_completion_rate": float(best_key[0]),
                "episode_reward_mean": float(best_key[1]),
            },
            "config_path": str(case.config_path),
            "checkpoint_path": str(case.checkpoint_path),
        },
        "nodes": nodes,
        "edges": edges,
        "frames": best_trace,
    }
    return raw_payload, {"rows": rows}


def _render_video_from_payload(
    raw_payload: Dict[str, Any],
    out_mp4: Path,
    fps: int = 5,
) -> None:
    nodes: List[Dict[str, Any]] = raw_payload["nodes"]
    edges: List[List[int]] = raw_payload["edges"]
    frames: List[Dict[str, Any]] = raw_payload["frames"]

    node_xy: Dict[int, Tuple[float, float]] = {
        int(n["id"]): (float(n["x"]), float(n["y"])) for n in nodes
    }
    xs = [p[0] for p in node_xy.values()]
    ys = [p[1] for p in node_xy.values()]
    margin = 220.0
    xlim = (min(xs) - margin, max(xs) + margin)
    ylim = (min(ys) - margin, max(ys) + margin)

    writer = imageio.get_writer(str(out_mp4), fps=int(fps))
    transit_total_cache: Dict[str, Tuple[int, int, float]] = {}
    try:
        for fr in frames:
            fig, ax = plt.subplots(figsize=(12, 8), dpi=120)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_aspect("equal", adjustable="box")
            ax.set_facecolor("#0b1220")
            fig.patch.set_facecolor("#0b1220")
            ax.grid(color="#1f2a40", alpha=0.15, linewidth=0.5)

            hazard = fr.get("hazard_field", {}) or {}
            storm = hazard.get("storm_center")
            rmw = float(hazard.get("rmw", 0.0) or 0.0)
            if isinstance(storm, list) and len(storm) >= 2 and rmw > 1.0:
                scx, scy = float(storm[0]), float(storm[1])
                c1 = plt.Circle((scx, scy), rmw * 1.1, color=(0.20, 0.55, 1.0, 0.10))
                c2 = plt.Circle((scx, scy), rmw * 0.65, color=(0.55, 0.85, 1.0, 0.08))
                ax.add_patch(c1)
                ax.add_patch(c2)
                ax.text(scx, scy, "🌀", fontsize=14, ha="center", va="center")

            quake = hazard.get("quake_epicenter")
            if isinstance(quake, list) and len(quake) >= 2:
                qx, qy = float(quake[0]), float(quake[1])
                for rr, aa in [(250.0, 0.35), (520.0, 0.22), (860.0, 0.14)]:
                    cc = plt.Circle(
                        (qx, qy),
                        rr,
                        fill=False,
                        linestyle="--",
                        linewidth=1.1,
                        color=(1.0, 0.25, 0.20, aa),
                    )
                    ax.add_patch(cc)
                ax.text(qx, qy, "💥", fontsize=13, ha="center", va="center")

            blocked_set = {
                (min(int(a), int(b)), max(int(a), int(b)))
                for (a, b) in (fr.get("blocked_edges") or [])
            }
            normal_lines = []
            blocked_lines = []
            for a, b in edges:
                p0 = node_xy[int(a)]
                p1 = node_xy[int(b)]
                k = (min(int(a), int(b)), max(int(a), int(b)))
                if k in blocked_set:
                    blocked_lines.append([p0, p1])
                else:
                    normal_lines.append([p0, p1])
            if normal_lines:
                lc = LineCollection(normal_lines, colors="#4e5f7d", linewidths=0.9, alpha=0.45)
                ax.add_collection(lc)
            if blocked_lines:
                lc2 = LineCollection(
                    blocked_lines,
                    colors="#ff5252",
                    linewidths=1.6,
                    alpha=0.9,
                    linestyles="dashed",
                )
                ax.add_collection(lc2)

            ax.scatter(
                [node_xy[i][0] for i in sorted(node_xy.keys())],
                [node_xy[i][1] for i in sorted(node_xy.keys())],
                s=9,
                c="#b7c3d8",
                alpha=0.72,
                zorder=3,
            )

            tasks = fr.get("tasks", {}) or {}
            for tid, t in tasks.items():
                st = str(t.get("status", "pending"))
                if st in {"delivered", "failed"}:
                    continue
                nid = int(t.get("demand_node", 0))
                tx, ty = node_xy.get(nid, (0.0, 0.0))
                kind = str(t.get("kind", "normal"))
                if kind == "emergency":
                    ax.scatter([tx], [ty], s=62, c="#ff4d4f", marker="P", zorder=5)
                else:
                    ax.scatter([tx], [ty], s=42, c="#48dbfb", marker="D", zorder=5)
                ax.text(tx + 18, ty + 16, str(tid), fontsize=6.8, color="#f4f7ff", zorder=6)

            agents = fr.get("agents", {}) or {}
            follower_counts: Dict[str, int] = {}
            for _, aa in agents.items():
                ft = aa.get("follow_target")
                if ft is not None:
                    k = str(ft)
                    follower_counts[k] = follower_counts.get(k, 0) + 1
            for aid, a in agents.items():
                dt = float((fr.get("info", {}) or {}).get("dt_seconds", 1.0) or 1.0)
                x, y = _resolve_agent_xy(
                    aid=aid,
                    a=a,
                    node_xy=node_xy,
                    transit_total_cache=transit_total_cache,
                    dt_seconds=dt,
                )
                kind = str(a.get("kind", "uav"))
                if kind == "truck":
                    ax.scatter(
                        [x],
                        [y],
                        s=185,
                        c="#16e0c2",
                        marker="s",
                        zorder=7,
                        edgecolors="#00322a",
                        linewidths=1.1,
                    )
                    piggy = int(follower_counts.get(str(aid), 0))
                    if piggy > 0:
                        ax.text(
                            x,
                            y + 34,
                            f"载机:{piggy}",
                            fontsize=7.2,
                            color="#ffe29a",
                            ha="center",
                            va="bottom",
                            zorder=10,
                        )
                else:
                    ax.scatter([x], [y], s=72, c="#ffd166", marker="o", zorder=8, edgecolors="#4d3f00")
                    battery = a.get("battery")
                    if battery is not None:
                        ax.text(
                            x + 14,
                            y - 10,
                            f"{float(battery)*100:.0f}%",
                            fontsize=6.5,
                            color="#ffe6a6",
                            zorder=9,
                        )
                ax.text(x + 12, y + 12, str(aid), fontsize=6.7, color="#e8f0ff", zorder=9)

            info = fr.get("info", {}) or {}
            step_idx = int(fr.get("step", 0))
            completion = float(info.get("task_completion_rate", 0.0))
            invalid_cnt = int(info.get("invalid_action_count", 0))
            bind_cnt = int(info.get("uav_follow_bind_count_total", 0))
            title = (
                f"{raw_payload['meta'].get('case', '')} | step={step_idx} | "
                f"completion={completion:.3f} | invalid={invalid_cnt} | bind_total={bind_cnt}"
            )
            ax.set_title(title, color="#dbe7ff", fontsize=10)
            ax.tick_params(colors="#8ea2c9", labelsize=7)

            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            arr = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
            writer.append_data(arr)
            plt.close(fig)
    finally:
        writer.close()


def _write_case_outputs(
    out_dir: Path,
    panel_template: Path,
    raw_payload: Dict[str, Any],
    extra_rows: Dict[str, Any],
    fps: int,
) -> Dict[str, str]:
    case_name = str(raw_payload["meta"]["case"])
    case_dir = out_dir / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    raw_path = case_dir / "best_episode_raw_trace.json"
    raw_path.write_text(json.dumps(raw_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    panel_data = adapt_trace_to_command_center_data(raw_payload)
    html = build_viewer_html_with_template(panel_data, panel_template)
    html_path = case_dir / "best_episode_command_center.html"
    html_path.write_text(html, encoding="utf-8")

    rows_path = case_dir / "eval_rows.json"
    rows_path.write_text(json.dumps(extra_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    mp4_path = case_dir / "best_episode.mp4"
    _render_video_from_payload(raw_payload, mp4_path, fps=fps)

    meta_path = case_dir / "summary.json"
    meta_path.write_text(
        json.dumps(raw_payload.get("meta", {}), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "case_dir": str(case_dir),
        "video": str(mp4_path),
        "html": str(html_path),
        "summary": str(meta_path),
    }


def main() -> None:
    args = parse_args()
    root = Path(args.results_root)
    panel_template = Path(args.panel_template)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = root / f"best_scene_videos_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        EvalCase(
            name="phase1_bind340_best",
            config_path=Path(r"E:\HetGAT\HetGAT-HRL\code_and_models\configs\calibrated_short_v1_bind340.yaml"),
            checkpoint_path=Path(r"E:\HetGAT\HetGAT-HRL\training_results\allocator_phase1_bind340_final.pt"),
            seed=0,
            decision_interval=1,
        ),
        EvalCase(
            name="phase2_bind170_best",
            config_path=Path(r"E:\HetGAT\HetGAT-HRL\code_and_models\configs\calibrated_short_v1.yaml"),
            checkpoint_path=Path(r"E:\HetGAT\HetGAT-HRL\training_results\allocator_phase2_bind170_final.pt"),
            seed=0,
            decision_interval=1,
        ),
    ]

    outputs = []
    for case in cases:
        raw_payload, extra_rows = _collect_case_best_episode(case=case, episodes=int(args.episodes))
        out = _write_case_outputs(
            out_dir=out_dir,
            panel_template=panel_template,
            raw_payload=raw_payload,
            extra_rows=extra_rows,
            fps=int(args.fps),
        )
        outputs.append(out)
        print(
            json.dumps(
                {"case": case.name, "video": out["video"], "html": out["html"]},
                ensure_ascii=False,
            )
        )

    index = out_dir / "index.json"
    index.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved index: {index}")


if __name__ == "__main__":
    main()
