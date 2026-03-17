# Step 1: MDP/SMDP Formal Specification

## 1. Objective

Build a physically grounded MARL decision process for post-earthquake emergency logistics:

- Heterogeneous agents: trucks (graph-constrained) + UAVs (continuous flight).
- Task classes: normal + emergency.
- Dynamic hazards: rainfall, wind, road blockage risk.

## 2. Decision Process

We use a two-time-scale process:

- **Low-level MDP** (every simulation step):
  - Trucks choose legal graph actions (neighbor node move / stay / service).
  - UAVs choose continuous control actions (vx, vy, mode switch).
- **High-level SMDP** (every `K` steps or risk trigger):
  - Select/refresh sub-goals and task routing plans.
  - Triggered by fixed interval or `risk_spike`.

## 3. State \(s_t\)

Joint state includes:

- Agent runtime states:
  - `position`, `velocity`, `battery`, `crashed`, `follow_target`, `transit`.
- Task pool states:
  - `task_id`, `kind`, `location`, `deadline`, `status`, `assigned_to`.
- Hazard field snapshot:
  - node rainfall, wind, blockage posterior/probability.
- Graph topology snapshot:
  - road connectivity, blocked edges.
- Episode meta:
  - `step_index`, `remaining_time`, `done_flags`.

## 4. Action \(a_t\)

- Truck action space: discrete topology actions.
- UAV action space: continuous control + discrete mode toggles.
- Joint action: dictionary keyed by `agent_id`.

## 5. Transition \(P(s_{t+1}|s_t,a_t)\)

Transition order:

1. Validate and apply per-agent action.
2. Advance physical states (truck graph transition, UAV dynamics integration).
3. Update hazards and road conditions.
4. Resolve task pickup/delivery/failure.
5. Resolve battery, crash, and follow/charging effects.

## 6. Reward \(R_t\)

Per-agent reward (no shared reward by default):

- Delivery reward: normal, emergency.
- Penalties: invalid action, crash, timeout, idle-under-task.
- Optional potential shaping under target-locked phase.

## 7. Termination

Episode terminates when any condition is met:

- Time budget exhausted.
- All tasks delivered.
- Optional catastrophic abort rule.

## 8. Interfaces

Implemented in:

- `code_and_models/hetgat_hrl/core/mdp_spec.py`
- `code_and_models/hetgat_hrl/envs/base_env.py`

