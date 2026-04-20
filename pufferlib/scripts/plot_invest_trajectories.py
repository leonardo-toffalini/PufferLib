import argparse
import os
from dataclasses import dataclass
from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

import pufferlib
from pufferlib.pufferl import load_env as _load_env, load_policy as _load_policy
from pufferlib.scripts.eval_wandb_invest import find_wandb_run_id, safe_load_config


MAX_PRICE = 1.0
MAX_RISKLESS = 400.0
MAX_RISKY = 400.0


@dataclass
class Trajectory:
    T: int
    steps: np.ndarray
    prices: np.ndarray
    riskless: np.ndarray
    risky_units: np.ndarray
    risky_value: np.ndarray
    total_wealth: np.ndarray
    actions: np.ndarray
    terminal_riskless: float


def _extract_single_observation(ob) -> np.ndarray:
    ob_arr = np.asarray(ob, dtype=np.float32)
    if ob_arr.ndim == 1:
        return ob_arr
    return ob_arr[0]


def _decode_observation(obs: np.ndarray) -> tuple[float, float, float]:
    price = float(obs[1]) * MAX_PRICE
    riskless = float(obs[2]) * MAX_RISKLESS
    risky_units = float(obs[3]) * MAX_RISKY
    return price, riskless, risky_units


def _choose_horizons(t_min: int, t_max: int, count: int) -> List[int]:
    if count <= 1 or t_min == t_max:
        return [int(t_min)]

    values = np.linspace(t_min, t_max, num=count)
    horizons: List[int] = []
    seen = set()
    for value in values:
        T = int(round(float(value)))
        T = max(t_min, min(t_max, T))
        if T not in seen:
            horizons.append(T)
            seen.add(T)

    candidate = t_min
    while len(horizons) < count and candidate <= t_max:
        if candidate not in seen:
            horizons.append(candidate)
            seen.add(candidate)
        candidate += 1

    horizons.sort()
    return horizons[:count]


@torch.no_grad()
def rollout_for_T(
    env_name: str,
    args_template: dict,
    policy: torch.nn.Module,
    T: int,
    device: str,
) -> Trajectory:
    args = args_template.copy()
    env_kwargs = dict(args["env"])
    env_kwargs["time_horizon"] = int(T)
    env_kwargs["t_min"] = int(T)
    env_kwargs["t_max"] = int(T)
    args["env"] = env_kwargs
    args["vec"] = dict(backend="Serial", num_envs=1)

    vecenv = _load_env(env_name, {"package": args["package"], "env_name": env_name, **args})
    ob, _ = vecenv.reset()

    steps: List[int] = []
    prices: List[float] = []
    riskless_values: List[float] = []
    risky_units_values: List[float] = []
    risky_value_values: List[float] = []
    total_wealth_values: List[float] = []
    actions: List[float] = []
    terminal_riskless = float("nan")

    state = {}
    use_rnn = args["train"].get("use_rnn", False)
    hidden_size = getattr(policy, "hidden_size", None)
    if use_rnn and hidden_size is not None:
        state = dict(
            lstm_h=torch.zeros(vecenv.num_agents, hidden_size, device=device),
            lstm_c=torch.zeros(vecenv.num_agents, hidden_size, device=device),
        )

    while True:
        obs_single = _extract_single_observation(ob)
        price, riskless, risky_units = _decode_observation(obs_single)
        risky_value = risky_units * price
        total_wealth = riskless + risky_value

        ob_t = torch.as_tensor(ob, device=device)
        logits, _ = policy.forward_eval(ob_t, state)
        action, _, _ = pufferlib.pytorch.sample_logits(logits)
        action_np = action.cpu().numpy().reshape(vecenv.action_space.shape)
        if isinstance(vecenv.action_space, pufferlib.spaces.Box):
            action_np = np.clip(action_np, vecenv.action_space.low, vecenv.action_space.high).astype(np.float32)
        plotted_action = float(np.asarray(action_np).reshape(-1)[0])

        steps.append(len(steps))
        prices.append(price)
        riskless_values.append(riskless)
        risky_units_values.append(risky_units)
        risky_value_values.append(risky_value)
        total_wealth_values.append(total_wealth)
        actions.append(plotted_action)

        ob, _r, _d, _t, info = vecenv.step(action_np)
        if info and isinstance(info, (list, tuple)) and info[0] and int(info[0].get("n", 0)) > 0:
            terminal_riskless = float(info[0].get("terminal_riskless", float("nan")))
            break

    vecenv.close()
    return Trajectory(
        T=int(T),
        steps=np.asarray(steps, dtype=np.int32),
        prices=np.asarray(prices, dtype=np.float32),
        riskless=np.asarray(riskless_values, dtype=np.float32),
        risky_units=np.asarray(risky_units_values, dtype=np.float32),
        risky_value=np.asarray(risky_value_values, dtype=np.float32),
        total_wealth=np.asarray(total_wealth_values, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        terminal_riskless=terminal_riskless,
    )


def plot_trajectories(trajectories: Sequence[Trajectory], out_path: str, rows: int, cols: int) -> None:
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows), squeeze=False)
    flat_axes = axes.reshape(-1)

    for ax, traj in zip(flat_axes, trajectories):
        # ax.plot(traj.steps, traj.riskless, color="tab:blue", label="riskless")
        # ax.plot(traj.steps, traj.risky_units, color="tab:orange", label="risky units")
        ax.plot(traj.steps[1:], np.diff(traj.risky_units), color="tab:purple", label="risky units speed")
        print(np.diff(traj.risky_units))
        # ax.plot(traj.steps, traj.total_wealth, color="tab:green", linewidth=2, label="total wealth")
        ax.set_xlabel("trade step")
        ax.set_ylabel("portfolio value")
        ax.grid(True, alpha=0.3)
        ax.hlines(100, 0, len(traj.steps), color="black", linestyle="--", alpha=0.3)
        ax.hlines(-100, 0, len(traj.steps), color="black", linestyle="--", alpha=0.3)

        price_ax = ax.twinx()
        price_ax.plot(traj.steps, traj.prices, color="0.35", linestyle="--", alpha=0.8, label="price")
        price_ax.set_ylabel("price")

        title = f"T={traj.T}, steps={len(traj.steps)}, terminal={traj.terminal_riskless:.1f}"
        ax.set_title(title)

        handles, labels = ax.get_legend_handles_labels()
        price_handles, price_labels = price_ax.get_legend_handles_labels()
        ax.legend(handles + price_handles, labels + price_labels, loc="best", fontsize=9)

    for ax in flat_axes[len(trajectories):]:
        ax.axis("off")

    fig.suptitle("Invest policy example trading trajectories", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser(
        description="Plot example Invest trading trajectories for the same Wandb model used in evaluation"
    )
    parser.add_argument("--env-name", type=str, default="puffer_invest")
    parser.add_argument("--wandb-project", type=str, default="pufferlib")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-tag", type=str, default=None, help="Tag to locate the run (optional if --run-id)")
    parser.add_argument("--run-id", type=str, default=None, help="Direct Wandb run id (overrides --wandb-tag)")
    parser.add_argument("--t-min", type=int, default=None, help="Override lower horizon bound")
    parser.add_argument("--t-max", type=int, default=None, help="Override upper horizon bound")
    parser.add_argument("--Ts", type=int, nargs="*", default=None, help="Exact horizons to plot")
    parser.add_argument("--rows", type=int, default=2, help="Number of subplot rows")
    parser.add_argument("--cols", type=int, default=2, help="Number of subplot columns")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output image path. Defaults to results/invest_trajectories_<run>.png",
    )
    args = parser.parse_args()

    if args.run_id is None and args.wandb_tag is None:
        raise ValueError("Provide either --run-id or --wandb-tag to locate the Wandb run")
    if args.rows <= 0 or args.cols <= 0:
        raise ValueError("rows and cols must both be positive")

    cfg = safe_load_config(args.env_name)
    env_defaults = dict(cfg["env"])
    default_t_min = int(env_defaults.get("t_min", env_defaults.get("time_horizon", 1)))
    default_t_max = int(env_defaults.get("t_max", env_defaults.get("time_horizon", 1)))
    t_min = int(args.t_min) if args.t_min is not None else default_t_min
    t_max = int(args.t_max) if args.t_max is not None else default_t_max
    if t_max < t_min:
        raise ValueError(f"t_max {t_max} must be >= t_min {t_min}")

    run_id = args.run_id or find_wandb_run_id(args.wandb_entity, args.wandb_project, args.wandb_tag)
    out_path = args.out or f"results/invest_trajectories_{args.wandb_tag or run_id}.png"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    args_template = cfg.copy()
    args_template.setdefault("policy", {})
    args_template.setdefault("rnn", {})
    args_template["package"] = cfg["package"]
    args_template["wandb_project"] = args.wandb_project
    args_template["wandb_group"] = cfg.get("wandb_group", "debug")
    args_template["wandb"] = True
    args_template["neptune"] = False
    args_template["load_id"] = run_id
    train_cfg = dict(args_template.get("train", {}))
    train_cfg.setdefault("device", args.device)
    args_template["train"] = train_cfg

    panel_count = args.rows * args.cols
    horizons = list(args.Ts) if args.Ts else _choose_horizons(t_min, t_max, panel_count)
    if not horizons:
        raise ValueError("No horizons selected to plot")

    bootstrap_T = int(horizons[0])
    bootstrap_args = args_template.copy()
    bootstrap_env_kwargs = dict(bootstrap_args["env"])
    bootstrap_env_kwargs["time_horizon"] = bootstrap_T
    bootstrap_env_kwargs["t_min"] = bootstrap_T
    bootstrap_env_kwargs["t_max"] = bootstrap_T
    bootstrap_args["env"] = bootstrap_env_kwargs
    bootstrap_args["vec"] = dict(backend="Serial", num_envs=1)

    bootstrap_env = _load_env(
        args.env_name, {"package": bootstrap_args["package"], "env_name": args.env_name, **bootstrap_args}
    )
    policy = _load_policy({"package": bootstrap_args["package"], **bootstrap_args}, bootstrap_env)
    policy.eval()
    bootstrap_env.close()

    trajectories = [
        rollout_for_T(
            env_name=args.env_name,
            args_template=args_template,
            policy=policy,
            T=int(T),
            device=args.device,
        )
        for T in horizons[:panel_count]
    ]
    plot_trajectories(trajectories, out_path=out_path, rows=args.rows, cols=args.cols)
    print(f"Saved trajectory plot to {out_path}")


if __name__ == "__main__":
    main()
