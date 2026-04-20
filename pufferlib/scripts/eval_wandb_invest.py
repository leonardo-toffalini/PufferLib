import argparse
import csv
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch

import pufferlib
from pufferlib.pufferl import load_env as _load_env, load_policy as _load_policy, load_config as _load_config


def find_wandb_run_id(entity: Optional[str], project: str, tag: str) -> str:
    """Find the most recent finished wandb run id with the given tag."""
    import wandb

    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    # Filter runs containing the tag; prefer finished or running
    runs = api.runs(
        path=path,
        filters={
            "$and": [
                {"tags": {"$in": [tag]}},
                {"state": {"$in": ["finished", "crashed", "failed", "running"]}},
            ]
        },
        order="-updated_at",
    )
    if not runs:
        raise RuntimeError(f"No runs found for tag '{tag}' in '{path}'")
    return runs[0].id


def safe_load_config(env_name: str) -> dict:
    """Load default config for env_name without consuming this script's CLI args."""
    saved_argv = list(sys.argv)
    try:
        sys.argv = [saved_argv[0]]
        return _load_config(env_name)
    finally:
        sys.argv = saved_argv


@torch.no_grad()
def evaluate_for_T(
    env_name: str,
    args_template: dict,
    run_id: str,
    T: int,
    episodes: int,
    device: str,
) -> List[float]:
    """
    Evaluate the trained policy for a fixed time horizon T (by setting t_min=t_max=T).
    Returns a list of terminal_riskless values, one per episode.
    """
    # Prepare args for this T
    args = args_template.copy()
    # Ensure required nested dicts exist even if not present in config
    args.setdefault("policy", {})
    args.setdefault("rnn", {})
    args["wandb"] = True
    args["neptune"] = False
    args["load_id"] = run_id

    # Use a serial vecenv with a single environment
    args["vec"] = dict(backend="Serial", num_envs=1)

    # Fix the horizon by setting both bounds to T
    env_kwargs = dict(args["env"])
    env_kwargs["time_horizon"] = int(T)
    env_kwargs["t_min"] = int(T)
    env_kwargs["t_max"] = int(T)
    args["env"] = env_kwargs

    # Minimal train config needed by loader/policy
    train_cfg = dict(args.get("train", {}))
    train_cfg.setdefault("device", device)
    args["train"] = train_cfg

    # Build env and policy
    vecenv = _load_env(env_name, {"package": args["package"], "env_name": env_name, **args})
    policy = _load_policy({"package": args["package"], **args}, vecenv)

    # Eval loop closely follows pufferlib.pufferl.eval, simplified for 1 env
    ob, _ = vecenv.reset()
    collected = 0
    results: List[float] = []

    use_rnn = args["train"].get("use_rnn", False)
    hidden_size = getattr(policy, "hidden_size", None)
    state = {}
    if use_rnn and hidden_size is not None:
        state = dict(
            lstm_h=torch.zeros(vecenv.num_agents, hidden_size, device=device),
            lstm_c=torch.zeros(vecenv.num_agents, hidden_size, device=device),
        )

    while collected < episodes:
        ob_t = torch.as_tensor(ob, device=device)
        logits, _ = policy.forward_eval(ob_t, state)
        action, _, _ = pufferlib.pytorch.sample_logits(logits)
        action_np = action.cpu().numpy().reshape(vecenv.action_space.shape)
        if isinstance(vecenv.action_space, pufferlib.spaces.Box):
            action_np = np.clip(action_np, vecenv.action_space.low, vecenv.action_space.high).astype(np.float32)

        ob, _r, _d, _t, info = vecenv.step(action_np)

        # Invest env returns list with a dict; dict contains 'n' and metrics
        if info and isinstance(info, (list, tuple)) and info[0] and "n" in info[0]:
            n = int(info[0]["n"])
            if n > 0 and "terminal_riskless" in info[0]:
                # For Serial env, n should be 1; handle >1 defensively by duplicating
                val = float(info[0]["terminal_riskless"])
                results.extend([val] * n)
                collected += n

    vecenv.close()
    return results[:episodes]


def main():
    parser = argparse.ArgumentParser(description="Evaluate Invest policy from Wandb tag across horizons")
    parser.add_argument("--env-name", type=str, default="puffer_invest")
    parser.add_argument("--wandb-project", type=str, default="pufferlib")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-tag", type=str, default=None, help="Tag to locate the run (optional if --run-id)")
    parser.add_argument("--run-id", type=str, default=None, help="Direct Wandb run id (overrides --wandb-tag)")
    parser.add_argument("--episodes-per-T", type=int, default=32)
    parser.add_argument("--t-min", type=int, default=None, help="Override lower horizon bound")
    parser.add_argument("--t-max", type=int, default=None, help="Override upper horizon bound")
    parser.add_argument("--k", type=int, default=10, help="Stride for T sweep; evaluate T in [t_min, t_max] with step k")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-csv", type=str, default=None, help="Output CSV path")
    args = parser.parse_args()

    # Load base config for env and extract default T range if not overridden
    cfg = safe_load_config(args.env_name)

    # Package comes from config
    package = cfg["package"]
    env_defaults = dict(cfg["env"])
    default_t_min = int(env_defaults.get("t_min", env_defaults.get("time_horizon", 1)))
    default_t_max = int(env_defaults.get("t_max", env_defaults.get("time_horizon", 1)))
    t_min = int(args.t_min) if args.t_min is not None else default_t_min
    t_max = int(args.t_max) if args.t_max is not None else default_t_max
    if t_max < t_min:
        raise ValueError(f"t_max {t_max} must be >= t_min {t_min}")
    if args.k <= 0:
        raise ValueError(f"k must be > 0, got {args.k}")

    # Prepare template args for loaders
    # Preserve default dict behavior where missing nested keys default to {}
    args_template = cfg.copy()
    # Ensure required nested dicts exist
    args_template.setdefault("policy", {})
    args_template.setdefault("rnn", {})
    args_template["package"] = package
    args_template["wandb_project"] = args.wandb_project
    args_template["wandb_group"] = cfg.get("wandb_group", "debug")

    # Ensure at least one locator is provided
    if args.run_id is None and args.wandb_tag is None:
        raise ValueError("Provide either --run-id or --wandb-tag to locate the Wandb run")

    # Resolve run id
    run_id = args.run_id or find_wandb_run_id(args.wandb_entity, args.wandb_project, args.wandb_tag)

    # Output CSV
    out_csv = args.out_csv or f"eval_invest_{args.wandb_project}_{args.wandb_tag or run_id}.csv"
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    # Sweep horizons and evaluate
    rows: List[Tuple[int, float]] = []
    for T in range(t_min, t_max + 1, args.k):
        vals = evaluate_for_T(
            env_name=args.env_name,
            args_template=args_template,
            run_id=run_id,
            T=T,
            episodes=args.episodes_per_T,
            device=args.device,
        )
        rows.extend((T, v) for v in vals)
        print(f"T={T}: collected {len(vals)} terminal_riskless values")

    # Write CSV
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["T", "terminal_riskless"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    main()

