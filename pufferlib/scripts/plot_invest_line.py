import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def read_grouped(csv_path: str) -> Tuple[List[int], List[float], List[int]]:
    """Read CSV rows and return (sorted_Ts, mean_per_T, count_per_T)."""
    groups: Dict[int, List[float]] = defaultdict(list)
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if "T" not in reader.fieldnames or "terminal_riskless" not in reader.fieldnames:
            raise ValueError(f"CSV must have columns 'T' and 'terminal_riskless', got {reader.fieldnames}")
        for row in reader:
            try:
                T = int(float(row["T"]))
                v = float(row["terminal_riskless"])
            except Exception:
                continue
            groups[T].append(v)

    if not groups:
        raise ValueError("No valid rows found in CSV")

    Ts = sorted(groups.keys())
    means = [float(np.mean(groups[T])) for T in Ts]
    counts = [len(groups[T]) for T in Ts]
    return Ts, means, counts


def main():
    parser = argparse.ArgumentParser(description="Plot mean terminal_riskless vs time horizon (T) from evaluation CSV")
    parser.add_argument("--csv", required=True, help="Path to CSV produced by eval_wandb_invest.py")
    parser.add_argument("--out", type=str, default=None, help="Output image path (png). Defaults to <csv>_line.png")
    parser.add_argument("--marker", type=str, default="o", help="Matplotlib marker style (default 'o')")
    parser.add_argument("--H", type=float, default=0.1, help="H parameter for overlay T^(2H+1)")
    parser.add_argument("--show", action="store_true", help="Show the plot window")
    args = parser.parse_args()

    Ts, means, counts = read_grouped(args.csv)
    Ts = [t * 2 for t in Ts]
    out_path = args.out or os.path.splitext(args.csv)[0] + "_line.png"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(Ts, means, marker=args.marker, label="avg terminal_riskless")
    # Overlay T^(2H+1)
    exponent = 2 * args.H + 1.0
    overlay = [t ** exponent for t in Ts]
    plt.plot(Ts, overlay, linestyle="--", color="tab:red", label=f"T^(2H+1), H={args.H:g}")
    plt.xlabel("time_horizon (T)")
    plt.ylabel("avg terminal_riskless")
    total_rows = sum(counts)
    plt.title(f"Avg terminal_riskless vs T (rows={total_rows}, unique T={len(Ts)})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    if args.show:
        plt.show()
    print(f"Saved line plot to {out_path}")


if __name__ == "__main__":
    main()

