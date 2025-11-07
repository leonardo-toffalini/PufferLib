import argparse
import os
import struct
import sys

import numpy as np


def write_fbm_file(
    output_path: str,
    num_paths: int,
    num_points: int,
    H: float,
    T: float,
    dtype: str = "float64",
    seed: int | None = None,
    method: str = "daviesharte",
) -> None:
    try:
        from fbm import FBM
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "The 'fbm' package is required. Install with: pip install fbm"
        ) from exc

    if num_points < 2:
        raise ValueError("num_points must be >= 2 (includes the initial 0 point)")
    if not (0.0 < H < 1.0):
        raise ValueError("H must be in (0, 1)")

    rng = np.random.default_rng(seed)

    # Binary header (little-endian):
    # uint32 magic ('FBM1' -> 0x314D4246),
    # uint16 version (1), uint16 dtype (1=float64, 2=float32),
    # uint32 num_paths, uint32 num_points,
    # float64 H, float64 T,
    # 32 bytes reserved (zeros)
    magic = 0x314D4246
    version = 1
    dtype_code = 1 if dtype == "float64" else 2 if dtype == "float32" else None
    if dtype_code is None:
        raise ValueError("dtype must be 'float64' or 'float32'")

    header_fmt = "<I H H I I d d 32s"
    reserved = b"\x00" * 32

    # Ensure parent directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    # Precompute scale for [0, T]
    scale = float(T) ** float(H)

    # Prepare file for writing
    with open(output_path, "wb") as f:
        f.write(
            struct.pack(
                header_fmt,
                magic,
                version,
                dtype_code,
                int(num_paths),
                int(num_points),
                float(H),
                float(T),
                reserved,
            )
        )

        # Write paths sequentially in row-major: path 0 [M], path 1 [M], ...
        for _ in range(num_paths):
            # fbm() returns array of length n+1 on [0,1]
            n = int(num_points) - 1
            # Provide a per-path seed by advancing the RNG
            path_seed = int(rng.integers(0, np.iinfo(np.uint32).max)) if seed is not None else None
            fbm = FBM(n=n, hurst=H, length=1.0, method=method, seed=path_seed)
            series = fbm.fbm()  # numpy array shape (n+1,)

            # Scale to [0, T] using self-similarity: B_H(t) = T^H * B_H(t/T)
            series = (series.astype(np.float64, copy=False) * scale)

            if dtype == "float32":
                arr = series.astype("<f4", copy=False)
            else:
                arr = series.astype("<f8", copy=False)

            # Ensure little-endian when writing
            if not arr.dtype.isnative:
                arr = arr.byteswap().newbyteorder()

            arr.tofile(f)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate N fractional Brownian motion (fBm) paths using the 'fbm' "
            "Python package and write them to a compact little-endian binary file "
            "that is easy to read from C."
        )
    )
    parser.add_argument("--output", "-o", required=True, help="Output file path")
    parser.add_argument("--num-paths", "-N", type=int, default=1, help="Number of paths (rows)")
    parser.add_argument(
        "--num-points", "-M", type=int, default=257, help="Points per path (columns), including t=0"
    )
    parser.add_argument("--H", type=float, default=0.5, help="Hurst exponent in (0,1)")
    parser.add_argument("--T", type=float, default=1.0, help="Time horizon length")
    parser.add_argument(
        "--dtype",
        choices=["float64", "float32"],
        default="float64",
        help="Data precision to store (binary).",
    )
    parser.add_argument(
        "--method",
        choices=["daviesharte", "hosking"],
        default="daviesharte",
        help="fBm generation method",
    )
    parser.add_argument("--seed", type=int, default=None, help="Base RNG seed for reproducibility")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        write_fbm_file(
            output_path=args.output,
            num_paths=args.num_paths,
            num_points=args.num_points,
            H=args.H,
            T=args.T,
            dtype=args.dtype,
            seed=args.seed,
            method=args.method,
        )
    except Exception as exc:  # pragma: no cover
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


