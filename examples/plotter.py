from pathlib import Path
import argparse
import re
from typing import List, Optional, Tuple

def _lazy_imports():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    return np, pd, plt

"""python plotter.py --glob "../data/logs/*/*" --out ../data/logs/ALL/all_algos.png --smooth 5"""

def _find_runs_from_glob(glob_exprs: List[str]) -> List[Path]:
    runs = []
    for expr in glob_exprs:
        for p in Path().glob(expr):
            p = p.resolve()
            if p.is_file() and p.name == "policy_training_progress.csv" and p.parent.name == "record":
                runs.append(p.parent.parent)  
            elif p.is_dir() and (p / "record" / "policy_training_progress.csv").exists():
                runs.append(p)
            elif p.is_dir() and (p / "policy_training_progress.csv").exists():
                if p.name == "record":
                    runs.append(p.parent)
            else:
                rec = list(p.rglob("record/policy_training_progress.csv"))
                runs.extend([q.parent.parent for q in rec])

    seen = set()
    uniq = []
    for r in runs:
        if r not in seen:
            uniq.append(r)
            seen.add(r)
    return uniq


def _label_from_path(run_dir: Path, algo_hint: Optional[str] = None) -> str:
    if algo_hint:
        return algo_hint
    parent = run_dir.parent.name if run_dir.parent else ""
    if re.match(r"^\d{8}[_-]\d{6}$", run_dir.name) and parent:
        return parent
    return run_dir.name


def _read_progress_csv(run_dir: Path, metric: str):
    np, pd, plt = _lazy_imports()
    candidates = [
        run_dir / "record" / "policy_training_progress.csv",
        run_dir / "policy_training_progress.csv",
    ]
    for csv_path in candidates:
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                return None
            if metric in df.columns:
                return df, csv_path
            key = metric.split("/")[-1]
            similar = [c for c in df.columns if key in c]
            if similar:
                return df.rename(columns={similar[0]: metric}), csv_path
            return None
    return None


def _extract_xy(df, metric: str):
    np, pd, plt = _lazy_imports()
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not in columns: {list(df.columns)}")
    y = df[metric].dropna().to_numpy()

    for xc in ["step", "timestep", "global_step"]:
        if xc in df.columns:
            x = df[xc].iloc[:len(y)].to_numpy()
            return x, y, xc

    x = np.arange(len(y))
    return x, y, "Eval #"


def _smooth(y, window: int):
    np, pd, plt = _lazy_imports()  
    if window <= 1 or y.size < window:
        return y
    k = np.ones(window, dtype=float) / window
    return np.convolve(y, k, mode="valid")


def _plot_multi(
    runs,
    out: Path,
    metric: str,
    smooth: int,
    title: Optional[str],
    ylim: Optional[Tuple[float, float]],
    width: float,
    height: float,
) -> None:
    np, pd, plt = _lazy_imports()
    plt.figure(figsize=(width, height))
    any_curve = False
    xlabel_final = None

    for run_dir, label in runs:
        read = _read_progress_csv(run_dir, metric)
        if not read:
            print(f"[warn] skip: {run_dir} (no CSV or metric missing)")
            continue
        df, csv_path = read
        try:
            x, y, xlabel = _extract_xy(df, metric)
            y = _smooth(y, smooth)
            if y.shape[0] != x.shape[0]:
                x = x[:y.shape[0]]
            if xlabel_final is None:
                xlabel_final = xlabel
            elif xlabel_final != xlabel:
                xlabel_final = "Step" 
            lab = _label_from_path(run_dir, label)
            plt.plot(x, y, label=lab)
            any_curve = True
        except Exception as e:
            print(f"[warn] skip: {run_dir} ({e})")

    plt.xlabel(xlabel_final or "Step")
    plt.ylabel(metric)
    plt.title(title or f"Evaluation — {metric}")
    if ylim is not None:
        plt.ylim(ylim)
    plt.grid(alpha=0.3)
    plt.legend()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="plotter",
        description="Plot multiple algorithms' reward curves on a single figure."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--dirs", nargs="+", help="Run directories that contain record/policy_training_progress.csv")
    src.add_argument("--glob", nargs="+", help="Glob(s) to discover run directories, e.g. 'data/logs/*/*' or '**/2025*'")

    parser.add_argument("--labels", nargs="+", help="Legend labels; if omitted, inferred from directory names")
    parser.add_argument("--out", type=Path, default=Path("ALL_algos_reward.png"), help="Output image path")
    parser.add_argument("--metric", type=str, default="eval/normalized_episode_reward", help="CSV column name")
    parser.add_argument("--smooth", type=int, default=1, help="Moving-average window (>=1 means no smoothing)")
    parser.add_argument("--title", type=str, default=None, help="Figure title")
    parser.add_argument("--ylim", type=float, nargs=2, default=None, help="y-axis: --ylim MIN MAX")
    parser.add_argument("--size", type=float, nargs=2, default=[9, 5.5], help="Figure size in inches: --size W H")

    args = parser.parse_args(argv)

    if args.dirs:
        run_dirs = [Path(d).resolve() for d in args.dirs]
    else:
        run_dirs = _find_runs_from_glob(args.glob)

    if not run_dirs:
        raise SystemExit("No run directories found. Check your --dirs or --glob patterns.")

    if args.labels and len(args.labels) != len(run_dirs):
        raise SystemExit(f"--labels length ({len(args.labels)}) must match number of runs ({len(run_dirs)})")

    runs = list(zip(run_dirs, args.labels if args.labels else [None] * len(run_dirs)))

    _plot_multi(
        runs=runs,
        out=args.out.resolve(),
        metric=args.metric,
        smooth=max(1, int(args.smooth)),
        title=args.title,
        ylim=tuple(args.ylim) if args.ylim else None,
        width=float(args.size[0]),
        height=float(args.size[1]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
