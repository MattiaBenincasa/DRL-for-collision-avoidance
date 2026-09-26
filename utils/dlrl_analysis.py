#!/usr/bin/env python3
"""dlrl_analysis.py — improved analysis and reporting for DLRL training metrics.

Usage:
    python dlrl_analysis.py --csv path/to/metrics.csv --out report.pdf --window 10
"""

import os
import sys
import argparse
import warnings
from textwrap import fill

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import json

# === DEFAULT CONFIG ===
DEFAULT_CSV = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "metrics", "ddqn_metrics_episodes_20260520_212153.csv"))
DEFAULT_OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".", "out", "report_" + os.path.splitext(os.path.basename(DEFAULT_CSV))[0] + ".pdf"))
DEFAULT_ROLLING = 10
SEPS = [",", ";", "\t"]

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "lines.linewidth": 2.2,
})

# Helpful column name candidates (keeps backwards compatibility)
COL_CANDIDATES = {
    "episode": ["episode", "Episode", "ep", "episode_number", "episode_idx"],
    "reward": ["total_reward", "reward", "episode_reward", "reward_total"],
    "steps": ["steps", "step_count", "episode_steps"],
    "epsilon": ["epsilon", "eps"],
    "time": ["episode_time_sec", "episode_time", "time", "duration"],
    "distance": ["distance_to_target", "final_distance", "dist_to_target", "distance"],
    "target_reached": ["target_reached", "success", "reached", "goal_reached"],
    "collided": ["collided", "collision", "collisions", "did_collide"],
}

# === UTILITIES ===

def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def to_bool_series(s):
    if s.dtype == bool:
        return s.fillna(False)
    s_str = s.astype(str).str.lower().str.strip()
    mapping = {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False, "y": True, "n": False}
    return s_str.map(mapping).fillna(False)


def try_read_csv(path, skip_any_hash=False, return_comments=False):
    """Read CSV while optionally skipping comment lines and returning leading comments.

    If skip_any_hash is False (default), lines whose first non-whitespace
    character is '#' are skipped. If True, any line containing '#' is skipped.
    If return_comments is True, also return a list of leading commented lines
    (without the leading '#'). Tries multiple separators from SEPS.
    """
    from io import StringIO
    last_err = None
    # Read file once
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception as e:
        lines = None
        last_err = e

    leading_comments = []
    if lines is not None:
        # collect initial contiguous commented lines (starting with '#')
        for line in lines:
            if line.lstrip().startswith("#"):
                leading_comments.append(line.lstrip()[1:].rstrip("\n"))
            elif line.strip() == "":
                # allow blank lines before header
                continue
            else:
                break

        if skip_any_hash:
            filtered = [line for line in lines if "#" not in line]
        else:
            filtered = [line for line in lines if not line.lstrip().startswith("#")]
        data = StringIO("".join(filtered))
    else:
        data = None

    for sep in SEPS:
        try:
            if data is not None:
                data.seek(0)
                df = pd.read_csv(data, sep=sep)
            else:
                df = pd.read_csv(path, sep=sep)
            if return_comments:
                return df, leading_comments
            return df
        except Exception as e:
            last_err = e
    raise last_err


def compute_slope_and_r2(x, y):
    # x, y: 1D numpy arrays
    if len(x) < 2 or np.all(np.isnan(y)):
        return 0.0, np.nan
    # remove NaNs
    mask = ~np.isnan(y) & ~np.isnan(x)
    if mask.sum() < 2:
        return 0.0, np.nan
    x_m = x[mask]
    y_m = y[mask]
    # linear fit
    p = np.polyfit(x_m, y_m, 1)
    slope = float(p[0])
    intercept = float(p[1])
    y_pred = slope * x_m + intercept
    ss_res = np.sum((y_m - y_pred) ** 2)
    ss_tot = np.sum((y_m - y_m.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return slope, r2


def pct_change(start, end):
    try:
        if start == 0 or np.isnan(start):
            return np.nan
        return (end - start) / abs(start) * 100.0
    except Exception:
        return np.nan


def add_wrapped_text(ax, text, y, size=10, width=95):
    ax.text(0.05, y, fill(str(text), width=width), fontsize=size, va="top")


def parse_config_lines(lines):
    """Parse leading comment lines into a flat list of (key, value) pairs.

    Tries to detect embedded JSON and parse it. If JSON parsing fails,
    falls back to simple line parsing.
    """
    import json

    s = "\n".join(lines).strip()
    # try to extract JSON object from the commented block
    try:
        start = s.find('{')
        end = s.rfind('}')
        if start != -1 and end != -1 and end > start:
            json_str = s[start:end+1]
            obj = json.loads(json_str)
            # flatten
            def flatten(o, parent=''):
                items = []
                if isinstance(o, dict):
                    for k, v in o.items():
                        new_key = f"{parent}.{k}" if parent else k
                        if isinstance(v, (dict, list)):
                            items.extend(flatten(v, new_key))
                        else:
                            items.append((new_key, v))
                elif isinstance(o, list):
                    # if primitives, join; else enumerate
                    if all(not isinstance(el, (dict, list)) for el in o):
                        items.append((parent, ', '.join(map(str, o))))
                    else:
                        for i, el in enumerate(o):
                            items.extend(flatten(el, f"{parent}[{i}]"))
                else:
                    items.append((parent, o))
                return items
            return flatten(obj)
    except Exception:
        pass

    # fallback: simple parsing into pairs
    pairs = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        ln = ln.strip(',')
        if ln in ('{', '}'):
            continue
        if ':' in ln:
            k, v = ln.split(':', 1)
        elif '=' in ln:
            k, v = ln.split('=', 1)
        else:
            parts = ln.split(None, 1)
            k, v = (parts[0], parts[1] if len(parts) > 1 else '')
        pairs.append((k.strip().strip('"'), v.strip().strip('", ')))
    return pairs


def plot_line(pdf, df, xcol, ycol, title, rolling_window=DEFAULT_ROLLING, color="tab:blue"):
    if xcol not in df.columns or ycol not in df.columns:
        return
    plt.figure(figsize=(8.5, 4.5))
    plt.plot(df[xcol], df[ycol], color=color, alpha=0.25, linewidth=1)
    smooth = df[ycol].rolling(rolling_window, min_periods=1).mean()
    plt.plot(df[xcol], smooth, color=color, linewidth=2.5)
    plt.title(title)
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    pdf.savefig()
    plt.close()


# === REPORTING / ANALYSIS ===

def create_report(csv_path, out_pdf, rolling_window=DEFAULT_ROLLING, skip_any_hash=False):
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        sys.exit(1)

    result = try_read_csv(csv_path, skip_any_hash=skip_any_hash, return_comments=True)
    if isinstance(result, tuple):
        df, config_lines = result
    else:
        df = result
        config_lines = []
    if df.empty:
        print("CSV contains no rows. Exiting.")
        sys.exit(1)

    # normalize boolean-ish columns
    for key in ("target_reached", "collided"):
        candidates = COL_CANDIDATES.get(key, [])
        found = find_col(df, candidates)
        if found:
            df[found] = to_bool_series(df[found])
            COL_CANDIDATES[key] = [found]

    # determine key columns
    ep_col = find_col(df, COL_CANDIDATES["episode"]) or "episode"
    if ep_col not in df.columns:
        # create episode index starting at 1
        df["episode"] = np.arange(len(df)) + 1
        ep_col = "episode"

    reward_col = find_col(df, COL_CANDIDATES["reward"]) or None
    steps_col = find_col(df, COL_CANDIDATES["steps"]) or None
    eps_col = find_col(df, COL_CANDIDATES["epsilon"]) or None
    # episode_time deprecated — do not use in analysis
    time_col = None
    dist_col = find_col(df, COL_CANDIDATES["distance"]) or None
    reached_col = find_col(df, COL_CANDIDATES["target_reached"]) or None
    collided_col = find_col(df, COL_CANDIDATES["collided"]) or None

    # coerce numeric columns where appropriate
    for c in [reward_col, steps_col, eps_col, dist_col]:
        if c and c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # compute slopes and pct changes
    notes = []
    engineering_notes = [
        "Check reward scale: large constant penalties can mask small improvements.",
        "Verify epsilon decay schedule: too slow decay can prevent exploitation.",
        "Inspect collision rate vs. reward: frequent collisions may indicate sensor noise or poor obstacle avoidance.",
        "Consider normalizing inputs and using reward clipping to stabilize learning.",
        "If reward variance is high, increase replay buffer or batch size.",
        "Compare performance by spawn_id/target_id to detect scenario bias.",
        "Use moving average (e.g., 20 episodes) to assess learning stability.",
    ]

    x = df[ep_col].to_numpy().astype(float)

    def safe_metrics(col, name):
        if not col or col not in df.columns:
            return None
        y = df[col].to_numpy().astype(float)
        slope, r2 = compute_slope_and_r2(x, y)
        pct = pct_change(y[~np.isnan(y)][0] if (~np.isnan(y)).any() else np.nan, y[~np.isnan(y)][-1] if (~np.isnan(y)).any() else np.nan)
        notes.append(f"{name} slope: {slope:.4g} per episode (r2={r2 if not np.isnan(r2) else 'NA'}), Δ% = {pct:.2f}%")
        return slope

    reward_slope = safe_metrics(reward_col, "Reward")
    steps_slope = safe_metrics(steps_col, "Steps")
    dist_slope = safe_metrics(dist_col, "Distance to target")

    # short textual interpretation
    if reward_slope is not None:
        if reward_slope > 0:
            notes.append("Reward increasing — policy improving across episodes.")
        else:
            notes.append("Reward flat/decreasing — consider reward shaping or tuning.")

    if steps_slope is not None:
        if steps_slope < 0:
            notes.append("Steps decreasing — agent converges faster (likely good).")
        else:
            notes.append("Steps flat/increasing — learning may be slow or unstable.")

    if dist_slope is not None:
        if dist_slope < 0:
            notes.append("Distance to target decreasing — navigation improving.")
        else:
            notes.append("Distance flat/increasing — consider curriculum or sensor preprocessing.")

    # success/collision stats
    success_stats = []
    if reached_col and reached_col in df.columns:
        mask = df[reached_col].astype(bool)
        total = len(df)
        success = int(mask.sum())
        fail = total - success
        success_rate = (success / total) * 100 if total > 0 else 0.0
        success_stats.extend([f"Successes: {success}/{total} ({success_rate:.2f}%)", f"Failures: {fail}/{total} ({100-success_rate:.2f}%)"])

        # run-length streaks
        runs = (mask != mask.shift()).cumsum()
        try:
            max_success_streak = int(df[mask].groupby(runs[mask]).size().max()) if success > 0 else 0
        except Exception:
            max_success_streak = 0
        try:
            max_fail_streak = int(df[~mask].groupby(runs[~mask]).size().max()) if fail > 0 else 0
        except Exception:
            max_fail_streak = 0
        success_stats.extend([f"Max success streak: {max_success_streak}", f"Max fail streak: {max_fail_streak}"])

        for col in [reward_col, steps_col, dist_col]:
            if col and col in df.columns:
                s_mean = df[mask][col].mean() if success > 0 else np.nan
                f_mean = df[~mask][col].mean() if fail > 0 else np.nan
                success_stats.append(f"{col} | success mean: {s_mean:.3f} | failure mean: {f_mean:.3f}")

    if collided_col and collided_col in df.columns:
        col_mask = df[collided_col].astype(bool)
        collisions = int(col_mask.sum())
        collision_rate = collisions / len(df) * 100 if len(df) > 0 else 0.0
        engineering_notes.insert(0, f"Collisions: {collisions}/{len(df)} ({collision_rate:.2f}%) — check obstacle handling.")

    # ensure output directory exists
    out_dir = os.path.dirname(out_pdf)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # build PDF
    with PdfPages(out_pdf) as pdf:
        # Title
        fig = plt.figure(figsize=(8.27, 11.69))
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.95, "dlr_report", ha="center", fontsize=20, weight="bold")
        ax.text(0.5, 0.91, f"File: {os.path.basename(csv_path)}", ha="center", fontsize=11)
        ax.text(0.5, 0.88, f"Episodes: {len(df)}", ha="center", fontsize=11)

        ax.text(0.05, 0.82, "Key Trend Findings:", fontsize=13, weight="bold")
        y = 0.78
        for n in notes:
            add_wrapped_text(ax, f"- {n}", y, size=10)
            y -= 0.05

        ax.text(0.05, y - 0.02, "Engineering Notes (Potential Improvements):", fontsize=13, weight="bold")
        y -= 0.06
        for en in engineering_notes:
            add_wrapped_text(ax, f"- {en}", y, size=10)
            y -= 0.05

        pdf.savefig()
        plt.close()

        # Configuration page (from CSV leading comments)
        if 'config_lines' in locals() and config_lines:
            parsed = parse_config_lines(config_lines)
            if parsed:
                fig = plt.figure(figsize=(8.27, 11.69))
                ax = fig.add_subplot(111)
                ax.axis("off")
                ax.text(0.5, 0.95, "Experiment Configuration", ha="center", fontsize=16, weight="bold")
                y = 0.90
                for k, v in parsed:
                    k_disp = str(k).replace('.', ' -> ')
                    if isinstance(v, (dict, list)):
                        val_disp = json.dumps(v, ensure_ascii=False)
                    else:
                        val_disp = str(v)
                    add_wrapped_text(ax, f"- {k_disp}: {val_disp}", y, size=9, width=90)
                    y -= 0.04
                    if y < 0.12:
                        pdf.savefig()
                        plt.close()
                        fig = plt.figure(figsize=(8.27, 11.69))
                        ax = fig.add_subplot(111)
                        ax.axis("off")
                        y = 0.95
                pdf.savefig()
                plt.close()
            else:
                fig = plt.figure(figsize=(8.27,11.69))
                ax = fig.add_subplot(111)
                ax.axis("off")
                ax.text(0.5,0.95, "Experiment Configuration (raw)", ha="center", fontsize=16, weight="bold")
                y = 0.90
                for ln in config_lines:
                    add_wrapped_text(ax, ln, y, size=9, width=95)
                    y -= 0.035
                    if y < 0.12:
                        pdf.savefig()
                        plt.close()
                        fig = plt.figure(figsize=(8.27, 11.69))
                        ax = fig.add_subplot(111)
                        ax.axis("off")
                        y = 0.95
                pdf.savefig()
                plt.close()

        # Success / collision stats
        if success_stats:
            fig = plt.figure(figsize=(8.27, 11.69))
            ax = fig.add_subplot(111)
            ax.axis("off")
            ax.text(0.5, 0.95, "Destination / Success Statistics", ha="center", fontsize=16, weight="bold")
            y = 0.90
            for s in success_stats:
                add_wrapped_text(ax, f"- {s}", y, size=10)
                y -= 0.04
            pdf.savefig()
            plt.close()

        # Summary stats
        numeric_cols = [c for c in [reward_col, steps_col, eps_col, dist_col] if c and c in df.columns]
        if numeric_cols:
            fig = plt.figure(figsize=(8.27, 11.69))
            ax = fig.add_subplot(111)
            ax.axis("off")
            ax.text(0.5, 0.95, "Summary Statistics", ha="center", fontsize=16, weight="bold")
            stats = df[numeric_cols].describe()
            ax.text(0.05, 0.9, stats.to_string(), family="monospace", fontsize=9, va="top")
            pdf.savefig()
            plt.close()

        # Plots
        plot_line(pdf, df, ep_col, reward_col, "Total Reward per Episode", rolling_window, color="tab:blue")
        plot_line(pdf, df, ep_col, steps_col, "Steps per Episode", rolling_window, color="tab:green")
        plot_line(pdf, df, ep_col, eps_col, "Epsilon per Episode", rolling_window, color="black")
        plot_line(pdf, df, ep_col, dist_col, "Final Distance to Target", rolling_window, color="tab:purple")

        # Reward distribution
        if reward_col and reward_col in df.columns:
            plt.figure(figsize=(8.5, 4))
            sns.histplot(data=df, x=reward_col, bins=30, kde=True, color="tab:blue")
            plt.title("Reward Distribution")
            plt.tight_layout()
            pdf.savefig()
            plt.close()

        # Correlation matrix
        if len(numeric_cols) >= 2:
            plt.figure(figsize=(6.5, 5.5))
            corr = df[numeric_cols].corr()
            sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
            plt.title("Correlation Matrix")
            plt.tight_layout()
            pdf.savefig()
            plt.close()

    print(f"PDF report saved to: {out_pdf}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DLRL training CSV analysis and PDF report generator")
    parser.add_argument("--csv", "-c", default=DEFAULT_CSV, help="Path to metrics CSV")
    parser.add_argument("--out", "-o", default=DEFAULT_OUT, help="Output PDF path")
    parser.add_argument("--window", "-w", type=int, default=DEFAULT_ROLLING, help="Rolling window for smoothing")
    parser.add_argument("--skip-any-hash", action="store_true", help="Skip lines that contain '#' anywhere (may remove data with '#')")
    args = parser.parse_args()

    try:
        create_report(args.csv, args.out, rolling_window=args.window, skip_any_hash=args.skip_any_hash)
    except Exception as e:
        warnings.warn(f"Analysis failed: {e}")
        raise
