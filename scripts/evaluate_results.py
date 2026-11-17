"""Score saved predictions and plot model comparisons.

Converted from notebooks/evaluation.ipynb, extended with TEDS, bootstrap confidence intervals, several
results files and paired comparisons. CPU only, no GPU needed.

Inputs: JSONL result files written by `run.py` -> src/inference.py. By default:
  data/evaluation/results.jsonl      the saved baseline predictions (Pix2Struct, Llama-90B, Granite 3.2 runs)
  data/evaluation/results_*.jsonl    this project's experiments, one file each (Granite Vision 4.1 runs, zero-shot 3.2)
A model's label is "<model_name>[_<adapter>]". If the same label occurs in several files (e.g. Granite 3.2
zero-shot: the saved run and ours), the file name is appended in brackets; --drop removes labels by regex.

Computed per prediction: HTML Similarity (variant "fixed" and/or "legacy", src/utils/metrics.py) and
TEDS / TEDS-Struct / strict TEDS (src/utils/teds.py). Written to --out-dir (default outputs/evaluation):
  summary.csv               mean + 95% bootstrap CI per model (+ tokens/s when available)
  scores.jsonl              per-sample scores and components
  html_similarity.png       bar chart of the primary metric with CIs
  tokens_per_second.png     bar chart of single-image speed (rows with batch size 1, or --speed-results)
  eval_loss_vs_metric.png   best eval loss of each fine-tuned adapter vs the primary metric
  paired_vs_<reference>.csv paired bootstrap differences (with --reference)

Usage (from repo root; `bash scripts/tmux_jobs.sh report` runs the project's standard command):
  python scripts/evaluate_results.py --compare-variants --primary teds \\
      --reference KennethTM/pix2struct-base-table2html \\
      --drop 'granite-vision-3\\.2-2b_' --drop 'granite-vision-3\\.2-2b \\[results\\]$'
"""
import argparse
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from src.utils.helpers import ADAPTERS_BASE_PATH
from src.utils.metrics import html_similarity
from src.utils.stats import bootstrap_ci, paired_bootstrap_diff
from src.utils.teds import teds

DEFAULT_RESULTS = ["data/evaluation/results.jsonl", "data/evaluation/results_*.jsonl"]


def _score_row(args):
    gt, pred, variants, with_teds = args
    out = {}
    for v in variants:
        c = html_similarity(gt, pred, variant=v, return_components=True)
        out.update({f"{v}_{k}": val for k, val in c.items()})
    if with_teds:
        out["teds"] = teds(pred, gt)
        out["teds_struct"] = teds(pred, gt, structure_only=True)
        out["teds_strict"] = teds(pred, gt, recover=False)   # truncated / unwrapped outputs count as 0
    return out


def score(results: pd.DataFrame, variants, with_teds: bool, workers: int) -> pd.DataFrame:
    jobs = [(g, p, variants, with_teds) for g, p in zip(results["html"], results["model_prediction"])]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = list(tqdm(ex.map(_score_row, jobs, chunksize=8), total=len(jobs), desc="Scoring"))
    return pd.DataFrame(rows, index=results.index)


def _label(df: pd.DataFrame) -> pd.Series:
    return df["model_name"] + df["adapter"].apply(lambda x: "" if pd.isna(x) else f"_{x}")


def load_results(patterns, drop):
    """Concatenate result files (globs allowed), label models, apply --drop. Adds columns: label (in
    `model_name`, to keep the rest of the script unchanged), `source`, and the raw `adapter`."""
    paths = []
    for pat in patterns:
        paths += sorted(glob.glob(pat)) or ([pat] if os.path.exists(pat) else [])
    paths = list(dict.fromkeys(paths))
    if not paths:
        raise SystemExit(f"No results files found for {patterns}")
    frames = []
    for p in paths:
        df = pd.read_json(p, lines=True)
        df["source"] = Path(p).stem
        frames.append(df)
        print(f"  {p}: {len(df)} predictions")
    df = pd.concat(frames, ignore_index=True)
    df["base_label"] = _label(df)

    def relabel(frame):
        multi = frame.groupby("base_label")["source"].transform("nunique") > 1
        return frame["base_label"].where(~multi, frame["base_label"] + " [" + frame["source"] + "]")

    df["model_name"] = relabel(df)
    for pattern in drop:
        df = df[~df["model_name"].str.contains(pattern, regex=True)]
    df["model_name"] = relabel(df)   # suffixes are only kept where two sources still collide
    if df.empty:
        raise SystemExit("--drop removed every model")
    return df.reset_index(drop=True)


def speed_table(path, tok):
    """Mean tokens/s per base label from a batch-size-1 results file."""
    df = pd.read_json(path, lines=True)
    df["label"] = _label(df)
    df["tokens_per_second"] = df["model_prediction"].apply(
        lambda x: len(tok.tokenize(x, add_special_tokens=False))) / df["execution_time"]
    return df.groupby("label")["tokens_per_second"].mean()


def barplot(df, x, y, title, xlabel, path, xlim=None, labels=None):
    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(14, 0.9 * len(df) + 2.5))
    sns.barplot(x=x, y=y, hue=y, data=df, palette="viridis", dodge=False, legend=False)
    plt.title(title, fontsize=16, weight="bold")
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Model", fontsize=12)
    if xlim:
        plt.xlim(*xlim)
    if labels is not None:
        for i, text in enumerate(labels):
            plt.text(df[x].iloc[i] + 0.01, i, text, color="black", va="center", fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", nargs="+", default=DEFAULT_RESULTS, help="results JSONL files or globs")
    ap.add_argument("--drop", action="append", default=[], help="regex; drop models whose label matches (repeatable)")
    ap.add_argument("--speed-results", default=None, help="batch-size-1 results file used for tokens/s")
    ap.add_argument("--out-dir", default="outputs/evaluation")
    ap.add_argument("--variant", default="fixed", choices=["fixed", "legacy", "demo"])
    ap.add_argument("--compare-variants", action="store_true", help="score with both fixed and legacy")
    ap.add_argument("--no-teds", action="store_true", help="skip TEDS (the slowest metric)")
    ap.add_argument("--primary", default=None, help="metric to rank/plot by: <variant>_score, teds or teds_struct")
    ap.add_argument("--reference", default=None, help="model label for paired bootstrap comparisons")
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    ap.add_argument("--no-speed", action="store_true", help="skip tokens/sec (avoids loading the tokenizer)")
    args = ap.parse_args()

    os.chdir(REPO_ROOT)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- Load results
    print("Loading results:")
    results = load_results(args.results, args.drop)
    print(f"Using {len(results)} predictions for {results['model_name'].nunique()} models")

    # ---- Apply metrics
    variants = ["fixed", "legacy"] if args.compare_variants else [args.variant]
    results = results.join(score(results, variants, not args.no_teds, args.workers))
    metrics = [f"{v}_score" for v in variants] + ([] if args.no_teds else ["teds", "teds_struct", "teds_strict"])
    primary = args.primary or f"{variants[0]}_score"
    if primary not in metrics:
        raise SystemExit(f"--primary {primary!r} not computed; available: {metrics}")

    # ---- Summary with 95% bootstrap CIs
    rows = []
    for name, g in results.groupby("model_name"):
        row = {"model_name": name, "count": len(g)}
        for m in metrics:
            mean, lo, hi = bootstrap_ci(g[m].values)
            row.update({m: mean, f"{m}_lo": lo, f"{m}_hi": hi})
        rows.append(row)
    summary = pd.DataFrame(rows).set_index("model_name")

    # ---- Inference speed: only single-image runs are comparable (batched runs divide the batch time)
    if not args.no_speed:
        from transformers import AutoProcessor

        tok = AutoProcessor.from_pretrained("ibm-granite/granite-vision-3.2-2b", use_fast=True).tokenizer
        single = results[results["batch_size"].fillna(1) == 1] if "batch_size" in results else results
        single = single.assign(tokens_per_second=single["model_prediction"].apply(
            lambda x: len(tok.tokenize(x, add_special_tokens=False))) / single["execution_time"])
        summary["tokens_per_second"] = single.groupby("model_name")["tokens_per_second"].mean()
        if args.speed_results and os.path.exists(args.speed_results):
            for label, tps in speed_table(args.speed_results, tok).items():
                for name in summary.index:
                    if name.split(" [")[0] == label:
                        summary.loc[name, "tokens_per_second"] = tps

    summary = summary.sort_values(primary, ascending=False)
    summary.to_csv(out / "summary.csv")
    results.drop(columns=["html", "model_prediction"]).to_json(out / "scores.jsonl", orient="records", lines=True)
    show = summary[["count"] + [c for m in metrics for c in (m, f"{m}_lo", f"{m}_hi")]]
    print("\n" + show.round(4).to_string())

    # ---- Paired comparisons against a reference model (same images)
    if args.reference:
        wide = results.pivot_table(index="imgid", columns="model_name", values=primary)
        if args.reference not in wide:
            raise SystemExit(f"--reference {args.reference!r} not in {list(wide.columns)}")
        pair_rows = []
        for name in wide.columns:
            if name == args.reference:
                continue
            both = wide[[name, args.reference]].dropna()
            d, lo, hi, p = paired_bootstrap_diff(both[name].values, both[args.reference].values)
            pair_rows.append({"model": name, "n": len(both), "diff_vs_reference": d, "lo": lo, "hi": hi, "p_leq_0": p})
        pairs = pd.DataFrame(pair_rows).sort_values("diff_vs_reference", ascending=False)
        safe = args.reference.replace("/", "_")
        pairs.to_csv(out / f"paired_vs_{safe}.csv", index=False)
        print(f"\nPaired difference in {primary} vs {args.reference} (positive = better than reference):")
        print(pairs.round(4).to_string(index=False))

    # ---- Plot: primary metric per model
    plot_df = summary.reset_index()[["model_name", primary, f"{primary}_lo", f"{primary}_hi", "count"]]
    barplot(
        plot_df, primary, "model_name",
        f"Mean {primary} by Model (95% bootstrap CI)", primary, out / "html_similarity.png", xlim=(0, 1),
        labels=[f"{s:.3f} [{lo:.3f}, {hi:.3f}] n={n}" for s, lo, hi, n in
                zip(plot_df[primary], plot_df[f"{primary}_lo"], plot_df[f"{primary}_hi"], plot_df["count"])],
    )

    # ---- Plot: tokens per second
    if not args.no_speed and "tokens_per_second" in summary:
        speed = summary.reset_index()[["model_name", "tokens_per_second"]].dropna().sort_values(
            "tokens_per_second", ascending=False)
        if len(speed):
            barplot(speed, "tokens_per_second", "model_name", "Mean Tokens per Second (single-image runs)",
                    "Mean Tokens per Second", out / "tokens_per_second.png")

    # ---- Plot: best eval loss of each fine-tuned adapter vs the primary metric (needs run_params.json)
    points = []
    for name, adapter in results.drop_duplicates("model_name")[["model_name", "adapter"]].itertuples(index=False):
        if pd.isna(adapter):
            continue
        params = REPO_ROOT / ADAPTERS_BASE_PATH.format(adapter) / "run_params.json"
        if params.exists():
            loss = json.loads(params.read_text()).get("best_eval_loss")
            if loss is not None:
                points.append((adapter, loss, summary.loc[name, primary]))
    if len(points) >= 2:
        plt.figure(figsize=(8, 6))
        plt.scatter([p[1] for p in points], [p[2] for p in points], color="steelblue")
        for adapter, loss, metric in points:
            plt.annotate(adapter, (loss, metric), textcoords="offset points", xytext=(5, 5), fontsize=9)
        plt.xlabel("Best eval loss (250 held-out training-split samples)")
        plt.ylabel(primary)
        plt.title("Eval loss vs test metric (per fine-tuned model)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out / "eval_loss_vs_metric.png", dpi=150)
        plt.close()

    print(f"\nWrote outputs to {out}/")


if __name__ == "__main__":
    main()
