"""Single-sample qualitative comparison across models.

Converted from notebooks/inference.ipynb. Runs the example, PubTabNet validation image 618932,
through each model. Instead of rendering inline (IPython display), every prediction is written to
outputs/demo/<model>.html (open in a browser) and scored with HTML Similarity and TEDS.

Models (choose with --models, comma separated):
  granite32   ibm-granite/granite-vision-3.2-2b, zero-shot (+ a question-answering check: it can read the
              table but does not write HTML for it)
  granite41   ibm-granite/granite-vision-4.1-4b, zero-shot with its native <tables_html> task tag
  finetuned   Granite Vision 4.1 + one of our adapters (--adapter, default lang_table_only_4); the adapter
              must have been trained first (configs/lang_table_only_4.yaml)
  pix2struct  KennethTM/pix2struct-base-table2html

Usage (from repo root):
  CUDA_VISIBLE_DEVICES=0 python scripts/inference_demo.py --models granite32,granite41,pix2struct
"""
import argparse
import gc
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
from datasets import load_dataset

from src.utils.metrics import html_similarity
from src.utils.teds import teds

TABLE_PROMPT = "Convert table to HTML (<table> ... </table>)"
QA_PROMPT = 'What is the Kappa value for the question "Do you communicate with this power?" in the present study?'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def clean_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_sample(dataset: str, sample_id: int, max_index: int):
    test = load_dataset(dataset, split="validation", streaming=True)
    for i, sample in enumerate(test):
        if sample["imgid"] == sample_id or i == max_index:
            break
    print(f"Using sample imgid={sample['imgid']} size={sample['image'].size}")
    return sample


def _granite(model_name, adapter=None):
    from src.models.granite_vision.transformers_library import LLM

    return LLM(model_name, adapter=adapter)


def run_granite32(sample, args):
    model = _granite("ibm-granite/granite-vision-3.2-2b")
    html = model.predict(sample["image"], max_new_tokens=1500, query=TABLE_PROMPT)
    print("QA check ->", model.predict(sample["image"], max_new_tokens=64, query=QA_PROMPT))
    del model
    return html


def run_granite41(sample, args):
    model = _granite("ibm-granite/granite-vision-4.1-4b")
    html = model.predict(sample["image"], max_new_tokens=2048)   # default prompt: <tables_html>
    del model
    return html


def run_finetuned(sample, args):
    model = _granite("ibm-granite/granite-vision-4.1-4b", adapter=args.adapter)
    html = model.predict(sample["image"], max_new_tokens=2048, query=args.prompt)
    del model
    return html


def run_pix2struct(sample, args):
    from transformers import AutoProcessor, Pix2StructForConditionalGeneration

    path = "KennethTM/pix2struct-base-table2html"
    processor = AutoProcessor.from_pretrained(path)
    model = Pix2StructForConditionalGeneration.from_pretrained(path).to(DEVICE).eval()
    encoding = processor(sample["image"], return_tensors="pt", max_patches=1024)
    with torch.inference_mode():
        preds = model.generate(
            flattened_patches=encoding.pop("flattened_patches").to(DEVICE),
            attention_mask=encoding.pop("attention_mask").to(DEVICE),
            max_new_tokens=1024,
        )
    del model
    return processor.tokenizer.batch_decode(preds, skip_special_tokens=True)[0]


RUNNERS = {
    "granite32": run_granite32,
    "granite41": run_granite41,
    "finetuned": run_finetuned,
    "pix2struct": run_pix2struct,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="granite32,granite41,pix2struct")
    ap.add_argument("--dataset", default="apoidea/pubtabnet-html")
    ap.add_argument("--sample-id", type=int, default=618932)
    ap.add_argument("--max-index", type=int, default=25, help="fallback: stop at this stream index")
    ap.add_argument("--adapter", default="lang_table_only_4", help="local adapter for the `finetuned` model")
    ap.add_argument("--prompt", default=None, help="prompt for `finetuned`; must match the one it was trained with "
                                                   "(None = <tables_html>)")
    ap.add_argument("--variant", default="fixed", choices=["fixed", "legacy", "demo"], help="HTML-Sim variant")
    ap.add_argument("--out-dir", default="outputs/demo")
    args = ap.parse_args()

    out = REPO_ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    sample = load_sample(args.dataset, args.sample_id, args.max_index)
    sample["image"].save(out / f"{sample['imgid']}.png")
    (out / "ground_truth.html").write_text(sample["html_table"])

    scores = {}
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"\n===== {name}")
        start = time.time()
        html = RUNNERS[name](sample, args)
        elapsed = time.time() - start
        (out / f"{name}.html").write_text(html)
        scores[name] = (
            html_similarity(html, sample["html_table"], variant=args.variant),
            teds(html, sample["html_table"]),
            teds(html, sample["html_table"], structure_only=True),
        )
        print(f"{name}: HTML-Sim({args.variant})={scores[name][0]:.4f} TEDS={scores[name][1]:.4f} "
              f"TEDS-Struct={scores[name][2]:.4f}  ({elapsed:.1f}s incl. model load) -> {out / (name + '.html')}")
        clean_gpu_memory()

    print(f"\nSummary (HTML-Sim {args.variant} | TEDS | TEDS-Struct):")
    for k, (h, t, ts) in scores.items():
        print(f"  {k:12s} {h:.4f} | {t:.4f} | {ts:.4f}")


if __name__ == "__main__":
    main()
