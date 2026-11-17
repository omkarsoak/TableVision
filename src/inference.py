"""Run a model over the first N validation images and append predictions to a JSONL results file."""
import hashlib
import importlib
import math
import os
import sys
from itertools import islice
from timeit import default_timer as timer

current_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(current_dir, '..'))

from datasets import load_dataset
from tqdm import tqdm

from src.utils.helpers import append_json_line, read_jsonl, timeout

DEFAULT_RESULTS_PATH = 'data/evaluation/results.jsonl'

# (module, class) per model, imported lazily so that e.g. the watsonx SDK is only needed for Llama.
GRANITE = ("src.models.granite_vision.transformers_library", "LLM")
PIX2STRUCT = ("src.models.pix2struct.transformers", "LLM")
LLAMA = ("src.models.llama.watsonxai", "LLM")
MODELS = {
    "KennethTM/pix2struct-base-table2html": PIX2STRUCT,
    "meta-llama/llama-3-2-11b-vision-instruct": LLAMA,
    "meta-llama/llama-3-2-90b-vision-instruct": LLAMA,
}


def resolve_model_class(model_name: str):
    if model_name in MODELS:
        module, cls = MODELS[model_name]
    elif "granite-vision" in model_name.lower():   # ibm-granite/* base models and merged fine-tunes
        module, cls = GRANITE
    else:
        raise ValueError(f"Model {model_name} not supported. Supported: granite-vision-* and {list(MODELS)}")
    return getattr(importlib.import_module(module), cls), module == GRANITE[0]


def _token_budget(tokenizer, html: str) -> int:
    """Original behavior: next power of two above the reference length (a mild length oracle)."""
    return 2 ** math.ceil(math.log2(max(2, len(tokenizer.encode(html, truncation=False)))))


@timeout(seconds=360)
def predict(model, img, max_new_tokens, query):
    return model.predict(img, max_new_tokens=max_new_tokens, query=query)


@timeout(seconds=1800)
def predict_batch(model, imgs, max_new_tokens, query):
    return model.predict_batch(imgs, max_new_tokens=max_new_tokens, query=query)


def main(model_name: str,
         dataset: str,
         num_test_images: int,
         system_message: str = None,
         prompt: str = None,
         adapter: str = None,
         results_path: str = DEFAULT_RESULTS_PATH,
         batch_size: int = 1,
         max_new_tokens: int = None,
         prompt_family: str = None,
         split: str = 'validation',
         ) -> None:
    """
    :param prompt: instruction text (alias: system_message, the old key). None -> the model's default prompt.
    :param max_new_tokens: fixed generation budget. None -> next power of two above the reference length
        (the earlier behavior; it hints the model's budget, so prefer a fixed value for new comparisons).
    :param batch_size: >1 uses batched generation when the model supports it. `execution_time` is then the
        batch time divided by the batch size, so do not compare speeds across different batch sizes.
    """
    query = prompt or system_message
    model_cls, is_granite = resolve_model_class(model_name)
    kwargs = {"prompt_family": prompt_family} if (is_granite and prompt_family) else {}
    model = model_cls(model_name, adapter=adapter, **kwargs)
    tokenizer = model.processor.tokenizer

    test_ds = load_dataset(dataset, split=split, streaming=True)
    print(f"Streaming {num_test_images} test images from {dataset}[{split}]")

    done = set(res['inference_id'] for res in read_jsonl(results_path)) if os.path.exists(results_path) else set()

    def save(sample, inference_id, output, seconds):
        append_json_line(results_path, {
            'inference_id': inference_id,
            'dataset': dataset,
            'imgid': sample['imgid'],
            'html': sample['html_table'],
            'model_name': model_name,
            'adapter': adapter,
            'prompt': query,
            'model_prediction': output,
            'model_backend': model.device_name,
            'batch_size': batch_size,
            'execution_time': seconds,
        })

    def run_single(sample, inference_id):
        budget = max_new_tokens or _token_budget(tokenizer, sample['html_table'])
        start = timer()
        try:
            output = predict(model, sample['image'], max_new_tokens=budget, query=query)
        except Exception as e:
            print(f"Error processing image {sample['imgid']}: {e}")
            return
        save(sample, inference_id, output, timer() - start)

    def run_batch(batch):
        if len(batch) == 1 or not hasattr(model, 'predict_batch'):
            for sample, inference_id in batch:
                run_single(sample, inference_id)
            return
        budget = max_new_tokens or max(_token_budget(tokenizer, s['html_table']) for s, _ in batch)
        start = timer()
        try:
            outputs = predict_batch(model, [s['image'] for s, _ in batch], max_new_tokens=budget, query=query)
        except Exception as e:   # e.g. out of memory on a very long table: retry one by one
            print(f"Batch failed ({e}); retrying sample by sample")
            for sample, inference_id in batch:
                run_single(sample, inference_id)
            return
        per_sample = (timer() - start) / len(batch)
        for (sample, inference_id), output in zip(batch, outputs):
            save(sample, inference_id, output, per_sample)

    pending = []
    for sample in tqdm(islice(test_ds, num_test_images), total=num_test_images):
        inference_id = hashlib.md5(f"{dataset}_{sample['imgid']}_{model_name}_{adapter}".encode('utf-8')).hexdigest()
        if inference_id in done:
            continue
        pending.append((sample, inference_id))
        if len(pending) >= batch_size:
            run_batch(pending)
            pending = []
    if pending:
        run_batch(pending)
