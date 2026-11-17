import os
import sys

current_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(current_dir, '..'))

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

from src.models.granite_vision.transformers_library import _dtype_kw
from src.utils.helpers import resolve_adapter_path


def main(model_name: str,
         adapter_path: str,
         repo_name: str = None,
         push: bool = False,
         private: bool = False,
         save_dir: str = None,
         dtype: str = "bfloat16") -> None:
    """Merge a trained adapter into its base model, save it locally, and optionally publish it.

    Nothing is uploaded unless `push: true`. The merge runs in fp32 on CPU, then the weights are cast to `dtype`.
    :param adapter_path: experiment name or directory (see helpers.resolve_adapter_path)
    :param repo_name: Hub repo name; default '<base model name>-table2html' under the logged-in account
    :param save_dir: local output directory; default 'src/models/granite_vision/merged/<repo_name>'
    """
    adapter_dir = resolve_adapter_path(adapter_path)
    repo_name = repo_name or model_name.split('/')[-1] + '-table2html'
    save_dir = save_dir or os.path.join('src', 'models', 'granite_vision', 'merged', repo_name)

    processor = AutoProcessor.from_pretrained(model_name)   # the processor is never modified by fine-tuning
    print("Processor loaded!")

    model = AutoModelForImageTextToText.from_pretrained(model_name, device_map='cpu', **_dtype_kw(torch.float32))
    print("Base model loaded!")

    model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
    print(f"Adapter loaded from {adapter_dir}!")

    model = model.merge_and_unload().to(getattr(torch, dtype))
    print("Adapter merged into base model!")

    model.save_pretrained(save_dir, max_shard_size="2GB")
    processor.save_pretrained(save_dir)
    print(f"Merged model saved to {save_dir}")

    if push:
        from huggingface_hub import get_token, login
        if not get_token():
            login()   # interactive prompt only when no token is configured
        model.push_to_hub(repo_name, private=private, commit_message="Upload LoRA model with merged adapter",
                          max_shard_size="2GB")
        processor.push_to_hub(repo_name, private=private)
        print(f"Merged model and processor uploaded to Hugging Face as {repo_name}!")
    else:
        print("push is false: nothing was uploaded.")
