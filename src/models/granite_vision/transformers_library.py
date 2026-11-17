import importlib.util
from threading import Thread

import torch
import transformers
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor, TextIteratorStreamer
from transformers.utils import logging

from src.models.granite_vision.prompts import DEFAULT_PROMPT, build_messages, get_family, processor_extra_kwargs
from src.utils.helpers import ADAPTERS_BASE_PATH, resolve_adapter_path

transformers_logger = logging.get_logger("transformers")
transformers_logger.setLevel(logging.ERROR)


def _dtype_kw(dtype):
    """`dtype=` on transformers>=5, `torch_dtype=` before."""
    return {"dtype": dtype} if int(transformers.__version__.split(".")[0]) >= 5 else {"torch_dtype": dtype}


def _best_attn_implementation() -> str:
    return "flash_attention_2" if importlib.util.find_spec("flash_attn") else "sdpa"


class LLM:
    """Granite Vision (3.x or 4.x) wrapper for inference, optionally with a merged LoRA/DoRA adapter."""

    ADAPTERS_BASE_PATH: str = ADAPTERS_BASE_PATH

    def __init__(self, model_name, adapter: str = None, prompt_family: str = None):
        self.model_name = model_name
        self.family = get_family(model_name, prompt_family)
        self.default_prompt = DEFAULT_PROMPT[self.family]

        # Load processor (left padding is required for batched generation)
        proc_kw = {"use_fast": True} if self.family == "granite3" else {}
        self.processor = AutoProcessor.from_pretrained(model_name, **proc_kw)
        self.processor.tokenizer.padding_side = "left"

        # With an adapter the base model is loaded in fp32 on CPU and merged there (as in the original
        # project, for a more exact merge), then moved to the accelerator in bf16.
        if adapter:
            self.device, self.device_name = 'cpu', 'CPU'
            device_map, dtype, attn = 'cpu', torch.float32, 'sdpa'
        else:
            self.device, self.device_name = self._select_device()
            on_cuda = self.device == 'cuda'
            device_map = {"": 0} if on_cuda else 'cpu'
            dtype = torch.bfloat16 if on_cuda else torch.float32
            attn = _best_attn_implementation() if on_cuda else 'sdpa'
            print(f"Using {self.device}: {self.device_name}")

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name, device_map=device_map, attn_implementation=attn, **_dtype_kw(dtype)
        )
        print("Model loaded")

        if adapter:
            adapter_path = resolve_adapter_path(adapter)
            self.model = PeftModel.from_pretrained(self.model, adapter_path, is_trainable=False)
            print(f"Adapter '{adapter}' loaded from {adapter_path}")
            self.model = self.model.merge_and_unload()
            print(f"Adapter '{adapter}' merged")
            self.device, self.device_name = self._select_device()
            self.model = self.model.to(dtype=torch.bfloat16, device=self.device)
            print(f"Using {self.device}: {self.device_name}")
        self.model.eval()

    @staticmethod
    def _select_device():
        if torch.backends.mps.is_available():
            return 'mps', "Apple device with MPS support"
        elif torch.cuda.is_available():
            return 'cuda', torch.cuda.get_device_name(torch.cuda.current_device())
        return 'cpu', "CPU"

    def _inputs(self, imgs, query):
        prompt = query or self.default_prompt
        text = self.processor.apply_chat_template(
            build_messages(self.family, prompt), tokenize=False, add_generation_prompt=True
        )
        return self.processor(
            images=[img.convert("RGB") for img in imgs],
            text=[text] * len(imgs),
            return_tensors="pt",
            padding=True,
            **processor_extra_kwargs(self.family),
        ).to(self.device)

    def predict_batch(self, imgs, max_new_tokens: int, query=None):
        """Greedy-decode one answer per image; returns a list of strings (prompt tokens excluded)."""
        inputs = self._inputs(imgs, query)
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generated = output[:, inputs["input_ids"].shape[1]:]
        return [t.strip() for t in self.processor.batch_decode(generated, skip_special_tokens=True)]

    def predict(self, img, max_new_tokens: int, query=None):
        return self.predict_batch([img], max_new_tokens, query)[0]

    def stream(self, img, max_new_tokens: int, query=None):
        streamer = TextIteratorStreamer(self.processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_args = {
            "max_new_tokens": max_new_tokens,
            "streamer": streamer,
            "do_sample": False,
            **self._inputs([img], query),
        }

        # Generate in a separate thread so tokens can be yielded as they arrive
        thread = Thread(target=self.model.generate, kwargs=generation_args)
        thread.start()
        for text_token in streamer:
            yield text_token
        thread.join()
