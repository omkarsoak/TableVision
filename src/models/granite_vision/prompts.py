"""Prompt formats, label masking and LoRA target selection shared by training and inference.

Two Granite Vision "families" use different chat formats:

  granite3  (3.1 / 3.2 / 3.3, LlavaNext-style): system = free-text instruction, user = image.
            Template: <|system|> .. <|user|><image> <|assistant|> ...
  granite4  (4.x):  one user turn = image + task tag such as "<tables_html>"; the chat template
            expands the tag into IBM's full instruction. Template: <|start_of_role|>user<|end_of_role|> ...

Keeping both here guarantees training and inference build identical prompts.
"""
import re
from typing import List, Optional

import torch

DEFAULT_PROMPT = {
    "granite3": "Convert table to HTML (<table> ... </table>)",
    "granite4": "<tables_html>",
}


def get_family(model_name: str, override: Optional[str] = None) -> str:
    """'granite4' for Granite Vision 4.x / Granite 4 models, else 'granite3'."""
    if override:
        if override not in DEFAULT_PROMPT:
            raise ValueError(f"prompt_family must be one of {list(DEFAULT_PROMPT)}, got {override!r}")
        return override
    return "granite4" if re.search(r"granite-vision-4|granite4|granite-4", model_name.lower()) else "granite3"


def build_messages(family: str, prompt: str, answer: Optional[str] = None) -> list:
    """Chat messages WITHOUT image data (a bare {"type": "image"} placeholder).

    Images are passed to the processor separately, so apply_chat_template never has to load them.
    """
    image = {"type": "image"}
    if family == "granite4":
        messages = [{"role": "user", "content": [image, {"type": "text", "text": prompt}]}]
    else:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": prompt}]},
            {"role": "user", "content": [image]},
        ]
    if answer is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return messages


def processor_extra_kwargs(family: str) -> dict:
    """Extra processor kwargs needed for correct batching (see the Granite Vision 4.1 model card)."""
    return {"do_pad": True} if family == "granite4" else {}


# --------------------------------------------------------------------------- label masking
def generation_suffix_ids(processor, family: str, prompt: str) -> List[int]:
    """Token ids of the text the template appends for add_generation_prompt=True,
    e.g. '<|assistant|>\\n' (granite3) or '<|start_of_role|>assistant<|end_of_role|>' (granite4)."""
    msgs = build_messages(family, prompt)
    with_gen = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    without = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    if not with_gen.startswith(without):
        raise ValueError("Chat template: the generation prompt is not a pure suffix; cannot build label mask")
    return processor.tokenizer(with_gen[len(without):], add_special_tokens=False)["input_ids"]


def mask_labels(input_ids: torch.Tensor, attention_mask: torch.Tensor, gen_ids: List[int], eos_id: int) -> torch.Tensor:
    """Labels = -100 everywhere except the assistant answer, up to and including its first EOS.

    Template-agnostic replacement for the original '<|assistant|>' search, which only worked for Granite 3.
    Padding, the prompt (system/user/image tokens) and anything after the answer's EOS (for example the
    '\\n' that Granite 4 templates append) are all excluded from the loss.
    """
    labels = torch.full_like(input_ids, -100)
    g = torch.tensor(gen_ids, dtype=input_ids.dtype)
    n = len(gen_ids)
    for i in range(input_ids.shape[0]):
        row = input_ids[i]
        hits = (row.unfold(0, n, 1) == g).all(dim=1).nonzero()
        if len(hits) == 0:
            raise ValueError(
                "Generation prompt not found in a training example (was it truncated?). "
                "Refusing to train with an empty loss mask."
            )
        start = hits[-1].item() + n
        eos = (row[start:] == eos_id).nonzero()
        end = start + eos[0].item() + 1 if len(eos) else len(row)
        labels[i, start:end] = row[start:end]
    labels[attention_mask == 0] = -100
    return labels


def make_collate_fn(processor, family: str, prompt: str):
    """Collator for training: examples are dicts with a PIL 'image' and the target 'html_table' string."""
    gen_ids = generation_suffix_ids(processor, family, prompt)
    eos_id = processor.tokenizer.eos_token_id
    extra = processor_extra_kwargs(family)

    def collate_fn(examples):
        texts = [
            processor.apply_chat_template(build_messages(family, prompt, answer=ex["html_table"]), tokenize=False)
            for ex in examples
        ]
        images = [[ex["image"].convert("RGB")] for ex in examples]
        batch = processor(text=texts, images=images, return_tensors="pt", padding=True, **extra)
        batch["labels"] = mask_labels(batch["input_ids"], batch["attention_mask"], gen_ids, eos_id)
        return batch

    return collate_fn


# --------------------------------------------------------------------------- LoRA targets
LM_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
VISION_SUFFIXES = ("q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2")
PROJECTOR_KEYS = ("multi_modal_projector", "layerwise_projectors", "spatial_projectors")

# Accepted `layers_to_tune` values. "vision_model" is the old config spelling: it only matched
# transformers<4.52 module names (vision_tower.vision_model...), so it is kept as an alias.
GROUP_ALIASES = {
    "language_model": "language_model",
    "vision_tower": "vision_tower",
    "vision_model": "vision_tower",
    "projector": "projector",
}


def select_target_modules(model: torch.nn.Module, layers_to_tune: List[str]) -> List[str]:
    """Full names of the nn.Linear modules to adapt, for the groups in `layers_to_tune`:
    language_model (attention + MLP projections), vision_tower (SigLIP attention + MLP, pooling head
    excluded), projector (multimodal projector / Q-Formers)."""
    groups = set()
    for key in layers_to_tune:
        if key not in GROUP_ALIASES:
            raise ValueError(f"Unknown layers_to_tune entry {key!r}; valid: {sorted(GROUP_ALIASES)}")
        groups.add(GROUP_ALIASES[key])

    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        leaf = name.split(".")[-1]
        if "language_model" in groups and "language_model" in name and leaf in LM_SUFFIXES:
            targets.append(name)
        elif "vision_tower" in groups and "vision_tower" in name and ".head." not in name and leaf in VISION_SUFFIXES:
            targets.append(name)
        elif "projector" in groups and any(k in name for k in PROJECTOR_KEYS):
            targets.append(name)
    if not targets:
        raise ValueError(f"No Linear modules matched layers_to_tune={layers_to_tune}")
    return targets
