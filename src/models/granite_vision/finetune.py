"""LoRA/DoRA fine-tuning of Granite Vision (3.x and 4.x) for table image -> HTML.

Differences from the original (TRL SFTTrainer) version:
  * plain transformers.Trainer + PEFT: TRL 1.x dropped `max_seq_length`/`skip_prepare_dataset`, and
    SFTTrainer added nothing this task needs (custom collator, no text packing);
  * prompt format and label masking are per model family (see prompts.py), so Granite 4.x trains correctly;
  * `layers_to_tune` accepts language_model / vision_tower / projector (alias: vision_model);
  * gradient checkpointing is NON-reentrant by default. With reentrant checkpointing the vision tower's
    input (pixel_values) does not require grad, so vision LoRA adapters can silently receive no gradient;
    set gradient_checkpointing_reentrant: true to reproduce the original behavior;
  * batch size, learning rate, eval/save cadence, LoRA hyper-parameters and max_steps are parameters.

`max_seq_length` keeps its original meaning: the HTML target (plus prompt) must be shorter than this many
tokens, otherwise the sample is skipped. It does not truncate and does not include image tokens.
"""
import gc
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor, Trainer, TrainingArguments

from src.models.granite_vision.prompts import (
    DEFAULT_PROMPT, build_messages, get_family, make_collate_fn, select_target_modules,
)
from src.models.granite_vision.transformers_library import _best_attn_implementation, _dtype_kw
from src.utils.helpers import ADAPTERS_BASE_PATH, extract_html_table


def load_process_filter_dataset(dataset, max_seq_length, num_train_images, num_test_images, processor, family, prompt):
    """Stream the train split, keep tables short enough, return (train, test) lists of
    {"image": PIL, "html_table": "<table>..</table>"}. The first N kept samples are used (deterministic)."""
    tokenizer = processor.tokenizer
    prompt_text = processor.apply_chat_template(build_messages(family, prompt), tokenize=False, add_generation_prompt=True)
    max_html_tokens = max_seq_length - len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
    num_total_needed = num_train_images + num_test_images

    kept = []
    p_bar = tqdm(total=num_total_needed, desc="Filtering dataset samples")
    for sample in load_dataset(dataset, split='train', streaming=True):
        table = extract_html_table(sample['html_table'])
        if table is None:
            continue
        if len(tokenizer(table, add_special_tokens=False)["input_ids"]) < max_html_tokens:
            kept.append({"image": sample["image"], "html_table": table})
            p_bar.update(1)
            if len(kept) >= num_total_needed:
                break
    p_bar.close()
    if len(kept) < num_total_needed:
        raise RuntimeError(f"Dataset ended after {len(kept)} usable samples; needed {num_total_needed}")
    return kept[:num_train_images], kept[num_train_images:]


def main(
        model_name: str,
        dataset: str,
        max_seq_length: int,
        gradient_accumulation_steps: int,
        num_train_images: int,
        num_test_images: int,
        layers_to_tune: list,
        experiment_name: str,
        system_message: str = None,
        prompt: str = None,
        prompt_family: str = None,
        debug: bool = False,
        use_wandb: bool = False,
        resume_id: str = None,
        rewind_step: int = None,
        per_device_train_batch_size: int = 1,
        learning_rate: float = 3e-4,
        warmup_steps: int = 10,
        weight_decay: float = 0.01,
        logging_steps: int = 25,
        eval_steps: int = None,
        save_steps: int = None,
        max_steps: int = -1,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        use_dora: bool = True,
        gradient_checkpointing_reentrant: bool = False,
        dataloader_num_workers: int = 4,
        seed: int = 42,
):
    family = get_family(model_name, prompt_family)
    prompt = prompt or system_message or DEFAULT_PROMPT[family]   # `system_message` is the old config key
    use_wandb = use_wandb and not debug   # W&B is opt-in: it needs `wandb login`, which would hang an unattended job
    eval_steps = eval_steps or logging_steps
    save_steps = save_steps or 2 * eval_steps                     # must be a multiple of eval_steps
    output_dir = ADAPTERS_BASE_PATH.format(experiment_name)
    print(f"Model family: {family} | prompt: {prompt!r} | output: {output_dir}")

    proc_kw = {"use_fast": True} if family == "granite3" else {}
    processor = AutoProcessor.from_pretrained(model_name, **proc_kw)
    processor.tokenizer.padding_side = "right"

    # Data
    train_dataset, test_dataset = load_process_filter_dataset(
        dataset, max_seq_length, num_train_images, num_test_images, processor, family, prompt
    )
    collate_fn = make_collate_fn(processor, family, prompt)

    # Model
    gc.collect()
    torch.cuda.empty_cache()
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        device_map={"": torch.cuda.current_device()},
        attn_implementation=_best_attn_implementation(),
        **_dtype_kw(torch.bfloat16),
    )
    model.config.use_cache = False
    if gradient_checkpointing_reentrant:
        model.enable_input_require_grads()   # reentrant checkpointing needs a grad-requiring input

    # LoRA / DoRA
    target_modules = select_target_modules(model, layers_to_tune)
    print(f"LoRA targets: {len(target_modules)} modules, e.g. {target_modules[0]}")
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        use_dora=use_dora,
        init_lora_weights="gaussian",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        max_steps=max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        logging_strategy="steps",
        logging_steps=logging_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=1,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True,
        optim="adamw_torch_fused",
        bf16=True,
        report_to="wandb" if use_wandb else "none",
        remove_unused_columns=False,
        label_names=["labels"],          # required for eval loss with PEFT-wrapped models
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": gradient_checkpointing_reentrant},
        dataloader_num_workers=dataloader_num_workers,
        seed=seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=collate_fn,
    )

    # Weights & Biases (optional)
    if use_wandb:
        import wandb
        wandb.login()
        init_args = {
            "project": "granite-vision",
            "name": experiment_name,
            "config": {
                "model_name": model_name, "dataset": dataset, "max_seq_length": max_seq_length,
                "num_train_images": num_train_images, "layers_to_tune": layers_to_tune,
                "prompt": prompt, "family": family, "learning_rate": learning_rate,
                "gradient_accumulation_steps": gradient_accumulation_steps,
            },
        }
        if resume_id is not None:
            init_args["id"] = resume_id
            if rewind_step is not None:
                init_args["resume_from"] = f"{resume_id}?_step={rewind_step}"
            else:
                init_args["resume"] = "must"
        wandb.init(**init_args)

    # Train
    print(f"Memory footprint: {model.get_memory_footprint() / (1024 ** 3):.2f} GB\n")
    if resume_id is None:
        trainer.evaluate()
        trainer.train()
    else:
        trainer.train(resume_from_checkpoint=True)

    # The best model (lowest eval loss) is loaded at this point; save it with the processor and the settings used.
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    with open(os.path.join(output_dir, "training_log.json"), "w") as f:   # train/eval loss curves, no W&B needed
        json.dump(trainer.state.log_history, f, indent=1)
    with open(os.path.join(output_dir, "run_params.json"), "w") as f:
        json.dump({
            "model_name": model_name, "family": family, "prompt": prompt, "layers_to_tune": layers_to_tune,
            "max_seq_length": max_seq_length, "num_train_images": num_train_images,
            "per_device_train_batch_size": per_device_train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps, "learning_rate": learning_rate,
            "lora_r": lora_r, "lora_alpha": lora_alpha, "use_dora": use_dora,
            "gradient_checkpointing_reentrant": gradient_checkpointing_reentrant,
            "transformers": transformers.__version__, "best_eval_loss": trainer.state.best_metric,
        }, f, indent=2)
    print(f"Saved adapter to {output_dir} (best eval loss {trainer.state.best_metric})")
