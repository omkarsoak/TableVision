# Table Vision

Table image to HTML with IBM Granite Vision 4.1 (4B), fine-tuned with LoRA/DoRA on PubTabNet and compared with zero-shot Granite models, Pix2Struct and Llama-3.2-90B-Vision.

The Pix2Struct and Llama baselines use saved predictions on the same 500 images; every Granite 4.1 number comes from the runs in this repository.

## Results

500 held-out PubTabNet validation images, same images for every model. Intervals are 95% bootstrap intervals.

| Model | TEDS | TEDS-Struct | HTML-Sim (fixed) |
|---|---|---|---|
| Granite 4.1 zero-shot | 0.956 [0.947, 0.964] | 0.968 | 0.851 |
| Granite 4.1 fine-tuned, `lang_table_only_4` | 0.954 [0.945, 0.963] | 0.965 | 0.940 |
| Granite 4.1 fine-tuned, `lang_table_only_2` | 0.953 [0.942, 0.962] | 0.964 | 0.931 |
| Granite 4.1 fine-tuned, `lang_table_only` | 0.940 [0.928, 0.951] | 0.956 | 0.925 |
| Pix2Struct table2html (saved predictions) | 0.909 [0.893, 0.925] | 0.929 | 0.904 |
| Granite 3.2 zero-shot | 0.870 [0.854, 0.886] | 0.892 | 0.825 |
| Llama-3.2-90B-Vision (saved predictions) | 0.720 [0.698, 0.742] | 0.790 | 0.727 |

- Fine-tuned Granite Vision 4.1 (4B) with DoRA adapters (32M trainable parameters, 0.8% of the model) on 10,000 PubTabNet table images. It reaches 0.954 TEDS, which is +0.045 over Pix2Struct (95% CI +0.030 to +0.061) and +0.23 over Llama-3.2-90B-Vision, a model about 26 times larger.
- Built an evaluation pipeline with its own TEDS implementation, bootstrap confidence intervals, paired significance tests and 36 unit tests. It fixed an inverted Jaccard term in the original custom metric, and the corrected HTML Similarity rises from 0.851 (zero-shot) to 0.940 (fine-tuned).
- Replaced the training loop (plain `Trainer` with PEFT), made loss masking independent of the chat template, and added batched inference that gives the same output as single images (4.9x faster).

## Experiments

| Name | Setup | Config |
|---|---|---|
| `lang_table_only_4` | 10,000 images, gradient accumulation 8, prompt `<tables_html>` | [configs/lang_table_only_4.yaml](configs/lang_table_only_4.yaml) |
| `lang_table_only_2` | 5,000 images, gradient accumulation 8, generic prompt | [configs/lang_table_only_2.yaml](configs/lang_table_only_2.yaml) |
| `lang_table_only` | 5,000 images, gradient accumulation 4, generic prompt | [configs/lang_table_only.yaml](configs/lang_table_only.yaml) |
| zero-shot 4.1 | no training | [configs/zeroshot_granite41.yaml](configs/zeroshot_granite41.yaml) |
| zero-shot 3.2 | no training | [configs/zeroshot_granite32.yaml](configs/zeroshot_granite32.yaml) |

Shared settings: one epoch, learning rate 3e-4, batch size 1 per device, `max_seq_length` 1024, bf16, table-only output, best checkpoint by eval loss on 250 held-out samples.

## Setup and usage

Python 3.11, one conda env. Trained on NVIDIA RTX 6000 Ada Generation GPUs (48 GB each, about 20 GB used per training run, one GPU per experiment). Set `HF_TOKEN` before downloading.

```bash
pip install -r requirements.txt
bash scripts/tmux_jobs.sh download            # models and datasets
bash scripts/tmux_jobs.sh smoke 0             # 10 minute end-to-end check
bash scripts/tmux_jobs.sh all                 # the five experiments, one GPU each
bash scripts/tmux_jobs.sh report              # scores, intervals, charts in outputs/evaluation/
python -m pytest tests -q
```

Use a fine-tuned adapter:

```python
from src.models.granite_vision.transformers_library import LLM

model = LLM("ibm-granite/granite-vision-4.1-4b", adapter="lang_table_only_4")
html = model.predict(image, max_new_tokens=1024, query="<tables_html>")
```

The adapters are written to `src/models/granite_vision/checkpoints/<experiment>/` and are not tracked in git.
