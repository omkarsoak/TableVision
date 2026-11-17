#!/usr/bin/env bash
# Launch project jobs in a detached tmux session, each in its own window and logging to logs/.
#
#   bash scripts/tmux_jobs.sh download        # models + datasets into the HF cache         (~20-45 min)
#   bash scripts/tmux_jobs.sh smoke [gpu]     # 200-image train + 5-image inference, 4.1    (~10 min)
#   bash scripts/tmux_jobs.sh demo [gpu]      # qualitative demo on image 618932            (~5 min)
#   bash scripts/tmux_jobs.sh run <config.yaml> [gpu]    # any run.py config (alias: train)
#   bash scripts/tmux_jobs.sh all [gpu gpu gpu gpu gpu]  # the 5 experiments in parallel, one GPU each
#   bash scripts/tmux_jobs.sh report          # score everything, CIs, paired tests, charts (CPU, ~5 min)
#   bash scripts/tmux_jobs.sh status          # list windows + tail of each log + GPU memory
#   bash scripts/tmux_jobs.sh setup           # only for a fresh machine: create the conda env
#
# Attach with:  tmux attach -t tv2h      Detach: Ctrl-b d      Switch window: Ctrl-b <n>
#
# The 5 experiments (configs/): lang_table_only_4, lang_table_only_2, lang_table_only (Granite Vision 4.1
# fine-tunes), zeroshot_granite41 and zeroshot_granite32 (no fine-tuning). Each writes its own results file.
#
# GPUs on this server are shared. The script picks the least-used GPU(s) unless you pass them
# explicitly (or set GPU=...).
set -euo pipefail

SESSION="${SESSION:-tv2h}"
ENV_NAME="${ENV_NAME:-tb-vision}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO/logs"
mkdir -p "$LOG_DIR"

# Keep HF downloads on the large /home volume (dataset is ~13 GB compressed).
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# GPU indices, least used first
gpus_by_free() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | sort -t, -k2 -n | cut -d, -f1 | tr -d ' '
}
pick_gpu() { gpus_by_free | head -1; }

ACTIVATE="source \"\$(conda info --base)/etc/profile.d/conda.sh\" && conda activate $ENV_NAME"

launch() {  # launch <window-name> <command>
  local name="$1" cmd="$2"
  local log="$LOG_DIR/${name}_$(date +%Y%m%d_%H%M%S).log"
  tmux has-session -t "$SESSION" 2>/dev/null || tmux new-session -d -s "$SESSION" -n main -c "$REPO"
  # kill a finished window with the same name first (duplicate names make tmux targets ambiguous)
  tmux kill-window -t "$SESSION:$name" 2>/dev/null || true
  tmux new-window -t "$SESSION" -n "$name" -c "$REPO"
  tmux send-keys -t "$SESSION:$name" \
    "export HF_HOME=$HF_HOME; ( $cmd ) 2>&1 | tee $log; echo \"[exit \${PIPESTATUS[0]}] $name finished at \$(date)\" | tee -a $log" C-m
  echo "Started '$name' in tmux session '$SESSION' -> $log"
}

run_config() {  # run_config <config.yaml> <gpu>
  local cfg="$1" gpu="$2" name
  name="$(basename "$cfg" .yaml)"
  launch "$name" "$ACTIVATE && CUDA_VISIBLE_DEVICES=$gpu python run.py --config $cfg"
}

case "${1:-}" in
  setup)
    launch setup "conda create -y -n $ENV_NAME python=3.11 && source \"\$(conda info --base)/etc/profile.d/conda.sh\" && conda activate $ENV_NAME && \
      pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128 && \
      pip install -r requirements.txt && \
      python -c 'import torch, transformers; print(torch.__version__, torch.cuda.is_available(), transformers.__version__)'"
    ;;
  download)
    launch download "$ACTIVATE && \
      hf download ibm-granite/granite-vision-4.1-4b && \
      hf download ibm-granite/granite-vision-3.2-2b && \
      hf download KennethTM/pix2struct-base-table2html && \
      hf download apoidea/pubtabnet-html --repo-type dataset && \
      hf download kensho/PubTables-v2 --repo-type dataset --include 'PubTables-v2_Cropped-Tables_test_*' --local-dir data/pubtables-v2"
    ;;
  smoke)
    run_config configs/smoke_granite41.yaml "${2:-${GPU:-$(pick_gpu)}}"
    ;;
  demo)
    gpu="${2:-${GPU:-$(pick_gpu)}}"
    launch demo "$ACTIVATE && CUDA_VISIBLE_DEVICES=$gpu python scripts/inference_demo.py --models granite32,granite41,finetuned,pix2struct"
    ;;
  run|train)
    cfg="${2:?usage: tmux_jobs.sh run <config.yaml> [gpu]}"
    run_config "$cfg" "${3:-${GPU:-$(pick_gpu)}}"
    ;;
  all)
    shift
    GPUS=("$@")
    if [ "${#GPUS[@]}" -lt 5 ]; then
      mapfile -t AUTO < <(gpus_by_free)
      GPUS=("${AUTO[@]:0:5}")
    fi
    if [ "${#GPUS[@]}" -lt 5 ]; then echo "Need 5 GPUs, found ${#GPUS[@]}; pass them explicitly or run configs one by one."; exit 1; fi
    i=0
    # longest job first
    for cfg in lang_table_only_4 lang_table_only_2 lang_table_only zeroshot_granite41 zeroshot_granite32; do
      run_config "configs/$cfg.yaml" "${GPUS[$i]}"
      i=$((i + 1))
    done
    ;;
  report)
    launch report "$ACTIVATE && python scripts/evaluate_results.py --compare-variants --primary teds \
      --reference KennethTM/pix2struct-base-table2html \
      --drop 'granite-vision-3\\.2-2b_' --drop 'granite-vision-3\\.2-2b \\[results\\]\$' \
      --speed-results data/evaluation/speed_bs1.jsonl"
    ;;
  status)
    tmux list-windows -t "$SESSION" 2>/dev/null || echo "No session '$SESSION'"
    for f in $(ls -t "$LOG_DIR"/*.log 2>/dev/null | head -6); do
      echo "== $f"; tail -c 600 "$f" | tr '\r' '\n' | grep -v '^\s*$' | tail -n 2 | cut -c1-200
    done
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
    ;;
  *)
    sed -n '2,18p' "$0"; exit 1
    ;;
esac
