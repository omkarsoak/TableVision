import re
import json
import signal
from typing import List
from pathlib import Path


def append_json_line(file_path: str, data: dict):
    """
    Appends a dictionary as a new JSON line in a JSONL file.
    If the file doesn't exist, it creates it.

    :param file_path: Path to the JSONL file.
    :param data: Dictionary to append as a new JSON line.
    """
    # Ensure the data is a dictionary
    if not isinstance(data, dict):
        raise ValueError("Data must be a dictionary.")

    # Create parent directories if they don't exist
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    # Open the file in append mode and write the dictionary as a JSON line
    with open(file_path, 'a', encoding='utf-8') as file:
        json_line = json.dumps(data, ensure_ascii=False)
        file.write(json_line + '\n')


def read_jsonl(file_path: str) -> List[dict]:
    """
    Reads a JSON Lines file and returns a list of dictionaries.

    :param file_path: The path to the JSON Lines file.
    :return: list of dictionaries parsed from the file.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON line: {line.strip()}")
                raise Exception(f"Error decoding JSON line: {str(e)}")
    return data

class TimeoutException(Exception):
    pass

def timeout(seconds=10):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutException(f"Function call timed out after {seconds} seconds")
        
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                return func(*args, **kwargs)
            finally:
                signal.alarm(0)  # Disable alarm
        return wrapper
    return decorator


def extract_html_table(html):
    """Extracts the first <table>...</table> block from text.

    Returns None when there is no table (callers that train on the result must skip those samples).
    Note: src.utils.metrics.extract_html_table instead returns its input unchanged in that case.
    """
    match = re.search(r'<table\b.*?</table>', html, re.DOTALL | re.IGNORECASE)
    if match:
        table_html = match.group()
        return table_html
    else:
        return None


def recover_table_html(text: str):
    """Best-effort table markup from raw model output, or None if it has no table-like content.

    1. the first complete <table>...</table>;
    2. else, from an unclosed '<table' (output cut off by the token limit) to the end, closed with </table>;
    3. else bare <tr>/<td> rows without any wrapper, wrapped in <table>...</table>.
    Used by the 'fixed' HTML-Sim variant and by TEDS so that missing wrappers and truncation are
    scored on what the model actually produced (the legacy code gives such outputs zero text credit).
    """
    text = text or ""
    match = re.search(r'<table\b.*?</table>', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group()
    start = re.search(r'<table\b', text, re.IGNORECASE)
    if start:
        return text[start.start():] + "</table>"
    if re.search(r'<t[rd]\b', text, re.IGNORECASE):
        return f"<table>{text}</table>"
    return None


ADAPTERS_BASE_PATH = "src/models/granite_vision/checkpoints/{}"


def resolve_adapter_path(adapter: str) -> str:
    """Find the directory holding a trained LoRA/DoRA adapter.

    `adapter` is an experiment name (looked up under ADAPTERS_BASE_PATH) or a directory.
    Preference order:
      1. the directory itself, if it contains adapter_config.json (training saves the BEST model
         there at the end, because load_best_model_at_end=True);
      2. the best checkpoint recorded in trainer_state.json of the newest checkpoint;
      3. the newest checkpoint-N directory.
    """
    p = Path(adapter) if Path(adapter).exists() else Path(ADAPTERS_BASE_PATH.format(adapter))
    if not p.exists():
        raise FileNotFoundError(f"Adapter '{adapter}' not found (looked in {p})")
    if (p / "adapter_config.json").exists():
        return str(p)
    ckpts = sorted(p.glob("checkpoint-*"), key=lambda d: int(d.name.split("-")[-1]))
    if not ckpts:
        raise FileNotFoundError(f"No adapter_config.json or checkpoint-* directories in {p}")
    state_file = ckpts[-1] / "trainer_state.json"
    if state_file.exists():
        best = json.loads(state_file.read_text()).get("best_model_checkpoint")
        if best and Path(best).exists():
            return str(best)
    return str(ckpts[-1])