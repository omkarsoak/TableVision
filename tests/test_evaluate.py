import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("evaluate_results", REPO / "scripts" / "evaluate_results.py")
evaluate_results = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate_results)


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _row(model, adapter, imgid):
    return {"model_name": model, "adapter": adapter, "imgid": imgid, "html": "<table></table>",
            "model_prediction": "<table></table>", "execution_time": 1.0}


def test_labels_and_collisions(tmp_path):
    a, b = tmp_path / "results.jsonl", tmp_path / "results_zeroshot.jsonl"
    _write(a, [_row("m/granite-3.2", None, 1), _row("m/granite-3.2", "ft", 1), _row("pix", None, 1)])
    _write(b, [_row("m/granite-3.2", None, 1), _row("m/granite-4.1", "ft", 1)])
    df = evaluate_results.load_results([str(a), str(b)], drop=[])
    assert set(df["model_name"]) == {
        "m/granite-3.2 [results]", "m/granite-3.2 [results_zeroshot]", "m/granite-3.2_ft", "pix", "m/granite-4.1_ft"}


def test_drop_removes_suffix_once_collision_is_gone(tmp_path):
    a, b = tmp_path / "results.jsonl", tmp_path / "results_zeroshot.jsonl"
    _write(a, [_row("m/granite-3.2", None, 1), _row("m/granite-3.2", "ft", 1)])
    _write(b, [_row("m/granite-3.2", None, 1)])
    df = evaluate_results.load_results([str(a), str(b)], drop=[r"granite-3\.2_", r"granite-3\.2 \[results\]$"])
    assert list(df["model_name"]) == ["m/granite-3.2"]
    assert list(df["source"]) == ["results_zeroshot"]   # the saved row was dropped, ours was kept


def test_glob_patterns(tmp_path):
    for n in ("results_a.jsonl", "results_b.jsonl"):
        _write(tmp_path / n, [_row(n, None, 1)])
    df = evaluate_results.load_results([str(tmp_path / "results_*.jsonl")], drop=[])
    assert len(df) == 2
