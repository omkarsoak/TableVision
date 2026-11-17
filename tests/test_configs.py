"""Every experiment config must match the real function signatures and be internally consistent,
so that a typo is caught here and not 4 hours into a training run."""
import glob
import importlib
import inspect

import pytest
import yaml

MODULES = {
    "inference": "src.inference",
    "granite_vision_finetune": "src.models.granite_vision.finetune",
    "merge_and_upload": "src.merge_and_upload",
}
CONFIGS = sorted(glob.glob("configs/*.yaml"))


def _steps(path):
    return yaml.safe_load(open(path))["modules"]


def test_configs_exist():
    names = {p.split("/")[-1] for p in CONFIGS}
    assert {"lang_table_only.yaml", "lang_table_only_2.yaml", "lang_table_only_4.yaml",
            "zeroshot_granite41.yaml", "zeroshot_granite32.yaml"} <= names


@pytest.mark.parametrize("path", CONFIGS)
def test_params_match_signatures(path):
    for step in _steps(path):
        main = importlib.import_module(MODULES[step["module"]]).main
        inspect.signature(main).bind(**step.get("params", {}))   # raises TypeError on unknown/missing params


@pytest.mark.parametrize("path", CONFIGS)
def test_finetune_and_inference_agree(path):
    """The inference step must evaluate the adapter that was just trained, with the same prompt and base model."""
    steps = _steps(path)
    train = [s["params"] for s in steps if s["module"] == "granite_vision_finetune"]
    infer = [s["params"] for s in steps if s["module"] == "inference"]
    if not train:
        return
    t, i = train[0], infer[0]
    assert i["adapter"] == t["experiment_name"]
    assert i["model_name"] == t["model_name"]
    assert i.get("prompt") == t.get("prompt")
    assert i["dataset"] == t["dataset"]


def test_experiment_results_files_are_unique():
    """One writer per results file (/home is a network filesystem)."""
    paths = []
    for p in CONFIGS:
        if p.endswith("speed_bs1.yaml") or "smoke" in p:
            continue
        paths += [s["params"]["results_path"] for s in _steps(p) if s["module"] == "inference"]
    assert len(paths) == len(set(paths))
    assert all(x.startswith("data/evaluation/results_") for x in paths)
