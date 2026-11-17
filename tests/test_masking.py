"""Label masking must train on exactly '<answer> + EOS' for the models and prompts used in the experiments
(needs the HF cache for the processors)."""
import pytest
from PIL import Image

from src.models.granite_vision.prompts import DEFAULT_PROMPT, get_family, make_collate_fn

CASES = [
    ("ibm-granite/granite-vision-4.1-4b", "<tables_html>"),
    ("ibm-granite/granite-vision-4.1-4b", "Convert user's image to HTML"),
    ("ibm-granite/granite-vision-3.2-2b", DEFAULT_PROMPT["granite3"]),
]
ANSWERS = [
    "<table><tr><td>a</td><td>b</td></tr></table>",
    "<table><thead><tr><td><b>Name</b></td></tr></thead><tbody><tr><td>x y z</td></tr></tbody></table>",
]


def test_get_family():
    assert get_family("ibm-granite/granite-vision-4.1-4b") == "granite4"
    assert get_family("ibm-granite/granite-vision-3.2-2b") == "granite3"
    assert get_family("anything", override="granite4") == "granite4"


@pytest.mark.parametrize("model_name,prompt", CASES)
def test_loss_covers_only_answer_and_eos(model_name, prompt):
    from transformers import AutoProcessor

    family = get_family(model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    processor.tokenizer.padding_side = "right"
    collate = make_collate_fn(processor, family, prompt)
    examples = [{"image": Image.new("RGB", (300 + 40 * i, 80), "white"), "html_table": a} for i, a in enumerate(ANSWERS)]
    batch = collate(examples)   # two examples of different length -> padding is exercised
    for i, answer in enumerate(ANSWERS):
        labels = batch["labels"][i]
        trained = processor.tokenizer.decode(labels[labels != -100])
        assert trained == answer + processor.tokenizer.eos_token
