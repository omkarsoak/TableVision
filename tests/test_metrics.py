import pytest

from src.utils.helpers import recover_table_html
from src.utils.metrics import html_similarity
from src.utils.teds import teds

GT = "<table><tr><td>a</td><td>b</td></tr><tr><td>c</td><td>d</td></tr></table>"
UNWRAPPED = "<tr><td>a</td><td>b</td></tr><tr><td>c</td><td>d</td></tr>"
TRUNCATED = "<table><tr><td>a</td><td>b</td></tr><tr><td>c</td>"


def test_recover_table_html():
    assert recover_table_html(GT) == GT
    assert recover_table_html(UNWRAPPED) == f"<table>{UNWRAPPED}</table>"
    assert recover_table_html(TRUNCATED) == TRUNCATED + "</table>"
    assert recover_table_html("just words") is None
    assert recover_table_html(None) is None


def test_fixed_gives_content_credit_to_unwrapped_rows_legacy_does_not():
    fixed = html_similarity(UNWRAPPED, GT, variant="fixed", return_components=True)
    assert fixed["content"] == pytest.approx(1.0) and fixed["jaccard"] == pytest.approx(1.0)
    legacy = html_similarity(UNWRAPPED, GT, variant="legacy", return_components=True)
    assert legacy["content"] == pytest.approx(0.0, abs=1e-6)   # published behavior, kept for reproducing the reported numbers


def test_fixed_perfect_match_is_one_and_legacy_jaccard_is_inverted():
    assert html_similarity(GT, GT, variant="fixed") == pytest.approx(1.0)
    # identical text: the published formula adds (1 - 1) = 0 for the 20% token term, so even a perfect match loses 0.2
    assert html_similarity(GT, GT, variant="legacy") == pytest.approx(0.8, abs=1e-3)


def test_teds_recover_vs_strict():
    assert teds(UNWRAPPED, GT) == 1.0 and teds(UNWRAPPED, GT, recover=False) == 0.0
    assert 0.0 < teds(TRUNCATED, GT) < 1.0 and teds(TRUNCATED, GT, recover=False) == 0.0
