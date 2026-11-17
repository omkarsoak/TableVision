import pytest

from src.utils.stats import bootstrap_ci, paired_bootstrap_diff
from src.utils.teds import teds

T = "<table><tr><td>a</td><td>b</td></tr><tr><td>c</td><td>d</td></tr></table>"


def test_identical_is_one():
    assert teds(T, T) == 1.0
    assert teds(T, T, structure_only=True) == 1.0


def test_missing_row_costs_three_of_seven_nodes():
    # table + 2 tr + 4 td = 7 nodes; dropping a row removes tr + 2 td = 3 deletions
    one_row = "<table><tr><td>a</td><td>b</td></tr></table>"
    assert teds(one_row, T) == pytest.approx(1 - 3 / 7)


def test_text_matters_only_when_not_structure_only():
    wrong_text = T.replace(">d<", ">x<")
    assert teds(wrong_text, T, structure_only=True) == 1.0
    assert teds(wrong_text, T) == pytest.approx(1 - 1 / 7)   # one cell, normalized edit distance 1


def test_colspan_mismatch_is_structural():
    spanned = "<table><tr><td colspan='2'>a</td></tr><tr><td>c</td><td>d</td></tr></table>"
    assert teds(spanned, T, structure_only=True) < 1.0


def test_normalization_ignores_thead_b_and_th():
    gt = "<table><thead><tr><td><b>A</b></td><td><b>B</b></td></tr></thead><tbody><tr><td>1</td><td>2</td></tr></tbody></table>"
    pred = "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>"
    assert teds(pred, gt, normalize=True) == 1.0
    assert teds(pred, gt, normalize=False) < 1.0


def test_wrapped_in_html_document_and_no_table():
    assert teds("<html><body>" + T + "</body></html>", T) == 1.0
    assert teds("no table here", T) == 0.0


def test_rows_without_table_wrapper_are_recovered():
    unwrapped = "<tr><td>a</td><td>b</td></tr><tr><td>c</td><td>d</td></tr>"
    assert teds(unwrapped, T) == 1.0


def test_order_does_not_matter():
    a = "<table><tr><td>a</td></tr></table>"
    assert teds(a, T) == pytest.approx(teds(T, a))


def test_bootstrap():
    mean, lo, hi = bootstrap_ci([0.5, 0.6, 0.7, 0.8, 0.9], n_boot=2000)
    assert lo < mean < hi
    diff, lo, hi, p = paired_bootstrap_diff([0.9] * 30, [0.5] * 30, n_boot=2000)
    assert diff == pytest.approx(0.4) and lo > 0 and p == 0.0
