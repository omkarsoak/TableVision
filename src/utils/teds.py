"""TEDS: Tree-Edit-Distance-based Similarity for tables (Zhong et al., 2020, the PubTabNet paper).

    TEDS(T_a, T_b) = 1 - EditDist(T_a, T_b) / max(|T_a|, |T_b|)

Tables are trees (table > [thead/tbody >] tr > td). Insert/delete cost 1. Renaming a node costs 1 when the
tags differ; between two cells it costs 1 if their colspan/rowspan differ, otherwise the normalized
Levenshtein distance of their text (0 for TEDS-Struct, which ignores text).

This is our own implementation of the metric as described in the paper, not the official script, so
absolute numbers can differ slightly from published ones.

normalize=True (default) makes the score robust to harmless formatting differences: inline tags (<b>, <i>,
<sup>, <span> ...) are flattened into the cell text, <thead>/<tbody>/<tfoot> wrappers are removed and <th>
is treated as <td>. This matters for zero-shot Granite Vision 4.x, which emits correct tables without
PubTabNet's <thead> and <b> conventions. normalize=False keeps those wrappers and th/td distinctions.
"""
import re
from typing import Optional

from apted import APTED, Config
from lxml import etree, html as lhtml

from src.utils.helpers import recover_table_html

try:
    from rapidfuzz.distance import Levenshtein as _Lev
except ImportError:   # slow pure-python fallback
    _Lev = None

INLINE_TAGS = ("b", "i", "u", "s", "strike", "strong", "em", "sup", "sub", "span", "font", "a", "small", "big", "mark", "p")
WRAPPER_TAGS = ("thead", "tbody", "tfoot")
STRUCT_TAGS = {"table", "thead", "tbody", "tfoot", "tr", "td", "th"}


def _levenshtein(a: str, b: str) -> int:
    if _Lev is not None:
        return _Lev.distance(a, b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _norm_lev(a: str, b: str) -> float:
    if not a and not b:
        return 0.0
    return _levenshtein(a, b) / max(len(a), len(b))


class _Node:
    __slots__ = ("tag", "colspan", "rowspan", "content", "children")

    def __init__(self, tag):
        self.tag, self.colspan, self.rowspan, self.content, self.children = tag, 1, 1, "", []


class _Config(Config):
    def __init__(self, structure_only: bool):
        self.structure_only = structure_only

    def rename(self, a, b):
        if a.tag != b.tag:
            return 1.0
        if a.tag in ("td", "th"):
            if a.colspan != b.colspan or a.rowspan != b.rowspan:
                return 1.0
            return 0.0 if self.structure_only else _norm_lev(a.content, b.content)
        return 0.0

    def children(self, node):
        return node.children


def _span(value) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return 1


def _build(el, normalize: bool) -> _Node:
    tag = el.tag.lower()
    if normalize and tag == "th":
        tag = "td"
    node = _Node(tag)
    if tag in ("td", "th"):
        node.colspan, node.rowspan = _span(el.get("colspan")), _span(el.get("rowspan"))
        node.content = " ".join("".join(el.itertext()).split())   # nested tables are flattened into text
        return node
    node.children = [_build(c, normalize) for c in el if isinstance(c.tag, str) and c.tag.lower() in STRUCT_TAGS]
    return node


def _count(node: _Node) -> int:
    return 1 + sum(_count(c) for c in node.children)


def _parse(html_str: str, normalize: bool, recover: bool) -> Optional[_Node]:
    if recover:
        table_html = recover_table_html(html_str)
    else:   # strict: only a complete <table>...</table> counts
        match = re.search(r"<table\b.*?</table>", html_str or "", re.DOTALL | re.IGNORECASE)
        table_html = match.group() if match else None
    if table_html is None:
        return None
    try:
        root = lhtml.fromstring(table_html)
    except (etree.ParserError, etree.XMLSyntaxError):
        return None
    table = root if root.tag == "table" else root.find(".//table")
    if table is None:
        return None
    etree.strip_tags(table, *INLINE_TAGS)
    if normalize:
        etree.strip_tags(table, *WRAPPER_TAGS)
    return _build(table, normalize)


def teds(pred_html: str, true_html: str, structure_only: bool = False, normalize: bool = True,
         recover: bool = True) -> float:
    """TEDS in [0, 1] between the first <table> of each input; 0.0 if either has no parseable table.

    recover=True also scores outputs whose table is truncated (no </table>) or missing its <table> wrapper
    (see helpers.recover_table_html); recover=False gives those 0.0 ("strict")."""
    pred, true = _parse(pred_html, normalize, recover), _parse(true_html, normalize, recover)
    if pred is None or true is None:
        return 0.0
    distance = APTED(pred, true, _Config(structure_only)).compute_edit_distance()
    return max(0.0, 1.0 - distance / max(_count(pred), _count(true)))
