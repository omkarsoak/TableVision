"""HTML Similarity metric for table-image -> HTML generation.

Single source of truth for the metric that previously lived (in two different
versions) inside notebooks/evaluation.ipynb and notebooks/inference.ipynb.

Variants
--------
- "fixed"  (default): the 4-term metric from the original project with two corrections.
  (1) The original code computed ``1 - (|A∩B| / |A∪B| + 1e-10)``, i.e. the Jaccard
  *distance*, so a 20%-weighted term rewarded token mismatch.
  (2) Outputs that are truncated (no </table>) or lack the <table> wrapper are
  recovered before scoring; the original extracted text only from a literal <table>,
  so such outputs got zero content/token credit however good their rows were.
- "legacy": the 4-term metric exactly as published (including the inverted
  Jaccard term). Use this to reproduce the previously reported numbers (e.g. 0.77).
- "demo":   the older 3-term metric used in notebooks/inference.ipynb
  (0.10 style + 0.50 structure + 0.40 content). The qualitative
  example scores (0.3619 -> 0.9615) were produced with this variant.

Score components
----------------
S  style similarity      - niteru: Jaccard over CSS class sets
T  structural similarity - niteru: sequence similarity over tag sequences
C  content similarity    - 1 - normalized char-level Levenshtein of cell text
J  token overlap         - Jaccard over whitespace tokens of cell text
"""
import re
from typing import Dict, Union

from bs4 import BeautifulSoup
from niteru import style_similarity, structural_similarity
from torchmetrics.functional.text import edit_distance

from src.utils.helpers import recover_table_html

WEIGHTS = {
    "fixed": {"style": 0.10, "struct": 0.40, "content": 0.30, "jaccard": 0.20},
    "legacy": {"style": 0.10, "struct": 0.40, "content": 0.30, "jaccard": 0.20},
    "demo": {"style": 0.10, "struct": 0.50, "content": 0.40, "jaccard": 0.00},
}


def extract_table_text(html: str) -> str:
    """Extracts only the text from an HTML table in row-wise space-separated format."""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")  # Find the first table
    if not table:
        return ""
    return "\n".join(
        " ".join(cell.get_text(strip=True) for cell in row.find_all(["th", "td"]))
        for row in table.find_all("tr")
    )


def extract_html_table(html: str) -> str:
    """Extracts the first <table>...</table> block; returns the input unchanged if none."""
    match = re.search(r"<table\b.*?</table>", html, re.DOTALL | re.IGNORECASE)
    return match.group() if match else html


def html_similarity(
    html1: str,
    html2: str,
    variant: str = "fixed",
    return_components: bool = False,
    verbose: bool = False,
) -> Union[float, Dict[str, float]]:
    """Weighted HTML similarity in [0, 1]. Argument order does not matter."""
    if variant not in WEIGHTS:
        raise ValueError(f"Unknown variant {variant!r}; choose from {list(WEIGHTS)}")
    w = WEIGHTS[variant]

    if variant == "fixed":
        # also score truncated tables and bare rows (no <table> wrapper) on their content, see recover_table_html
        html1, html2 = (recover_table_html(h) or h for h in (html1, html2))
    elif variant == "legacy":  # exactly as published: no recovery, so unwrapped outputs get no text credit
        html1, html2 = extract_html_table(html1), extract_html_table(html2)
    # "demo" compared the raw strings

    style_sim = float(style_similarity(html1, html2))
    struct_sim = float(structural_similarity(html1, html2))

    txt1, txt2 = extract_table_text(html1), extract_table_text(html2)
    ed = float(edit_distance(txt1, txt2)) if (txt1 or txt2) else 0.0
    content_sim = 1 - ed / max(len(txt1), len(txt2) + 1e-10)  # original guard kept as-is

    tok1, tok2 = set(txt1.split()), set(txt2.split())
    union = tok1 | tok2
    jaccard = len(tok1 & tok2) / len(union) if union else 1.0
    jaccard_term = (1 - jaccard) if variant == "legacy" else jaccard

    score = (
        w["style"] * style_sim
        + w["struct"] * struct_sim
        + w["content"] * content_sim
        + w["jaccard"] * jaccard_term
    )
    score = max(0.0, min(1.0, score))

    if verbose:
        print(f"Style similarity: {style_sim:.4f}")
        print(f"Structural similarity: {struct_sim:.4f}")
        print(f"Lev-Edit Distance: {content_sim:.4f}")
        if w["jaccard"]:
            print(f"Token Jaccard: {jaccard:.4f}")
        print(f"Final HTML Similarity Score ({variant}): {score:.4f}")

    if return_components:
        return {
            "style": style_sim,
            "struct": struct_sim,
            "content": content_sim,
            "jaccard": jaccard,
            "score": score,
        }
    return score
