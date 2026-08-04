#!/usr/bin/python3
"""Render a Markdown ledger or report to PDF: /usr/bin/python3 render_pdf.py FILE.md

WeasyPrint is installed against the system interpreter, not the project venv,
so run this with /usr/bin/python3.

Replaces the ad-hoc conversions that produced the checked-in PDFs.  Those were
written with a latin-1 read, which mangled every non-ASCII character
("Nystrom" and the multiplication sign in particular); this reads UTF-8.
"""
import sys
from pathlib import Path

import markdown
from weasyprint import HTML

CSS = """
@page { size: A4; margin: 18mm 16mm; @bottom-center { content: counter(page);
        font: 9pt "DejaVu Sans"; color: #666; } }
body { font: 10.5pt/1.45 "DejaVu Sans"; color: #111; }
h1 { font-size: 19pt; margin: 0 0 .6em; }
h2 { font-size: 14pt; margin: 1.4em 0 .5em; border-bottom: 1px solid #ccc;
     padding-bottom: .15em; }
h3 { font-size: 11.5pt; margin: 1.1em 0 .4em; }
h2, h3 { page-break-after: avoid; }
code { font-family: "DejaVu Sans Mono"; font-size: .87em; }
pre { background: #f5f5f5; padding: .6em .8em; font-size: .85em;
      white-space: pre-wrap; page-break-inside: avoid; }
table { border-collapse: collapse; width: 100%; font-size: 8.6pt;
        margin: .7em 0; page-break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 3px 5px; vertical-align: top;
         word-break: break-word; }
th { background: #eee; text-align: left; }
blockquote { border-left: 3px solid #ccc; margin-left: 0; padding-left: .8em;
             color: #444; }
"""


def render(path: Path) -> Path:
    html = markdown.markdown(
        path.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "sane_lists", "attr_list"],
    )
    out = path.with_suffix(".pdf")
    HTML(string=f"<style>{CSS}</style>{html}", base_url=str(path.parent)).write_pdf(out)
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for name in sys.argv[1:]:
        print(render(Path(name)))
