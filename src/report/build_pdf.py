"""Render the preprint Markdown to an academic-styled PDF via bundled Chromium.

Markdown -> styled HTML (serif, A4, figure/table styling) -> PDF (Playwright).
Relative image paths (figures/*.png) resolve to absolute file:// URLs so they embed.

Usage: .venv/bin/python src/report/build_pdf.py paper/paper.md paper/paper.pdf
"""
import argparse
import pathlib
import re

import markdown
from playwright.sync_api import sync_playwright

ROOT = pathlib.Path(__file__).resolve().parents[2]

CSS = """
@page { size: A4; margin: 22mm 20mm; }
* { box-sizing: border-box; }
body { font-family: Georgia, 'Times New Roman', serif; font-size: 10.6pt; line-height: 1.5;
       color: #1a1a1a; max-width: 100%; }
h1 { font-size: 19pt; text-align: center; line-height: 1.25; margin: 0 0 4px; }
h2 { font-size: 13.5pt; border-bottom: 1px solid #ccc; padding-bottom: 3px; margin: 20px 0 8px; }
h3 { font-size: 11.5pt; margin: 14px 0 5px; color: #222; }
p { margin: 0 0 9px; text-align: justify; }
a { color: #16607a; text-decoration: none; }
strong { color: #111; }
em { color: #333; }
hr { border: none; border-top: 1px solid #ddd; margin: 14px 0; }
code { background: #f3f1ec; padding: 0 3px; border-radius: 3px; font-size: 9pt; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 9.2pt; }
th, td { border: 1px solid #ccc; padding: 4px 7px; text-align: left; }
th { background: #f3f1ec; }
img { max-width: 92%; display: block; margin: 6px auto; border: 1px solid #e4e1da; }
ul, ol { margin: 0 0 9px; padding-left: 22px; }
li { margin: 2px 0; }
blockquote { color: #555; border-left: 3px solid #ccc; margin: 8px 0; padding: 2px 12px; }
h2, h3 { page-break-after: avoid; }
img, table, figure { page-break-inside: avoid; }
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("outp")
    a = ap.parse_args()
    src = pathlib.Path(a.inp).resolve()
    text = src.read_text(encoding="utf-8")
    html_body = markdown.markdown(text, extensions=["tables", "fenced_code", "attr_list", "sane_lists"])
    figdir = (ROOT / "figures").resolve()
    html_body = re.sub(r'src="(?:\.\./)*figures/([^"]+)"',
                       lambda m: f'src="file://{figdir}/{m.group(1)}"', html_body)
    doc = (f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head>"
           f"<body>{html_body}</body></html>")
    htmlpath = src.with_suffix(".rendered.html")
    htmlpath.write_text(doc, encoding="utf-8")
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page()
        pg.goto("file://" + str(htmlpath))
        pg.pdf(path=str(pathlib.Path(a.outp).resolve()), format="A4", print_background=True,
               margin={"top": "20mm", "bottom": "18mm", "left": "18mm", "right": "18mm"})
        b.close()
    print("PDF ->", a.outp)


if __name__ == "__main__":
    main()
