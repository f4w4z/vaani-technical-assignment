"""Regenerate docs/assets/search-index.js and check internal links.

Run from anywhere:  python docs/build_search_index.py

Each page is split at its h2/h3 headings (which must carry ids). Every section
becomes one search entry holding its heading and the plain text beneath it.
The same pass verifies that every local href and src in the pages resolves to
an existing file, and that every #fragment exists as an id on its target page.
Exit status is 1 if any link is broken.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
import re
import sys

DOCS = Path(__file__).resolve().parent
MAIN = re.compile(r'<main id="content">(.*?)</main>', re.S)
HEAD = re.compile(r'<h([23])\s+id="([^"]+)"[^>]*>(.*?)</h\1>', re.S)
H1 = re.compile(r"<h1[^>]*>(.*?)</h1>", re.S)
TAGS = re.compile(r"<[^>]+>")
SKIP = re.compile(r"<(script|style)\b.*?</\1>", re.S)
LINK = re.compile(r'(?:href|src)="([^"]+)"')
ID = re.compile(r'\bid="([^"]+)"')
MAX_TEXT = 700


def plain(fragment: str) -> str:
    text = TAGS.sub(" ", SKIP.sub(" ", fragment))
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def build_entries(pages: list[Path]) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for page in pages:
        source = page.read_text(encoding="utf-8")
        body = MAIN.search(source)
        if not body:
            raise ValueError(f"{page.name}: missing <main id=\"content\">")
        content = body.group(1)
        title = plain(H1.search(content).group(1))
        marks = list(HEAD.finditer(content))
        intro_end = marks[0].start() if marks else len(content)
        intro = plain(content[:intro_end])
        entries.append({"p": page.stem, "t": title, "h": title, "id": "", "x": intro[:MAX_TEXT]})
        for index, mark in enumerate(marks):
            end = marks[index + 1].start() if index + 1 < len(marks) else len(content)
            entries.append({
                "p": page.stem, "t": title, "h": plain(mark.group(3)),
                "id": mark.group(2), "x": plain(content[mark.end():end])[:MAX_TEXT],
            })
    return entries


def check_links(pages: list[Path]) -> list[str]:
    ids = {page.name: set(ID.findall(page.read_text(encoding="utf-8"))) for page in pages}
    problems: list[str] = []
    for page in pages:
        for target in LINK.findall(page.read_text(encoding="utf-8")):
            if re.match(r"(https?:|mailto:|#$)", target):
                continue
            path, _, fragment = target.partition("#")
            if not path:  # same-page fragment
                if fragment and fragment not in ids[page.name]:
                    problems.append(f"{page.name}: missing anchor #{fragment}")
                continue
            resolved = (DOCS / path).resolve()
            if not resolved.is_file():
                problems.append(f"{page.name}: missing file {path}")
            elif fragment and resolved.suffix == ".html" and fragment not in ids.get(resolved.name, set()):
                problems.append(f"{page.name}: {path} has no anchor #{fragment}")
    return problems


def main() -> None:
    pages = sorted(DOCS.glob("*.html"))
    entries = build_entries(pages)
    (DOCS / "assets" / "search-index.js").write_text(
        "window.SEARCH_INDEX = " + json.dumps(entries, ensure_ascii=False, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )
    print(f"{len(pages)} pages, {len(entries)} search entries")
    problems = check_links(pages)
    for problem in problems:
        print("BROKEN:", problem)
    if problems:
        sys.exit(1)
    print("all internal links resolve")


if __name__ == "__main__":
    main()
