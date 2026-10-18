"""Vendor the English text of "Local AI Field Guide" into docs/corpus/.

Shallow-clones https://github.com/antoinelucasfra/local-ai-field-guide into
`<system temp dir>/book` (reusing a previous clone when present), then extracts
only the English (`::: {.en}`) blocks of each `chapters/_src/*.qmd` chapter, in
book order (`chapters/_order.txt`). One markdown file per chapter lands in
`docs/corpus/`, next to an ATTRIBUTION.md recording provenance.

The output is a reading corpus, not a buildable Quarto project: shortcodes and
attribute braces are dropped, figure targets are dropped (alt text is kept),
everything else (headings, lists, tables, code fences, footnote markers) is
kept verbatim.

    uv run tools/extract_book.py
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from datetime import date
from pathlib import Path

REPO_URL = "https://github.com/antoinelucasfra/local-ai-field-guide"
CLONE_DIR = Path(tempfile.gettempdir()) / "book"
OUT_DIR = Path("docs/corpus")
AUTHOR = "Antonin Lucas"
LICENSE = "CC BY 4.0"

FENCE_RE = re.compile(r"^(`{3,}|~{3,})")
SHORTCODE_RE = re.compile(r"\{\{[<%].*?[%>]\}\}")
IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
ATTR_RE = re.compile(r"\{[.#][^{}]*\}")
FENCE_LANG_RE = re.compile(r"^\{\.(\w+)\b[^}]*\}")
FENCE_INCLUDE_RE = re.compile(r"^\{\.(\w+)\b[^}]*include=\"([^\"]+)\"[^}]*\}")


def _run(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def ensure_clone() -> Path:
    """Return the book clone, shallow-cloning into the temp dir if needed."""
    if (CLONE_DIR / ".git").exists() and (CLONE_DIR / "chapters" / "_order.txt").exists():
        return CLONE_DIR
    if CLONE_DIR.exists():
        shutil.rmtree(CLONE_DIR)
    _run("git", "clone", "--depth", "1", REPO_URL, str(CLONE_DIR))
    return CLONE_DIR


def read_titles(src_dir: Path) -> dict[str, str]:
    """Map chapter filenames to their English titles from _order.txt."""
    titles: dict[str, str] = {}
    for line in (src_dir.parent / "_order.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("FILE ") and "|" in line:
            name, en_title, _fr = line[len("FILE ") :].split("|", 2)
            titles[name.strip()] = en_title.strip()
    return titles


def clean_line(line: str) -> str:
    """Drop shortcodes, image targets and attribute braces from one line."""
    line = SHORTCODE_RE.sub("", line)
    line = IMAGE_RE.sub(lambda m: f"![{m.group(1)}]", line)
    line = ATTR_RE.sub("", line)
    return line.rstrip()


def extract_english(path: Path, book_root: Path) -> str:
    """Concatenate the `::: {.en}` blocks of one chapter, cleaned up.

    Block nesting is tracked with a depth counter: a closing `:::` at column 0
    may close a nested `::: {.al-take}` block rather than the `.en` block
    itself. Lines inside a nested `::: {.fr}` block are skipped. Fenced code is
    passed through untouched, except `include=` fences, which are expanded from
    the referenced file so the corpus carries the actual code.
    """
    out: list[str] = []
    block_depth = 0  # ::: depth inside the .en block; 0 when outside .en
    stack: list[str] = []  # classes of enclosing blocks, innermost last
    fence = ""  # opening fence marker while inside a code fence
    include_target: str | None = None
    include_lang = ""

    for line in path.read_text(encoding="utf-8").splitlines():
        in_en = bool(block_depth) and ".en" in stack[0]

        if line.startswith(":::") and not fence:
            if line.strip() == ":::":
                if block_depth:
                    block_depth -= 1
                    stack.pop()
            else:
                block_depth += 1
                stack.append(line)
            continue

        if fence:
            if line.startswith(fence):
                fence = ""
                if include_target is not None:
                    inc = path.parent.joinpath(include_target)
                    if not inc.exists():
                        inc = book_root / include_target.replace("../", "")
                    code = inc.read_text(encoding="utf-8").rstrip()
                    out.append(f"```{include_lang}\n{code}\n```")
                    include_target = None
            elif in_en and include_target is None:
                out.append(line)
            continue

        if FENCE_RE.match(line) and in_en:
            rest = line.lstrip("`~").strip()
            if rest.startswith("{"):
                inc = FENCE_INCLUDE_RE.match(rest)
                if inc:
                    include_target, include_lang = inc.group(2), inc.group(1)
                else:
                    lang = FENCE_LANG_RE.match(rest)
                    out.append(f"```{lang.group(1)}" if lang else "```")
            else:
                out.append(line)
            fence = line[: len(line) - len(line.lstrip("`~"))]
            continue

        if in_en:
            out.append(clean_line(line))

    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def write_chapter(src: Path, book_root: Path, title: str, dest: Path) -> None:
    body = extract_english(src, book_root)
    dest.write_text(f"# {title}\n\n{body}\n", encoding="utf-8", newline="\n")


def write_attribution(book: Path, commit: str) -> None:
    text = (
        "# Local AI Field Guide — English corpus\n\n"
        f"- Book: *Local AI Field Guide*\n"
        f"- Repository: {REPO_URL}\n"
        f"- Author: {AUTHOR}\n"
        f"- License: {LICENSE}\n"
        f"- Snapshot commit: {commit}\n"
        f"- Extracted: {date.today().isoformat()}\n\n"
        "English (`::: {.en}`) blocks extracted verbatim from `chapters/_src/*.qmd`;\n"
        "shortcodes, attribute braces and figure targets removed. Extracted by\n"
        "`tools/extract_book.py`.\n"
    )
    (OUT_DIR / "ATTRIBUTION.md").write_text(text, encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    book = ensure_clone()
    commit = _run("git", "rev-parse", "HEAD", cwd=book)
    src_dir = book / "chapters" / "_src"
    titles = read_titles(src_dir)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, title in titles.items():
        src = src_dir / name
        if not src.exists():
            raise FileNotFoundError(f"chapter listed in _order.txt is missing: {name}")
        slug = src.stem
        write_chapter(src, book, title, OUT_DIR / f"ch-{slug}.md")
    write_attribution(book, commit)

    files = sorted(OUT_DIR.glob("*.md"))
    words = sum(len(f.read_text(encoding="utf-8").split()) for f in files)
    print(f"wrote {len(files)} files ({words} words) to {OUT_DIR} from {commit}")


if __name__ == "__main__":
    main()
