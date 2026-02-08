from __future__ import annotations

from pathlib import Path

import pdfplumber
from docx import Document


TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".py",
    ".java",
    ".js",
    ".ts",
    ".tsx",
    ".go",
    ".sql",
    ".json",
    ".yaml",
    ".yml",
    ".ini",
    ".toml",
    ".sh",
}


def parse_file(path: str) -> str:
    source = Path(path)
    suffix = source.suffix.lower()

    if suffix in TEXT_EXTENSIONS:
        return _read_text_file(source)
    if suffix == ".pdf":
        return _read_pdf(source)
    if suffix == ".docx":
        return _read_docx(source)

    # Fallback: try text mode, ignore undecodable chars.
    return source.read_text(encoding="utf-8", errors="ignore")


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf(path: Path) -> str:
    pages: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return "\n".join(pages)


def _read_docx(path: Path) -> str:
    doc = Document(path)
    return "\n".join(paragraph.text for paragraph in doc.paragraphs)
