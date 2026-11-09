"""Splits documents into chunks. Each chunk keeps its own text and a copy of the document's metadata."""
from dataclasses import dataclass

from rag.loader import Doc


@dataclass
class Chunk:
    id: str            # "<doc id>#<number>", e.g. "of-210-city#2"
    doc_id: str
    text: str          # the original text, exactly as it will be shown to the model
    meta: dict


def chunk_fixed(doc: Doc, size: int = 350, overlap: int = 50) -> list[Chunk]:
    """Baseline: cut every `size` characters, with `overlap` characters repeated between neighbours."""
    chunks = []
    step = size - overlap
    for start in range(0, max(len(doc.text), 1), step):
        piece = doc.text[start:start + size]
        if piece.strip():
            chunks.append(Chunk(f"{doc.id}#{len(chunks)}", doc.id, piece, dict(doc.meta)))
        if start + size >= len(doc.text):
            break
    return chunks


def chunk_by_headings(doc: Doc) -> list[Chunk]:
    """One chunk per "## " section, each prefixed with its "title › heading" path line."""
    sections = [(None, [])]
    for line in doc.text.splitlines():
        if line.startswith("## "):
            sections.append((line[3:].strip(), []))
        elif not line.startswith("# "):
            sections[-1][1].append(line)

    chunks = []
    for heading, lines in sections:
        body = "\n".join(lines).strip()
        # A title-only document (e.g. a failed scan) still gets one chunk so it stays findable.
        if not body and not (heading is None and len(sections) == 1):
            continue
        path = doc.title if heading is None else f"{doc.title} › {heading}"
        meta = dict(doc.meta, section=heading or "")
        chunks.append(Chunk(f"{doc.id}#{len(chunks)}", doc.id, f"{path}\n\n{body}".strip(), meta))
    return chunks
