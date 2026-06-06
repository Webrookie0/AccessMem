"""One-shot loader: read ./data, chunk, embed, and upsert to Chroma with the
right tenant + clearance metadata.

Document files use a tiny YAML-ish front matter:

    tenant: acme
    clearance: CONFIDENTIAL
    title: Some Title
    ---
    ... body ...
"""
from __future__ import annotations

import re
from pathlib import Path

from .auth.models import Clearance
from .services.vector_store import add_documents

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHUNK_CHARS = 800
CHUNK_OVERLAP = 100


def _parse_front_matter(text: str) -> tuple[dict, str]:
    if "---" not in text:
        raise ValueError("missing '---' separator")
    head, body = text.split("---", 1)
    meta: dict[str, str] = {}
    for line in head.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        k, v = line.split(":", 1)
        meta[k.strip().lower()] = v.strip()
    return meta, body.strip()


def _chunk(text: str, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def main() -> None:
    files = sorted(DATA_DIR.glob("*.md"))
    if not files:
        print(f"No .md files found in {DATA_DIR}")
        return

    ids: list[str] = []
    texts: list[str] = []
    metas: list[dict] = []

    for path in files:
        raw = path.read_text(encoding="utf-8")
        meta, body = _parse_front_matter(raw)
        tenant = meta.get("tenant")
        clearance_name = (meta.get("clearance") or "").upper()
        title = meta.get("title", path.stem)
        if not tenant or not clearance_name:
            raise ValueError(f"{path.name}: missing tenant or clearance")
        clearance = Clearance.from_name(clearance_name)

        for i, chunk in enumerate(_chunk(body)):
            ids.append(f"{path.stem}::chunk{i}")
            texts.append(chunk)
            metas.append({
                "source": path.name,
                "title": title,
                "tenant_id": tenant,
                "clearance_level": int(clearance),
                "clearance_name": clearance.name,
                "chunk_index": i,
            })
        print(f"  prepared {path.name}  tenant={tenant}  clearance={clearance.name}")

    print(f"\nUpserting {len(texts)} chunks from {len(files)} documents...")
    add_documents(ids, texts, metas)
    print("Done.")


if __name__ == "__main__":
    main()
