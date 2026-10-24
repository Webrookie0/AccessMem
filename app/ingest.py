"""Ingest and synchronize a metadata-tagged text or Markdown corpus."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

from .auth.models import Clearance
from .config import get_settings
from .services.vector_store import replace_corpus

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHUNK_CHARS = 800
CHUNK_OVERLAP = 100


def _parse_front_matter(text: str) -> tuple[dict, str]:
    parts = re.split(r"(?m)^---\s*$", text.lstrip("\ufeff"), maxsplit=1)
    if len(parts) != 2:
        raise ValueError("Missing front matter separator")
    metadata: dict[str, str] = {}
    for line in parts[0].splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError("Invalid front matter line")
        key, value = line.split(":", 1)
        key = key.strip().lower()
        if key in metadata:
            raise ValueError(f"Duplicate front matter key: {key}")
        metadata[key] = value.strip()
    body = parts[1].strip()
    if not body:
        raise ValueError("Document body must not be empty")
    return metadata, body


def _chunk(text: str, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if size < 1 or not 0 <= overlap < size:
        raise ValueError("Chunk size must be positive and overlap smaller than size")
    text = text.strip()
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # Prefer a paragraph, line or word boundary in the latter half.
            for separator in ("\n\n", "\n", " "):
                boundary = text.rfind(separator, start + size // 2, end)
                if boundary > start:
                    end = boundary + len(separator)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


def ingest_directory(directory: Path, *, adopt_legacy: bool = False) -> dict:
    if not directory.is_dir():
        raise ValueError(f"Data directory does not exist: {directory}")
    settings = get_settings()
    paths = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".md", ".txt"}
    )
    ids, texts, metadatas = [], [], []
    for path in paths:
        raw = path.read_text(encoding="utf-8")
        meta, body = _parse_front_matter(raw)
        tenant = meta.get("tenant")
        if not tenant or not meta.get("clearance"):
            raise ValueError(f"{path.name}: tenant and clearance are required")
        try:
            clearance = Clearance.from_name(meta["clearance"])
        except KeyError as exc:
            raise ValueError(f"{path.name}: invalid clearance") from exc
        document_hash = hashlib.sha256(raw.encode()).hexdigest()
        for index, chunk in enumerate(_chunk(body, settings.chunk_chars, settings.chunk_overlap)):
            ids.append(f"{settings.ingest_corpus_id}:{tenant}:{path.stem}::chunk{index}")
            texts.append(chunk)
            metadatas.append(
                {
                    "source": path.name,
                    "title": meta.get("title", path.stem),
                    "tenant_id": tenant,
                    "clearance_level": int(clearance),
                    "clearance_name": clearance.name,
                    "chunk_index": index,
                    "corpus_id": settings.ingest_corpus_id,
                    "document_hash": document_hash,
                    "content_hash": hashlib.sha256(chunk.encode()).hexdigest(),
                }
            )
    removed = replace_corpus(
        ids,
        texts,
        metadatas,
        corpus_id=settings.ingest_corpus_id,
        sources=[path.name for path in paths],
        adopt_legacy=adopt_legacy,
    )
    return {"documents": len(paths), "chunks": len(texts), "replaced_chunks": removed}


def main() -> None:
    parser = argparse.ArgumentParser(description="Synchronize AccessMem's managed document corpus")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--adopt-legacy",
        action="store_true",
        help="Also replace untagged documents from the original ingestion implementation",
    )
    args = parser.parse_args()
    try:
        result = ingest_directory(args.data_dir, adopt_legacy=args.adopt_legacy)
    except (ValueError, RuntimeError) as exc:
        parser.exit(1, f"Ingestion failed: {exc}\n")
    print(
        f"Indexed {result['chunks']} chunks from {result['documents']} documents; "
        f"replaced {result['replaced_chunks']} existing chunks."
    )


if __name__ == "__main__":
    main()
