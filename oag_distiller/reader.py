"""Phase 0: Document reading and preprocessing."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from .llm import DistillerLLM
from .prompts import DOCUMENT_SUMMARY_SYSTEM, DOCUMENT_SUMMARY_USER

log = logging.getLogger(__name__)


def _chunk_markdown(text: str, doc_name: str) -> list[dict]:
    """Split markdown by heading hierarchy, preserving section paths."""
    chunks: list[dict] = []
    lines = text.split("\n")
    current_path: list[str] = []
    current_content: list[str] = []
    current_level = 0

    def flush():
        if current_content:
            content = "\n".join(current_content).strip()
            if content:
                chunks.append({
                    "doc_name": doc_name,
                    "section_path": " > ".join(current_path) if current_path else doc_name,
                    "content": content,
                    "char_count": len(content),
                })

    for line in lines:
        m = re.match(r"^(#{1,4})\s+(.+)", line)
        if m:
            flush()
            current_content = []
            level = len(m.group(1))
            title = m.group(2).strip()
            if level <= current_level:
                current_path = current_path[: level - 1]
            current_path = current_path[: level - 1] + [title]
            current_level = level
            current_content.append(line)
        else:
            current_content.append(line)

    flush()
    return chunks


class Reader:
    """Read and preprocess documents from a directory."""

    def __init__(self, docs_dir: Path, llm: DistillerLLM):
        self.docs_dir = docs_dir
        self.llm = llm

    def run(self) -> dict:
        md_files = sorted(self.docs_dir.glob("*.md"))
        if not md_files:
            raise FileNotFoundError(f"No .md files found in {self.docs_dir}")

        log.info("Phase 0: Reading %d documents from %s", len(md_files), self.docs_dir)

        documents = []
        for fp in md_files:
            text = fp.read_text(encoding="utf-8")
            chunks = _chunk_markdown(text, fp.name)
            documents.append({
                "filename": fp.name,
                "full_text": text,
                "char_count": len(text),
                "chunks": chunks,
                "chunk_count": len(chunks),
            })

        doc_list = "\n".join(
            f"- {d['filename']} ({d['char_count']} chars): {d['full_text'][:500]}..."
            for d in documents
        )
        summaries = self.llm.call_json(
            DOCUMENT_SUMMARY_SYSTEM,
            DOCUMENT_SUMMARY_USER.format(doc_list=doc_list),
        )

        summary_map = {
            s["filename"]: s
            for s in summaries.get("documents", [])
        }
        for doc in documents:
            info = summary_map.get(doc["filename"], {})
            doc["summary"] = info.get("summary", "")
            doc["doc_type"] = info.get("doc_type", "其他")
            doc["priority"] = info.get("priority", 3)

        documents.sort(key=lambda d: d["priority"])

        log.info(
            "Phase 0 complete: %d documents, %d total chunks",
            len(documents),
            sum(d["chunk_count"] for d in documents),
        )
        for d in documents:
            log.info(
                "  [P%d] %s (%s): %s",
                d["priority"], d["filename"], d["doc_type"], d["summary"],
            )

        return {
            "documents": documents,
            "doc_count": len(documents),
            "total_chars": sum(d["char_count"] for d in documents),
        }
