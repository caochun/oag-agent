"""Phase 1: Per-document knowledge extraction with accumulating context."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .llm import DistillerLLM
from .prompts import (
    EXTRACTION_SYSTEM,
    EXTRACTION_USER,
    METHODOLOGY,
    get_metamodel_spec,
)

log = logging.getLogger(__name__)


def _build_accumulated_context(extractions: list[dict]) -> str:
    """Build accumulated knowledge summary from previous extractions."""
    if not extractions:
        return "当前已从前序文档积累的领域知识: 暂无，这是第一篇文档。"

    lines = ["当前已从前序文档积累的领域知识:\n"]

    all_objects = []
    for ext in extractions:
        for obj in ext.get("objects", []):
            all_objects.append(f"  - {obj['name']}: {obj.get('summary', '')}")
    if all_objects:
        lines.append("已识别的对象类型:")
        lines.extend(all_objects[:50])

    all_functions = []
    for ext in extractions:
        for fn in ext.get("functions", []):
            all_functions.append(f"  - {fn['name']}: {fn.get('summary', '')}")
    if all_functions:
        lines.append("\n已识别的函数:")
        lines.extend(all_functions[:30])

    all_rules = []
    for ext in extractions:
        for rule in ext.get("rules", []):
            all_rules.append(f"  - {rule['name']}: {rule.get('description', '')}")
    if all_rules:
        lines.append("\n已识别的规则:")
        lines.extend(all_rules[:20])

    all_links = []
    for ext in extractions:
        for link in ext.get("links", []):
            all_links.append(f"  - {link.get('source', '')} → {link.get('target', '')}: {link.get('description', '')}")
    if all_links:
        lines.append("\n已识别的关系:")
        lines.extend(all_links[:20])

    lines.append("\n请在阅读新文档时:")
    lines.append("- 为已有对象补充新发现的属性")
    lines.append("- 发现新的对象/关系/函数/规则/工作流")
    lines.append("- 建立跨文档的关系(如新文档中的概念关联到已有对象)")

    return "\n".join(lines)


class Extractor:
    """Extract ontology materials from each document iteratively."""

    def __init__(self, doc_data: dict, llm: DistillerLLM, state_dir: Path):
        self.documents = doc_data["documents"]
        self.llm = llm
        self.state_dir = state_dir
        self.extractions_dir = state_dir / "extractions"
        self.extractions_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> list[dict]:
        log.info("Phase 1: Extracting from %d documents", len(self.documents))

        system_prompt = EXTRACTION_SYSTEM.format(
            metamodel_spec=get_metamodel_spec(),
            methodology=METHODOLOGY,
        )

        extractions: list[dict] = []

        for i, doc in enumerate(self.documents):
            log.info(
                "  [%d/%d] Extracting from %s (%d chars)...",
                i + 1, len(self.documents), doc["filename"], doc["char_count"],
            )

            cache_file = self.extractions_dir / f"{Path(doc['filename']).stem}.json"
            if cache_file.exists():
                log.info("    Using cached extraction: %s", cache_file)
                extraction = json.loads(cache_file.read_text(encoding="utf-8"))
                extractions.append(extraction)
                continue

            accumulated = _build_accumulated_context(extractions)
            user_prompt = EXTRACTION_USER.format(
                accumulated_context=accumulated,
                doc_name=doc["filename"],
                doc_content=doc["full_text"],
            )

            extraction = self.llm.call_json(system_prompt, user_prompt)

            extraction["_meta"] = {
                "doc_name": doc["filename"],
                "doc_type": doc.get("doc_type", ""),
                "doc_summary": doc.get("summary", ""),
            }

            cache_file.write_text(
                json.dumps(extraction, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            extractions.append(extraction)

            n_obj = len(extraction.get("objects", []))
            n_fn = len(extraction.get("functions", []))
            n_rule = len(extraction.get("rules", []))
            n_link = len(extraction.get("links", []))
            n_wf = len(extraction.get("workflows", []))
            log.info(
                "    Extracted: %d objects, %d functions, %d rules, %d links, %d workflows",
                n_obj, n_fn, n_rule, n_link, n_wf,
            )

        total = {
            "objects": sum(len(e.get("objects", [])) for e in extractions),
            "functions": sum(len(e.get("functions", [])) for e in extractions),
            "rules": sum(len(e.get("rules", [])) for e in extractions),
            "links": sum(len(e.get("links", [])) for e in extractions),
            "workflows": sum(len(e.get("workflows", [])) for e in extractions),
        }
        log.info("Phase 1 complete. Total raw: %s", total)
        return extractions
