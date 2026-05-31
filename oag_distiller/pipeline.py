"""Domain Distiller Pipeline — orchestrates all phases."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from .assembler import Assembler
from .extractor import Extractor
from .llm import DistillerLLM
from .reader import Reader
from .reviewer import Reviewer

log = logging.getLogger(__name__)


def _save_state(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        path.write_text(data, encoding="utf-8")
    else:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


class DistillerPipeline:
    """Full pipeline: docs → ontology.yaml in 4 phases."""

    def __init__(self, docs_dir: str, output_dir: str | None = None, llm_config: dict | None = None):
        self.docs_dir = Path(docs_dir)
        if not self.docs_dir.exists():
            raise FileNotFoundError(f"Docs directory not found: {self.docs_dir}")

        self.output_dir = Path(output_dir) if output_dir else self.docs_dir.parent
        self.state_dir = self.output_dir / "state"
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.llm = DistillerLLM(llm_config or {})

    def run(self, up_to_phase: int = 3) -> Path:
        """Run the pipeline up to the specified phase (0-3).

        Returns path to the generated ontology.yaml.
        """
        t0 = time.time()
        log.info("=" * 60)
        log.info("Domain Distiller Pipeline starting")
        log.info("  docs_dir: %s", self.docs_dir)
        log.info("  output_dir: %s", self.output_dir)
        log.info("  model: %s", self.llm.model)
        log.info("=" * 60)

        doc_data = self._phase0_read()
        if up_to_phase < 1:
            return self.state_dir / "documents.json"

        extractions = self._phase1_extract(doc_data)
        if up_to_phase < 2:
            return self.state_dir / "extractions"

        yaml_text = self._phase2_assemble(extractions)
        if up_to_phase < 3:
            return self.state_dir / "assembled.yaml"

        final_yaml = self._phase3_review(yaml_text, doc_data)

        model_tag = self.llm.model.split("/")[-1].split(".")[0].replace(" ", "_")
        output_name = f"ontology_{model_tag}.yaml"
        output_path = self.output_dir / output_name
        output_path.write_text(final_yaml, encoding="utf-8")

        elapsed = time.time() - t0
        _save_state(
            self.state_dir / "generation_log.json",
            {
                "elapsed_seconds": round(elapsed, 1),
                "llm_usage": self.llm.usage,
                "call_log": self.llm.call_log,
                "model": self.llm.model,
                "doc_count": doc_data["doc_count"],
                "total_doc_chars": doc_data["total_chars"],
            },
        )

        log.info("=" * 60)
        log.info("Pipeline complete in %.0fs", elapsed)
        log.info("Output: %s", output_path)
        log.info(self.llm.usage_summary())
        log.info("=" * 60)

        return output_path

    def _phase0_read(self) -> dict:
        cache = self.state_dir / "documents.json"
        if cache.exists():
            log.info("Phase 0: Using cached documents.json")
            return json.loads(cache.read_text(encoding="utf-8"))
        reader = Reader(self.docs_dir, self.llm)
        doc_data = reader.run()
        _save_state(cache, doc_data)
        return doc_data

    def _phase1_extract(self, doc_data: dict) -> list[dict]:
        extractor = Extractor(doc_data, self.llm, self.state_dir)
        return extractor.run()

    def _phase2_assemble(self, extractions: list[dict]) -> str:
        cache = self.state_dir / "assembled.yaml"
        if cache.exists():
            log.info("Phase 2: Using cached assembled.yaml")
            return cache.read_text(encoding="utf-8")
        assembler = Assembler(extractions, self.llm)
        yaml_text = assembler.run()
        _save_state(cache, yaml_text)
        return yaml_text

    def _phase3_review(self, yaml_text: str, doc_data: dict) -> str:
        reviewer = Reviewer(yaml_text, doc_data, self.llm)
        final = reviewer.run()
        _save_state(self.state_dir / "reviewed.yaml", final)
        _save_state(
            self.state_dir / "review_issues.json",
            reviewer.review_issues,
        )
        return final

    def status(self) -> str:
        """Return a human-readable status of the pipeline state."""
        lines = [f"State directory: {self.state_dir}\n"]

        checks = [
            ("Phase 0: Documents", "documents.json"),
            ("Phase 2: Assembly", "assembled.yaml"),
            ("Phase 3: Review issues", "review_issues.json"),
            ("Phase 3: Reviewed", "reviewed.yaml"),
        ]
        for label, filename in checks:
            path = self.state_dir / filename
            if path.exists():
                size = path.stat().st_size
                lines.append(f"  ✓ {label}: {filename} ({size:,} bytes)")
            else:
                lines.append(f"  ✗ {label}: not yet generated")

        ext_dir = self.state_dir / "extractions"
        if ext_dir.exists():
            n = len(list(ext_dir.glob("*.json")))
            lines.append(f"  ✓ Phase 1: Extractions: {n} files")
        else:
            lines.append("  ✗ Phase 1: Extractions: not yet generated")

        output = self.output_dir / "ontology.yaml"
        if output.exists():
            lines.append(f"\n  ✓ Final output: {output} ({output.stat().st_size:,} bytes)")
        else:
            lines.append(f"\n  ✗ Final output: {output} (not yet generated)")

        log_path = self.state_dir / "generation_log.json"
        if log_path.exists():
            log_data = json.loads(log_path.read_text(encoding="utf-8"))
            lines.append(f"\n  Elapsed: {log_data.get('elapsed_seconds', 0):.0f}s")
            usage = log_data.get("llm_usage", {})
            lines.append(
                f"  LLM: {usage.get('calls', 0)} calls, "
                f"{usage.get('prompt_tokens', 0):,} prompt tokens, "
                f"{usage.get('completion_tokens', 0):,} completion tokens"
            )

        return "\n".join(lines)
