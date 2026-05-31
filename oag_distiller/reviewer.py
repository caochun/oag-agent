"""Phase 3: Validate, self-review, and fix the assembled ontology."""

from __future__ import annotations

import json
import logging

import yaml

from oag.ontology.schema import Ontology
from .llm import DistillerLLM
from .prompts import FIX_SYSTEM, FIX_USER, REVIEW_SYSTEM, REVIEW_USER

log = logging.getLogger(__name__)


def _validate_schema(yaml_text: str) -> tuple[Ontology | None, list[str]]:
    """Try to parse YAML and validate against Ontology schema."""
    errors: list[str] = []

    try:
        raw = yaml.safe_load(yaml_text)
    except yaml.YAMLError as e:
        return None, [f"YAML syntax error: {e}"]

    if not isinstance(raw, dict):
        return None, ["YAML root is not a dict"]

    try:
        ont = Ontology.model_validate(raw)
    except Exception as e:
        return None, [f"Schema validation error: {e}"]

    return ont, errors


def _check_cross_references(ont: Ontology) -> list[str]:
    """Check that all cross-references between layers are valid."""
    errors: list[str] = []
    obj_names = set(ont.objects.keys())
    fn_names = set(ont.functions.keys())

    for name, link in ont.links.items():
        if link.source not in obj_names:
            errors.append(f"link '{name}': source '{link.source}' not in objects")
        if link.target not in obj_names:
            errors.append(f"link '{name}': target '{link.target}' not in objects")

    for name, fn in ont.functions.items():
        for wt in fn.writes_to:
            if wt not in obj_names:
                errors.append(f"function '{name}': writes_to '{wt}' not in objects")
        for io in fn.involves_objects:
            if io not in obj_names:
                errors.append(f"function '{name}': involves_objects '{io}' not in objects")
        for pc in fn.preconditions:
            if pc.object not in obj_names:
                errors.append(f"function '{name}': precondition object '{pc.object}' not in objects")
        for eff in fn.effects:
            if eff.object not in obj_names:
                errors.append(f"function '{name}': effect object '{eff.object}' not in objects")

    for name, rule in ont.rules.items():
        for at in rule.applies_to:
            if at not in obj_names:
                errors.append(f"rule '{name}': applies_to '{at}' not in objects")

    for name, wf in ont.workflows.items():
        for step in wf.steps:
            if step.function and step.function not in fn_names:
                errors.append(f"workflow '{name}' step '{step.name}': function '{step.function}' not in functions")
        for io in wf.involves_objects:
            if io not in obj_names:
                errors.append(f"workflow '{name}': involves_objects '{io}' not in objects")

    for name, obj in ont.objects.items():
        has_pk = any(p.required for p in obj.properties.values())
        if not has_pk:
            errors.append(f"object '{name}': no required=true property (missing primary key)")

    return errors


class Reviewer:
    """Validate, self-review via LLM, and fix the ontology."""

    def __init__(
        self, yaml_text: str, doc_data: dict, llm: DistillerLLM, *, max_fix_rounds: int = 2
    ):
        self.yaml_text = yaml_text
        self.doc_data = doc_data
        self.llm = llm
        self.max_fix_rounds = max_fix_rounds
        self.review_issues: list[dict] = []

    def run(self) -> str:
        log.info("Phase 3: Reviewing and validating ontology")

        yaml_text = self.yaml_text

        for attempt in range(self.max_fix_rounds + 1):
            ont, parse_errors = _validate_schema(yaml_text)

            if parse_errors:
                log.warning("Validation errors (attempt %d): %s", attempt + 1, parse_errors)
                if attempt < self.max_fix_rounds:
                    yaml_text = self._fix_with_llm(yaml_text, parse_errors)
                    continue
                else:
                    log.error("Could not fix YAML after %d attempts", self.max_fix_rounds)
                    break

            xref_errors = _check_cross_references(ont)
            if xref_errors:
                log.warning("Cross-reference errors: %d issues", len(xref_errors))
                for err in xref_errors[:10]:
                    log.warning("  %s", err)

            all_errors = parse_errors + xref_errors
            review_issues = self._llm_review(yaml_text, all_errors)
            self.review_issues = review_issues

            if review_issues and attempt < self.max_fix_rounds:
                log.info("LLM review found %d issues, fixing...", len(review_issues))
                yaml_text = self._fix_with_llm(yaml_text, [i["issue"] for i in review_issues])
            else:
                break

        ont_final, final_errors = _validate_schema(yaml_text)
        if final_errors:
            log.error("Final validation still has errors: %s", final_errors[:5])
        else:
            xref = _check_cross_references(ont_final)
            log.info(
                "Phase 3 complete. Objects: %d, Links: %d, Functions: %d, Rules: %d, Workflows: %d. Cross-ref issues: %d",
                len(ont_final.objects),
                len(ont_final.links),
                len(ont_final.functions),
                len(ont_final.rules),
                len(ont_final.workflows),
                len(xref),
            )

        return yaml_text

    def _llm_review(self, yaml_text: str, validation_errors: list[str]) -> list[dict]:
        doc_summaries = "\n".join(
            f"- {d['filename']}: {d.get('summary', '')}"
            for d in self.doc_data.get("documents", [])
        )
        errors_str = "\n".join(f"- {e}" for e in validation_errors) or "无"

        result = self.llm.call_json(
            REVIEW_SYSTEM,
            REVIEW_USER.format(
                ontology_yaml=yaml_text,
                document_summaries=doc_summaries,
                validation_errors=errors_str,
            ),
        )
        return result.get("issues", [])

    def _fix_with_llm(self, yaml_text: str, issues: list[str]) -> str:
        issues_json = json.dumps(
            [{"issue": i} if isinstance(i, str) else i for i in issues[:15]],
            ensure_ascii=False,
            indent=2,
        )
        yaml_truncated = yaml_text[:30000] if len(yaml_text) > 30000 else yaml_text
        fixed = self.llm.call(
            FIX_SYSTEM,
            FIX_USER.format(ontology_yaml=yaml_truncated, issues_json=issues_json),
        )
        fixed = fixed.strip()
        if fixed.startswith("```"):
            lines = fixed.split("\n")
            fixed = "\n".join(lines[1:])
            if fixed.endswith("```"):
                fixed = fixed[:-3].strip()
        if len(fixed) < len(yaml_text) * 0.5:
            log.warning("LLM fix output too short (%d vs %d chars), keeping original", len(fixed), len(yaml_text))
            return yaml_text
        return fixed
