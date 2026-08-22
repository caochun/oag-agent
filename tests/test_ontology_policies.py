from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from oag.ontology.bindings import RuntimeBindings
from oag.ontology.inspector import OntologyInspector
from oag.ontology.prompt_builder import OntologyPromptBuilder
from oag.ontology.schema import Ontology, Precondition


def make_policy_ontology() -> Ontology:
    return Ontology.model_validate({
        "name": "PolicyDomain",
        "data_sources": {"memory": {"type": "memory"}},
        "objects": {
            "ResultCell": {
                "summary": "结果单元",
                "binding": {"source": "memory"},
            },
        },
        "interaction_policies": {
            "user_chat": {
                "description": "用户交互",
                "include_in_system_prompt": True,
                "instructions": ["只使用已验证的领域事实。"],
            },
        },
    })


def test_interaction_policy_is_injected_into_system_prompt():
    builder = OntologyPromptBuilder(make_policy_ontology(), RuntimeBindings())

    prompt = builder.build_system_prompt()

    assert "## 领域交互策略" in prompt
    assert "只使用已验证的领域事实" in prompt


def test_inspect_exposes_interaction_policy():
    ontology = make_policy_ontology()
    inspector = OntologyInspector(ontology, RuntimeBindings())

    result = json.loads(inspector.inspect("user_chat"))

    assert result["kind"] == "interaction_policy"
    assert result["instructions"] == ["只使用已验证的领域事实。"]


@pytest.mark.parametrize("unsupported_field", ["presentation_tools", "event_policies"])
def test_ontology_rejects_runtime_or_ui_specific_policy_fields(unsupported_field):
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Ontology.model_validate({
            "name": "InvalidPolicyDomain",
            unsupported_field: {"unsupported": {}},
        })


def test_interaction_policy_rejects_keyword_intent_routing():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Ontology.model_validate({
            "name": "InvalidInteractionDomain",
            "interaction_policies": {
                "user_chat": {
                    "intents": {
                        "map": {
                            "keywords": ["显示"],
                            "tools": ["ui_typo"],
                        },
                    },
                },
            },
        })


def test_nested_metamodel_definitions_reject_unknown_fields():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Ontology.model_validate({
            "name": "StrictDomain",
            "data_sources": {"memory": {"type": "memory"}},
            "objects": {
                "Result": {
                    "binding": {"source": "memory"},
                    "unknown_policy": True,
                },
            },
        })


@pytest.mark.parametrize("section", ["properties", "object", "relation"])
def test_deprecated_is_not_part_of_the_oag_metamodel(section):
    payload = {
        "name": "StrictDomain",
        "data_sources": {"memory": {"type": "memory"}},
        "objects": {
            "Result": {
                "binding": {"source": "memory"},
                "properties": {"status": {"type": "str"}},
            },
        },
        "relations": {
            "points_to": {
                "binding": {"source": "memory"},
                "from_types": ["Result"],
                "to_types": ["Result"],
            },
        },
    }
    if section == "properties":
        payload["objects"]["Result"]["properties"]["status"]["deprecated"] = True
    elif section == "object":
        payload["objects"]["Result"]["deprecated"] = True
    else:
        payload["relations"]["points_to"]["deprecated"] = True

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Ontology.model_validate(payload)


def test_precondition_rejects_unsupported_operator():
    with pytest.raises(ValidationError):
        Precondition(object="Result", field="status", operator="ne", value="bad")


def test_precondition_field_filter_requires_an_explicit_value_source():
    with pytest.raises(ValidationError, match="value or value_from_param"):
        Precondition(object="Result", field="id", operator="exists")


def test_workflow_can_reference_an_action():
    ontology = Ontology.model_validate({
        "name": "ActionWorkflow",
        "actions": {"approve": {"display_name": "Approve"}},
        "workflows": {
            "approval": {
                "steps": [{"name": "Approval", "action": "approve"}],
            },
        },
    })

    assert ontology.workflows["approval"].steps[0].action == "approve"
