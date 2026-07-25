from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from oag.ontology.inspector import OntologyInspector
from oag.ontology.prompt_builder import OntologyPromptBuilder
from oag.ontology.registry import FunctionRegistry
from oag.ontology.schema import Ontology


def make_policy_ontology() -> Ontology:
    return Ontology.model_validate({
        "name": "PolicyDomain",
        "objects": {
            "ResultCell": {
                "summary": "结果单元",
            },
        },
        "functions": {
            "analyze": {
                "summary": "分析结果",
            },
        },
        "presentation_tools": {
            "ui_show_objects": {
                "summary": "展示结果对象",
                "description": "在前端展示领域对象。",
                "side_effect_scope": "frontend_map",
                "object_scope": "listed",
                "allowed_objects": ["ResultCell"],
            },
        },
        "interaction_policies": {
            "user_chat": {
                "description": "用户交互",
                "include_in_system_prompt": True,
                "instructions": ["只使用已验证的领域事实。"],
            },
        },
        "event_policies": {
            "ResultGenerated": {
                "role": "领域事件智能体",
                "allowed_tools": ["analyze", "ui_show_objects"],
                "required_functions": ["analyze"],
                "automatic_map": {
                    "mode": "when_relevant",
                    "tool": "ui_show_objects",
                    "objects": [{
                        "object_type": "ResultCell",
                        "filters": {"result_id": "latest"},
                        "refresh": True,
                    }],
                    "allowed_action_types": ["apply_result"],
                    "other_objects": "on_user_request",
                },
            },
        },
    })


def test_interaction_policy_is_injected_into_system_prompt():
    builder = OntologyPromptBuilder(make_policy_ontology(), FunctionRegistry())

    prompt = builder.build_system_prompt()

    assert "## 领域交互策略" in prompt
    assert "只使用已验证的领域事实" in prompt


def test_event_prompt_is_rendered_from_typed_policy():
    builder = OntologyPromptBuilder(make_policy_ontology(), FunctionRegistry())

    prompt = builder.build_event_prompt(
        "ResultGenerated", {"event_type": "ResultGenerated", "event_id": "evt_1"},
    )

    assert "必须调用以下函数：analyze" in prompt
    assert '"object_type": "ResultCell"' in prompt
    assert "只有用户在普通对话中明确请求时才可展示" in prompt


def test_event_policy_rejects_required_function_outside_allowlist():
    with pytest.raises(ValidationError, match="required_functions must be included"):
        Ontology.model_validate({
            "name": "InvalidPolicyDomain",
            "event_policies": {
                "ResultGenerated": {
                    "allowed_tools": [],
                    "required_functions": ["analyze"],
                },
            },
        })


def test_inspect_exposes_event_policy():
    ontology = make_policy_ontology()
    inspector = OntologyInspector(ontology, FunctionRegistry())

    result = json.loads(inspector.inspect("ResultGenerated"))

    assert result["kind"] == "event_policy"
    assert result["required_functions"] == ["analyze"]


def test_inspect_exposes_presentation_tool():
    ontology = make_policy_ontology()
    inspector = OntologyInspector(ontology, FunctionRegistry())

    result = json.loads(inspector.inspect("ui_show_objects"))

    assert result["kind"] == "presentation_tool"
    assert result["side_effect_scope"] == "frontend_map"
    assert result["mutates_domain"] is False


def test_event_policy_rejects_unknown_presentation_tool():
    with pytest.raises(ValidationError, match="unknown presentation tool"):
        Ontology.model_validate({
            "name": "InvalidPresentationDomain",
            "objects": {"ResultCell": {"summary": "结果单元"}},
            "functions": {"analyze": {"summary": "分析结果"}},
            "event_policies": {
                "ResultGenerated": {
                    "allowed_tools": ["analyze", "ui_missing"],
                    "required_functions": ["analyze"],
                    "automatic_map": {
                        "mode": "always",
                        "tool": "ui_missing",
                        "objects": [{"object_type": "ResultCell"}],
                        "allowed_action_types": ["apply_result"],
                    },
                },
            },
        })


def test_interaction_policy_rejects_unknown_tool():
    with pytest.raises(ValidationError, match="interaction policy user_chat.map references unknown tools"):
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
