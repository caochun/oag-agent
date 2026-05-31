"""Phase 2: Merge per-document extractions and assemble into ontology.yaml.

Uses layer-by-layer assembly to avoid output token limits.
"""

from __future__ import annotations

import json
import logging

from .llm import DistillerLLM
from .prompts import METHODOLOGY, get_metamodel_spec

log = logging.getLogger(__name__)

_LAYER_SYSTEM = """\
你是 OAG 本体组装专家。你的任务是将从多篇文档提取的原始素材组装为 ontology.yaml 的一个层。

## 建模方法论
{methodology}"""

_OBJECTS_USER = """\
以下是从 {n_docs} 篇领域文档中提取的对象(objects)素材:
{objects_json}

散布在文档中的约束性条款:
{key_constraints}

请组装为 ontology.yaml 的 objects 层。要求:
1. 合并同名对象的属性，去重，保留最详细的描述
2. 每个对象须有 kind(entity/rule_table/lookup_table)、data_source、mutability
3. 有状态流转的对象须有 status_transitions
4. 从约束条款中提取 constraints(when + excluded_functions + reason)
5. 每个对象至少一个 required: true 的属性作为主键
6. 为整个领域取一个 name 和 description

输出完整合法的 YAML 片段，包含 name、description 和 objects 三个顶层键。
不要包含 ```yaml 标记，直接输出。以 "name: " 开头。"""

_LINKS_USER = """\
当前已组装的 objects:
{objects_yaml}

从文档提取的关系(links)素材:
{links_json}

请组装为 ontology.yaml 的 links 层。要求:
1. 去重，确保 source/target 引用的对象在上面的 objects 中存在
2. 每个 link 须有 link_type(contains/enables/causal/prevents) 和 cardinality(1..1/1..n/0..n/0..1)
3. join 的 source_key/target_key 须是对应对象中实际存在的属性名
4. 如果素材中的 link 引用了不存在的对象，丢弃它

只输出 links 部分的 YAML (以 "links:" 开头)。不要包含 ```yaml 标记。"""

_FUNCTIONS_USER = """\
当前已组装的 objects 名称列表:
{object_names}

从文档提取的函数(functions)素材:
{functions_json}

约束条款(用于提取 preconditions/effects/temporal_constraints):
{key_constraints}

请组装为 ontology.yaml 的 functions 层。要求:
1. 每个函数须有 function_type: get(查询外部资产)/lookup(查询规则表)/business(写入业务对象)
2. business 函数须有 writes_to(引用的对象须存在于 objects 中)
3. 涉及状态变更的 business 函数须有 effects
4. 从约束条款中提取 preconditions(object+field+operator+value) 和 temporal_constraints(when+deadline+sla)
5. 复杂业务逻辑写入 hint 字段
6. involves_objects 中的对象须存在

只输出 functions 部分的 YAML (以 "functions:" 开头)。不要包含 ```yaml 标记。"""

_RULES_USER = """\
当前已组装的 objects 名称列表:
{object_names}

从文档提取的规则(rules)素材:
{rules_json}

请组装为 ontology.yaml 的 rules 层。要求:
1. 每个规则须有 rule_type(classification/judgment/qualification/threshold)
2. applies_to 引用的对象须存在
3. 每个 condition 须有 field/operator/value/result
4. 只包含可确定性执行的条件→结果映射

只输出 rules 部分的 YAML (以 "rules:" 开头)。不要包含 ```yaml 标记。"""

_WORKFLOWS_USER = """\
当前已组装的 functions 名称列表:
{function_names}

当前已组装的 objects 名称列表:
{object_names}

从文档提取的工作流(workflows)素材:
{workflows_json}

请组装为 ontology.yaml 的 workflows 层。要求:
1. steps 中引用的 function 须存在于 functions 中(人工步骤的 function 为空字符串)
2. involves_objects 中的对象须存在
3. 有条件分支的步骤用 dict 表示 next
4. 有时限的步骤填写 sla

只输出 workflows 部分的 YAML (以 "workflows:" 开头)。不要包含 ```yaml 标记。"""


def _merge_extractions(extractions: list[dict]) -> dict:
    """Merge raw extractions into per-layer collections."""
    objects: dict[str, dict] = {}
    links: dict[str, dict] = {}
    functions: dict[str, dict] = {}
    rules: dict[str, dict] = {}
    workflows: dict[str, dict] = {}
    key_constraints: list[str] = []

    for ext in extractions:
        for obj in ext.get("objects", []):
            name = obj.get("name", "")
            if not name:
                continue
            if name in objects:
                existing_props = {p["name"] for p in objects[name].get("properties", [])}
                for prop in obj.get("properties", []):
                    if prop.get("name") not in existing_props:
                        objects[name].setdefault("properties", []).append(prop)
                if obj.get("status_transitions") and not objects[name].get("status_transitions"):
                    objects[name]["status_transitions"] = obj["status_transitions"]
                for c in obj.get("constraints", []):
                    objects[name].setdefault("constraints", []).append(c)
            else:
                objects[name] = obj

        for link in ext.get("links", []):
            link_name = link.get("name", "")
            if link_name and link_name not in links:
                links[link_name] = link

        for fn in ext.get("functions", []):
            fn_name = fn.get("name", "")
            if fn_name and fn_name not in functions:
                functions[fn_name] = fn

        for rule in ext.get("rules", []):
            rule_name = rule.get("name", "")
            if rule_name and rule_name not in rules:
                rules[rule_name] = rule

        for wf in ext.get("workflows", []):
            wf_name = wf.get("name", "")
            if wf_name and wf_name not in workflows:
                workflows[wf_name] = wf

        for kc in ext.get("key_constraints", []):
            if kc and kc not in key_constraints:
                key_constraints.append(kc)

    log.info(
        "Merged: %d objects, %d links, %d functions, %d rules, %d workflows, %d constraints",
        len(objects), len(links), len(functions), len(rules), len(workflows), len(key_constraints),
    )

    return {
        "objects": list(objects.values()),
        "links": list(links.values()),
        "functions": list(functions.values()),
        "rules": list(rules.values()),
        "workflows": list(workflows.values()),
        "key_constraints": key_constraints,
    }


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])
        if text.endswith("```"):
            text = text[:-3].strip()
    return text


class Assembler:
    """Merge extractions and assemble ontology.yaml layer-by-layer."""

    def __init__(self, extractions: list[dict], llm: DistillerLLM):
        self.extractions = extractions
        self.llm = llm

    def run(self) -> str:
        log.info("Phase 2: Assembling ontology layer-by-layer from %d extractions", len(self.extractions))

        merged = _merge_extractions(self.extractions)
        system = _LAYER_SYSTEM.format(methodology=METHODOLOGY)
        constraints_str = "\n".join(f"- {kc}" for kc in merged["key_constraints"]) or "无"

        # Layer 1: objects (+ name/description)
        log.info("  Assembling objects layer...")
        objects_yaml = self.llm.call(system, _OBJECTS_USER.format(
            n_docs=len(self.extractions),
            objects_json=json.dumps(merged["objects"], ensure_ascii=False, indent=2),
            key_constraints=constraints_str,
        ))
        objects_yaml = _strip_fences(objects_yaml)

        # Extract object names for downstream layers
        object_names = self._extract_names(objects_yaml, "objects")
        log.info("  Objects assembled: %d types", len(object_names))

        # Layer 2: links
        log.info("  Assembling links layer...")
        links_yaml = self.llm.call(system, _LINKS_USER.format(
            objects_yaml=objects_yaml[:8000],
            links_json=json.dumps(merged["links"], ensure_ascii=False, indent=2),
        ))
        links_yaml = _strip_fences(links_yaml)

        # Layer 3: functions
        log.info("  Assembling functions layer...")
        functions_yaml = self.llm.call(system, _FUNCTIONS_USER.format(
            object_names=", ".join(object_names),
            functions_json=json.dumps(merged["functions"], ensure_ascii=False, indent=2),
            key_constraints=constraints_str,
        ))
        functions_yaml = _strip_fences(functions_yaml)

        function_names = self._extract_names(functions_yaml, "functions")
        log.info("  Functions assembled: %d", len(function_names))

        # Layer 4: rules
        log.info("  Assembling rules layer...")
        rules_yaml = self.llm.call(system, _RULES_USER.format(
            object_names=", ".join(object_names),
            rules_json=json.dumps(merged["rules"], ensure_ascii=False, indent=2),
        ))
        rules_yaml = _strip_fences(rules_yaml)

        # Layer 5: workflows
        log.info("  Assembling workflows layer...")
        workflows_yaml = self.llm.call(system, _WORKFLOWS_USER.format(
            function_names=", ".join(function_names),
            object_names=", ".join(object_names),
            workflows_json=json.dumps(merged["workflows"], ensure_ascii=False, indent=2),
        ))
        workflows_yaml = _strip_fences(workflows_yaml)

        # Combine all layers
        combined = f"{objects_yaml}\n\n{links_yaml}\n\n{functions_yaml}\n\n{rules_yaml}\n\n{workflows_yaml}"
        log.info("Phase 2 complete. Generated %d chars of YAML across 5 layers", len(combined))
        return combined

    @staticmethod
    def _extract_names(yaml_text: str, section: str) -> list[str]:
        """Extract top-level key names from a YAML section (simple heuristic)."""
        import re
        names: list[str] = []
        in_section = False
        for line in yaml_text.split("\n"):
            if line.startswith(f"{section}:"):
                in_section = True
                continue
            if in_section:
                m = re.match(r"^  (\w[\w\d_]*):.*", line)
                if m:
                    names.append(m.group(1))
                elif line and not line.startswith(" ") and not line.startswith("#"):
                    break
        return names
