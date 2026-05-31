"""All LLM prompt templates for the distiller pipeline."""

from __future__ import annotations

from pathlib import Path

_METAMODEL_SPEC: str | None = None


def get_metamodel_spec() -> str:
    global _METAMODEL_SPEC
    if _METAMODEL_SPEC is None:
        spec_path = Path(__file__).resolve().parents[3] / "docs" / "metamodel-spec.md"
        _METAMODEL_SPEC = spec_path.read_text(encoding="utf-8")
    return _METAMODEL_SPEC


METHODOLOGY = """\
建模方法论（严格遵循）:

1. 实体分类三层:
   - 外部资产(external_api, read_only): 系统外管理的已有实体。查询用 get_xxx 函数
   - 规则配置(human_confirmed, read_only): 从法规/标准提取的稳定配置表。查询用 lookup_xxx 函数
   - 业务过程(agent_generated, append_only/mutable): 智能体在业务交互中产出的记录。由 business 函数写入，可能有状态流转

2. 判断实体 vs 属性:
   - 有独立查询需求，或被多个函数引用 → 独立实体
   - 只被一个函数产出/消费，无独立查询场景 → 作为属性

3. 约束分配规则:
   - "X 状态不能做 Y" → objects.constraints (when + excluded_functions)
   - "做 A 前必须先有 B" → functions.preconditions (object + field + operator + value)
   - "做 A 后 B 变为 C" → functions.effects (object + field + set_to)
   - "X 状态只能转到 Y/Z" → objects.status_transitions
   - "必须在 N 时间内完成" → functions.temporal_constraints (when + deadline + sla)
   - "如果 X=a 则结果=b" 的确定性逻辑 → rules.conditions (不让 LLM 推理)

4. 确定性系统做计算:
   - 能写成 field+operator+value → result 的判断 → 放 rules 层
   - 需要语义理解或模糊判断 → 放 function 的 hint

5. 函数分类:
   - function_type=get: 查询外部资产，无副作用
   - function_type=lookup: 查询规则配置表，无副作用
   - function_type=business: 写入业务过程对象，须有 writes_to，可能有 preconditions/effects

6. 命名规范:
   - 对象名: PascalCase (如 FlightMission)
   - 属性名/函数名: snake_case (如 mission_id, plan_flight_mission)
   - 关系名: snake_case 描述 (如 mission_has_approval)
"""


DOCUMENT_SUMMARY_SYSTEM = """\
你是文档分析专家。为每篇文档生成一行摘要，说明它的核心主题和内容类型（法规/技术标准/操作规程/管理预案等）。"""

DOCUMENT_SUMMARY_USER = """\
以下是一批领域文档的标题和开头内容。请为每篇文档输出一行摘要和文档类型分类。

输出 JSON 格式:
{{
  "documents": [
    {{"filename": "文件名", "summary": "一行摘要", "doc_type": "法规|技术标准|操作规程|管理预案|其他", "priority": 1-5}}
  ]
}}

priority 说明: 1=最高优先(定义基础概念的法规), 5=最低(背景资料)。法规/条例优先于技术标准，技术标准优先于操作规程。

文档列表:
{doc_list}"""


EXTRACTION_SYSTEM = """\
你是 OAG (Ontology-Agent-Graph) 本体建模专家。你的任务是从领域文档中提取建模素材。

## OAG 元模型规范

{metamodel_spec}

## 建模方法论

{methodology}"""


EXTRACTION_USER = """\
{accumulated_context}

现在阅读以下文档，提取所有可建模的素材:

## 文档: {doc_name}

{doc_content}

---

请严格按以下 JSON 结构输出提取结果。每个元素都必须标注来源(source 字段,格式为"文档名§章节号")。

{{
  "objects": [
    {{
      "name": "PascalCase对象名",
      "kind": "entity|rule_table|lookup_table",
      "data_source": "external_api|agent_generated|human_confirmed",
      "mutability": "read_only|append_only|mutable",
      "summary": "一行摘要(出现在 system prompt 对象列表中)",
      "description": "详细描述",
      "source": "文档名§章节号",
      "properties": [
        {{"name": "snake_case属性名", "type": "str|int|float|bool", "required": false, "description": "属性描述(含单位/取值范围/枚举值)", "source": "§x.x"}}
      ],
      "status_transitions": {{"状态A": ["状态B", "状态C"]}},
      "constraints": [
        {{"when": {{"status": "某状态"}}, "excluded_functions": ["不可调用的函数"], "reason": "原因(引用法规条款)"}}
      ]
    }}
  ],
  "links": [
    {{
      "name": "snake_case关系名",
      "source": "源对象名",
      "target": "目标对象名",
      "join": {{"source_key": "源字段", "target_key": "目标字段"}},
      "link_type": "contains|enables|causal|prevents",
      "cardinality": "1..1|1..n|0..n|0..1",
      "description": "关系描述"
    }}
  ],
  "functions": [
    {{
      "name": "snake_case函数名",
      "summary": "一行摘要",
      "description": "详细描述",
      "function_type": "get|lookup|business",
      "group": "功能分组名",
      "params": [{{"name": "参数名", "type": "str|int|float", "description": "描述", "default": null}}],
      "writes_to": ["写入的对象类型(business函数必填)"],
      "involves_objects": ["涉及的对象类型"],
      "preconditions": [
        {{"object": "对象类型", "field": "字段名", "operator": "eq|ne|in|exists", "value": "期望值"}}
      ],
      "effects": [
        {{"object": "对象类型", "field": "字段名", "set_to": "变更后的值"}}
      ],
      "temporal_constraints": [
        {{"when": {{}}, "deadline": "时间表达式", "sla": "人类可读SLA描述"}}
      ],
      "hint": "详细执行规则(仅business函数,首次调用后注入LLM上下文)",
      "source": "§x.x"
    }}
  ],
  "rules": [
    {{
      "name": "snake_case规则名",
      "description": "规则描述",
      "rule_type": "classification|judgment|qualification|threshold",
      "applies_to": ["适用的对象类型"],
      "result_field": "结果写入的字段名",
      "source": "§x.x",
      "conditions": [
        {{"field": "字段名", "operator": "eq|ne|gt|gte|lt|lte|in|between", "value": "比较值", "result": "匹配结果"}}
      ]
    }}
  ],
  "workflows": [
    {{
      "name": "snake_case工作流名",
      "description": "工作流描述",
      "trigger": "触发条件",
      "involves_objects": ["涉及的对象类型"],
      "steps": [
        {{"name": "步骤名", "function": "函数名(人工步骤为空字符串)", "description": "步骤描述", "next": "下一步骤名或条件分支dict", "sla": "时限(如有)"}}
      ]
    }}
  ],
  "key_constraints": [
    "散布在文档中的约束性条款(时限/前置条件/禁止行为/SLA等),以原文引述+条款号的形式记录"
  ]
}}

要求:
1. 只提取文档中明确写的内容,不推断或补全
2. 每个属性的 required 字段: 第一个作为业务主键的属性设为 true
3. status_transitions 只在对象有明确状态流转时填写
4. constraints 只在文档明确说"X状态下不可做Y"时填写
5. rules 只提取可以确定性执行的条件→结果映射
6. 如果发现已累积的对象需要补充新属性,在 objects 中包含该对象(只列新增属性)"""


ASSEMBLY_SYSTEM = """\
你是 OAG 本体组装专家。你的任务是将从多篇文档提取的原始素材合并为一份完整的 ontology.yaml。

## OAG 元模型规范

{metamodel_spec}

## 建模方法论

{methodology}"""


ASSEMBLY_USER = """\
以下是从 {n_docs} 篇领域文档中提取并初步合并的建模素材:

{merged_materials}

---

散布在各文档中的约束性条款:
{key_constraints}

---

请将以上素材组装为完整的 ontology.yaml。要求:

1. name 和 description: 为整个领域取一个简洁名称和描述
2. objects: 合并同名对象的属性,去重,保留最详细的描述。每个对象须有 kind/data_source/mutability
3. links: 去重,确保 source/target 引用的对象存在。须有 link_type 和 cardinality
4. functions:
   - get 函数: 每个 external_api 对象至少一个
   - lookup 函数: 每个 rule_table/lookup_table 对象至少一个
   - business 函数: 须有 writes_to; 有状态变更的须有 effects
   - 从 key_constraints 中提取 preconditions 和 temporal_constraints
5. rules: 须有 rule_type 和 applies_to。每个条件须有 field/operator/value/result
6. workflows: steps 中引用的 function 须存在于 functions 中

输出合法的 YAML。不要包含 ```yaml 标记,直接输出 YAML 内容。
以 "name: " 开头。"""


REVIEW_SYSTEM = """\
你是 OAG 本体质量审查员。你的任务是审查自动生成的 ontology.yaml,找出缺陷并提出修复建议。"""

REVIEW_USER = """\
以下是自动生成的 ontology.yaml:

{ontology_yaml}

---

源文档摘要:
{document_summaries}

---

以下是 schema 验证发现的问题:
{validation_errors}

---

请从以下维度审查并输出 JSON:

1. **覆盖完整性**: 源文档中的核心概念是否都已建模为 objects?
2. **约束完整性**: 文档中的时限要求、前置条件、禁止行为是否都已捕获到 preconditions/temporal_constraints/constraints?
3. **函数-对象一致性**: business 函数的 writes_to 引用的对象是否存在? hint 中引用的字段是否存在于对应对象?
4. **状态机完整性**: 有状态流转的对象是否都定义了 status_transitions?
5. **规则完整性**: 确定性判断逻辑(如分类标准、等级划分)是否都放入了 rules 而非 function hint?
6. **主键完整性**: 每个对象是否至少有一个 required: true 的属性?

输出格式:
{{
  "issues": [
    {{"layer": "objects|links|functions|rules|workflows", "target": "名称", "issue": "问题描述", "fix": "修复建议"}}
  ]
}}"""


FIX_SYSTEM = """\
你是 OAG 本体修复专家。根据审查发现的问题修复 ontology.yaml。"""

FIX_USER = """\
当前 ontology.yaml:

{ontology_yaml}

---

审查发现的问题:
{issues_json}

---

请修复以上问题,输出完整的修复后 ontology.yaml。不要包含 ```yaml 标记,直接输出 YAML 内容。
以 "name: " 开头。

只修复指出的问题,不要做其他无关的修改。"""
