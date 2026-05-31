# OAG Agent

OAG Agent 是一个本体驱动的在线智能体运行时。它把业务领域定义为
`ontology.yaml`，再将对象查询、规则执行、工作流推进、业务函数和用户确认
统一封装为 LLM 可调用的工具。

这个包本身定位为 Python runtime/library：调用方负责准备领域目录、OpenAI
兼容客户端、模型名以及外层服务接口。

## 功能概览

- 基于 `ontology.yaml` 构建领域对象、关系、规则、工作流和业务函数。
- 自动注册查询、统计、搜索、规则、工作流、写入和业务函数工具。
- 对写操作、业务操作和用户提问提供确认流程。
- 支持流式文本、reasoning 事件、工具调用事件和 SSE 事件转换。
- 支持长上下文压缩、历史协议修复、工具输入 schema 校验。
- 支持工具执行超时、Worker 策略限制、大工具结果落盘。
- Prompt 采用分层装配：静态领域摘要常驻，完整本体详情通过 `inspect` 按需获取。

## 安装与验证

```bash
cd agent
uv sync
uv run pytest
uv run python -m compileall -q oag
```

## 领域目录

`load_domain()` 读取一个领域目录，约定结构如下：

```text
my_domain/
  ontology.yaml
  data/
    assets.json
  functions/
    __init__.py
```

`ontology.yaml` 描述领域模型。一个最小示例：

```yaml
name: AssetOps
description: 资产运维领域

objects:
  Asset:
    summary: 资产基础信息
    mutability: read_only
    properties:
      asset_id:
        type: str
        required: true
        description: 资产编号
      status:
        type: str
        description: 当前状态

  WorkOrder:
    summary: 工单
    mutability: mutable
    properties:
      order_id:
        type: str
        required: true
        description: 工单编号
      asset_id:
        type: str
        description: 资产编号
      status:
        type: str
        description: 工单状态

functions:
  lookup_asset:
    summary: 查询资产详情
    function_type: get
    involves_objects: [Asset]
    params:
      asset_id:
        type: str
        description: 资产编号

  create_work_order:
    summary: 创建维修工单
    description: 为指定资产创建工单
    usage_prompt: |
      只有在用户明确要求创建工单时调用。
      调用前应确认 asset_id 指向真实资产，并说明会写入 WorkOrder。
    function_type: business
    writes_to: [WorkOrder]
    involves_objects: [Asset, WorkOrder]
    params:
      asset_id:
        type: str
        description: 资产编号
```

`functions/__init__.py` 负责绑定 Python 实现：

```python
DATA_FILES = {
    "Asset": "assets.json",
}

FIELD_MAPPINGS = {}


def register(registry, store, ontology):
    def lookup_asset(asset_id: str):
        return store.query_by_id("Asset", asset_id) or {"error": "not found"}

    def create_work_order(asset_id: str):
        return {
            "order_id": "WO-001",
            "asset_id": asset_id,
            "status": "created",
        }

    registry.register("lookup_asset", lookup_asset, ontology.functions["lookup_asset"])
    registry.register(
        "create_work_order",
        create_work_order,
        ontology.functions["create_work_order"],
    )
```

## 最小运行示例

```python
from openai import OpenAI

from oag.agent import Agent
from oag.harness import Harness
from oag.ontology.loader import load_domain
from oag.runtime import HarnessConfig


ontology, store, registry = load_domain("my_domain")

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy",
)

harness = Harness(
    ontology=ontology,
    store=store,
    registry=registry,
    llm_client=client,
    model="your-model",
    config=HarnessConfig(
        enable_write_confirmation=True,
        runtime_context={"deployment": "local"},
        append_system_prompt="请优先使用只读工具获取事实，再执行写入操作。",
    ),
)

agent = Agent(harness, client, model="your-model")

for event in agent.chat_stream("查询资产 A1 的状态", session_id="demo"):
    print(event)
```

如果工具需要用户确认，`chat_stream()` 会返回确认事件；调用方应保存
`session_id`，再调用：

```python
for event in agent.confirm_tool("demo", approved=True):
    print(event)
```

## Prompt 分层

`Harness.build_system_prompt()` 会组装以下层：

- `base_system_prompt`：领域身份和领域说明。
- `ontology_summary`：对象、关系、规则、工作流、函数摘要。
- `tool_usage_rules`：通用工具选择规则。
- `runtime_context`：当前运行模式、审计、轮次限制、部署上下文等动态信息。
- `append_system_prompt`：调用方追加的部署或业务策略。

默认情况下不会把完整函数和对象定义塞进 system prompt。模型需要详情时应调用
`inspect` 工具。若要兼容旧行为，可以设置：

```python
HarnessConfig(include_ontology_full_context=True)
```

## 架构设计

OAG Agent 的核心设计目标是把“领域知识、LLM 对话循环、工具执行策略、会话状态”
拆开。LLM 负责理解用户意图和选择工具；领域事实、业务规则、写入约束、确认流程
放在模型外侧的确定性 runtime 里。

整体分层如下：

```text
Agent
  ├─ QueryLoop
  ├─ ConfirmationFlow
  └─ SessionStore

Harness
  ├─ OntologyRuntime
  │   ├─ OntologyPromptBuilder
  │   ├─ OntologyInspector
  │   ├─ OntologyValidator
  │   ├─ RuleEngine
  │   ├─ WorkflowRuntime
  │   └─ OntologyToolRegistrar
  ├─ DataExecutor / Store
  ├─ ToolRegistry
  ├─ ToolExecutionPipeline
  ├─ RuntimeTools
  ├─ ContextManager
  ├─ HookRegistry / AuditLog
  └─ TraceRecorder
```

### Agent 层

`Agent` 是调用方面向的会话 API。它负责：

- 接收用户消息并加载会话历史。
- 首次会话时生成并写入 system prompt。
- 将对话执行交给 `QueryLoop`。
- 在工具需要确认时保存 pending 状态。
- 用户确认或拒绝后交给 `ConfirmationFlow` 继续。
- 持久化和读取历史记录。
- 将内部事件转换成 SSE 友好的字典格式。

`Agent` 不直接做工具策略判断，也不直接理解 ontology。它更像会话协调器。

### Harness 层

`Harness` 是模型外侧的执行边界。它把 ontology、工具、上下文管理、hook、trace 和
执行管线组合起来，并向 `QueryLoop` 暴露少量稳定接口：

- `build_system_prompt()`
- `build_tools()`
- `execute_tool()`
- `maybe_compact()`
- `force_compact()`
- `run_stop_check()`

这样 `QueryLoop` 不需要知道工具背后是数据库查询、业务函数、规则引擎还是工作流。
所有工具调用都必须经过 `Harness.execute_tool()`，从而保证校验、确认、审计、超时、
缓存和大结果处理不会被绕过。

### OntologyRuntime 层

`OntologyRuntime` 是本体能力的 facade。它不直接承载大量逻辑，而是把职责分给几个
小模块：

- `OntologyPromptBuilder`：构建 prompt 静态层和 ontology 摘要。
- `OntologyInspector`：按需返回函数、对象、规则的完整定义。
- `OntologyValidator`：在工具执行前做领域约束校验。
- `RuleEngine`：执行确定性业务规则。
- `WorkflowRuntime`：启动、推进工作流，并暴露 SLA 定义。
- `OntologyToolRegistrar`：把 ontology 能力注册成工具。

这个拆法的好处是：ontology schema 增长时，不会把所有逻辑塞进一个大 runtime 类里；
后续要扩展规则、工作流或 prompt 策略，也能在对应模块里改。

### DataExecutor 与 Store

`Store` 是 SQLite 对象存储层，根据 ontology 中的对象定义创建表，提供：

- 对象数据导入。
- 条件查询、计数、排序、分页。
- 关系查询。
- 全文搜索。
- 创建、更新、删除对象实例。

`DataExecutor` 是工具层调用 `Store` 的适配器。它把工具名和 JSON 参数转换为具体的
数据操作，并统一把结果序列化为字符串。

领域级校验不放在 `Store`，而是放在 `OntologyValidator`。这样底层存储保持简单，
面向用户的错误消息和业务约束可以在更靠近工具执行的位置处理。

### ToolRegistry 与 ToolDef

`ToolRegistry` 保存所有可供模型调用的工具定义。每个工具用 `ToolDef` 描述：

- `name`：工具名。
- `description`：能力摘要。
- `parameters`：OpenAI function calling JSON schema。
- `handler`：实际执行函数。
- `usage_prompt`：复杂工具的使用约束。
- `policy`：执行策略。

`ToolRegistry.build_tools()` 会把内部 `ToolDef` 转成 OpenAI 工具 schema，并缓存稳定
结果。注册新工具时版本号递增，缓存自动失效。

`usage_prompt` 的设计意图是让复杂工具自描述，而不是把所有规则都写进全局 prompt。
例如 `ask_user` 可以说明什么时候才应该问用户，业务函数可以说明调用前置条件和副作用。

### ToolExecutionPipeline

`ToolExecutionPipeline` 是工具执行的唯一通道。一次工具调用会经过这些步骤：

1. 查找工具定义。
2. 记录 trace。
3. 校验 JSON 参数 schema。
4. 执行 ontology 约束校验。
5. 检查 worker/main、只读、写入、确认等策略。
6. 命中只读缓存时直接返回。
7. 触发 pre-hook。
8. 带超时执行 handler。
9. 截断或持久化过大的工具结果。
10. 写入只读缓存。
11. 触发 post-hook 和审计。
12. 返回统一的 `ToolResult`。

这个管线让工具 handler 可以保持简单。handler 只关心“怎么做事”，不需要自己处理
确认、缓存、审计、超时和大结果落盘。

### QueryLoop

`QueryLoop` 是主 LLM 回合循环。它负责：

- 在每轮请求前修复历史协议问题。
- 必要时压缩上下文。
- 向模型发送 messages 和 tools。
- 消费流式响应，产出文本、reasoning 和 debug 事件。
- 聚合 streaming tool calls。
- 将 assistant tool-call envelope 和 tool result 按 OpenAI 协议写回历史。
- 处理非法 JSON 参数，把错误作为工具结果反馈给模型。
- 在确认工具出现时暂停本轮，并保存现场。
- 最终回答后运行 stop check。

`QueryLoop` 不直接执行工具逻辑，而是调用 `Harness.execute_tool()`。这保证主 agent、
确认恢复流程和 worker 都共享同一套工具执行语义。

### ConfirmationFlow

写操作、业务操作和 `ask_user` 可能需要暂停等待用户确认。`ConfirmationFlow` 负责：

- 处理用户批准或拒绝。
- 为被拒绝工具写入 tool result，避免破坏 OpenAI tool-call 协议。
- 恢复 pending 时的 `RunState`。
- 继续交给 `QueryLoop` 执行后续回合。

当模型一次返回多个工具调用，而前一个工具需要确认时，后续未执行的工具调用会被写入
“skipped”结果。这样历史始终满足“每个 tool_call_id 都有对应 tool result”的协议。

### SessionStore 与 Message Sanitizer

`SessionStore` 使用 SQLite 保存会话消息。读取历史时会调用 message sanitizer 修复：

- 孤立的 tool result。
- 缺失的 tool result。
- 重复 tool result。
- 空 assistant 消息。

保存和运行中修复会更保守，避免在 pending confirmation 状态下提前补齐工具结果，
破坏等待确认的现场。

### ContextManager

`ContextManager` 负责上下文长度控制：

- 估算消息 token。
- 对旧工具结果做轻量压缩。
- 在接近上下文窗口时调用 LLM 摘要旧历史。
- 保留 system prompt 和最近消息。
- 调整压缩边界，避免 assistant tool call 和 tool result 被拆开。
- 遇到 context overflow 时支持强制压缩并重试。

压缩后的历史会插入 `[前置对话摘要]`，同时保留最近交互，让模型能继续任务而不丢关键状态。

### Prompt 设计

Prompt 不再采用“大本体全量前置注入”作为默认模式。默认 system prompt 只常驻：

- 领域身份。
- ontology 摘要。
- 通用工具规则。
- 动态运行时上下文。
- 调用方追加策略。

完整函数、对象、规则详情通过 `inspect` 工具按需获取。这样可以减少 prompt 体积，
也能避免 ontology 变大后每轮请求都携带大量不相关细节。

静态 prompt sections 会缓存；动态 runtime context 每次构建。这样既保持稳定前缀，
也允许部署信息、运行模式等动态状态及时进入 prompt。

### Worker / Subagent

`dispatch_workers` 用于并行处理独立子任务。Worker 的设计边界更窄：

- 不继承主会话完整历史。
- 只接收主 agent 显式传入的 `context`。
- 使用精简 system prompt：worker 身份、领域摘要、背景信息和执行要求。
- 根据任务类型过滤工具列表。
- 通过 `ToolUseContext(source="worker")` 进入同一条工具执行管线。
- 默认不允许执行需要用户确认或写入的工具。

这能避免 worker 误用主会话隐含上下文，也降低并行任务的 prompt 成本。

### Hooks、Audit 和 Trace

运行时提供三个侧面的可观测与控制机制：

- `HookRegistry`：在工具前后、查询完成等时机插入策略。
- `AuditLog`：记录工具调用和结果摘要。
- `TraceRecorder`：记录 agent turn、工具开始/结束、阻止原因、缓存命中等事件。

默认 hook 包括写入确认、审计记录、业务复核和最终回答完整性检查。

### 一次用户请求的流转

```text
user message
  -> Agent.chat_stream()
  -> SessionStore.get()
  -> 首次会话构建 system prompt
  -> QueryLoop.run()
  -> sanitize messages
  -> maybe compact
  -> LLM request(messages + tools)
  -> final answer
       -> stop check
       -> SessionStore.save()
  -> tool calls
       -> append assistant tool-call envelope
       -> Harness.execute_tool()
       -> ToolExecutionPipeline
       -> append tool result
       -> next LLM turn
  -> confirmation required
       -> save pending state
       -> wait for Agent.confirm_tool()
       -> ConfirmationFlow
       -> QueryLoop continues
```

这个流转的关键不变量是：模型看到的历史始终满足工具协议；所有工具执行都经过同一条
pipeline；业务规则和副作用控制都在模型外侧。

## 工具与策略

工具由 `ToolDef` 描述：

- `description`：工具能力摘要。
- `parameters`：OpenAI function calling JSON schema。
- `usage_prompt`：复杂工具的使用约束，会拼入工具描述。
- `policy`：只读、是否需要确认、是否允许 Worker、是否破坏性、超时等策略。

领域函数可以在 `ontology.yaml` 中声明 `usage_prompt`。这比把所有细节塞进全局
prompt 更清晰，也更容易按函数维护。

内置工具包括：

- `inspect`
- `query`
- `count`
- `query_links`
- `describe`
- `pivot`
- `distribution`
- `mutate`
- `search`
- `apply_rule`
- `apply_rule_batch`
- `start_workflow`
- `check_sla`
- `summarize_progress`
- `ask_user`
- `dispatch_workers`

实际可用工具取决于 ontology 中是否声明了关系、规则、工作流和业务函数。

## 运行时行为

- `SessionStore` 持久化会话历史，并在读取时修复孤立工具结果、缺失工具结果等协议问题。
- `ContextManager` 会在上下文变长时压缩旧历史，同时保护 system prompt 和最近工具调用对。
- `ToolExecutionPipeline` 统一处理工具校验、确认、策略限制、缓存、审计、超时和大结果落盘。
- Worker 只接收精简 prompt、显式传入的 `context` 和经过过滤的工具列表，不继承主会话完整历史。

## 测试

```bash
cd agent
uv run pytest
uv run python -m compileall -q oag
```

测试覆盖 harness runtime、工具策略、确认流程、上下文压缩、历史修复、工具超时、
大结果落盘和 prompt 分层等行为。
