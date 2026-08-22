# OAG Agent

OAG Agent 是领域无关的本体增强智能体运行时。它接收 Provider 已经编译好的
`Ontology`，把对象/关系访问、无副作用 Function、有副作用 Action、工具策略和会话循环
装配成 LLM 可理解且可执行的 Agent。

OAG 不读取领域 YAML，不发现领域目录，不实现具体数据库 adapter，也不通过业务关键词猜测
意图。UOM 等上层负责领域建模、编译、物理存储和具体业务实现。

## 职责边界

OAG 负责：

- Agent、流式事件、会话历史和用户确认流程。
- Object、Relation、Function、Action、Rule、Workflow 的运行时元模型。
- `DomainProvider`、`ObjectQuerySource`、`ObjectSearchSource`、`RelationSource` 和 `ActionRuntime` 协议。
- `SourceManager` 与按 binding 访问逻辑数据的 `OntologyRepository`。
- Prompt、工具注册、输入校验、执行策略、超时、缓存、审计和 trace。
- LLM 重试、上下文估算、压缩和工具消息协议修复。

OAG 不负责：

- `model.yaml` 或其他领域 DSL 的解析、合并与编译。
- 领域关键词路由、固定业务意图分类或业务结果启发式判断。
- SQLite/HTTP/第三方系统等具体 Source 实现。
- ChangeSet、revision、retire、history 等领域数据治理语义。
- 具体 Web UI、地图、表单实现或领域目录发现规则。
- 绕过 Action 的通用业务 CRUD 工具。

## Provider 协议

Provider 返回最终 `Ontology`，再绑定该本体声明的运行时实现：

```python
from oag.ontology.domain import DomainContext


class Provider:
    def load_ontology(self):
        # 可以由领域 DSL 编译、代码构造或外部服务获取。
        return ontology

    def register(self, context: DomainContext):
        context.sources.register("asset_api", source_factory)
        context.bindings.register(
            "lookup_asset",
            lookup_asset,
            context.ontology.functions["lookup_asset"],
        )
        context.bindings.register_action_runtime(action_runtime)
```

装配顺序是：

```text
provider.load_ontology()
  -> SourceManager(ontology)
  -> OntologyRepository(ontology, sources)
  -> provider.register(DomainContext)
  -> 校验全部 Function 和 Action 已绑定实现
```

`DomainContext` 只包含 `ontology`、`bindings`、`sources` 和 `repository`。Provider 自己持有
目录、连接或部署配置，OAG 不假设领域来自文件系统。

## 本体元模型

- `ObjectTypeDef`：对象类型、属性和数据 binding。
- `RelationTypeDef`：一等关系、端点类型、属性和数据 binding。
- `FunctionDef`：无领域副作用的查询、计算或校验能力。
- `ActionDef`：会改变业务状态的操作及其公开输入、副作用摘要。
- `RuleDef`：确定性规则声明。
- `WorkflowDef`：供模型理解的业务步骤和分支，不在 OAG 内保存业务进度。
- `InteractionPolicyDef`：领域提供的稳定自然语言指令，不包含关键词意图路由。

OAG 元模型中的 Object/Relation 必须声明 `binding.source`。Function 声明后必须由 Provider
注册 callable；存在 Action 时必须注册 `ActionRuntime`。Action 的真实变更模板和事务语义
属于领域实现，OAG 只读取公开契约。

## 数据访问

`OntologyRepository` 根据每个类型的 binding 调用命名 Source：

```text
OntologyRepository
  -> SourceManager.require(binding.source, capability)
  -> ObjectQuerySource / ObjectSearchSource / RelationSource
  -> 外部系统或领域存储
```

逻辑协议包括对象/关系的 query、get，以及可选的 search 和写能力。OAG 不提供具体 adapter；统计和聚合
应由领域 Function 或外部系统提供，不作为所有 Source 都必须实现的基础协议。
Provider 可以从同一 Source 获取额外的领域能力并包装为自己的服务，但这些协议不能倒灌进 OAG。

## Function 与 Action

Function 始终按只读工具注册，适合查询、聚合、追溯和确定性校验。业务副作用只能经 Action：

```text
LLM -> request_action_input -> InteractionEvent
UI  -> ActionRuntime.preview_action
UI  -> 用户确认
UI  -> ActionRuntime.execute_action
```

`get_available_actions` 和 `request_action_input` 是 OAG 根据 Action 目录生成的通用桥接工具。
交互 payload 描述用户需要补充的输入，不绑定 Web 表单；具体渲染和 Action 接口由应用负责。

## 工具执行

所有主 Agent 和 Worker 工具调用都经过 `ToolExecutionPipeline`：

1. 查找工具并校验 JSON 参数。
2. 执行 Worker 权限和确认策略。
3. 触发 pre-hook。
4. 命中当前 Agent run 内的只读缓存。
5. 执行本体前置条件。
6. 带超时调用 handler，并把异常转成 blocked `ToolResult`。
7. 截断过大的工具结果；启用结果读取能力时通过不透明 `result_ref` 持久化。
8. 触发 post-hook、审计和 trace。

`ToolRegistry` 拒绝同名覆盖，避免领域 Function 静默替换内置工具。Worker 不按任务文本分类，
只看到 `ToolPolicy.worker_allowed=True` 的工具；需要用户确认或写入的工具默认不能由 Worker 执行。

核心运行时工具只有 `ask_user`；核心本体工具是 `inspect`、`query` 和按模型需要注册的
`query_relations`。`search` 只在 Source 声明搜索能力时注册，规则工具和 Action 工具只在本体
具备对应能力时注册。`read_tool_result` 与 `dispatch_workers` 由 Harness 配置显式开启。

## Agent 循环

```text
Agent.chat_stream
  -> SessionStore
  -> QueryLoop
       -> sanitize messages
       -> compact context when needed
       -> LLM(messages + allowed tools)
       -> ToolExecutionPipeline
       -> confirmation pause/resume
  -> stream Event / SSE dict
```

`allowed_tools` 是调用方显式传入的运行时工具范围，不是 OAG 的意图识别结果。`ask_user`
通过独立的用户输入状态暂停并恢复会话，不借用写操作确认。Action 输入请求产生通用
`InteractionEvent`，OAG 不解释应用如何渲染其 payload。

## Hooks 与状态

默认 hook 只包含通用横切策略：写工具确认和工具调用审计。领域复核和最终回答检查可以
由调用方注册 hook，但 OAG 不内置中文短语、业务字段或成功/失败关键词启发式。

`SessionStore` 使用 SQLite 保存 OAG 自己的对话状态；这不是领域数据 adapter。

## 最小运行示例

```python
from openai import OpenAI

from oag.agent import Agent
from oag.harness import Harness
from oag.ontology.loader import load_domain
from oag.runtime import HarnessConfig


provider = Provider(...)
ontology, repository, bindings = load_domain(provider)
client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

harness = Harness(
    ontology,
    repository,
    bindings,
    client,
    "your-model",
    HarnessConfig(enable_write_confirmation=True),
)
agent = Agent(harness, client, "your-model")

for event in agent.chat_stream("查询资产 A1", session_id="demo"):
    print(event)
```

## 源码职责清单

本轮按文件逐一检查后的归属如下：

- `agent.py`：会话 API；`harness.py`：运行时门面。
- `llm/*`：重试、上下文压缩和用量估算。
- `loop/query_loop.py`：主回合状态循环；`response_parser.py`：完整/流式模型响应解析；
  `tool_call_coordinator.py`：工具批次、结果消息、确认和用户输入暂停；`tool_executor.py`：同回合工具并发；
  `confirmation_flow.py`：暂停后的恢复；`worker.py`：受策略限制的独立子任务。
- `ontology/schema.py`：元模型；`domain.py`/`loader.py`：Provider 生命周期；
  `source.py`/`repository.py`：逻辑数据协议；`bindings.py`：Function/ActionRuntime 实现绑定。
- `ontology/data_executor.py`：查询和 Function 的统一序列化；`tool_registrars.py`：按查询、规则、
  Function、Action/交互分别注册工具；`runtime.py`：本体能力门面。
- `ontology/prompt_builder.py`、`inspector.py`、`validators.py`、`rules.py`：各自的确定性能力。
- `runtime/*`：配置、状态、事件、hook、会话、trace 和大结果存储。
- `tools/*`：工具定义、注册、执行管线和 Agent 控制工具。

已删除职责不成立的文件：具体数据库 adapters、伪工作流状态运行时和关键词式 stop check。

## 验证

```bash
uv sync
uv run pytest
uv run python -m compileall -q oag
```
