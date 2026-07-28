from __future__ import annotations

import asyncio
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parents[5]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

os.environ.setdefault("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY")

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

try:
    from opentelemetry.instrumentation.genai.langchain import LangChainInstrumentor
except Exception:
    try:
        from opentelemetry.instrumentation.langchain import LangChainInstrumentor
    except Exception:
        from openinference.instrumentation.langchain import LangChainInstrumentor

from langchain_core.messages import AIMessage, HumanMessage

from open_deep_research.deep_researcher import deep_researcher as _agent_graph

_exporter = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(_exporter))
trace.set_tracer_provider(_provider)
_instrumentor = LangChainInstrumentor()
_instrumented = False
try:
    _instrumentor.instrument(tracer_provider=_provider)
    _instrumented = True
except Exception:
    _instrumented = False

_BASE_ENV_KEYS = [
    "OPENAI_API_BASE",
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "MODEL_NAME",
    "GET_API_KEYS_FROM_CONFIG",
    "TAVILY_API_KEY",
]
_ENV_SNAPSHOT = {key: os.environ.get(key) for key in _BASE_ENV_KEYS}
_RUNTIME_ENV_SNAPSHOT: dict[str, str | None] = {}
_CASE_RUNTIME: dict[str, Any] = {
    "input_state": None,
    "configurable": {},
    "metadata": {},
}
_BASE_CASE_RUNTIME = deepcopy(_CASE_RUNTIME)


def _extract_vars(options: dict | None, context: dict | None) -> dict:
    options = options or {}
    context = context or {}
    candidates = [
        context.get("vars"),
        options.get("vars"),
        context.get("test", {}).get("vars") if isinstance(context.get("test"), dict) else None,
        options.get("test", {}).get("vars") if isinstance(options.get("test"), dict) else None,
    ]
    merged: dict[str, Any] = {}
    for candidate in candidates:
        if isinstance(candidate, dict):
            merged.update(candidate)
    return merged


def _read_test_case_id(options: dict | None, context: dict | None) -> str:
    vars_ = _extract_vars(options, context)
    test_case_id = vars_.get("test_case_id")
    if test_case_id is None:
        return ""
    return str(test_case_id)


def _read_precondition(options: dict | None, context: dict | None) -> Any:
    vars_ = _extract_vars(options, context)
    if "precondition" in vars_:
        return vars_.get("precondition")
    if "preconditions" in vars_:
        return vars_.get("preconditions")
    return None


def _snapshot_runtime_env() -> None:
    global _RUNTIME_ENV_SNAPSHOT
    _RUNTIME_ENV_SNAPSHOT = {key: os.environ.get(key) for key in _BASE_ENV_KEYS}


def _restore_runtime_env() -> None:
    for key, value in _RUNTIME_ENV_SNAPSHOT.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _set_env(key: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value


def _resolve_model(config: dict) -> str:
    return str(config.get("model") or "")


def _configure_model_env(config: dict) -> None:
    base_url = os.environ["LITELLM_BASE_URL"]
    api_key = os.environ["LITELLM_API_KEY"]
    _set_env("OPENAI_API_BASE", base_url)
    _set_env("OPENAI_BASE_URL", base_url)
    _set_env("OPENAI_API_KEY", api_key)
    _set_env("ANTHROPIC_API_KEY", api_key)
    _set_env("GOOGLE_API_KEY", api_key)
    _set_env("GET_API_KEYS_FROM_CONFIG", "false")
    model_name = _resolve_model(config)
    if model_name:
        _set_env("MODEL_NAME", model_name)


def _base_configurable(config: dict) -> dict:
    model_name = _resolve_model(config)
    configurable = {
        "research_model": model_name,
        "final_report_model": model_name,
        "compression_model": model_name,
        "summarization_model": model_name,
        "research_model_max_tokens": 800,
        "final_report_model_max_tokens": 1200,
        "compression_model_max_tokens": 800,
        "summarization_model_max_tokens": 600,
        "max_structured_output_retries": 1,
        "max_researcher_iterations": 2,
        "max_concurrent_research_units": 2,
        "max_react_tool_calls": 2,
        "max_content_length": 4000,
        "search_api": "none",
        "allow_clarification": True,
    }
    return configurable


def _seed_happy_path_full_workflow() -> None:
    _CASE_RUNTIME["input_state"] = {
        "messages": [
            HumanMessage(
                content=(
                    "I need a comprehensive research report on the current state of quantum "
                    "computing applications in drug discovery, focusing on pharmaceutical "
                    "industry adoption and future prospects."
                )
            )
        ]
    }
    _CASE_RUNTIME["configurable"] = {
        **_base_configurable({"model": os.environ.get("MODEL_NAME", "")}),
        "allow_clarification": True,
        "search_api": "none",
    }


def _seed_happy_path_skip_clarification() -> None:
    _CASE_RUNTIME["input_state"] = {
        "messages": [
            HumanMessage(
                content="Research the competitive landscape of electric vehicle battery manufacturers in 2024."
            )
        ]
    }
    _CASE_RUNTIME["configurable"] = {
        **_base_configurable({"model": os.environ.get("MODEL_NAME", "")}),
        "allow_clarification": False,
        "search_api": "none",
    }


def _seed_missing_info_vague_request() -> None:
    _CASE_RUNTIME["input_state"] = {
        "messages": [HumanMessage(content="Research AI.")]
    }
    _CASE_RUNTIME["configurable"] = {
        **_base_configurable({"model": os.environ.get("MODEL_NAME", "")}),
        "allow_clarification": True,
        "search_api": "none",
    }


def _seed_happy_path_specific_research_brief() -> None:
    _CASE_RUNTIME["input_state"] = {
        "messages": [
            HumanMessage(
                content=(
                    "I need to understand the regulatory framework for cryptocurrency "
                    "exchanges in the European Union, specifically focusing on MiCA "
                    "regulation compliance requirements for 2025."
                )
            )
        ]
    }
    _CASE_RUNTIME["configurable"] = {
        **_base_configurable({"model": os.environ.get("MODEL_NAME", "")}),
        "allow_clarification": True,
        "search_api": "none",
    }


def _seed_happy_path_final_report_generation() -> None:
    _CASE_RUNTIME["input_state"] = {
        "messages": [
            HumanMessage(content="Generate the final report for my research on sustainable packaging innovations in the food industry.")
        ],
        "research_brief": (
            "Research sustainable packaging innovations in the food industry, emphasizing "
            "material advances, commercial adoption, and operational trade-offs."
        ),
        "notes": [
            "Biodegradable films and fiber-based packaging are increasingly adopted for food products.",
            "Barrier performance, shelf life, and recycling infrastructure remain central constraints.",
            "Brands are balancing regulatory pressure, consumer demand, and cost-to-scale considerations.",
        ],
    }
    _CASE_RUNTIME["configurable"] = {
        **_base_configurable({"model": os.environ.get("MODEL_NAME", "")}),
        "allow_clarification": False,
        "search_api": "none",
    }


def _seed_edge_case_token_limit_retry() -> None:
    _CASE_RUNTIME["input_state"] = {
        "messages": [HumanMessage(content="Complete the research report with all detailed findings.")],
        "research_brief": "Produce a comprehensive final report from the accumulated findings.",
        "notes": ["Detailed finding segment {} {}".format(i, "x" * 400) for i in range(1, 60)],
    }
    _CASE_RUNTIME["configurable"] = {
        **_base_configurable({"model": os.environ.get("MODEL_NAME", "")}),
        "allow_clarification": False,
        "search_api": "none",
    }


def _seed_happy_path_supervisor_decomposition() -> None:
    _CASE_RUNTIME["input_state"] = {
        "messages": [
            HumanMessage(
                content=(
                    "I need research on renewable energy across solar, wind, and hydroelectric "
                    "power generation technologies, including efficiency comparisons."
                )
            )
        ]
    }
    _CASE_RUNTIME["configurable"] = {
        **_base_configurable({"model": os.environ.get("MODEL_NAME", "")}),
        "allow_clarification": True,
        "search_api": "none",
        "max_concurrent_research_units": 3,
    }


def _seed_tool_selection_clarify_vs_direct() -> None:
    _CASE_RUNTIME["input_state"] = {
        "messages": [
            HumanMessage(
                content=(
                    "Please research the top 5 cybersecurity frameworks used in healthcare "
                    "organizations for HIPAA compliance in 2024."
                )
            )
        ]
    }
    _CASE_RUNTIME["configurable"] = {
        **_base_configurable({"model": os.environ.get("MODEL_NAME", "")}),
        "allow_clarification": True,
        "search_api": "none",
    }


def _seed_edge_case_empty_findings() -> None:
    _CASE_RUNTIME["input_state"] = {
        "messages": [HumanMessage(content="Research obscure topic xyz123 that has no available information.")],
        "research_brief": "Investigate topic xyz123 and report any reliable information or acknowledge lack of evidence.",
        "notes": [],
    }
    _CASE_RUNTIME["configurable"] = {
        **_base_configurable({"model": os.environ.get("MODEL_NAME", "")}),
        "allow_clarification": True,
        "search_api": "none",
    }


def setup_dependencies(test_case_id: str, precondition: Any, config: dict) -> None:
    del precondition
    _snapshot_runtime_env()
    _CASE_RUNTIME.clear()
    _CASE_RUNTIME.update(deepcopy(_BASE_CASE_RUNTIME))
    _configure_model_env(config)

    dispatch = {
        "happy_path_full_workflow": _seed_happy_path_full_workflow,
        "happy_path_skip_clarification": _seed_happy_path_skip_clarification,
        "missing_info_vague_request": _seed_missing_info_vague_request,
        "happy_path_specific_research_brief": _seed_happy_path_specific_research_brief,
        "happy_path_final_report_generation": _seed_happy_path_final_report_generation,
        "edge_case_token_limit_retry": _seed_edge_case_token_limit_retry,
        "happy_path_supervisor_decomposition": _seed_happy_path_supervisor_decomposition,
        "tool_selection_clarify_vs_direct": _seed_tool_selection_clarify_vs_direct,
        "edge_case_empty_findings": _seed_edge_case_empty_findings,
    }
    seed_fn = dispatch.get(test_case_id)
    if seed_fn is not None:
        seed_fn()
    else:
        _CASE_RUNTIME["input_state"] = {
            "messages": [HumanMessage(content="")]
        }
        _CASE_RUNTIME["configurable"] = _base_configurable(config)

    model_name = _resolve_model(config)
    if model_name:
        _CASE_RUNTIME["configurable"]["research_model"] = model_name
        _CASE_RUNTIME["configurable"]["final_report_model"] = model_name
        _CASE_RUNTIME["configurable"]["compression_model"] = model_name
        _CASE_RUNTIME["configurable"]["summarization_model"] = model_name


def cleanup_dependencies() -> None:
    _CASE_RUNTIME.clear()
    _CASE_RUNTIME.update(deepcopy(_BASE_CASE_RUNTIME))
    _restore_runtime_env()
    try:
        if _instrumented:
            _instrumentor.uninstrument()
            _instrumentor.instrument(tracer_provider=_provider)
    except Exception:
        pass


def _normalize_attributes(attrs: Any) -> dict[str, Any]:
    if not attrs:
        return {}
    try:
        return {str(k): _jsonable(v) for k, v in dict(attrs).items()}
    except Exception:
        return {}


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    content = getattr(value, "content", None)
    if content is not None:
        return _jsonable(content)
    return str(value)


def _pick_first(attrs: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in attrs and attrs[key] not in (None, "", [], {}):
            return attrs[key]
    return None


def _map_genai_attributes(attrs: dict[str, Any]) -> dict[str, Any]:
    mapped: dict[str, Any] = {}
    alias_map = {
        "gen_ai.system": [
            "gen_ai.system",
            "llm.system",
            "ai.system",
        ],
        "gen_ai.request.model": [
            "gen_ai.request.model",
            "llm.request.model",
            "llm.model_name",
            "model",
            "openinference.llm.model_name",
        ],
        "gen_ai.response.model": [
            "gen_ai.response.model",
            "llm.response.model",
            "response.model",
        ],
        "gen_ai.operation.name": [
            "gen_ai.operation.name",
            "llm.operation.name",
            "openinference.span.kind",
        ],
        "gen_ai.prompt": [
            "gen_ai.prompt",
            "input.value",
            "llm.prompts",
            "prompt",
            "gen_ai.input.messages",
            "input",
        ],
        "gen_ai.completion": [
            "gen_ai.completion",
            "output.value",
            "response",
            "completion",
            "gen_ai.output.messages",
            "output",
        ],
        "gen_ai.usage.input_tokens": [
            "gen_ai.usage.input_tokens",
            "llm.token_count.prompt",
            "input_tokens",
            "prompt_tokens",
            "usage.prompt_tokens",
        ],
        "gen_ai.usage.output_tokens": [
            "gen_ai.usage.output_tokens",
            "llm.token_count.completion",
            "output_tokens",
            "completion_tokens",
            "usage.completion_tokens",
        ],
    }
    for target_key, aliases in alias_map.items():
        value = _pick_first(attrs, aliases)
        if value is not None:
            mapped[target_key] = value
    return mapped


def _guess_node_type(span_name: str, attrs: dict[str, Any], gen_ai: dict[str, Any]) -> str:
    operation = str(gen_ai.get("gen_ai.operation.name", "")).lower()
    if operation == "execute_tool":
        return "tool"
    if operation in {"chat", "completion", "generate"}:
        return "llm"
    tool_name = _pick_first(attrs, ["gen_ai.tool.name", "tool.name", "name"])
    if tool_name and span_name != "user_input":
        return "tool"
    if "agent" in span_name.lower() or "graph" in span_name.lower() or "workflow" in span_name.lower() or "langgraph" in span_name.lower():
        return "agent"
    if any(k.startswith("gen_ai") or k.startswith("llm") for k in attrs):
        return "llm"
    return "span"


def _span_to_node(span: Any) -> dict[str, Any]:
    attrs = _normalize_attributes(getattr(span, "attributes", {}))
    gen_ai = _map_genai_attributes(attrs)
    parent_span_id = None
    if getattr(span, "parent", None) is not None:
        parent_span_id = format(span.parent.span_id, "x")
    node = {
        "type": _guess_node_type(getattr(span, "name", ""), attrs, gen_ai),
        "name": getattr(span, "name", ""),
        "span_id": format(span.context.span_id, "x"),
        "parent_span_id": parent_span_id,
        "attributes": attrs,
        "gen_ai_attributes": gen_ai,
        "children": [],
    }
    return node


def _spans_to_tree(spans: list[Any], *, exclude_names: set[str]) -> list[dict[str, Any]]:
    filtered = sorted(
        [span for span in spans if getattr(span, "name", "") not in exclude_names],
        key=lambda span: getattr(span, "start_time", 0) or 0,
    )
    nodes = {span.context.span_id: _span_to_node(span) for span in filtered}
    child_ids: dict[int, list[int]] = {}
    roots: list[int] = []
    span_ids = set(nodes)

    for span in filtered:
        sid = span.context.span_id
        parent_sid = span.parent.span_id if getattr(span, "parent", None) is not None else None
        if parent_sid is not None and parent_sid in span_ids:
            child_ids.setdefault(parent_sid, []).append(sid)
        else:
            roots.append(sid)

    def attach(sid: int) -> dict[str, Any]:
        node = nodes[sid]
        node["children"] = [attach(child_sid) for child_sid in child_ids.get(sid, [])]
        return node

    return [attach(root_sid) for root_sid in roots]


def _build_trace(user_input: str, answer: str, spans: list[Any]) -> dict[str, Any]:
    return {
        "type": "user_input",
        "input": user_input,
        "output": answer,
        "children": _spans_to_tree(spans, exclude_names={"user_input"}),
    }


def _coerce_answer(result: Any) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        for key in ("final_report", "output", "answer", "content"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value
            if hasattr(value, "content"):
                return _coerce_answer(value)
        messages = result.get("messages")
        if isinstance(messages, list) and messages:
            return _coerce_answer(messages[-1])
        return json.dumps(_jsonable(result), ensure_ascii=False)
    content = getattr(result, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        try:
            return "\n".join(str(_jsonable(item)) for item in content if item is not None)
        except Exception:
            pass
    messages = getattr(result, "messages", None)
    if messages:
        return _coerce_answer(messages[-1])
    return str(result)


async def _ainvoke_agent(user_input: str, config: dict) -> tuple[str, dict[str, Any]]:
    configurable = deepcopy(_CASE_RUNTIME.get("configurable") or _base_configurable(config))
    metadata = deepcopy(_CASE_RUNTIME.get("metadata") or {})
    state = deepcopy(_CASE_RUNTIME.get("input_state") or {"messages": [HumanMessage(content=user_input)]})
    if not state.get("messages"):
        state["messages"] = [HumanMessage(content=user_input)]

    tracer = trace.get_tracer("promptfoo-eval")
    _exporter.clear()
    with tracer.start_as_current_span("user_input") as root:
        root.set_attribute("input", user_input)
        result = await _agent_graph.ainvoke(
            state,
            config={
                "configurable": configurable,
                "metadata": metadata,
            },
        )
        answer = _coerce_answer(result)
        root.set_attribute("output", answer)
    spans = list(_exporter.get_finished_spans())
    trace_tree = _build_trace(user_input, answer, spans)
    return answer, trace_tree


def _invoke_agent(user_input: str, config: dict) -> tuple[str, dict[str, Any]]:
    return asyncio.run(_ainvoke_agent(user_input, config))


def call_api(prompt: str, options: dict, context: dict) -> dict:
    options = options or {}
    context = context or {}
    config = options.get("config", {}) or {}
    test_case_id = _read_test_case_id(options, context)
    precondition = _read_precondition(options, context)

    setup_dependencies(test_case_id, precondition, config)
    try:
        answer, trace_tree = _invoke_agent(prompt, config)
        return {
            "output": json.dumps({"answer": answer, "trace": trace_tree}, ensure_ascii=False)
        }
    finally:
        cleanup_dependencies()
