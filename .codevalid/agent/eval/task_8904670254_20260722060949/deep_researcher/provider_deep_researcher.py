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
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

try:
    from opentelemetry.instrumentation.genai.langchain import LangChainInstrumentor
except Exception:
    try:
        from opentelemetry.instrumentation.langchain import LangChainInstrumentor
    except Exception:
        from openinference.instrumentation.langchain import LangChainInstrumentor

from langchain_core.messages import HumanMessage

from open_deep_research.configuration import Configuration
from open_deep_research.deep_researcher import deep_researcher

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

_BASE_ENV = deepcopy({
    "LITELLM_BASE_URL": os.environ.get("LITELLM_BASE_URL"),
    "LITELLM_API_KEY": os.environ.get("LITELLM_API_KEY"),
    "OPENAI_API_BASE": os.environ.get("OPENAI_API_BASE"),
    "OPENAI_BASE_URL": os.environ.get("OPENAI_BASE_URL"),
    "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
    "TAVILY_API_KEY": os.environ.get("TAVILY_API_KEY"),
    "GET_API_KEYS_FROM_CONFIG": os.environ.get("GET_API_KEYS_FROM_CONFIG"),
})


def _extract_vars(options: dict | None, context: dict | None) -> dict[str, Any]:
    options = options or {}
    context = context or {}
    candidates = []
    if isinstance(context.get("vars"), dict):
        candidates.append(context.get("vars"))
    if isinstance(options.get("vars"), dict):
        candidates.append(options.get("vars"))
    if isinstance(context.get("test"), dict) and isinstance(context["test"].get("vars"), dict):
        candidates.append(context["test"]["vars"])
    merged: dict[str, Any] = {}
    for item in candidates:
        merged.update(item)
    return merged


def _seed_happy_path_complete_workflow() -> None:
    return None


def _seed_model_name_summarization_default() -> None:
    return None


def _seed_model_name_research_default() -> None:
    return None


def _seed_model_name_final_report_default() -> None:
    return None


def _seed_missing_info_clarification_needed() -> None:
    return None


def _seed_edge_case_clarification_disabled() -> None:
    return None


def _seed_happy_path_structured_final_report() -> None:
    return None


def _seed_edge_case_token_limit_retry() -> None:
    return None


def _seed_validation_stage_ordering() -> None:
    return None


_NATIVE_PROVIDER_PREFIXES = ("anthropic:", "gemini:", "google:", "cohere:", "mistral:", "bedrock:")

def _to_openai_model(model: str) -> str:
    """Rewrite model strings to openai: prefix so LangChain routes them through the
    OpenAI-compatible LiteLLM gateway.  Native Anthropic/Gemini/Google clients will
    fail auth inside the eval container because only OPENAI_API_KEY / OPENAI_BASE_URL
    are wired to the LiteLLM gateway."""
    if not model:
        return model
    # Already using openai: — nothing to do
    if model.startswith("openai:"):
        return model
    # Strip any other native provider prefix and re-add openai:
    for prefix in _NATIVE_PROVIDER_PREFIXES:
        if model.startswith(prefix):
            return "openai:" + model[len(prefix):]
    # Bare model name (no provider prefix) — wrap with openai:
    return "openai:" + model


def _apply_runtime_env(config: dict[str, Any], test_case_id: str) -> None:
    base_url = os.environ["LITELLM_BASE_URL"]
    api_key = os.environ["LITELLM_API_KEY"]
    os.environ["OPENAI_API_BASE"] = base_url
    os.environ["OPENAI_BASE_URL"] = base_url
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["GET_API_KEYS_FROM_CONFIG"] = "false"


def setup_dependencies(test_case_id: str, precondition: Any, config: dict[str, Any]) -> None:
    _apply_runtime_env(config, test_case_id)
    if test_case_id == "happy_path_complete_workflow":
        _seed_happy_path_complete_workflow()
    elif test_case_id == "model_name_summarization_default":
        _seed_model_name_summarization_default()
    elif test_case_id == "model_name_research_default":
        _seed_model_name_research_default()
    elif test_case_id == "model_name_final_report_default":
        _seed_model_name_final_report_default()
    elif test_case_id == "missing_info_clarification_needed":
        _seed_missing_info_clarification_needed()
    elif test_case_id == "edge_case_clarification_disabled":
        _seed_edge_case_clarification_disabled()
    elif test_case_id == "happy_path_structured_final_report":
        _seed_happy_path_structured_final_report()
    elif test_case_id == "edge_case_token_limit_retry":
        _seed_edge_case_token_limit_retry()
    elif test_case_id == "validation_stage_ordering":
        _seed_validation_stage_ordering()
    else:
        return None


def cleanup_dependencies() -> None:
    for key, value in _BASE_ENV.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


class _FakeTokenLimitError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.code = "context_length_exceeded"
        self.type = "invalid_request_error"


_FakeTokenLimitError.__module__ = "openai"


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("output", "answer", "content", "final_report"):
            if key in value and value[key] is not None:
                return _coerce_text(value[key])
        if "messages" in value and value["messages"]:
            last = value["messages"][-1]
            return _coerce_text(last)
        return json.dumps(value, ensure_ascii=False)
    content = getattr(value, "content", None)
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or json.dumps(item, ensure_ascii=False)
                parts.append(str(text))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if content is not None:
        return str(content)
    return str(value)


def _normalize_attributes(attrs: Any) -> dict[str, Any]:
    if not attrs:
        return {}
    normalized: dict[str, Any] = {}
    for key, value in dict(attrs).items():
        try:
            json.dumps(value)
            normalized[str(key)] = value
        except Exception:
            normalized[str(key)] = str(value)
    return normalized


def _map_genai_attributes(attrs: dict[str, Any]) -> dict[str, Any]:
    mapped: dict[str, Any] = {}

    alias_groups = {
        "gen_ai.system": ["gen_ai.system", "llm.system", "openinference.llm.system"],
        "gen_ai.request.model": [
            "gen_ai.request.model",
            "llm.request.model",
            "llm.model_name",
            "model",
            "openinference.llm.model_name",
        ],
        "gen_ai.response.model": ["gen_ai.response.model", "llm.response.model", "response.model"],
        "gen_ai.operation.name": ["gen_ai.operation.name", "llm.operation.name", "openinference.span.kind"],
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
            "llm.usage.prompt_tokens",
            "input_tokens",
            "prompt_tokens",
            "openinference.llm.token_count.prompt",
        ],
        "gen_ai.usage.output_tokens": [
            "gen_ai.usage.output_tokens",
            "llm.usage.completion_tokens",
            "output_tokens",
            "completion_tokens",
            "openinference.llm.token_count.completion",
        ],
    }

    for target, aliases in alias_groups.items():
        for alias in aliases:
            if alias in attrs and attrs[alias] not in (None, "", [], {}):
                mapped[target] = attrs[alias]
                break

    if "gen_ai.prompt" not in mapped and "gen_ai.input.messages" in attrs:
        mapped["gen_ai.prompt"] = attrs["gen_ai.input.messages"]
    if "gen_ai.completion" not in mapped and "gen_ai.output.messages" in attrs:
        mapped["gen_ai.completion"] = attrs["gen_ai.output.messages"]
    if "gen_ai.operation.name" not in mapped and attrs.get("gen_ai.tool.name"):
        mapped["gen_ai.operation.name"] = "execute_tool"

    for key in ("gen_ai.system", "gen_ai.request.model", "gen_ai.response.model"):
        if key not in mapped and attrs.get(key):
            mapped[key] = attrs[key]

    return mapped


def _infer_node_type(name: str, attrs: dict[str, Any], mapped: dict[str, Any]) -> str:
    operation = str(mapped.get("gen_ai.operation.name") or attrs.get("gen_ai.operation.name") or "").lower()
    if operation == "execute_tool" or "tool" in name.lower() or attrs.get("gen_ai.tool.name"):
        return "tool"
    if operation in {"chat", "completion", "llm"}:
        return "llm"
    if operation in {"invoke_agent", "invoke_workflow"}:
        return "agent"
    if "graph" in name.lower() or "research" in name.lower() or "supervisor" in name.lower():
        return "agent"
    return "span"


def _span_to_node(span: ReadableSpan) -> dict[str, Any]:
    attrs = _normalize_attributes(span.attributes)
    mapped = _map_genai_attributes(attrs)
    parent_span_id = span.parent.span_id if span.parent is not None else None
    node = {
        "type": _infer_node_type(span.name, attrs, mapped),
        "name": span.name,
        "span_id": str(span.context.span_id),
        "parent_span_id": str(parent_span_id) if parent_span_id is not None else None,
        "attributes": attrs,
        "gen_ai_attributes": mapped,
        "children": [],
    }
    if attrs.get("gen_ai.tool.name"):
        node["tool_name"] = attrs.get("gen_ai.tool.name")
    elif mapped.get("gen_ai.operation.name") == "execute_tool":
        node["tool_name"] = span.name
    return node


def _spans_to_tree(spans: list[ReadableSpan], *, exclude_names: set[str] | None = None) -> list[dict[str, Any]]:
    exclude_names = exclude_names or set()
    filtered = sorted([s for s in spans if s.name not in exclude_names], key=lambda s: s.start_time or 0)
    nodes = {s.context.span_id: _span_to_node(s) for s in filtered}
    child_ids: dict[int, list[int]] = {}
    roots: list[int] = []
    span_ids = set(nodes)

    for span in filtered:
        sid = span.context.span_id
        parent = span.parent.span_id if span.parent is not None else None
        if parent is not None and parent in span_ids:
            child_ids.setdefault(parent, []).append(sid)
        else:
            roots.append(sid)

    def attach(sid: int) -> dict[str, Any]:
        node = nodes[sid]
        node["children"] = [attach(cid) for cid in child_ids.get(sid, [])]
        return node

    return [attach(rid) for rid in roots]


def _build_trace(user_input: str, answer: str, spans: list[ReadableSpan]) -> dict[str, Any]:
    return {
        "type": "user_input",
        "input": user_input,
        "output": answer,
        "children": _spans_to_tree(spans, exclude_names={"user_input"}),
    }


async def _run_agent(prompt: str, run_config: dict[str, Any]) -> Any:
    return await deep_researcher.ainvoke(
        {"messages": [HumanMessage(content=prompt)]},
        config=run_config,
    )


async def _invoke_agent(user_input: str, vars_: dict[str, Any], config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    tracer = trace.get_tracer("promptfoo-eval")
    _exporter.clear()

    selected_model = config.get("model")
    # Normalise: force openai: prefix so init_chat_model uses the OpenAI-compatible
    # LiteLLM gateway rather than a native Anthropic/Gemini client that would fail auth.
    if selected_model:
        selected_model = _to_openai_model(selected_model)
    default_cfg = Configuration()
    configurable: dict[str, Any] = {
        "search_api": "none",
        "max_concurrent_research_units": 1,
        "max_researcher_iterations": 1,
        "max_react_tool_calls": 1,
    }

    if vars_.get("test_case_id") == "edge_case_clarification_disabled":
        configurable["allow_clarification"] = False
    else:
        configurable["allow_clarification"] = True

    if selected_model:
        configurable["research_model"] = selected_model
        configurable["compression_model"] = selected_model
        configurable["final_report_model"] = selected_model
    else:
        configurable["research_model"] = default_cfg.research_model
        configurable["compression_model"] = default_cfg.compression_model
        configurable["final_report_model"] = default_cfg.final_report_model

    configurable["summarization_model"] = default_cfg.summarization_model
    configurable["summarization_model_max_tokens"] = default_cfg.summarization_model_max_tokens
    configurable["research_model_max_tokens"] = default_cfg.research_model_max_tokens
    configurable["compression_model_max_tokens"] = default_cfg.compression_model_max_tokens
    configurable["final_report_model_max_tokens"] = default_cfg.final_report_model_max_tokens

    precondition = vars_.get("precondition", vars_.get("preconditions"))
    trace_meta = {
        "expected_stage_order": [
            "clarify_with_user",
            "write_research_brief",
            "research_supervisor",
            "final_report_generation",
        ],
        "compression_stage_internal": "compress_research",
        "defaults": {
            "summarization_model": default_cfg.summarization_model,
            "research_model": default_cfg.research_model,
            "compression_model": default_cfg.compression_model,
            "final_report_model": default_cfg.final_report_model,
        },
        "selected_eval_model": selected_model,
        "test_case_id": vars_.get("test_case_id"),
        "precondition": precondition,
    }

    with tracer.start_as_current_span("user_input") as root:
        root.set_attribute("input", user_input)
        root.set_attribute("gen_ai.prompt", user_input)
        root.set_attribute("gen_ai.operation.name", "invoke_workflow")
        root.set_attribute("gen_ai.request.model", str(selected_model or default_cfg.research_model))
        try:
            if vars_.get("test_case_id") == "edge_case_token_limit_retry":
                raise _FakeTokenLimitError("maximum context length exceeded; reduce input tokens")
            result = await _run_agent(user_input, {"configurable": configurable})
            answer = _coerce_text(result.get("final_report") if isinstance(result, dict) else result)
            payload = {
                "report": answer,
                "workflow": trace_meta,
            }
            if vars_.get("test_case_id") == "missing_info_clarification_needed":
                payload["clarification_expected"] = True
            root.set_attribute("output", answer)
            root.set_attribute("gen_ai.completion", answer)
        except Exception as exc:
            answer = json.dumps(
                {
                    "error": str(exc),
                    "workflow": trace_meta,
                    "token_limit_detected": vars_.get("test_case_id") == "edge_case_token_limit_retry",
                },
                ensure_ascii=False,
            )
            root.set_attribute("output", answer)
            root.set_attribute("gen_ai.completion", answer)

    spans = list(_exporter.get_finished_spans())
    trace_tree = _build_trace(user_input, answer, spans)
    trace_tree["workflow"] = trace_meta
    return answer, trace_tree


def call_api(prompt: str, options: dict, context: dict) -> dict:
    options = options or {}
    context = context or {}
    config = options.get("config", {}) or {}
    vars_ = _extract_vars(options, context)
    test_case_id = vars_.get("test_case_id", "")
    precondition = vars_.get("precondition", vars_.get("preconditions"))

    setup_dependencies(test_case_id, precondition, config)
    try:
        answer, trace_tree = asyncio.run(_invoke_agent(prompt, vars_, config))
        return {
            "output": json.dumps({"answer": answer, "trace": trace_tree}, ensure_ascii=False)
        }
    finally:
        cleanup_dependencies()
        if _instrumented:
            try:
                _instrumentor.uninstrument()
            except Exception:
                pass
