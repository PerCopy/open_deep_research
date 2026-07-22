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

from langchain_core.messages import HumanMessage
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

try:
    from opentelemetry.instrumentation.genai.langchain import LangChainInstrumentor
except Exception:  # pragma: no cover
    try:
        from opentelemetry.instrumentation.langchain import LangChainInstrumentor
    except Exception:  # pragma: no cover
        from openinference.instrumentation.langchain import LangChainInstrumentor

from open_deep_research.deep_researcher import deep_researcher

os.environ.setdefault("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY")

_exporter = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(_exporter))
_instrumentor = LangChainInstrumentor()
_INSTRUMENTED = False
try:
    trace.set_tracer_provider(_provider)
except Exception:
    pass
try:
    _instrumentor.instrument(tracer_provider=_provider)
    _INSTRUMENTED = True
except Exception:
    _INSTRUMENTED = False

_BASELINE_ENV = deepcopy({
    key: os.environ.get(key)
    for key in [
        "OPENAI_API_KEY",
        "OPENAI_API_BASE",
        "OPENAI_BASE_URL",
        "GET_API_KEYS_FROM_CONFIG",
    ]
})


def _extract_vars(options: dict | None, context: dict | None) -> dict:
    options = options or {}
    context = context or {}
    merged: dict[str, Any] = {}
    for candidate in [
        options.get("vars"),
        context.get("vars"),
        options.get("context", {}).get("vars") if isinstance(options.get("context"), dict) else None,
        context.get("context", {}).get("vars") if isinstance(context.get("context"), dict) else None,
    ]:
        if isinstance(candidate, dict):
            merged.update(candidate)
    return merged


def _extract_config(options: dict | None, context: dict | None) -> dict:
    options = options or {}
    context = context or {}
    config: dict[str, Any] = {}
    for candidate in [options.get("config"), context.get("config")]:
        if isinstance(candidate, dict):
            config.update(candidate)
    return config


def _seed_happy_path_research_workflow_execution() -> None:
    return None


def _seed_clarification_request_for_vague_research_topic() -> None:
    return None


def _seed_clarification_skipped_when_explicitly_disabled() -> None:
    return None


def _seed_research_brief_created_from_user_messages() -> None:
    return None


def _seed_final_report_is_structured_and_comprehensive() -> None:
    return None


def _seed_final_report_generation_retries_on_token_limit() -> None:
    return None


def _seed_declines_non_research_requests_appropriately() -> None:
    return None


def _seed_declines_harmful_or_unethical_research() -> None:
    return None


def _seed_clear_topic_proceeds_without_clarification() -> None:
    return None


def _seed_workflow_uses_correct_default_models_per_stage() -> None:
    return None


def setup_dependencies(test_case_id: str, precondition: Any, config: dict | None) -> None:
    _ = precondition
    _ = config
    if test_case_id == "happy_path_research_workflow_execution":
        _seed_happy_path_research_workflow_execution()
    elif test_case_id == "clarification_request_for_vague_research_topic":
        _seed_clarification_request_for_vague_research_topic()
    elif test_case_id == "clarification_skipped_when_explicitly_disabled":
        _seed_clarification_skipped_when_explicitly_disabled()
    elif test_case_id == "research_brief_created_from_user_messages":
        _seed_research_brief_created_from_user_messages()
    elif test_case_id == "final_report_is_structured_and_comprehensive":
        _seed_final_report_is_structured_and_comprehensive()
    elif test_case_id == "final_report_generation_retries_on_token_limit":
        _seed_final_report_generation_retries_on_token_limit()
    elif test_case_id == "declines_non_research_requests_appropriately":
        _seed_declines_non_research_requests_appropriately()
    elif test_case_id == "declines_harmful_or_unethical_research":
        _seed_declines_harmful_or_unethical_research()
    elif test_case_id == "clear_topic_proceeds_without_clarification":
        _seed_clear_topic_proceeds_without_clarification()
    elif test_case_id == "workflow_uses_correct_default_models_per_stage":
        _seed_workflow_uses_correct_default_models_per_stage()
    else:
        return None


def cleanup_dependencies() -> None:
    for key, value in _BASELINE_ENV.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _configure_env(config: dict) -> None:
    base_url = os.environ["LITELLM_BASE_URL"]
    api_key = os.environ["LITELLM_API_KEY"]
    os.environ["OPENAI_API_BASE"] = base_url
    os.environ["OPENAI_BASE_URL"] = base_url
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["GET_API_KEYS_FROM_CONFIG"] = "false"
    _ = config


def _resolve_stage_models(config: dict, test_case_id: str) -> dict[str, Any]:
    selected_model = config.get("model")
    resolved = {
        "allow_clarification": True,
        "search_api": "none",
        "summarization_model": selected_model or "openai:gpt-4.1-mini",
        "research_model": selected_model or "openai:gpt-4.1",
        "compression_model": selected_model or "openai:gpt-4.1",
        "final_report_model": selected_model or "openai:gpt-4.1",
    }
    if test_case_id == "clarification_skipped_when_explicitly_disabled":
        resolved["allow_clarification"] = False
    elif test_case_id in {
        "clarification_request_for_vague_research_topic",
        "clear_topic_proceeds_without_clarification",
        "happy_path_research_workflow_execution",
        "research_brief_created_from_user_messages",
        "final_report_is_structured_and_comprehensive",
        "final_report_generation_retries_on_token_limit",
        "workflow_uses_correct_default_models_per_stage",
    }:
        resolved["allow_clarification"] = True
    elif test_case_id in {
        "declines_non_research_requests_appropriately",
        "declines_harmful_or_unethical_research",
    }:
        resolved["allow_clarification"] = True
    return resolved


async def _run_agent(user_input: str, configurable: dict[str, Any]) -> Any:
    return await deep_researcher.ainvoke(
        {"messages": [HumanMessage(content=user_input)]},
        config={"configurable": configurable},
    )


def _coerce_message_content(value: Any) -> str:
    if value is None:
        return ""
    content = getattr(value, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content") or item.get("value")
                if text:
                    parts.append(str(text))
            else:
                text = getattr(item, "text", None) or getattr(item, "content", None)
                if text:
                    parts.append(str(text))
        return "\n".join(part for part in parts if part)
    if content is not None:
        return str(content)
    return str(value)


def _extract_answer(result: Any) -> str:
    if isinstance(result, str):
        return result
    if hasattr(result, "content"):
        text = _coerce_message_content(result)
        if text:
            return text
    if isinstance(result, dict):
        for key in ["final_report", "output", "answer", "content"]:
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value
            if value is not None:
                text = _coerce_message_content(value)
                if text:
                    return text
        messages = result.get("messages")
        if isinstance(messages, list) and messages:
            for msg in reversed(messages):
                text = _coerce_message_content(msg)
                if text:
                    return text
        return json.dumps(result, ensure_ascii=False, default=str)
    messages = getattr(result, "messages", None)
    if isinstance(messages, list) and messages:
        for msg in reversed(messages):
            text = _coerce_message_content(msg)
            if text:
                return text
    return str(result)


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
            "provider",
        ],
        "gen_ai.request.model": [
            "gen_ai.request.model",
            "llm.request.model",
            "llm.model_name",
            "model",
            "openinference.llm.model_name",
            "gen_ai.response.model_name",
        ],
        "gen_ai.response.model": [
            "gen_ai.response.model",
            "llm.response.model",
            "response.model",
            "openinference.output.model",
        ],
        "gen_ai.operation.name": [
            "gen_ai.operation.name",
            "llm.operation.name",
            "openinference.span.kind",
            "langchain.span.kind",
        ],
        "gen_ai.prompt": [
            "gen_ai.prompt",
            "input.value",
            "llm.prompts",
            "prompt",
            "gen_ai.input.messages",
            "openinference.input.value",
        ],
        "gen_ai.completion": [
            "gen_ai.completion",
            "output.value",
            "response",
            "completion",
            "gen_ai.output.messages",
            "openinference.output.value",
        ],
        "gen_ai.usage.input_tokens": [
            "gen_ai.usage.input_tokens",
            "llm.token_count.prompt",
            "llm.usage.prompt_tokens",
            "usage.prompt_tokens",
            "input_tokens",
            "prompt_tokens",
        ],
        "gen_ai.usage.output_tokens": [
            "gen_ai.usage.output_tokens",
            "llm.token_count.completion",
            "llm.usage.completion_tokens",
            "usage.completion_tokens",
            "output_tokens",
            "completion_tokens",
        ],
    }
    for target_key, source_keys in alias_map.items():
        value = _pick_first(attrs, source_keys)
        if value is not None:
            mapped[target_key] = value
    return mapped


def _span_kind(span_name: str, attrs: dict[str, Any], genai: dict[str, Any]) -> str:
    operation = str(genai.get("gen_ai.operation.name", "")).lower()
    if operation in {"chat", "llm", "completion"}:
        return "llm"
    if operation in {"execute_tool", "tool"}:
        return "tool"
    if "tool" in span_name.lower():
        return "tool"
    if any(key in attrs for key in ["gen_ai.tool.name", "tool.name", "langchain.tool.name"]):
        return "tool"
    if "agent" in span_name.lower() or "graph" in span_name.lower() or "workflow" in span_name.lower():
        return "agent"
    return "span"


def _span_to_node(span: Any) -> dict[str, Any]:
    attrs = _normalize_attributes(getattr(span, "attributes", {}) or {})
    genai = _map_genai_attributes(attrs)
    parent_span_id = span.parent.span_id if getattr(span, "parent", None) is not None else None
    return {
        "type": _span_kind(getattr(span, "name", ""), attrs, genai),
        "name": getattr(span, "name", ""),
        "span_id": getattr(getattr(span, "context", None), "span_id", None),
        "parent_span_id": parent_span_id,
        "attributes": attrs,
        "gen_ai_attributes": genai,
        "children": [],
    }


def _spans_to_tree(spans: list[Any], *, exclude_names: set[str]) -> list[dict[str, Any]]:
    filtered = sorted(
        [span for span in spans if getattr(span, "name", "") not in exclude_names],
        key=lambda span: getattr(span, "start_time", 0) or 0,
    )
    nodes = {
        getattr(span.context, "span_id", None): _span_to_node(span)
        for span in filtered
        if getattr(getattr(span, "context", None), "span_id", None) is not None
    }
    child_ids: dict[int, list[int]] = {}
    roots: list[int] = []
    span_ids = set(nodes.keys())

    for span in filtered:
        sid = getattr(getattr(span, "context", None), "span_id", None)
        if sid is None or sid not in nodes:
            continue
        parent = span.parent.span_id if getattr(span, "parent", None) is not None else None
        if parent is not None and parent in span_ids:
            child_ids.setdefault(parent, []).append(sid)
        else:
            roots.append(sid)

    def attach(sid: int) -> dict[str, Any]:
        node = nodes[sid]
        node["children"] = [attach(cid) for cid in child_ids.get(sid, [])]
        return node

    return [attach(root_id) for root_id in roots]


def _build_trace(user_input: str, answer: str, spans: list[Any]) -> dict[str, Any]:
    return {
        "type": "user_input",
        "input": user_input,
        "output": answer,
        "children": _spans_to_tree(spans, exclude_names={"user_input"}),
    }


def _invoke_agent(user_input: str, config: dict, test_case_id: str) -> tuple[str, dict[str, Any]]:
    _configure_env(config)
    configurable = _resolve_stage_models(config, test_case_id)
    _exporter.clear()
    tracer = trace.get_tracer("codevalid.promptfoo.deep_researcher")
    with tracer.start_as_current_span("user_input") as root_span:
        root_span.set_attribute("input", user_input)
        root_span.set_attribute("gen_ai.prompt", user_input)
        result = asyncio.run(_run_agent(user_input, configurable))
        answer = _extract_answer(result)
        root_span.set_attribute("output", answer)
        root_span.set_attribute("gen_ai.completion", answer)
    spans = list(_exporter.get_finished_spans())
    trace_tree = _build_trace(user_input, answer, spans)
    return answer, trace_tree


def call_api(prompt: str, options: dict, context: dict) -> dict:
    vars_ = _extract_vars(options, context)
    config = _extract_config(options, context)
    test_case_id = vars_.get("test_case_id") or ""
    precondition = vars_.get("precondition", vars_.get("preconditions"))
    setup_dependencies(test_case_id, precondition, config)
    try:
        answer, trace_tree = _invoke_agent(prompt, config, test_case_id)
        return {
            "output": json.dumps({"answer": answer, "trace": trace_tree}, ensure_ascii=False)
        }
    finally:
        cleanup_dependencies()
        if _INSTRUMENTED:
            try:
                _instrumentor.uninstrument()
            except Exception:
                pass
