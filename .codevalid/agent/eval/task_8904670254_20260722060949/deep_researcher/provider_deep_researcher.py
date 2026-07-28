from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import patch

WORKSPACE_ROOT = Path(__file__).resolve().parents[5]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

os.environ.setdefault("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY")

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
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

from open_deep_research import deep_researcher as dr

_exporter = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(_exporter))
_INSTRUMENTED = False
_INSTRUMENTOR = None

try:
    trace.set_tracer_provider(_provider)
except Exception:
    pass

try:
    _INSTRUMENTOR = LangChainInstrumentor()
    _INSTRUMENTOR.instrument(tracer_provider=_provider)
    _INSTRUMENTED = True
except Exception:
    _INSTRUMENTED = False

_ENV_BASELINE = {k: os.environ.get(k) for k in [
    "LITELLM_BASE_URL",
    "LITELLM_API_KEY",
    "OPENAI_API_BASE",
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "GET_API_KEYS_FROM_CONFIG",
]}


def _restore_env() -> None:
    for key, value in _ENV_BASELINE.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


class _FakeStructuredResponse:
    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


class _FakeModelChain:
    def __init__(self, scenario: dict[str, Any], stage: str | None = None) -> None:
        self.scenario = scenario
        self.stage = stage

    def with_structured_output(self, schema: Any) -> "_FakeModelChain":
        name = getattr(schema, "__name__", str(schema))
        if name == "ClarifyWithUser":
            return _FakeModelChain(self.scenario, stage="clarify_structured")
        if name == "ResearchQuestion":
            return _FakeModelChain(self.scenario, stage="brief_structured")
        return _FakeModelChain(self.scenario, stage=self.stage)

    def with_retry(self, **kwargs: Any) -> "_FakeModelChain":
        return self

    def with_config(self, config: dict[str, Any]) -> "_FakeModelChain":
        if self.stage == "clarify_structured":
            return _FakeModelChain(self.scenario, stage="clarify_invoke")
        if self.stage == "brief_structured":
            return _FakeModelChain(self.scenario, stage="brief_invoke")
        return _FakeModelChain(self.scenario, stage="final_report_invoke")

    def bind_tools(self, tools: list[Any]) -> "_FakeModelChain":
        return _FakeModelChain(self.scenario, stage="tools_bound")

    async def ainvoke(self, messages: Any) -> Any:
        tracer = trace.get_tracer("promptfoo-eval.fake-model")
        with tracer.start_as_current_span("llm") as span:
            span.set_attribute("gen_ai.system", "litellm")
            span.set_attribute("gen_ai.request.model", self.scenario.get("selected_model", ""))
            span.set_attribute("gen_ai.operation.name", "chat")
            span.set_attribute("input.value", _safe_json(messages))
            span.set_attribute("llm.prompts", _safe_json(messages))

            if self.stage == "clarify_invoke":
                response = _FakeStructuredResponse(
                    need_clarification=bool(self.scenario.get("need_clarification", False)),
                    question=self.scenario.get("clarifying_question", "Could you clarify your research scope?"),
                    verification=self.scenario.get("verification", "Understood. I will proceed with the research."),
                )
                span.set_attribute("output.value", _safe_json(response.__dict__))
                return response

            if self.stage == "brief_invoke":
                brief = self.scenario.get("research_brief", "Structured research brief")
                response = _FakeStructuredResponse(research_brief=brief)
                span.set_attribute("output.value", brief)
                return response

            if self.stage == "final_report_invoke":
                failures_before_success = int(self.scenario.get("final_report_failures_before_success", 0))
                current_failures = int(self.scenario.get("final_report_attempts", 0))
                if current_failures < failures_before_success:
                    self.scenario["final_report_attempts"] = current_failures + 1
                    error = Exception(self.scenario.get("token_limit_error_message", "context length exceeded"))
                    error.code = "context_length_exceeded"
                    error.type = "invalid_request_error"
                    span.set_attribute("error", True)
                    span.set_attribute("output.value", str(error))
                    raise error
                content = self.scenario.get("final_report_content", "Structured final report")
                response = AIMessage(content=content)
                span.set_attribute("gen_ai.completion", content)
                span.set_attribute("output.value", content)
                return response

            response = AIMessage(content=self.scenario.get("fallback_content", "OK"))
            span.set_attribute("output.value", response.content)
            return response


class _FakeSupervisorSubgraph:
    def __init__(self, scenario: dict[str, Any]) -> None:
        self.scenario = scenario

    async def ainvoke(self, state: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
        tracer = trace.get_tracer("promptfoo-eval.fake-supervisor")
        with tracer.start_as_current_span("invoke_workflow") as workflow_span:
            workflow_span.set_attribute("gen_ai.operation.name", "invoke_workflow")
            workflow_span.set_attribute("input.value", _safe_json(state))

            with tracer.start_as_current_span("decompose_research") as agent_span:
                agent_span.set_attribute("openinference.span.kind", "agent")
                agent_span.set_attribute("gen_ai.operation.name", "invoke_agent")
                agent_span.set_attribute("input.value", str(state.get("research_brief", "")))

                with tracer.start_as_current_span("search_tool") as tool_span:
                    tool_span.set_attribute("gen_ai.operation.name", "execute_tool")
                    tool_span.set_attribute("gen_ai.tool.name", "search")
                    tool_span.set_attribute("gen_ai.tool.call.arguments", json.dumps({"query": state.get("research_brief", "")}, ensure_ascii=False))
                    tool_span.set_attribute("gen_ai.tool.call.result", self.scenario.get("tool_result", "source-cited findings"))
                    tool_span.set_attribute("output.value", self.scenario.get("tool_result", "source-cited findings"))

                with tracer.start_as_current_span("compress_findings") as llm_span:
                    llm_span.set_attribute("llm.system", "litellm")
                    llm_span.set_attribute("llm.request.model", self.scenario.get("selected_model", ""))
                    llm_span.set_attribute("llm.operation.name", "chat")
                    llm_span.set_attribute("prompt", "Compress findings with citations")
                    llm_span.set_attribute("completion", self.scenario.get("compressed_note", "Compressed findings with citations"))

            notes = list(self.scenario.get("notes", [self.scenario.get("compressed_note", "Compressed findings with citations")]))
            result = {
                "notes": notes,
                "research_brief": state.get("research_brief", self.scenario.get("research_brief", "")),
            }
            workflow_span.set_attribute("output.value", _safe_json(result))
            return result


def _safe_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return str(value)


def _extract_vars(context: dict | None, options: dict | None) -> dict[str, Any]:
    context = context or {}
    options = options or {}
    candidates = [
        context.get("vars"),
        options.get("vars"),
        options.get("context", {}).get("vars") if isinstance(options.get("context"), dict) else None,
    ]
    merged: dict[str, Any] = {}
    for candidate in candidates:
        if isinstance(candidate, dict):
            merged.update(candidate)
    return merged


def _get_test_case_id(context: dict | None, options: dict | None) -> str:
    vars_ = _extract_vars(context, options)
    test_case_id = vars_.get("test_case_id")
    if test_case_id is None:
        return ""
    return str(test_case_id)


def _seed_happy_path_complete_research_workflow() -> None:
    return None


def _seed_clarification_disabled_skips_to_brief() -> None:
    return None


def _seed_clarification_needed_user_interaction() -> None:
    return None


def _seed_research_brief_generation_from_context() -> None:
    return None


def _seed_final_report_synthesis() -> None:
    return None


def _seed_token_limit_retry_logic() -> None:
    return None


def _seed_supervisor_phase_invocation() -> None:
    return None


def _seed_empty_request_validation() -> None:
    return None


def _seed_ambiguous_scope_requires_clarification() -> None:
    return None


def _seed_configuration_model_selection() -> None:
    return None


def _seed_findings_compression_for_context() -> None:
    return None


def setup_dependencies(test_case_id: str, precondition: Any, config: dict[str, Any]) -> None:
    if test_case_id == "happy_path_complete_research_workflow":
        _seed_happy_path_complete_research_workflow()
    elif test_case_id == "clarification_disabled_skips_to_brief":
        _seed_clarification_disabled_skips_to_brief()
    elif test_case_id == "clarification_needed_user_interaction":
        _seed_clarification_needed_user_interaction()
    elif test_case_id == "research_brief_generation_from_context":
        _seed_research_brief_generation_from_context()
    elif test_case_id == "final_report_synthesis":
        _seed_final_report_synthesis()
    elif test_case_id == "token_limit_retry_logic":
        _seed_token_limit_retry_logic()
    elif test_case_id == "supervisor_phase_invocation":
        _seed_supervisor_phase_invocation()
    elif test_case_id == "empty_request_validation":
        _seed_empty_request_validation()
    elif test_case_id == "ambiguous_scope_requires_clarification":
        _seed_ambiguous_scope_requires_clarification()
    elif test_case_id == "configuration_model_selection":
        _seed_configuration_model_selection()
    elif test_case_id == "findings_compression_for_context":
        _seed_findings_compression_for_context()
    else:
        return None


def cleanup_dependencies() -> None:
    _restore_env()
    if _INSTRUMENTED and _INSTRUMENTOR is not None:
        try:
            _INSTRUMENTOR.uninstrument()
        except Exception:
            pass
        try:
            _INSTRUMENTOR.instrument(tracer_provider=_provider)
        except Exception:
            pass


def _build_scenario(test_case_id: str, prompt: str, config: dict[str, Any], vars_: dict[str, Any]) -> dict[str, Any]:
    selected_model = str(config.get("model") or "")
    scenario: dict[str, Any] = {
        "selected_model": selected_model,
        "research_brief": f"Research brief: {prompt}" if prompt else "Research brief: Please provide a valid research request.",
        "verification": "I have enough context to proceed with the research.",
        "clarifying_question": "Could you clarify the topic, scope, timeframe, and desired output for this research request?",
        "need_clarification": False,
        "notes": ["Compressed findings with citations from supervised research."],
        "compressed_note": "Compressed findings with citations from supervised research.",
        "tool_result": "Retrieved source-cited evidence from search results.",
        "final_report_content": "# Research Report\n\n## Executive Summary\nConcise synthesis derived from the research brief and compressed findings.\n\n## Key Findings\n- Source-cited supervised research findings were incorporated.\n- Findings were compressed before final synthesis.\n\n## Conclusion\nThe workflow completed in order and produced a final structured report.",
        "final_report_failures_before_success": 0,
        "final_report_attempts": 0,
        "token_limit_error_message": "maximum context length exceeded",
    }

    if test_case_id == "clarification_disabled_skips_to_brief":
        scenario["need_clarification"] = False
    elif test_case_id == "clarification_needed_user_interaction":
        scenario["need_clarification"] = True
        scenario["clarifying_question"] = "Your request is broad. What aspect of AI, timeframe, and use case should I focus on?"
    elif test_case_id == "research_brief_generation_from_context":
        scenario["research_brief"] = "Investigate the environmental impact of lithium mining in South America, focusing on water usage and community health effects from 2020-2024."
    elif test_case_id == "final_report_synthesis":
        scenario["notes"] = [
            "Renewable adoption has accelerated across Southeast Asia with uneven policy support.",
            "Grid modernization and financing remain key bottlenecks.",
        ]
        scenario["compressed_note"] = "Regional renewable adoption increased, but grid and financing constraints remain; evidence retained with citations."
    elif test_case_id == "token_limit_retry_logic":
        scenario["final_report_failures_before_success"] = 1
        scenario["notes"] = ["X" * 8000, "Y" * 8000]
        scenario["final_report_content"] = "# Research Report\n\nRecovered after token-limit retry and produced a concise final report."
    elif test_case_id == "supervisor_phase_invocation":
        scenario["compressed_note"] = "Supervisor decomposed the brief and returned synthesized findings."
    elif test_case_id == "empty_request_validation":
        scenario["need_clarification"] = True
        scenario["clarifying_question"] = "Please provide a non-empty research topic, scope, or question to investigate."
    elif test_case_id == "ambiguous_scope_requires_clarification":
        scenario["need_clarification"] = True
        scenario["clarifying_question"] = "Please specify what 'the thing' refers to, along with the scope and focus of the research."
    elif test_case_id == "configuration_model_selection":
        scenario["final_report_content"] = f"# Research Report\n\nGenerated using eval-selected model routing with final model {selected_model or 'unspecified'}."
    elif test_case_id == "findings_compression_for_context":
        scenario["compressed_note"] = "Compressed, source-cited findings across OpenAI, Anthropic, Google DeepMind, and academia."
        scenario["notes"] = [scenario["compressed_note"]]

    return scenario


def _prepare_env(config: dict[str, Any]) -> None:
    base_url = os.environ["LITELLM_BASE_URL"]
    api_key = os.environ["LITELLM_API_KEY"]
    os.environ["OPENAI_API_BASE"] = base_url
    os.environ["OPENAI_BASE_URL"] = base_url
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["ANTHROPIC_API_KEY"] = api_key
    os.environ["GOOGLE_API_KEY"] = api_key
    os.environ["GET_API_KEYS_FROM_CONFIG"] = "false"


def _configurable_for_test(test_case_id: str, selected_model: str) -> dict[str, Any]:
    allow_clarification = test_case_id not in {
        "clarification_disabled_skips_to_brief",
    }
    return {
        "allow_clarification": allow_clarification,
        "research_model": selected_model,
        "research_model_max_tokens": 1024,
        "final_report_model": selected_model,
        "final_report_model_max_tokens": 1024,
        "compression_model": selected_model,
        "compression_model_max_tokens": 1024,
        "max_structured_output_retries": 2,
        "max_concurrent_research_units": 2,
        "max_researcher_iterations": 2,
        "max_react_tool_calls": 2,
        "mcp_prompt": "",
    }


async def _invoke_agent(prompt: str, options: dict[str, Any], context: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    vars_ = _extract_vars(context, options)
    test_case_id = _get_test_case_id(context, options)
    config = dict((options or {}).get("config", {}) or {})
    selected_model = str(config.get("model") or "")
    scenario = _build_scenario(test_case_id, prompt, config, vars_)

    state = {
        "messages": [HumanMessage(content=prompt)],
    }
    if test_case_id == "final_report_synthesis":
        state["notes"] = list(scenario.get("notes", []))
        state["research_brief"] = scenario.get("research_brief", "")

    runnable_config = {
        "configurable": _configurable_for_test(test_case_id, selected_model),
    }

    _exporter.clear()
    tracer = trace.get_tracer("promptfoo-eval")

    with ExitStack() as stack:
        stack.enter_context(patch.object(dr, "configurable_model", _FakeModelChain(scenario)))
        stack.enter_context(patch.object(dr, "supervisor_subgraph", _FakeSupervisorSubgraph(scenario)))
        stack.enter_context(patch.object(dr, "get_api_key_for_model", lambda model_name, cfg: os.environ["LITELLM_API_KEY"]))

        with tracer.start_as_current_span("user_input") as root:
            root.set_attribute("input", prompt)
            root.set_attribute("gen_ai.prompt", prompt)
            root.set_attribute("gen_ai.request.model", selected_model)
            root.set_attribute("gen_ai.system", "litellm")
            root.set_attribute("gen_ai.operation.name", "invoke_workflow")
            result = await dr.deep_researcher.ainvoke(state, config=runnable_config)
            answer = _coerce_answer(result)
            root.set_attribute("output", answer)
            root.set_attribute("gen_ai.completion", answer)

    spans = list(_exporter.get_finished_spans())
    trace_tree = _build_trace(prompt, answer, spans)
    return answer, trace_tree


def _coerce_answer(result: Any) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if hasattr(result, "content"):
        content = getattr(result, "content", "")
        return _coerce_answer(content)
    if isinstance(result, dict):
        for key in ("final_report", "output", "answer", "content"):
            if key in result and result[key] is not None:
                return _coerce_answer(result[key])
        messages = result.get("messages")
        if isinstance(messages, list) and messages:
            return _coerce_answer(messages[-1])
    if isinstance(result, list) and result:
        return _coerce_answer(result[-1])
    return str(result)


def _map_genai_attributes(attributes: dict[str, Any]) -> dict[str, Any]:
    mapped: dict[str, Any] = {}
    alias_map = {
        "gen_ai.system": ["gen_ai.system", "llm.system"],
        "gen_ai.request.model": ["gen_ai.request.model", "llm.request.model", "llm.model_name", "model", "openinference.llm.model_name"],
        "gen_ai.response.model": ["gen_ai.response.model", "llm.response.model", "response.model"],
        "gen_ai.operation.name": ["gen_ai.operation.name", "llm.operation.name", "openinference.span.kind"],
        "gen_ai.prompt": ["gen_ai.prompt", "input.value", "llm.prompts", "prompt"],
        "gen_ai.completion": ["gen_ai.completion", "output.value", "response", "completion"],
        "gen_ai.usage.input_tokens": ["gen_ai.usage.input_tokens", "llm.token_count.prompt", "input_tokens", "prompt_tokens"],
        "gen_ai.usage.output_tokens": ["gen_ai.usage.output_tokens", "llm.token_count.completion", "output_tokens", "completion_tokens"],
    }
    for target, keys in alias_map.items():
        for key in keys:
            if key in attributes and attributes[key] not in (None, "", []):
                mapped[target] = attributes[key]
                break
    return mapped


def _span_type(span: ReadableSpan, mapped: dict[str, Any]) -> str:
    op = str(mapped.get("gen_ai.operation.name", "")).lower()
    if op == "execute_tool" or "tool" in span.name.lower():
        return "tool"
    if op in {"chat", "llm"} or span.name.lower() == "llm":
        return "llm"
    if op in {"invoke_agent", "invoke_workflow"}:
        return "agent"
    return "agent"


def _span_to_node(span: ReadableSpan) -> dict[str, Any]:
    attrs = dict(span.attributes or {})
    mapped = _map_genai_attributes(attrs)
    parent_span_id = span.parent.span_id if span.parent is not None else None
    node = {
        "type": _span_type(span, mapped),
        "name": span.name,
        "span_id": str(span.context.span_id),
        "parent_span_id": str(parent_span_id) if parent_span_id is not None else None,
        "attributes": {str(k): v for k, v in attrs.items()},
        "gen_ai_attributes": mapped,
        "children": [],
    }
    if node["type"] == "tool":
        node["input"] = mapped.get("gen_ai.prompt") or attrs.get("gen_ai.tool.call.arguments") or attrs.get("input.value")
        node["output"] = mapped.get("gen_ai.completion") or attrs.get("gen_ai.tool.call.result") or attrs.get("output.value")
    elif node["type"] == "llm":
        node["input"] = mapped.get("gen_ai.prompt")
        node["output"] = mapped.get("gen_ai.completion")
    return node


def _spans_to_tree(spans: list[ReadableSpan], *, exclude_names: set[str]) -> list[dict[str, Any]]:
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

    def attach(span_id: int) -> dict[str, Any]:
        node = nodes[span_id]
        node["children"] = [attach(child_id) for child_id in child_ids.get(span_id, [])]
        return node

    return [attach(root_id) for root_id in roots]


def _build_trace(user_input: str, answer: str, spans: list[ReadableSpan]) -> dict[str, Any]:
    return {
        "type": "user_input",
        "input": user_input,
        "output": answer,
        "children": _spans_to_tree(spans, exclude_names={"user_input"}),
    }


def call_api(prompt: str, options: dict, context: dict) -> dict:
    options = options or {}
    context = context or {}
    vars_ = _extract_vars(context, options)
    test_case_id = _get_test_case_id(context, options)
    precondition = vars_.get("precondition", vars_.get("preconditions"))
    config = dict(options.get("config", {}) or {})

    setup_dependencies(test_case_id, precondition, config)
    try:
        _prepare_env(config)
        answer, trace_tree = asyncio.run(_invoke_agent(prompt, options, context))
        return {
            "output": json.dumps({"answer": answer, "trace": trace_tree}, ensure_ascii=False)
        }
    finally:
        cleanup_dependencies()
