from pydantic import SecretStr

from kairorag.cloud.agent.planner import LLMToolPlanner
from kairorag.cloud.agent.schemas import AgentState
from kairorag.cloud.retriever import CloudReadChunk
from kairorag.config import KairoCloudSettings
from kairorag.providers.llm import LLMResponse, LLMToolCall, LLMToolSpec


class FakeLLMProvider:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, messages, *, temperature=0.0, response_format=None):
        raise AssertionError("planner 不应调用 complete")

    def complete_with_tools(self, messages, tools, *, temperature=0.0):
        self.calls.append((messages, tools, temperature))
        return self.responses.pop(0)


class FakeRegistry:
    def specs(self):
        return [LLMToolSpec("hybrid_search", "检索", {"type": "object"})]


def _settings():
    return KairoCloudSettings(
        openai_api_key=SecretStr("placeholder-openai"),
        qdrant_url="https://qdrant.example.invalid",
        qdrant_api_key=SecretStr("placeholder-qdrant"),
    )


def test_planner_returns_llm_tool_call_and_counts_round():
    tool_call = LLMToolCall("call-1", "hybrid_search", {"query": "RAG"})
    planner = LLMToolPlanner(
        _settings(),
        FakeLLMProvider([LLMResponse(content="检索", model="fake", usage={}, tool_calls=[tool_call])]),
        FakeRegistry(),
    )
    state = AgentState(query="RAG")

    calls = planner.next_tool_calls(state)

    assert calls == [tool_call]
    assert state.metrics["llm_planning_rounds"] == 1
    assert state.trace[-1].step == "llm_planning"


def test_planner_auto_repairs_to_hybrid_without_read_chunks():
    planner = LLMToolPlanner(
        _settings(),
        FakeLLMProvider([LLMResponse(content="无工具", model="fake", usage={})]),
        FakeRegistry(),
    )
    state = AgentState(query="RAG")

    calls = planner.next_tool_calls(state)

    assert calls[0].name == "hybrid_search"
    assert state.trace[-1].step == "planner_auto_repair"


def test_planner_auto_repairs_to_generate_answer_with_read_chunks():
    planner = LLMToolPlanner(
        _settings(),
        FakeLLMProvider([LLMResponse(content="无工具", model="fake", usage={})]),
        FakeRegistry(),
    )
    state = AgentState(query="RAG")
    state.read_chunks["c1"] = CloudReadChunk("c1", "d1", "标题", "完整文本", {})

    calls = planner.next_tool_calls(state)

    assert calls[0].name == "generate_grounded_answer"


def test_planner_marks_non_dict_arguments_as_invalid():
    bad_call = LLMToolCall("call-1", "hybrid_search", "不是 dict")  # type: ignore[arg-type]
    planner = LLMToolPlanner(
        _settings(),
        FakeLLMProvider([LLMResponse(content="", model="fake", usage={}, tool_calls=[bad_call])]),
        FakeRegistry(),
    )
    state = AgentState(query="RAG")

    calls = planner.next_tool_calls(state)

    assert calls[0].name == "__invalid_tool_arguments__"
