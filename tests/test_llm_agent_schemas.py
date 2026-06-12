from dataclasses import asdict

from kairorag.cloud.agent.schemas import AgentState, AgentTraceStep


def test_agent_state_initializes_default_metrics():
    state = AgentState(query="哪些岗位要求 RAG？")

    assert state.search_results == {}
    assert state.read_chunks == {}
    assert state.metrics["tool_call_count"] == 0
    assert state.metrics["invalid_tool_call_count"] == 0
    assert state.metrics["max_tool_calls_reached"] is False


def test_agent_trace_step_is_serializable():
    step = AgentTraceStep(step="tool_call", detail={"name": "hybrid_search", "chunk_ids": ["c1"]})

    assert asdict(step)["detail"]["chunk_ids"] == ["c1"]
