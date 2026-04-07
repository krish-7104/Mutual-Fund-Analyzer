from langgraph.graph import StateGraph, END
from langgraph.types import Send
from graph.state import AgentState
from graph.nodes.supervisor import supervisor_node
from graph.nodes.synthesizer import synthesizer_node
from graph.nodes.winner_extractor import extract_winner_node
from graph.nodes.fund_info import fund_info_node
from graph.nodes.fund_compare import compare_fund_node
from graph.nodes.sip_calculator import sip_calculator_node
from graph.nodes.qa_search import qa_search_node
from graph.nodes.news_agent import news_node
from graph.nodes.financial_advisor import financial_advisor_node
from graph.nodes.sentiment_agent import sentiment_node
from graph.nodes.portfolio import portfolio_node
from graph.nodes.goal_tracker import goal_tracker_node
from graph.nodes.out_of_scope import out_of_scope_node

NODE_MAP = {
    "fund_info":         fund_info_node,
    "fund_compare":      compare_fund_node,
    "sip_calculator":    sip_calculator_node,
    "qa_search":         qa_search_node,
    "news":              news_node,
    "financial_advisor": financial_advisor_node,
    "sentiment":         sentiment_node,
    "portfolio":         portfolio_node,
    "goal_tracker":      goal_tracker_node,
    "out_of_scope":      out_of_scope_node,
}

ROUTE_TARGETS = {
    "extract_winner": "extract_winner",
    "synthesizer": "synthesizer",
    **{tool: tool for tool in NODE_MAP},
}


def _planned_agents(state: AgentState):
    return state.get("next_agents") or [state.get("next_agent", "qa_search")]


def _pending_tool(task_chain, tool_results):
    for step in task_chain:
        tool = getattr(step, "tool", None)
        if not tool or tool in tool_results:
            continue
        needs_previous = (
            getattr(step, "depends_on_previous", False)
            or getattr(step, "use_winner_from_previous", False)
        )
        return tool, needs_previous
    return None, False

def fan_out_router(state: AgentState):
    agents = _planned_agents(state)
    print(f"[fan_out_router] parallel dispatch: {agents}")
    return [Send(agent, state) for agent in agents]


def sequential_start_router(state: AgentState):
    agents = _planned_agents(state)
    first = agents[0]
    print(f"[sequential_start_router] starting chain with: {first}")
    return first


def after_tool_router(state: AgentState):
    if not state.get("has_sequential"):
        return "synthesizer"

    task_chain = state.get("task_chain") or []
    tool_results = state.get("tool_results", {})

    next_tool, needs_previous = _pending_tool(task_chain, tool_results)
    if next_tool:
        return "extract_winner" if needs_previous else next_tool

    return "synthesizer"


def after_extractor_router(state: AgentState):
    task_chain = state.get("task_chain") or []
    tool_results = state.get("tool_results", {})

    next_tool, _ = _pending_tool(task_chain, tool_results)
    if next_tool:
        return next_tool

    return "synthesizer"


def supervisor_exit_router(state: AgentState):
    if state.get("has_sequential"):
        return sequential_start_router(state)
    return fan_out_router(state)

# ── Graph builder ────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(AgentState)

    # Core nodes
    g.add_node("supervisor",      supervisor_node)
    g.add_node("synthesizer",     synthesizer_node)
    g.add_node("extract_winner",  extract_winner_node)

    for name, fn in NODE_MAP.items():
        g.add_node(name, fn)

    g.set_entry_point("supervisor")

    g.add_conditional_edges(
        "supervisor",
        supervisor_exit_router,
    )

    for name in NODE_MAP:
        g.add_conditional_edges(
            name,
            after_tool_router,
            ROUTE_TARGETS,
        )

    g.add_conditional_edges(
        "extract_winner",
        after_extractor_router,
        {key: value for key, value in ROUTE_TARGETS.items() if key != "extract_winner"},
    )

    g.add_edge("synthesizer", END)

    return g.compile()

graph = build_graph()