from langgraph.graph import StateGraph, END
from langgraph.types import Send
from langgraph.checkpoint.memory import MemorySaver
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

# ── Routing helpers ──────────────────────────────────────────────────────────

def fan_out_router(state: AgentState):
    """Parallel: all agents run independently, no data dependency."""
    agents = state.get("next_agents") or [state.get("next_agent", "qa_search")]
    print(f"[fan_out_router] parallel dispatch: {agents}")
    return [Send(agent, state) for agent in agents]


def sequential_start_router(state: AgentState):
    """Sequential: run only the FIRST agent in the chain."""
    agents = state.get("next_agents") or ["qa_search"]
    first = agents[0]
    print(f"[sequential_start_router] starting chain with: {first}")
    return first  # simple string — single edge


def after_compare_router(state: AgentState):
    """
    After fund_compare runs:
    - If next step needs the winner → extract_winner
    - Otherwise → synthesizer (no further dependent steps)
    """
    task_chain = state.get("task_chain") or []
    # Find if any remaining step needs the winner
    needs_winner = any(
        getattr(s, "use_winner_from_previous", False)
        for s in task_chain[1:]  # skip the first (fund_compare already ran)
    )
    target = "extract_winner" if needs_winner else "synthesizer"
    print(f"[after_compare_router] → {target}")
    return target


def after_winner_router(state: AgentState):
    """After winner is extracted, route to the next dependent tool."""
    task_chain = state.get("task_chain") or []
    # Find the first step that depends on previous (it's the one waiting)
    for step in task_chain:
        tool = getattr(step, "tool", None)
        if getattr(step, "depends_on_previous", False) and tool:
            print(f"[after_winner_router] → {tool}")
            return tool
    # Fallback: nothing left, go to synthesizer
    return "synthesizer"


def supervisor_exit_router(state: AgentState):
    """Entry router: decides parallel vs sequential path."""
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

    # All tool nodes
    for name, fn in NODE_MAP.items():
        g.add_node(name, fn)

    # Entry point
    g.set_entry_point("supervisor")

    # supervisor decides to either fan out or start sequential
    g.add_conditional_edges(
        "supervisor",
        supervisor_exit_router,
    )

    # Sequential path: run first agent (fund_compare)
    # fund_compare → after_compare_router → extract_winner OR synthesizer
    g.add_conditional_edges(
        "fund_compare",
        after_compare_router,
        {
            "extract_winner": "extract_winner",
            "synthesizer":    "synthesizer",
        }
    )

    # extract_winner → next dependent tool (e.g. sip_calculator)
    g.add_conditional_edges(
        "extract_winner",
        after_winner_router,
        {tool: tool for tool in NODE_MAP},  # map every tool name to itself
    )

    # All parallel tool nodes → synthesizer
    for name in NODE_MAP:
        if name != "fund_compare":  # fund_compare has its own conditional edge
            g.add_edge(name, "synthesizer")

    g.add_edge("synthesizer", END)

    return g.compile()

graph = build_graph()