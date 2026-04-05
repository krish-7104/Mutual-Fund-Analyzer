from langgraph.graph import StateGraph, END
from langgraph.types import Send
from langgraph.checkpoint.memory import MemorySaver
from graph.state import AgentState
from graph.nodes.supervisor import supervisor_node
from graph.nodes.synthesizer import synthesizer_node
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
    "fund_info":      fund_info_node,
    "fund_compare":   compare_fund_node,
    "sip_calculator": sip_calculator_node,
    "qa_search":      qa_search_node,
    "news":              news_node,
    "financial_advisor": financial_advisor_node,
    "sentiment":         sentiment_node,
    "portfolio":         portfolio_node,
    "goal_tracker":      goal_tracker_node,
    "out_of_scope":      out_of_scope_node,
}


def fan_out_router(state: AgentState):
    agents = state.get("next_agents") or [state.get("next_agent", "qa_search")]
    print(f"[fan_out_router] dispatching to: {agents}")
    return [Send(agent, state) for agent in agents]


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("supervisor",  supervisor_node)
    g.add_node("synthesizer", synthesizer_node)

    for name, fn in NODE_MAP.items():
        g.add_node(name, fn)

    g.set_entry_point("supervisor")

    g.add_conditional_edges(
        "supervisor",
        fan_out_router,      
    )

    for name in NODE_MAP:
        g.add_edge(name, "synthesizer")

    g.add_edge("synthesizer", END)

    return g.compile()


graph = build_graph()
