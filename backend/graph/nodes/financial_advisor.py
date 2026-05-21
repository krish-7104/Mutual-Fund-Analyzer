from langchain_openai import ChatOpenAI
from graph.state import AgentState
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0.5)

ADVISOR_PROMPT = """
You are a holistic Financial Advisor for Indian investors covering mutual funds, stocks, and overall financial planning.

You help users with:
- Fund recommendations (by category, risk profile, goal)
- Stock investment strategies and sector selection
- Asset allocation (equity MF, debt MF, direct stocks, gold, etc.)
- Goal-based investing (retirement, child education, wealth creation)
- General investment strategies (SIP, lump sum, hybrid approaches)

RULES:
1. Provide actionable, specific advice based on the user's query.
2. NEVER hallucinate real-time data: no current NAVs, no specific stock prices, no exact historical returns.
3. When recommending specific funds, provide names with a brief rationale only — let data agents handle real numbers.
4. Use standard frameworks: 50/30/20 rule, 100-minus-age equity allocation, core-satellite strategy, etc.
5. Consider the user's risk profile, horizon, and goal if mentioned.
6. Keep language professional but accessible — avoid heavy jargon.
7. Always include a disclaimer at the end.
"""


def financial_advisor_node(state: AgentState) -> dict:
    print("financial_advisor_node")
    query = state["messages"][-1].content
    active_tools = state.get("next_agents") or []
    investment_type = state.get("investment_type") or "mutual_fund"

    prev_results = state.get("tool_results", {})
    context_str = ""
    if state.get("has_sequential") and prev_results:
        filtered = {k: v for k, v in prev_results.items() if k != "financial_advisor"}
        if filtered:
            context_str = (
                f"\n\nContext from previous tools:\n{filtered}\n"
                "Incorporate this context into your advice."
            )

    scope_guard = ""
    if "sip_calculator" in active_tools:
        scope_guard += (
            "\n\nScope rule: Do NOT calculate or mention specific monthly SIP amounts. "
            "Only recommend fund types/names and explain why they suit this goal."
        )
    if "lumpsum_calculator" in active_tools:
        scope_guard += (
            "\n\nScope rule: Do NOT calculate lump sum projections or corpus figures. "
            "Only recommend what to invest in and why."
        )

    type_context = ""
    if investment_type == "stock":
        type_context = "\nThe user is specifically asking about direct stock investments."
    elif investment_type == "both":
        type_context = "\nThe user is asking about both mutual funds and direct stocks — address both."

    human_message = query + type_context + context_str + scope_guard

    response = llm.invoke([
        ("system", ADVISOR_PROMPT),
        ("human", human_message),
    ])

    result = response.content
    if "disclaimer" not in result.lower() and "sebi" not in result.lower():
        result += "\n\nThis is general financial education, not personalised investment advice. Please consult a SEBI-registered advisor."

    return {
        "tool_result":  result,
        "tool_results": {"financial_advisor": result},
    }
