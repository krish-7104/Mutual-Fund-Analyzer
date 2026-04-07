from langchain_openai import ChatOpenAI
from graph.state import AgentState
from dotenv import load_dotenv
import json
load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0.2)


def _extract_holdings_from_query(query: str):
    response = llm.invoke(
        f"""
            Extract fund holdings from this message as JSON list:
            [{{"fund": "fund name", "amount": number}}, ...]
            Return ONLY the JSON, nothing else.
            Message: {query}
        """
    ).content
    cleaned = response.strip()
    cleaned = cleaned.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(cleaned)


def _normalize_holdings(holdings):
    normalized = []
    for holding in holdings:
        amount = float(str(holding["amount"]).replace(",", "").strip())
        normalized.append({"fund": holding["fund"], "amount": amount})
    return normalized


def _build_breakdown(holdings):
    total = sum(holding["amount"] for holding in holdings)
    if total <= 0:
        return total, []
    breakdown = []
    for holding in holdings:
        percentage = round((holding["amount"] / total) * 100, 1)
        breakdown.append(f"- {holding['fund']}: ₹{holding['amount']:,.0f} ({percentage}%)")
    return total, breakdown


def portfolio_node(state: AgentState) -> dict:
    holdings = state.get("portfolio", [])

    if not holdings:
        query = state["messages"][-1].content
        holdings = _extract_holdings_from_query(query)

    holdings = _normalize_holdings(holdings)
    total, breakdown = _build_breakdown(holdings)
    breakdown_text = "\n".join(breakdown) if breakdown else "- No valid holdings found."

    prompt = f"""
    The user has invested ₹{total:,.0f} across {len(holdings)} mutual funds:
    
    {breakdown_text}
    
    Give a plain English portfolio snapshot:
    1. Which fund takes the biggest share and is that too concentrated?
    2. Is there any diversification (different fund types / categories)?
    3. One friendly suggestion for improvement
    
    Keep it under 150 words. Simple language only.
    """
    response = llm.invoke(prompt)
    return {
        "tool_result":  response.content,
        "tool_results": {"portfolio": response.content},
    }
