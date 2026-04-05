from langchain_openai import ChatOpenAI
from graph.state import AgentState
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0.2)


def portfolio_node(state: AgentState) -> dict:
    holdings = state.get("portfolio", [])

    if not holdings:
        query = state["messages"][-1].content
        extract = llm.invoke(f"""
            Extract fund holdings from this message as JSON list:
            [{{"fund": "fund name", "amount": number}}, ...]
            Return ONLY the JSON, nothing else.
            Message: {query}
        """).content
        import json
        holdings = json.loads(extract.strip().strip("```json").strip("```"))

    for h in holdings:
        h["amount"] = float(str(h["amount"]).replace(",", "").strip())

    total = sum(h["amount"] for h in holdings)

    breakdown = []
    for h in holdings:
        percentage = round((h["amount"] / total) * 100, 1)
        breakdown.append(f"- {h['fund']}: ₹{h['amount']:,.0f} ({percentage}%)")

    prompt = f"""
    The user has invested ₹{total:,.0f} across {len(holdings)} mutual funds:
    
    {"\n".join(breakdown)}
    
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
