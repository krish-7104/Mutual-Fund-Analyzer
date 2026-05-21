from langchain_openai import ChatOpenAI
from graph.state import AgentState
from dotenv import load_dotenv
import json
load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0.2)


def _extract_holdings_from_query(query: str):
    response = llm.invoke(
        f"""
        Extract all investment holdings from this message.
        Holdings can be mutual funds OR stocks.
        Return a JSON list: [{{"name": "fund/stock name", "type": "mutual_fund" or "stock", "amount": number}}, ...]
        Return ONLY the JSON, nothing else.
        Message: {query}
        """
    ).content.strip()
    response = response.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(response)


def _normalize_holdings(holdings):
    normalized = []
    for h in holdings:
        try:
            amount = float(str(h.get("amount", 0)).replace(",", "").strip())
            normalized.append({
                "name":   h.get("name") or h.get("fund", "Unknown"),
                "type":   h.get("type", "mutual_fund"),
                "amount": amount,
            })
        except (ValueError, TypeError):
            continue
    return normalized


def _build_breakdown(holdings):
    total = sum(h["amount"] for h in holdings)
    if total <= 0:
        return total, [], 0, 0
    mf_total = sum(h["amount"] for h in holdings if h.get("type") != "stock")
    stock_total = sum(h["amount"] for h in holdings if h.get("type") == "stock")
    breakdown = []
    for h in holdings:
        pct = round((h["amount"] / total) * 100, 1)
        tag = "[Stock]" if h.get("type") == "stock" else "[MF]"
        breakdown.append(f"- {h['name']} {tag}: ₹{h['amount']:,.0f} ({pct}%)")
    return total, breakdown, mf_total, stock_total


def portfolio_node(state: AgentState) -> dict:
    print("portfolio_node")
    holdings = state.get("portfolio", [])

    if not holdings:
        query = state["messages"][-1].content
        try:
            holdings = _extract_holdings_from_query(query)
        except Exception as e:
            print(f"[portfolio] extraction error: {e}")
            holdings = []

    holdings = _normalize_holdings(holdings)
    total, breakdown, mf_total, stock_total = _build_breakdown(holdings)
    breakdown_text = "\n".join(breakdown) if breakdown else "- No valid holdings found."

    has_stocks = stock_total > 0
    has_mf = mf_total > 0

    prev_results = state.get("tool_results", {})
    context_str = ""
    if state.get("has_sequential") and prev_results:
        filtered = {k: v for k, v in prev_results.items() if k != "portfolio"}
        if filtered:
            context_str = f"Context from previous tools:\n{filtered}\n\n"

    query = state["messages"][-1].content

    PROMPT = f"""
    {context_str}
    You are a portfolio analyst for Indian investors.
    User asked: "{query}"

    The user holds a total of ₹{total:,.0f} across {len(holdings)} investments:

    {breakdown_text}

    {'Mutual Fund total: ₹' + f'{mf_total:,.0f}' + f' ({round(mf_total/total*100, 1)}%)' if has_mf else ''}
    {'Stock/Equity total: ₹' + f'{stock_total:,.0f}' + f' ({round(stock_total/total*100, 1)}%)' if has_stocks else ''}

    Provide a portfolio snapshot covering:
    1. **Concentration**: Is any single holding taking too large a share (>40%)?
    2. **Diversification**: Are different asset classes / sectors / fund categories represented?
    3. {'**MF vs Stock balance**: Is the ratio between mutual funds and direct stocks appropriate?' if has_stocks and has_mf else ''}
    4. **Top suggestion**: One clear, specific improvement recommendation.
    5. **Risk level**: Overall portfolio risk — Conservative / Moderate / Aggressive.

    Keep under 200 words. Use simple language. Use Indian number formatting.
    """
    response = llm.invoke(PROMPT)
    return {
        "tool_result":  response.content,
        "tool_results": {"portfolio": response.content},
    }
