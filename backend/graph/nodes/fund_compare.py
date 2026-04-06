from graph.nodes.fund_info import fetch_nav
from langchain_openai import ChatOpenAI
from graph.state import AgentState
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0.2)


def compare_fund_node(state: AgentState) -> dict:
    print("compare_fund_node")

    fund_names = state.get("fund_names", [])
    if len(fund_names) < 2:
        msg = "Please mention at least two fund names to compare."
        return {"tool_result": msg, "tool_results": {"fund_compare": msg}}

    fund_data = []
    errors = []
    for f in fund_names:
        data = fetch_nav(fund_name=f)
        if "error" in data:
            errors.append(f"Could not fetch data for '{f}': {data['error']}")
        else:
            fund_data.append(data)

    if len(fund_data) < 2:
        msg = "Not enough valid funds found to compare. " + " | ".join(errors)
        return {"tool_result": msg, "tool_results": {"fund_compare": msg}}

    def gain(data, idx):
        try:
            return round(
                ((data["latest_nav"] -
                  data[f"nav_{idx}_ago"]) / data[f"nav_{idx}_ago"]) * 100, 2
            )
        except (KeyError, ZeroDivisionError, TypeError):
            return "N/A"

    fund_details = ""
    for d in fund_data:
        fund_details += f"{d['name']} ({d['fund_house']})\nCategory: {d['category']} | NAV: ₹{d.get('latest_nav', 'N/A')}\n1Y return: {gain(d, '1y')}% | 3Y return: {gain(d, '3y')}%\n\n"

    prev_results = state.get("tool_results", {})
    context_str = ""
    if state.get("has_sequential") and prev_results:
        filtered_results = {k: v for k, v in prev_results.items() if k != "fund_compare"}
        if filtered_results:
            context_str = f"Context from previous tools:\n{filtered_results}\n\n"

    PROMPT = f"""
    {context_str}
    Compare the following mutual funds and give a clear verdict:

    {fund_details}

    Provide a comparative analysis. Which one is better for a beginner? Which for a long-term investor?
    Highlight the pros and cons based on available metrics.
    If the context mentions specific priorities (e.g. from a financial advisor), address them!
    End with a one-line overall verdict.
    """
    response = llm.invoke(PROMPT)
    return {
        "tool_result":  response.content,
        "tool_results": {"fund_compare": response.content},
    }
