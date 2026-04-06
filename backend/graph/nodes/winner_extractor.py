from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(model="gpt-4o", temperature=0)

def extract_winner_node(state: dict) -> dict:
    """
    Reads fund_compare output and extracts the recommended fund name.
    Only runs when next step has use_winner_from_previous=True.
    """
    compare_result = state.get("tool_results", {}).get("fund_compare", "")
    if not compare_result:
        return {}

    prompt = f"""
    From this fund comparison result, extract ONLY the name of the recommended/better fund.
    Return JUST the fund name, nothing else.
    
    Comparison result:
    {compare_result}
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    winner = response.content.strip()
    print(f"[winner_extractor] winner fund: {winner}")
    
    return {
        "fund_names": [winner],   # Override so sip_calculator uses winner
        "winner_fund": winner,
    }