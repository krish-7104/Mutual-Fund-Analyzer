from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import json

llm = ChatOpenAI(model="gpt-4o", temperature=0)

def extract_winner_node(state: dict) -> dict:
    tool_results = state.get("tool_results", {})
    if not tool_results:
        return {}
    
    last_content = list(tool_results.values())[0]

    prompt = f"""
    From this text, extract ALL mutual fund names mentioned as recommendations or winners.
    If it's comparing and declares one winner, extract that one.
    If it lists top 5 funds, extract all 5.
    Return ONLY a JSON array of strings (e.g. ["Fund A", "Fund B"]). Do NOT return markdown or backticks.
    
    Text:
    {last_content}
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content.strip()
    if content.startswith("```json"):
        content = content[7:-3]
    elif content.startswith("```"):
        content = content[3:-3]
        
    try:
        funds = json.loads(content)
        if not isinstance(funds, list):
            funds = [content]
    except:
        funds = [content]
        
    winner = funds[0] if funds else ""
    print(f"[winner_extractor] extracted funds: {funds}")
    
    return {
        "fund_names": funds,   # Override so next node uses these funds
        "winner_fund": winner,
    }