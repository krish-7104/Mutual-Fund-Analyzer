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
    From this text, extract the FINAL recommended winner funds only.
    Prefer explicit winners/recommended picks over broad top lists.
    If there is one winner, return one fund.
    If there are category-wise winners (e.g., one flexi-cap and one small-cap), return those winners only.
    If no explicit winners exist, then return recommendation candidates.
    Return all relevant recommended funds mentioned in the text.
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
    funds = [str(f).strip() for f in funds if str(f).strip()]
        
    winner = funds[0] if funds else ""
    print(f"[winner_extractor] extracted funds: {funds}")
    
    return {
        "fund_names": funds,   # Override so next node uses these funds
        "winner_fund": winner,
    }