import requests
import pandas as pd
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from graph.state import AgentState
from dotenv import load_dotenv
load_dotenv()


llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

def fetch_nav(fund_name: str, retries: int = 0) -> dict:
    print(f"fetch_nav for {fund_name}")
    try:
        results = requests.get(
            f"https://api.mfapi.in/mf/search?q={fund_name}"
        ).json()
    except Exception as e:
        return {"error": str(e)}

    if not results:
        if retries >= 2:
            return {"error": f"Fund not found after 3 attempts: {fund_name}"}
        
        print(f"Fund '{fund_name}' not found. Using AI to retry with explicit official name...")
        prompt = (
            f"The search for Indian mutual fund '{fund_name}' failed to match the API registry. "
            "Please provide the exact official/legal name of this fund (e.g. replacing 'IDFC' with 'Bandhan', or adding 'Direct Plan Growth'). "
            "Return ONLY the exact fund name without quotes, markdown, or extra text."
        )
        corrected_name = llm.invoke(prompt).content.strip()
        
        # Guard against loop
        if corrected_name.lower() == fund_name.lower():
            words = fund_name.split()
            if len(words) > 2:
                corrected_name = " ".join(words[:2])
            else:
                return {"error": f"Fund not found after AI fallback: {fund_name}"}

        return fetch_nav(corrected_name, retries + 1)

    code = results[0]["schemeCode"]
    data = requests.get(f"https://api.mfapi.in/mf/{code}").json()
    df = pd.DataFrame(data["data"])
    df["nav"] = df["nav"].astype(float)
    return {
        "name":       data["meta"]["scheme_name"],
        "fund_house": data["meta"]["fund_house"],
        "category":   data["meta"]["scheme_type"],
        "latest_nav": df["nav"].iloc[0],
        "nav_1y_ago": df["nav"].iloc[min(252, len(df)-1)],
        "nav_3y_ago": df["nav"].iloc[min(756, len(df)-1)],
    }


def fund_info_node(state: AgentState) -> dict:
    fund = state["fund_names"][0]
    data = fetch_nav(fund)
    print(data)
    
    if "error" in data:
        msg = data["error"]
        return {"tool_result": msg, "tool_results": {"fund_info": msg}}
        
    gain_1y = round(
        ((data["latest_nav"] - data["nav_1y_ago"]) / data["nav_1y_ago"]) * 100, 2
    )
    gain_3y = round(
        ((data["latest_nav"] - data["nav_3y_ago"]) / data["nav_3y_ago"]) * 100, 2
    )
    PROMPT = f"""
        Fund: {data['name']}
        House: {data['fund_house']} | Category: {data['category']}
        Latest NAV: ₹{data['latest_nav']}
        1-year gain: {gain_1y}%
        3-year gain: {gain_3y}%
        Give a plain-English summary of this fund. Cover what kind of fund it is,
        how it has performed, and who it might suit. Keep it conversational.
    """
    response = llm.invoke(PROMPT)
    return {
        "tool_result":  response.content,
        "tool_results": {"fund_info": response.content},
    }
