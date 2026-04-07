import requests
import pandas as pd
import json
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
    funds = state.get("fund_names", [])
    active_tools = state.get("next_agents") or []
    if not funds:
        return {
            "tool_result": "No specific fund names provided. Could not fetch fund info.",
            "tool_results": {"fund_info": "No specific fund names provided. Could not fetch fund info."}
        }

    all_results = []
    
    prev_results = state.get("tool_results", {})
    context_str = ""
    if state.get("has_sequential") and prev_results:
        # Don't include ourselves in the previous context
        filtered_results = {k: v for k, v in prev_results.items() if k != "fund_info"}
        if filtered_results:
            context_str = f"Context from previous tools:\n{filtered_results}\n\n"

    extracted_data = []
    for fund in funds:
        data = fetch_nav(fund)
        print(f"Data for {fund}:", data)
        
        if "error" in data:
            extracted_data.append({"fund": fund, "error": data['error']})
            continue
            
        gain_1y = round(
            ((data["latest_nav"] - data["nav_1y_ago"]) / data["nav_1y_ago"]) * 100, 2
        )
        gain_3y = round(
            ((data["latest_nav"] - data["nav_3y_ago"]) / data["nav_3y_ago"]) * 100, 2
        )
        extracted_data.append({
            "name": data['name'],
            "house": data['fund_house'],
            "category": data['category'],
            "latest_nav": data['latest_nav'],
            "1_year_gain": f"{gain_1y}%",
            "3_year_gain": f"{gain_3y}%"
        })

    data_dump = json.dumps(extracted_data, indent=2)
    query = state["messages"][-1].content
    sip_scope_guard = ""
    if "sip_calculator" in active_tools:
        sip_scope_guard = "- If sip_calculator is also active in this run, DO NOT provide SIP plans, split suggestions, or investment allocation advice.\n"

    PROMPT = f"""
        {context_str}
        
        You are a mutual fund data presenter.
        The user specifically asked: "{query}"
        
        Here is the real, fetched data for the underlying funds:
        {data_dump}

        Your job is to answer the user's EXACT query using ONLY this data. 
        - If the user ONLY asked for NAV data, output a clean, concise list or markdown table of just the NAVs.
        - If the user asked for a summary, give a brief plain-English summary.
        - DO NOT generate a huge multi-paragraph summary for every fund unless specifically asked.
        - Adapt your presentation format (table, bullet points, or short paragraph) to match what the user actually wants.
        - Keep it readable and directly to the point.
        {sip_scope_guard}
    """
    response = llm.invoke(PROMPT)
    final_result = response.content

    return {
        "tool_result":  final_result,
        "tool_results": {"fund_info": final_result},
    }
