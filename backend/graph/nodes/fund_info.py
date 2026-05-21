import requests
import pandas as pd
import json
from langchain_openai import ChatOpenAI
from graph.state import AgentState
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0.2)


def fetch_nav(fund_name: str, retries: int = 0) -> dict:
    print(f"fetch_nav for {fund_name}")
    try:
        results = requests.get(
            f"https://api.mfapi.in/mf/search?q={fund_name}", timeout=10
        ).json()
    except Exception as e:
        return {"error": str(e)}

    if not results:
        if retries >= 2:
            return {"error": f"Fund not found after 3 attempts: {fund_name}"}

        print(f"Fund '{fund_name}' not found. Using AI to suggest corrected name...")
        prompt = (
            f"The search for Indian mutual fund '{fund_name}' failed. "
            "Provide the exact official/legal fund name as registered with AMFI/MFAPI. "
            "Examples: 'IDFC' → 'Bandhan', replace short names with full names. "
            "Return ONLY the exact fund name — no quotes, markdown, or extra text."
        )
        corrected_name = llm.invoke(prompt).content.strip()

        if corrected_name.lower() == fund_name.lower():
            words = fund_name.split()
            corrected_name = " ".join(words[:2]) if len(words) > 2 else None
            if not corrected_name:
                return {"error": f"Fund not found after AI fallback: {fund_name}"}

        return fetch_nav(corrected_name, retries + 1)

    code = results[0]["schemeCode"]
    try:
        data = requests.get(f"https://api.mfapi.in/mf/{code}", timeout=10).json()
    except Exception as e:
        return {"error": str(e)}

    df = pd.DataFrame(data["data"])
    df["nav"] = df["nav"].astype(float)

    def safe_nav(idx):
        return df["nav"].iloc[min(idx, len(df) - 1)] if len(df) > 1 else df["nav"].iloc[0]

    return {
        "name":        data["meta"]["scheme_name"],
        "fund_house":  data["meta"]["fund_house"],
        "category":    data["meta"]["scheme_type"],
        "latest_nav":  df["nav"].iloc[0],
        "nav_1y_ago":  safe_nav(252),
        "nav_3y_ago":  safe_nav(756),
        "nav_5y_ago":  safe_nav(1260),
    }


def fund_info_node(state: AgentState) -> dict:
    funds = state.get("fund_names", [])
    active_tools = state.get("next_agents") or []

    if not funds:
        msg = "No specific fund names provided. Could not fetch fund info."
        return {
            "tool_result":  msg,
            "tool_results": {"fund_info": msg},
        }

    prev_results = state.get("tool_results", {})
    context_str = ""
    if state.get("has_sequential") and prev_results:
        filtered = {k: v for k, v in prev_results.items() if k != "fund_info"}
        if filtered:
            context_str = f"Context from previous tools:\n{filtered}\n\n"

    extracted_data = []
    for fund in funds:
        data = fetch_nav(fund)
        print(f"Data for {fund}:", data)

        if "error" in data:
            extracted_data.append({"fund": fund, "error": data["error"]})
            continue

        def gain(nav_key):
            nav_ago = data.get(nav_key)
            if nav_ago and nav_ago > 0:
                return round(((data["latest_nav"] - nav_ago) / nav_ago) * 100, 2)
            return None

        entry = {
            "name":          data["name"],
            "house":         data["fund_house"],
            "category":      data["category"],
            "latest_nav":    data["latest_nav"],
            "1_year_gain":   f"{gain('nav_1y_ago')}%" if gain("nav_1y_ago") is not None else "N/A",
            "3_year_gain":   f"{gain('nav_3y_ago')}%" if gain("nav_3y_ago") is not None else "N/A",
            "5_year_gain":   f"{gain('nav_5y_ago')}%" if gain("nav_5y_ago") is not None else "N/A",
        }
        extracted_data.append(entry)

    data_dump = json.dumps(extracted_data, indent=2)
    query = state["messages"][-1].content

    sip_scope_guard = ""
    if "sip_calculator" in active_tools or "lumpsum_calculator" in active_tools:
        sip_scope_guard = (
            "- Do NOT provide SIP plans, lump sum plans, or investment allocation advice — "
            "those are handled by separate calculator agents.\n"
        )

    PROMPT = f"""
    {context_str}
    You are a mutual fund data presenter for Indian investors.
    The user asked: "{query}"

    Here is the real, fetched data:
    {data_dump}

    Your job:
    - Answer the user's EXACT question using ONLY this data.
    - Present NAV, 1-year, 3-year, and 5-year returns clearly.
    - If multiple funds, use a table for easy comparison.
    - If the user asked for something specific (e.g., only NAV), show only that.
    - Do NOT generate unsolicited paragraphs — be concise and direct.
    - Use Indian number formatting (₹).
    {sip_scope_guard}
    """
    response = llm.invoke(PROMPT)
    final_result = response.content

    return {
        "tool_result":  final_result,
        "tool_results": {"fund_info": final_result},
    }
