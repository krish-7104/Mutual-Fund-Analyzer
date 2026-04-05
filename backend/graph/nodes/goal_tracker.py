from langchain_openai import ChatOpenAI
from graph.state import AgentState
import json
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0.2)


def goal_tracker_node(state: AgentState) -> dict:
    print("goal_tracker_node")
    query = state["messages"][-1].content

    extract = llm.invoke(f"""
        Extract from this message:
        - monthly_sip: how much they invest per month (number)
        - target_amount: what they want to reach (number)  
        - years_left: how many years remain (number)
        - assumed_return: annual return % (use 12 if not mentioned)
        
        Return ONLY JSON: {{"monthly_sip": X, "target": Y, "years": Z, "rate": R}}
        Message: {query}
    """).content
    p = json.loads(extract.strip().strip("```json").strip("```"))

    r = (p["rate"] / 100) / 12
    n = p["years"] * 12
    projected = p["monthly_sip"] * (((1 + r) ** n - 1) / r) * (1 + r)
    projected = round(projected)
    shortfall = p["target"] - projected
    on_track = projected >= p["target"]

    status = "on track" if on_track else "behind target"
    gap_text = (
        f"You will exceed your goal by ₹{abs(shortfall):,}."
        if on_track
        else f"You are short by ₹{abs(shortfall):,}."
    )

    prompt = f"""
    The user invests ₹{p['monthly_sip']:,}/month and wants to reach ₹{p['target']:,} in {p['years']} years.
    At {p['rate']}% assumed annual return, their projected corpus is ₹{projected:,}.
    Status: {status}. {gap_text}
    
    Give a friendly 3-line response:
    1. Tell them if they are on track or not (use simple language)
    2. If behind, suggest a small increase in monthly SIP to close the gap
    3. Encourage them regardless
    
    End with: "Projections assume steady returns which may vary in practice."
    """
    response = llm.invoke(prompt)
    return {
        "tool_result":  response.content,
        "tool_results": {"goal_tracker": response.content},
    }
