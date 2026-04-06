from langchain_openai import ChatOpenAI
from graph.state import AgentState
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0.5)

ADVISOR_PROMPT = """
You are a holistic Financial Advisor for Indian investors. 
The user is asking for financial advice, investment strategies, or fund recommendations.
Since this is a stateless, single-turn interaction, provide a comprehensive, actionable response based ONLY on the information provided in the user's query.

If they ask where to invest, suggest standard asset allocations based on typical scenarios (e.g., short-term vs long-term).
If they ask about budgeting, summarize a common rule like 50/30/20.
Address their specific question directly.

CRITICAL RULE: NEVER hallucinate or quote real-time data like current NAVs, specific historical percentage returns, or current AUM. If recommending funds, just provide the fund names and a brief rationale. Let the data nodes handle the exact real-time numbers!

Keep the language professional but accessible. Avoid jargon where possible.
Always include a clear disclaimer at the end: "This is general financial education, not personalized investment advice. Please consult a SEBI-registered advisor."
"""

def financial_advisor_node(state: AgentState) -> dict:
    print("financial_advisor_node")
    query = state["messages"][-1].content
    
    prev_results = state.get("tool_results", {})
    context_str = ""
    if state.get("has_sequential") and prev_results:
        filtered_results = {k: v for k, v in prev_results.items() if k != "financial_advisor"}
        if filtered_results:
            context_str = f"\n\nContext from previous tools:\n{filtered_results}\nPlease incorporate this context into your advice."

    human_message = query + context_str

    response = llm.invoke([
        ("system", ADVISOR_PROMPT),
        ("human", human_message)
    ])
    
    return {
        "tool_result":  response.content,
        "tool_results": {"financial_advisor": response.content},
    }
