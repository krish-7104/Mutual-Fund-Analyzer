from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from graph.state import AgentState
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
web = TavilySearchResults(max_results=3)

SEARCH_TRIGGERS = [
    "current", "latest", "today",  "2025", str(datetime.now().year),
    "budget", "sebi", "rule change", "interest rate", "recently"
]


def qa_search_node(state: AgentState) -> dict:
    print("qa_search_node")
    query = state["messages"][-1].content
    needs_search = any(t in query.lower() for t in SEARCH_TRIGGERS)

    context = ""
    if needs_search:
        results = web.invoke(query + " India mutual fund")
        context = "\n".join(r["content"][:250] for r in results)

    prev_results = state.get("tool_results", {})
    context_str = ""
    if state.get("has_sequential") and prev_results:
        filtered_results = {k: v for k, v in prev_results.items() if k != "qa_search"}
        if filtered_results:
            context_str = f"Context from previous tools:\n{filtered_results}\nPlease incorporate this context into your educational answer if relevant!"

    PROMPT = f"""
    You are a friendly mutual fund educator for Indian investors.
    Answer this question in plain, simple language with a real-life example.
    Always end with: "For personalised advice, consult a SEBI-registered advisor."
    
    {context_str}
    
    {"Latest info from web:\n" + context if context else ""}
    
    Question: {query}
    """
    response = llm.invoke(PROMPT)
    return {
        "tool_result":  response.content,
        "tool_results": {"qa_search": response.content},
    }
