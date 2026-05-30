from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from graph.state import AgentState
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
web = TavilySearch(max_results=4)

SEARCH_TRIGGERS = [
    "current", "latest", "today", "now", "recent", "this year",
    str(datetime.now().year), str(datetime.now().year - 1),
    "budget", "sebi", "rbi", "regulation", "rule change",
    "interest rate", "repo rate", "inflation", "gdp",
    "nse", "bse", "nifty", "sensex", "ipo",
    "recently", "new rule", "amendment",
]


def qa_search_node(state: AgentState) -> dict:
    print("qa_search_node")
    query = state["messages"][-1].content
    needs_search = any(t in query.lower() for t in SEARCH_TRIGGERS)

    context = ""
    if needs_search:
        search_q = query + " India mutual fund stock market"
        results = web.invoke(search_q)
        context = "\n".join(r.get("content", "")[:300] for r in results)

    prev_results = state.get("tool_results", {})
    context_str = ""
    if state.get("has_sequential") and prev_results:
        filtered = {k: v for k, v in prev_results.items() if k != "qa_search"}
        if filtered:
            context_str = (
                f"Context from previous tools:\n{filtered}\n"
                "Incorporate this context into your answer where relevant.\n\n"
            )

    web_context_str = f"Latest information from web:\n{context}\n\n" if context else ""

    PROMPT = f"""
    You are a friendly and knowledgeable financial educator for Indian investors.
    You cover mutual funds, stocks, ETFs, bonds, tax, financial planning, and the Indian economy.

    {context_str}
    {web_context_str}
    Question: {query}

    Answer in plain, simple language suitable for an Indian retail investor:
    - Use a real-life example or analogy where helpful.
    - If the question has a numeric answer, show the calculation.
    - If it's a concept, explain the mechanism and why it matters.
    - Keep it concise (under 200 words unless the question demands more detail).
    - Do NOT make up specific fund names, NAV data, or returns.

    End with: "For personalised advice, consult a SEBI-registered advisor."
    """
    response = llm.invoke(PROMPT)
    return {
        "tool_result":  response.content,
        "tool_results": {"qa_search": response.content},
    }
