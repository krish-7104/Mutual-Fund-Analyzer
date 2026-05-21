from datetime import datetime
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI
from graph.state import AgentState
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
web = TavilySearch(max_results=6)


def fund_screener_node(state: AgentState) -> dict:
    print("fund_screener_node")
    query = state["messages"][-1].content
    year = datetime.now().year

    prev_results = state.get("tool_results", {})
    context_str = ""
    if state.get("has_sequential") and prev_results:
        filtered = {k: v for k, v in prev_results.items() if k != "fund_screener"}
        if filtered:
            context_str = f"Context from previous tools:\n{filtered}\n\n"

    QUERY_PROMPT = (
        f"Convert this user question into a concise search query to find top Indian mutual fund recommendations.\n"
        f"User question: \"{query}\"\n\n"
        "Return ONLY the search string (max 12 words). Be specific about category/criteria.\n"
        f"Examples:\n"
        f"  'best small cap funds' -> 'best small cap mutual funds India {year} top performers'\n"
        f"  'top flexi cap for beginners' -> 'top flexi cap mutual funds India beginners {year}'\n"
        f"  'best index fund Nifty 50' -> 'best Nifty 50 index funds India {year} lowest expense ratio'\n"
    )
    search_q = llm.invoke(QUERY_PROMPT).content.strip()
    print(f"[fund_screener] search query: {search_q}")

    results = web.invoke(search_q)
    content = "\n\n".join(
        f"[Source {i+1}]: {r.get('content', '')[:500]}" for i, r in enumerate(results)
    )

    active_tools = state.get("next_agents") or []
    scope_guard = ""
    if "sip_calculator" in active_tools:
        scope_guard = "Do NOT include SIP plans or monthly investment amounts — the SIP calculator will handle that separately.\n"

    PROMPT = f"""
    {context_str}
    You are a mutual fund screener and analyst for Indian investors.
    User asked: "{query}"

    Based on these web search results:
    {content}

    Your response MUST include:
    1. **Top 5-6 funds** that match the user's criteria, each with:
       - Full official fund name
       - Fund house (AMC)
       - Why it fits (brief rationale: track record, ratings, expense ratio, category fit)
    2. **Category overview**: What type of investor should consider these funds and typical risk level
    3. **Key metrics to verify** before investing: CRISIL/Value Research rating, AUM, expense ratio, exit load
    4. If the user asked for a specific number of funds (e.g., "top 3"), respect that count.

    CRITICAL: Only list funds that appear in the search results. Do NOT hallucinate fund names.
    {scope_guard}
    End with: "Verify fund details on AMFI (amfiindia.com) or Value Research before investing."
    """
    response = llm.invoke(PROMPT)
    return {
        "tool_result": response.content,
        "tool_results": {"fund_screener": response.content},
    }
