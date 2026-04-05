from datetime import datetime
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from graph.state import AgentState
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
web = TavilySearchResults(max_results=5)


def news_node(state: AgentState) -> dict:
    print("news_node")
    fund = state["fund_names"][0] if state.get("fund_names") else None
    search_query = (
        f"{fund} mutual fund news India {datetime.now().year}"
        if fund
        else "Indian mutual fund market news today"
    )

    results = web.invoke(search_query)
    headlines = "\n".join(
        f"- {r['content'][:200]}" for r in results
    )

    PROMPT = f"""
    You are a news summarizer for mutual fund investors in India.
    
    Here are recent headlines/articles:
    {headlines}
    
    Your job:
    1. Summarize the top 3-4 key news points in plain English (2 lines each)
    2. Give an overall sentiment: Positive / Neutral / Cautious
    3. Explain in one sentence WHY the sentiment is what it is
    4. Add: "This is news summary only, not investment advice."
    
    {"Focus on: " + fund if fund else "Cover general market mood."}
    """
    response = llm.invoke(PROMPT)
    return {
        "tool_result":  response.content,
        "tool_results": {"news": response.content},
    }
