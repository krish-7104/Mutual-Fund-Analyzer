from datetime import datetime
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI
from graph.state import AgentState
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
web = TavilySearch(max_results=6)


def news_node(state: AgentState) -> dict:
    print("news_node")
    year = datetime.now().year

    fund_names = state.get("fund_names") or []
    stock_names = state.get("stock_names") or []
    investment_type = state.get("investment_type") or "mutual_fund"
    query = state["messages"][-1].content

    # Build the most targeted search query possible
    subjects = fund_names + stock_names
    primary_subject = subjects[0] if subjects else None

    if primary_subject:
        if investment_type == "stock" or (not fund_names and stock_names):
            search_query = f"{primary_subject} stock news India {year} NSE BSE"
        else:
            search_query = f"{primary_subject} mutual fund news India {year}"
    else:
        # General market news
        search_query = f"Indian stock market mutual fund latest news {year} Nifty Sensex"

    print(f"[news_node] search: {search_query}")
    results = web.invoke(search_query)
    headlines = "\n".join(f"- {r.get('content', '')[:300]}" for r in results)

    subject_focus = (
        f"Focus specifically on: {', '.join(subjects)}"
        if subjects else "Cover general Indian market and mutual fund news."
    )

    PROMPT = f"""
    You are a financial news summarizer for Indian investors covering mutual funds and stocks.

    Here are the latest headlines/articles:
    {headlines}

    User asked: "{query}"

    Your job:
    1. Summarize the top 4-5 key news points in plain English (2-3 lines each).
       Include fund/company name and why it matters to an investor.
    2. Give an overall sentiment: 🟢 Positive / 🟡 Neutral / 🔴 Cautious
    3. In one sentence, explain WHY the sentiment is what it is.
    4. One actionable takeaway for investors.

    {subject_focus}
    End with: "This is a news summary only, not investment advice."
    """
    response = llm.invoke(PROMPT)
    return {
        "tool_result":  response.content,
        "tool_results": {"news": response.content},
    }
