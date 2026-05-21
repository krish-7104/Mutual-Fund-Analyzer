from datetime import datetime
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI
from graph.state import AgentState
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
web = TavilySearch(max_results=5)


def sentiment_node(state: AgentState) -> dict:
    print("sentiment_node")
    year = datetime.now().year

    search_query = (
        f"Indian stock market sentiment Nifty Sensex FII DII mutual fund flows "
        f"India economy outlook {year}"
    )
    results = web.invoke(search_query)
    context = "\n".join(f"- {r.get('content', '')[:300]}" for r in results)

    query = state["messages"][-1].content

    PROMPT = f"""
    You are a Market Sentiment Analyst for Indian investors covering both mutual funds and stocks.

    Based on these recent web search results:
    {context}

    User asked: "{query}"

    Your response:
    1. **Overall Sentiment**: State clearly — Bullish / Bearish / Neutral / Volatile
    2. **Key Drivers** (3-4 sentences): What macro factors, FII/DII flows, or events are driving this sentiment?
    3. **Nifty/Sensex context**: Where are the indices, and what does the trend suggest?
    4. **For Mutual Fund investors**: Should they continue SIPs, pause, or increase allocation?
    5. **For Stock investors**: Is this a good time to buy, hold, or wait?
    6. **Risk factors to watch**: 2-3 things that could change the current sentiment.

    Conclude with: "Market conditions change rapidly. Focus on your long-term goals rather than timing the market."
    """
    response = llm.invoke(PROMPT)
    return {
        "tool_result":  response.content,
        "tool_results": {"sentiment": response.content},
    }
