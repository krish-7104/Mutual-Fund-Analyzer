from datetime import datetime
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from graph.state import AgentState
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
web = TavilySearchResults(max_results=4)

def sentiment_node(state: AgentState) -> dict:
    print("sentiment_node")
    
    search_query = f"Indian stock market mutual fund sentiment trend Nifty Sensex today {datetime.now().year}"
    results = web.invoke(search_query)
    context = "\n".join(f"- {r['content'][:250]}" for r in results)

    PROMPT = f"""
    You are a Market Sentiment Analyst for mutual fund investors in India.
    
    Based on the following recent web search results:
    {context}
    
    Your job:
    1. Determine the overall market sentiment right now: (Bullish / Bearish / Neutral / Volatile).
    2. Provide a 3-4 sentence explanation of the key drivers behind this sentiment.
    3. Suggest how mutual fund investors should behave in this current sentiment (e.g. 'Continue SIPs', 'Wait for dips').
    4. Conclude with: "Market conditions change rapidly. Do not time the market; focus on long-term goals."
    """
    response = llm.invoke(PROMPT)
    return {
        "tool_result":  response.content,
        "tool_results": {"sentiment": response.content},
    }
