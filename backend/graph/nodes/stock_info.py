import json
import yfinance as yf
from langchain_openai import ChatOpenAI
from graph.state import AgentState
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0.1)


def _resolve_ticker(name: str) -> str:
    prompt = (
        "Convert this Indian company/stock name to its NSE ticker symbol.\n"
        "Examples: 'Reliance Industries' -> 'RELIANCE', 'HDFC Bank' -> 'HDFCBANK', "
        "'Infosys' -> 'INFY', 'TCS' -> 'TCS', 'State Bank of India' -> 'SBIN', "
        "'Wipro' -> 'WIPRO', 'Bajaj Finance' -> 'BAJFINANCE', 'Nifty 50' -> 'NIFTY50'\n"
        f"Stock/Index: {name}\n"
        "Return ONLY the ticker symbol, nothing else."
    )
    return llm.invoke(prompt).content.strip().upper().split()[0]


def fetch_stock(name: str) -> dict:
    ticker_sym = _resolve_ticker(name)
    print(f"[stock_info] resolved '{name}' -> '{ticker_sym}'")

    for suffix in [".NS", ".BO"]:
        try:
            ticker = yf.Ticker(ticker_sym + suffix)
            hist = ticker.history(period="1y")
            if hist.empty:
                continue

            info = ticker.info
            cp = round(float(hist["Close"].iloc[-1]), 2)
            yap = round(float(hist["Close"].iloc[0]), 2)
            one_y = round(((cp - yap) / yap) * 100, 2) if yap else None

            six_m_idx = max(0, len(hist) - 126)
            sm = round(float(hist["Close"].iloc[six_m_idx]), 2)
            six_m = round(((cp - sm) / sm) * 100, 2) if sm else None

            one_m_idx = max(0, len(hist) - 21)
            om = round(float(hist["Close"].iloc[one_m_idx]), 2)
            one_m = round(((cp - om) / om) * 100, 2) if om else None

            market_cap = info.get("marketCap")
            pe = info.get("trailingPE")
            div_yield = info.get("dividendYield")

            return {
                "name": info.get("longName", name),
                "symbol": ticker_sym,
                "exchange": "NSE" if suffix == ".NS" else "BSE",
                "current_price": cp,
                "market_cap_cr": round(market_cap / 1e7, 0) if market_cap else None,
                "pe_ratio": round(pe, 2) if pe else None,
                "52w_high": info.get("fiftyTwoWeekHigh"),
                "52w_low": info.get("fiftyTwoWeekLow"),
                "1y_return_pct": one_y,
                "6m_return_pct": six_m,
                "1m_return_pct": one_m,
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "dividend_yield_pct": round(div_yield * 100, 2) if div_yield else None,
                "book_value": info.get("bookValue"),
                "pb_ratio": info.get("priceToBook"),
            }
        except Exception as e:
            print(f"[stock_info] error for {ticker_sym}{suffix}: {e}")
            continue

    return {"error": f"Could not find stock data for '{name}'. Verify the company name."}


def stock_info_node(state: AgentState) -> dict:
    print("stock_info_node")
    names = state.get("stock_names") or state.get("fund_names") or []
    query = state["messages"][-1].content
    active_tools = state.get("next_agents") or []

    if not names:
        msg = "No stock names provided. Please mention the company or stock name you want to look up."
        return {"tool_result": msg, "tool_results": {"stock_info": msg}}

    results = []
    for n in names:
        data = fetch_stock(n)
        print(f"[stock_info] data for '{n}': {data}")
        results.append(data)

    data_str = json.dumps(results, indent=2)

    prev_results = state.get("tool_results", {})
    context_str = ""
    if state.get("has_sequential") and prev_results:
        filtered = {k: v for k, v in prev_results.items() if k != "stock_info"}
        if filtered:
            context_str = f"Context from previous tools:\n{filtered}\n\n"

    scope_guard = ""
    if "sip_calculator" in active_tools or "lumpsum_calculator" in active_tools:
        scope_guard = "Do NOT include investment plans, SIP allocations, or lump sum projections — those are handled separately.\n"

    PROMPT = f"""
    {context_str}
    You are a stock market data presenter for Indian investors.
    User asked: "{query}"

    Fetched stock data:
    {data_str}

    Present clearly with:
    - Current price and where it stands vs 52-week high/low (as a % from high/low)
    - Returns: 1-month, 6-month, 1-year
    - Valuation: P/E ratio, P/B ratio, market cap in crores
    - Sector and industry
    - Dividend yield if available
    - One-line assessment: e.g. "Trading near 52-week high — momentum is strong" or "Near 52-week low — possible value zone"

    Use Indian number formatting (₹, crores, lakhs). Keep it factual and concise.
    {scope_guard}
    """
    response = llm.invoke(PROMPT)
    return {
        "tool_result": response.content,
        "tool_results": {"stock_info": response.content},
    }
