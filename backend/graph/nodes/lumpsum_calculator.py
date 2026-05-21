import json
import re
from langchain_openai import ChatOpenAI
from graph.state import AgentState
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

DEFAULT_RATE = 12.0
DEFAULT_YEARS = 10


def _parse_amount(text: str) -> float:
    t = text.lower().replace(",", "")
    m = re.search(r"([\d.]+)\s*(?:cr(?:ore)?)", t)
    if m:
        return float(m.group(1)) * 1_00_00_000
    m = re.search(r"([\d.]+)\s*(?:l(?:akh)?|lac)", t)
    if m:
        return float(m.group(1)) * 1_00_000
    m = re.search(r"[\d]+(?:\.\d+)?", t)
    if m:
        return float(m.group())
    return 0.0


def _extract_params(query: str, prev_results: dict) -> dict:
    EXTRACT_PROMPT = (
        "Extract these fields from the user message. Return ONLY valid JSON.\n"
        "Fields:\n"
        "  - amount: lump sum investment amount in rupees (plain number; 1 crore = 10000000, 1 lakh = 100000)\n"
        "  - years: investment tenure in years (integer, default 10)\n"
        "  - rate: expected annual return % (number, default 12)\n"
        "  - compare_with_sip: true if user explicitly asks to compare with SIP (bool)\n\n"
        f"Context from previous tools: {prev_results}\n\n"
        f"User message: {query}\n"
    )
    try:
        raw = llm.invoke(EXTRACT_PROMPT).content.strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
        return json.loads(raw)
    except Exception:
        return {}


def lumpsum_calculator_node(state: AgentState) -> dict:
    print("lumpsum_calculator_node")
    query = state["messages"][-1].content
    prev_results = state.get("tool_results", {})
    fund_names = state.get("fund_names") or []
    winner_fund = state.get("winner_fund")

    p = _extract_params(query, prev_results)
    print(f"[lumpsum] extracted: {p}")

    A = float(p.get("amount") or _parse_amount(query) or 1_00_000)
    Y = float(p.get("years") or DEFAULT_YEARS)
    R = float(p.get("rate") or DEFAULT_RATE)
    compare = bool(p.get("compare_with_sip", False))

    if A <= 0:
        A = 1_00_000
    if Y <= 0:
        Y = DEFAULT_YEARS
    if R <= 0:
        R = DEFAULT_RATE

    # Lump sum future value: FV = P * (1 + r/100)^n
    lumpsum_corpus = round(A * ((1 + R / 100) ** Y))
    lumpsum_gain = lumpsum_corpus - A
    cagr_check = round(((lumpsum_corpus / A) ** (1 / Y) - 1) * 100, 2)

    selected_funds = fund_names or ([winner_fund] if winner_fund else [])
    funds_text = ", ".join(selected_funds) if selected_funds else "equity mutual funds"

    context_str = ""
    if prev_results:
        filtered = {k: v for k, v in prev_results.items() if k != "lumpsum_calculator"}
        if filtered:
            context_str = f"Context from previous tools:\n{filtered}\n\n"

    sip_section = ""
    if compare:
        r_m = R / 100 / 12
        n = int(Y * 12)
        monthly_sip_for_same = round(lumpsum_corpus * r_m / (((1 + r_m) ** n) - 1))
        sip_total_invested = monthly_sip_for_same * n
        sip_corpus = round(monthly_sip_for_same * ((((1 + r_m) ** n) - 1) / r_m))
        sip_section = (
            f"\n\nSIP COMPARISON (to reach same ₹{lumpsum_corpus:,}):\n"
            f"- Monthly SIP required: ₹{monthly_sip_for_same:,}\n"
            f"- Total invested via SIP: ₹{sip_total_invested:,}\n"
            f"- Projected corpus via SIP: ₹{sip_corpus:,}\n"
            f"- SIP total investment is {'MORE' if sip_total_invested > A else 'LESS'} than lump sum by ₹{abs(sip_total_invested - A):,}\n"
        )

    PROMPT = f"""
    {context_str}
    You are a lump sum investment calculator for Indian investors.
    User asked: "{query}"

    LUMP SUM CALCULATION:
    - Investment amount: ₹{A:,.0f}
    - Fund(s): {funds_text}
    - Duration: {Y:.0f} years
    - Expected annual return: {R}%
    - Projected corpus: ₹{lumpsum_corpus:,}
    - Total wealth gained: ₹{lumpsum_gain:,}
    - Verified CAGR: {cagr_check}%
    {sip_section}

    Write a clear, friendly explanation:
    1. Summarize the lump sum plan with all numbers.
    {"2. Explain the SIP comparison: when is lump sum better (market dips, windfall) vs SIP (regular income, market volatility)." if compare else "2. Mention the power of compounding and the importance of timing for lump sum."}
    3. One practical tip about lump sum investing in the Indian context (e.g., using STPs).

    Use Indian number formatting (₹, crores, lakhs).
    End with: "Projections are based on assumed returns and may vary with actual market performance."
    """
    response = llm.invoke(PROMPT)
    return {
        "tool_result": response.content,
        "tool_results": {"lumpsum_calculator": response.content},
    }
