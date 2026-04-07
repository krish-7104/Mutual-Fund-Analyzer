import json
import re
from langchain_openai import ChatOpenAI
from graph.state import AgentState
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

DEFAULT_YEARS = 15
DEFAULT_RATE = 12
DEFAULT_TARGET = 10_000_000


def _parse_amount(text: str) -> float:
    text = text.lower().strip()
    m = re.search(r"([\d.]+)\s*(?:cr(?:ore)?)", text)
    if m:
        return float(m.group(1)) * 1_00_00_000
    m = re.search(r"([\d.]+)\s*(?:l(?:akh)?|lac)", text)
    if m:
        return float(m.group(1)) * 1_00_000
    m = re.search(r"[\d,]+", text.replace(",", ""))
    if m:
        return float(m.group())
    return DEFAULT_TARGET


def sip_calculator_node(state: AgentState) -> dict:
    print("sip_calculator_node")
    query = state["messages"][-1].content
    winner_fund = state.get("winner_fund")
    fund_names = state.get("fund_names") or []
    active_fund = winner_fund or (fund_names[0] if fund_names else None)

    fund_context = (
        f"This SIP plan is specifically for: {active_fund}\n"
        f"{'(This fund was selected as the winner from a comparison.)' if winner_fund else ''}\n\n"
    ) if active_fund else ""

    # ── 2. Build context string from previous tool results ───────────────────
    prev_results = state.get("tool_results", {})
    context_str = ""
    if prev_results:
        context_str = (
            f"Context from previous tools: {prev_results}\n"
            "CRITICAL: If the context specifies a winning fund with a specific historical return rate "
            "(e.g., a 3-year return), you MUST extract THAT numerical rate for 'rate' instead of the default 12.\n"
            "If the context specifies an investment amount, extract that for 'target' if a target wasn't mentioned.\n\n"
        )

    EXTRACT_PROMPT = (
        "From this user message and context, extract these three numbers:\n"
        "  - target: the corpus / amount they want to reach (in rupees as a plain integer) if not mention then assume 1,00,00,000,\n"
        "  - years: number of years for investment (integer; use 15 if not mentioned)\n"
        "  - rate: assumed annual return % (use 12 if not mentioned by user or previous tools)\n\n"
        f"{context_str}"
        f"{fund_context}"
        "Rules:\n"
        "  - '1CR' or '1 crore' = 10000000\n"
        "  - '50L' or '50 lakh' = 5000000\n"
        "  - If years is not mentioned, return 15\n"
        "  - If rate is not mentioned or available, return 12\n"
        "  - ALWAYS return all three keys.\n\n"
        "Return ONLY valid JSON, no markdown:\n"
        '{"target": <number>, "years": <number>, "rate": <number>}\n\n'
        f"User message: {query}"
    )

    try:
        raw = llm.invoke(EXTRACT_PROMPT).content.strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
        params = json.loads(raw)
    except Exception as e:
        print(
            f"[sip_calculator] JSON parse failed ({e}), using fallback extraction")
        params = {}

    T = float(params.get("target") or _parse_amount(query))
    Y = float(params.get("years") or DEFAULT_YEARS)
    rate = float(params.get("rate") or DEFAULT_RATE)

    if T <= 0:
        T = DEFAULT_TARGET
    if Y <= 0:
        Y = DEFAULT_YEARS
    if rate <= 0:
        rate = DEFAULT_RATE

    R = rate / 100 / 12
    N = int(Y * 12)

    monthly_sip = round(T * R / (((1 + R) ** N) - 1))
    total_invested = monthly_sip * N
    wealth_gained = round(T - total_invested)

    fund_label = f"in **{active_fund}**" if active_fund else "in an equity mutual fund"

    PROMPT = (
        f"{'Winner fund from comparison: ' + active_fund + chr(10) if winner_fund else ''}"
        f"The user wants to reach ₹{T:,.0f} in {Y:.0f} years investing {fund_label}.\n"
        f"Assuming {rate}% annual return:\n\n"
        f"  Recommended monthly SIP : ₹{monthly_sip:,}\n"
        f"  Total they will invest  : ₹{total_invested:,}\n"
        f"  Gain from market returns: ₹{wealth_gained:,}\n\n"
        "Explain this in a friendly, encouraging way. "
        "Mention that the actual return may vary and they should consult an advisor."
    )
    response = llm.invoke(PROMPT)
    return {
        "tool_result":  response.content,
        "tool_results": {"sip_calculator": response.content},
        "active_fund":  active_fund,
    }
