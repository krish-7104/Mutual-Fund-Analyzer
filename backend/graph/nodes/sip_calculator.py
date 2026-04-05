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

    EXTRACT_PROMPT = (
        "From this user message, extract these three numbers:\n"
        "  - target: the corpus / amount they want to reach (in rupees as a plain integer)\n"
        "  - years: number of years for investment (integer; use 15 if not mentioned)\n"
        "  - rate: assumed annual return % (use 12 if not mentioned)\n\n"
        "Rules:\n"
        "  - '1CR' or '1 crore' = 10000000\n"
        "  - '50L' or '50 lakh' = 5000000\n"
        "  - If years is not mentioned, return 15\n"
        "  - If rate is not mentioned, return 12\n"
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

    PROMPT = (
        f"The user wants to reach ₹{T:,.0f} in {Y:.0f} years.\n"
        f"Assuming {rate}% annual return (typical for equity mutual funds):\n\n"
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
    }
