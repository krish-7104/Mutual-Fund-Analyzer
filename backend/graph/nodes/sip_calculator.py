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


def _parse_recurring_amount(text: str):
    t = text.lower().replace(",", "")
    monthly_patterns = [
        r"(?:₹|rs\.?|inr)?\s*([\d.]+)\s*(?:per month|monthly|/month)",
        r"(?:monthly sip|sip)\s*(?:of|for)?\s*(?:₹|rs\.?|inr)?\s*([\d.]+)",
    ]
    yearly_patterns = [
        r"(?:₹|rs\.?|inr)?\s*([\d.]+)\s*(?:per year|yearly|annually|/year|a year)",
        r"(?:yearly sip|annual sip)\s*(?:of|for)?\s*(?:₹|rs\.?|inr)?\s*([\d.]+)",
    ]
    for pattern in monthly_patterns:
        m = re.search(pattern, t)
        if m:
            return {"amount": float(m.group(1)), "frequency": "monthly"}
    for pattern in yearly_patterns:
        m = re.search(pattern, t)
        if m:
            return {"amount": float(m.group(1)), "frequency": "yearly"}
    return None


def _parse_user_rate(text: str):
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", text.lower())
    return float(m.group(1)) if m else None


def _parse_years(text: str):
    m = re.search(r"(\d+)\s*(?:years?|yrs?)", text.lower())
    return float(m.group(1)) if m else None


def _infer_fund_type_caps(funds: list):
    if not funds:
        return 8.0, 15.0
    text = " ".join(funds).lower()
    has_small = "small cap" in text or "smallcap" in text
    has_flexi = "flexi cap" in text or "flexicap" in text
    if has_small and has_flexi:
        return 10.0, 15.0
    if has_small:
        return 12.0, 18.0
    if has_flexi:
        return 10.0, 14.0
    return 8.0, 15.0


def _extract_three_year_gains(prev_results: dict):
    values = []
    fund_info_text = str(prev_results.get("fund_info", ""))
    for match in re.findall(r"3[-\s]*Year\s*Gain:\s*(-?\d+(?:\.\d+)?)%", fund_info_text, flags=re.IGNORECASE):
        try:
            values.append(float(match))
        except Exception:
            continue
    return values


def _realistic_rate(query: str, prev_results: dict, selected_funds: list) -> float:
    user_rate = _parse_user_rate(query)
    if user_rate and user_rate > 0:
        return user_rate

    low_cap, high_cap = _infer_fund_type_caps(selected_funds)
    gains = _extract_three_year_gains(prev_results)
    if gains:
        positive = [g for g in gains if g > 0]
        if positive:
            avg = sum(positive) / len(positive)
            adjusted = avg * 0.6
            return round(min(max(adjusted, low_cap), high_cap), 2)
    return DEFAULT_RATE


def _allocation_with_ai(query: str, selected_funds: list, prev_results: dict):
    if len(selected_funds) <= 1:
        return {selected_funds[0]: 100} if selected_funds else {}
    prompt = (
        "Decide SIP allocation percentages across these funds.\n"
        f"User query: {query}\n"
        f"Funds: {selected_funds}\n"
        f"Context: {prev_results}\n"
        "Rules:\n"
        "- Return ONLY JSON object mapping fund -> percentage integer.\n"
        "- Total must be exactly 100.\n"
        "- Keep allocation realistic (avoid extreme concentration unless clearly justified).\n"
    )
    try:
        raw = llm.invoke(prompt).content.strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            cleaned = {}
            for fund in selected_funds:
                val = parsed.get(fund)
                if val is None:
                    continue
                cleaned[fund] = max(0, int(round(float(val))))
            if cleaned:
                total = sum(cleaned.values())
                if total == 100:
                    return cleaned
    except Exception:
        pass
    base = 100 // len(selected_funds)
    allocation = {fund: base for fund in selected_funds}
    allocation[selected_funds[-1]] += 100 - sum(allocation.values())
    return allocation


def _extract_inputs_with_rules(query: str):
    return {
        "target": _parse_amount(query),
        "years": _parse_years(query) or DEFAULT_YEARS,
        "rate": _parse_user_rate(query),
    }


def _wants_four_category_plan(query: str) -> bool:
    q = query.lower()
    return (
        ("flexi" in q and "small" in q and "mid" in q and "index" in q)
        or ("one flexi cap" in q and "one small cap" in q and "one mid cap" in q and "one index" in q)
    )


def _extract_suggested_funds(text: str) -> list:
    funds = []
    patterns = [
        r"Fund Suggestion:\s*([^\n\r]+)",
        r"-\s*([A-Za-z][A-Za-z0-9&().,\-\/ ]+(?:Fund|Index Fund))",
    ]
    for pattern in patterns:
        for match in re.findall(pattern, text, flags=re.IGNORECASE):
            name = " ".join(str(match).split()).strip(" -:")
            if name and name not in funds:
                funds.append(name)
    return funds


def _merge_funds(primary: list, secondary: list) -> list:
    merged = []
    for f in (primary or []) + (secondary or []):
        name = " ".join(str(f).split()).strip()
        if name and name not in merged:
            merged.append(name)
    return merged


def sip_calculator_node(state: AgentState) -> dict:
    print("sip_calculator_node")
    query = state["messages"][-1].content
    winner_fund = state.get("winner_fund")
    fund_names = state.get("fund_names") or []
    selected_funds = []
    prev_results = state.get("tool_results", {})
    if fund_names:
        selected_funds = fund_names
    elif winner_fund:
        selected_funds = [winner_fund]
    if _wants_four_category_plan(query):
        suggestions = _extract_suggested_funds(str(prev_results.get("financial_advisor", "")))
        selected_funds = _merge_funds(selected_funds, suggestions)

    funds_context = (
        f"This SIP plan should use these selected fund(s): {selected_funds}\n\n"
        if selected_funds else ""
    )

    # ── 2. Build context string from previous tool results ───────────────────
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
        "  - rate: assumed annual return % (use 12 if not mentioned by user)\n\n"
        f"{context_str}"
        f"{funds_context}"
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

    rule_inputs = _extract_inputs_with_rules(query)
    params = {}
    if rule_inputs["rate"] is None:
        try:
            raw = llm.invoke(EXTRACT_PROMPT).content.strip()
            raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
            params = json.loads(raw)
        except Exception as e:
            print(
                f"[sip_calculator] JSON parse failed ({e}), using fallback extraction")
            params = {}

    recurring = _parse_recurring_amount(query)
    T = float(rule_inputs["target"] or params.get("target") or DEFAULT_TARGET)
    Y = float(rule_inputs["years"] or params.get("years") or DEFAULT_YEARS)
    rate = float(rule_inputs["rate"] or _realistic_rate(query, prev_results, selected_funds))

    if T <= 0:
        T = DEFAULT_TARGET
    if Y <= 0:
        Y = DEFAULT_YEARS
    if rate <= 0:
        rate = DEFAULT_RATE

    monthly_sip = None
    if recurring:
        monthly_sip = recurring["amount"] if recurring["frequency"] == "monthly" else recurring["amount"] / 12
    R = rate / 100 / 12
    N = int(Y * 12)
    funds_text = ", ".join(selected_funds) if selected_funds else "equity mutual funds"
    allocation = _allocation_with_ai(query, selected_funds, prev_results)

    if monthly_sip is not None:
        monthly_sip = round(monthly_sip)
        total_invested = monthly_sip * N
        projected_corpus = round(monthly_sip * ((((1 + R) ** N) - 1) / R))
        wealth_gained = round(projected_corpus - total_invested)
        if projected_corpus > total_invested * 5:
            projected_corpus = total_invested * 5
            wealth_gained = projected_corpus - total_invested

        split_lines = []
        for fund, pct in allocation.items():
            split_lines.append(f"- {fund}: {pct}% (₹{round(monthly_sip * pct / 100):,}/month)")
        split_text = "\n".join(split_lines) if split_lines else "- No fund allocation available."

        PROMPT = (
            f"The user will invest ₹{monthly_sip:,} monthly for {Y:.0f} years.\n"
            f"Selected fund(s): {funds_text}\n"
            f"Allocation:\n{split_text}\n"
            f"Assumed annual return: {rate}%.\n"
            f"Projected corpus: ₹{projected_corpus:,}\n"
            f"Total invested: ₹{total_invested:,}\n"
            f"Projected gain: ₹{wealth_gained:,}\n\n"
            "Write one clear SIP plan. Do not provide alternative SIP plans."
        )
    else:
        monthly_sip = round(T * R / (((1 + R) ** N) - 1))
        total_invested = monthly_sip * N
        wealth_gained = round(T - total_invested)
        projected_corpus = total_invested + wealth_gained
        if projected_corpus > total_invested * 5:
            projected_corpus = total_invested * 5
            wealth_gained = projected_corpus - total_invested

        split_lines = []
        for fund, pct in allocation.items():
            split_lines.append(f"- {fund}: {pct}% (₹{round(monthly_sip * pct / 100):,}/month)")
        split_text = "\n".join(split_lines) if split_lines else "- No fund allocation available."

        PROMPT = (
            f"The user wants to reach ₹{T:,.0f} in {Y:.0f} years.\n"
            f"Selected fund(s): {funds_text}\n"
            f"Allocation:\n{split_text}\n"
            f"Assuming {rate}% annual return:\n"
            f"Recommended monthly SIP: ₹{monthly_sip:,}\n"
            f"Total invested: ₹{total_invested:,}\n"
            f"Projected corpus: ₹{projected_corpus:,}\n"
            f"Gain from market returns: ₹{wealth_gained:,}\n\n"
            "Write one clear SIP plan. Do not provide alternative SIP plans."
        )
    response = llm.invoke(PROMPT)
    return {
        "tool_result":  response.content,
        "tool_results": {"sip_calculator": response.content},
        "active_fund":  selected_funds[0] if selected_funds else None,
    }
