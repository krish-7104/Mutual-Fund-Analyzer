import re
from langchain_openai import ChatOpenAI
from graph.state import AgentState
from langchain_core.messages import AIMessage
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0.1)

SECTION_WORD_LIMITS = {
    "fund_compare":        500,
    "fund_info":           350,
    "fund_screener":       400,
    "sip_calculator":      300,
    "lumpsum_calculator":  300,
    "portfolio":           400,
    "financial_advisor":   400,
    "news":                300,
    "sentiment":           250,
    "goal_tracker":        300,
    "qa_search":           250,
    "stock_info":          350,
    "tax_calculator":      350,
    "out_of_scope":        100,
}
DEFAULT_WORD_LIMIT = 300

SECTION_LABELS = {
    "fund_info":           "Fund Overview",
    "fund_compare":        "Fund Comparison",
    "fund_screener":       "Top Fund Picks",
    "sip_calculator":      "SIP Plan",
    "lumpsum_calculator":  "Lump Sum Plan",
    "portfolio":           "Portfolio Snapshot",
    "financial_advisor":   "Financial Advice",
    "news":                "Latest News",
    "sentiment":           "Market Sentiment",
    "goal_tracker":        "Goal Tracker",
    "qa_search":           "Answer",
    "stock_info":          "Stock Overview",
    "tax_calculator":      "Tax Calculation",
    "out_of_scope":        "Out of Scope",
}

DISCLAIMER_MAP = {
    "fund_info":           "<p><em>Past performance is not indicative of future returns.</em></p>",
    "fund_compare":        "<p><em>Past performance is not indicative of future returns.</em></p>",
    "fund_screener":       "<p><em>Fund suggestions are based on web data. Verify on AMFI before investing.</em></p>",
    "sip_calculator":      "<p><em>Projections are estimates based on assumed returns and may vary.</em></p>",
    "lumpsum_calculator":  "<p><em>Projections are estimates based on assumed returns and may vary.</em></p>",
    "goal_tracker":        "<p><em>Projections are estimates based on assumed returns and may vary.</em></p>",
    "financial_advisor":   "<p><em>This is general information, not personalised investment advice.</em></p>",
    "qa_search":           "<p><em>This is general information, not personalised investment advice.</em></p>",
    "news":                "<p><em>News summaries are for informational purposes only, not investment advice.</em></p>",
    "stock_info":          "<p><em>Past performance is not indicative of future returns. Stock investments carry market risk.</em></p>",
    "tax_calculator":      "<p><em>Tax figures are estimates. Consult a CA/tax advisor for accurate tax filing.</em></p>",
}

SYNTH_PROMPT = """
You are a response formatter for a mutual fund & stock market AI assistant.
Your ONLY job is to convert raw tool output into clean, structured HTML.

Rules:
- Return ONLY valid HTML — no markdown, no code fences, no explanation
- Use semantic HTML:
    <h2> for section headings (only when multiple sections exist)
    <h3> for sub-headings within a section
    <p> for paragraphs
    <ul><li> for lists (max 6 items)
    <table> for comparisons, fund lists, or structured data with rows/columns
- Format numbers using Indian system: ₹1,00,000 not 100000; 12% not 0.12; "1.5 Crores" not 15000000
- Do NOT add inline styles
- Do NOT invent data not present in the raw input
- Respect the word limit per section
- For tax calculations: use <table> to show the step-by-step breakdown
- For stock data: use <table> to show price, returns, valuation metrics
- For fund screener results: use a clean <table> or structured <ul> with fund name and rationale
"""


def clean_html_output(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:html)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def build_disclaimers(active_tools: list) -> str:
    seen = set()
    disclaimers = []
    for tool in active_tools:
        d = DISCLAIMER_MAP.get(tool)
        if d and d not in seen:
            disclaimers.append(d)
            seen.add(d)
    return "\n".join(disclaimers)


def synthesizer_node(state: AgentState) -> dict:
    print("synthesizer_node")

    all_results: dict = state.get("tool_results") or {}
    active_tools: list = state.get("next_agents") or []
    winner_fund: str = state.get("winner_fund")

    print(f"[synthesizer] all_results keys: {list(all_results.keys())}")

    tool_results = all_results

    if not tool_results:
        fallback = "<p>I could not process that request. Please try again.</p>"
        return {"tool_result": fallback}

    query = state["messages"][-1].content
    disclaimers = build_disclaimers(active_tools)

    if len(tool_results) > 1:
        parts = []
        for tool_name, result in tool_results.items():
            label = SECTION_LABELS.get(tool_name, tool_name.replace("_", " ").title())
            word_limit = SECTION_WORD_LIMITS.get(tool_name, DEFAULT_WORD_LIMIT)
            parts.append(f"[SECTION: {label} | word_limit: {word_limit}]\n{result}")

        raw = "\n\n---\n\n".join(parts)

        winner_context = (
            f"\nNOTE: The recommended fund from the previous step is '{winner_fund}'. "
            f"Present any SIP/lump sum section as being for this fund.\n"
            if winner_fund else ""
        )

        instruction = (
            f"Original User Query: '{query}'\n"
            f"{winner_context}\n"
            "Format each section under its own <h2> heading. "
            "Respect each section's word limit. "
            "Do NOT reason about or recompute values — only format what is given.\n"
            "If a SIP or lump sum section exists, keep investment plans ONLY in that section.\n\n"
            f"{raw}\n\n"
            f"{disclaimers}"
        )
    else:
        tool_name = next(iter(tool_results))
        raw = next(iter(tool_results.values()))
        word_limit = SECTION_WORD_LIMITS.get(tool_name, DEFAULT_WORD_LIMIT)

        instruction = (
            f"Original Query: '{query}'\n"
            f"Word limit: {word_limit} words.\n"
            "No <h2> heading needed for a single section.\n\n"
            f"{raw}\n\n"
            f"{disclaimers}"
        )

    response = llm.invoke([
        ("system", SYNTH_PROMPT),
        ("human", instruction),
    ])

    cleaned = clean_html_output(response.content)

    return {"tool_result": cleaned}
