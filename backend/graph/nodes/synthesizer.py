import re
from langchain_openai import ChatOpenAI
from graph.state import AgentState
from langchain_core.messages import AIMessage
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0.1)

SECTION_WORD_LIMITS = {
    "fund_compare":      500,
    "fund_info":         350,
    "sip_calculator":    300,
    "portfolio":         400,
    "financial_advisor": 400,
    "news":              300,
    "sentiment":         250,
    "goal_tracker":      300,
    "qa_search":         250,
    "out_of_scope":      100,
}
DEFAULT_WORD_LIMIT = 300

SECTION_LABELS = {
    "fund_info":         "Fund Overview",
    "sip_calculator":    "SIP Plan",
    "fund_compare":      "Fund Comparison",
    "news":              "Latest News",
    "sentiment":         "Market Sentiment",
    "financial_advisor": "Financial Advice",
    "portfolio":         "Portfolio Snapshot",
    "goal_tracker":      "Goal Tracker",
    "qa_search":         "Answer",
    "out_of_scope":      "Out of Scope",
}

DISCLAIMER_MAP = {
    "fund_info":         "<p><em>Past performance is not indicative of future returns.</em></p>",
    "fund_compare":      "<p><em>Past performance is not indicative of future returns.</em></p>",
    "sip_calculator":    "<p><em>Projections are estimates based on assumed returns.</em></p>",
    "goal_tracker":      "<p><em>Projections are estimates based on assumed returns.</em></p>",
    "financial_advisor": "<p><em>This is general information, not personalised advice.</em></p>",
    "qa_search":         "<p><em>This is general information, not personalised advice.</em></p>",
    "news":              "<p><em>This is general information, not personalised advice.</em></p>",
}

SYNTH_PROMPT = """
You are a response formatter for a mutual fund AI assistant.
Your ONLY job is to convert raw tool output into clean, structured HTML.

Rules:
- Return ONLY valid HTML — no markdown, no code fences, no explanation
- Use semantic HTML:
    <h2> for section headings (only when multiple sections exist)
    <p> for paragraphs
    <ul><li> for lists only when genuinely list-like (max 5 items)
    <table> for comparisons with rows and columns
- Format numbers: ₹1,00,000 not 100000; 12% not 0.12
- Do NOT add inline styles
- Do NOT invent data not present in the raw input
- Word limit per section is provided in the instruction — respect it
"""


def clean_html_output(text: str) -> str:
    """Strip markdown code fences if model wraps output."""
    text = text.strip()
    # Remove ```html ... ``` or ``` ... ```
    text = re.sub(r"^```(?:html)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def build_disclaimers(active_tools: list) -> str:
    """Deduplicated disclaimers for all active tools."""
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
    
    # We use all_results directly since the state is scoped to a single run
    tool_results = all_results

    if not tool_results:
        fallback = "<p>I could not process that request. Please try again.</p>"
        return {
            "tool_result": fallback,
        }

    query = state["messages"][-1].content
    disclaimers = build_disclaimers(active_tools)

    if len(tool_results) > 1:
        parts = []
        for tool_name, result in tool_results.items():
            label = SECTION_LABELS.get(tool_name, tool_name.replace("_", " ").title())
            word_limit = SECTION_WORD_LIMITS.get(tool_name, DEFAULT_WORD_LIMIT)
            parts.append(
                f"[SECTION: {label} | word_limit: {word_limit}]\n{result}"
            )

        raw = "\n\n---\n\n".join(parts)

        winner_context = (
            f"\nNOTE: The recommended fund from the comparison is '{winner_fund}'. "
            f"The SIP section already uses this fund — present it as such.\n"
            if winner_fund else ""
        )

        instruction = (
            f"Original User Query: '{query}'\n"
            f"{winner_context}\n"
            "Format each section under its own <h2> heading. "
            "Respect each section's word limit. "
            "Do NOT reason about or recompute values — only format what is given.\n"
            "If a SIP section exists, keep SIP calculations/allocation ONLY in the SIP section. "
            "In other sections, remove SIP plans and SIP allocation lines.\n\n"
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

    return {
        "tool_result": cleaned,
    }