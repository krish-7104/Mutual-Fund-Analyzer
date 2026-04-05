from langchain_openai import ChatOpenAI
from graph.state import AgentState
from langchain_core.messages import AIMessage
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0.1)

SYNTH_PROMPT = """
        You are a response formatter for a mutual fund AI assistant.

        Your job is to convert raw tool output into clean, structured HTML.

        Rules:
        - Return ONLY valid HTML (no markdown, no explanations)
        - Use semantic HTML:
            <h2> for section headings
            <p> for paragraphs
            <ul><li> only if needed (avoid overuse)
        - Format numbers properly (₹1,00,000 not 100000)
        - Keep it clean and readable (no inline styles)

        Structure rules:
        - If multiple sections exist → each section must start with <h2>
        - If single response → no unnecessary heading

        Disclaimers:
        - Fund analysis → <p><em>Past performance is not indicative of future returns.</em></p>
        - Q&A/news → <p><em>This is general information, not personalised advice.</em></p>
        - SIP/goal → <p><em>Projections are estimates based on assumed returns.</em></p>
        - If multiple apply → include all as separate <p><em>...</em></p>

        Constraints:
        - Keep response under 400 words
"""

SECTION_LABELS = {
    "fund_info":      "Fund Overview",
    "sip_calculator": "SIP Plan",
    "fund_compare":      "Fund Comparison",
    "news":              "Latest News",
    "sentiment":         "Market Sentiment",
    "financial_advisor": "Financial Advice",
    "portfolio":         "Portfolio Snapshot",
    "goal_tracker":      "Goal Tracker",
    "qa_search":         "Answer",
}


def clean_html_output(text: str) -> str:
    if not text:
        return text

    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1] if "```" in text else text
        text = text.replace("html", "", 1).strip()

    if text.endswith("```"):
        text = text[:-3].strip()

    return text


def synthesizer_node(state: AgentState) -> dict:
    print("synthesizer_node")

    all_results: dict = state.get("tool_results") or {}
    active_tools = state.get("next_agents") or []

    # Filter: only keep results for tools that were active in THIS turn
    tool_results = {k: v for k, v in all_results.items() if k in active_tools}

    if len(tool_results) > 1:
        parts = []
        for tool_name, result in tool_results.items():
            label = SECTION_LABELS.get(
                tool_name, tool_name.replace("_", " ").title())
            parts.append(f"## {label}\n\n{result}")
        raw = "\n\n---\n\n".join(parts)
        instruction = (
            "The user asked for multiple things at once. "
            "Format this combined response clearly with the section headings provided. "
            "Keep each section focused. Use appropriate disclaimers at the end."
        )
    else:
        raw = (
            next(iter(tool_results.values()), None)
            or "I could not process that request."
        )
        instruction = "Format this response cleanly."

    response = llm.invoke([
        ("system", SYNTH_PROMPT),
        ("human", f"{instruction}\n\n{raw}")
    ])

    cleaned = clean_html_output(response.content)

    return {
        "tool_result": cleaned,
        "messages": [AIMessage(content=cleaned)],
    }
