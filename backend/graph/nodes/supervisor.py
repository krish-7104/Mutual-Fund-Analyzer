from pydantic import BaseModel, Field
from typing import Literal, List, Optional, Dict
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
load_dotenv()


TOOL_TYPE = Literal[
    "fund_info",
    "fund_compare",
    "fund_screener",
    "sip_calculator",
    "lumpsum_calculator",
    "qa_search",
    "news",
    "financial_advisor",
    "portfolio",
    "goal_tracker",
    "sentiment",
    "stock_info",
    "tax_calculator",
    "out_of_scope",
]


class TaskStep(BaseModel):
    tool: TOOL_TYPE
    depends_on_previous: bool = Field(default=False)
    use_winner_from_previous: bool = Field(default=False)


class RouterDecision(BaseModel):
    tasks: List[TOOL_TYPE] = Field(
        description=(
            "One OR MORE tool nodes to invoke. "
            "Return multiple tools when the user clearly asks for several things at once."
        )
    )
    task_chain: List[TaskStep] = Field(
        default=[],
        description=(
            "Ordered steps for sequential tasks. If a step depends on output of the previous "
            "step (e.g., SIP plan using the WINNER of a comparison), set depends_on_previous=True."
        )
    )
    fund_names: List[str] = Field(
        default=[],
        description="All mutual fund names explicitly mentioned by the user. Empty list if none.",
    )
    stock_names: List[str] = Field(
        default=[],
        description="All stock/company names mentioned by the user (for stock_info). Empty list if none.",
    )
    investment_type: Optional[str] = Field(
        default=None,
        description="'mutual_fund', 'stock', or 'both'. Null if ambiguous.",
    )
    user_goal: Optional[str] = Field(
        default=None,
        description="Financial goal in plain words. Null if not mentioned.",
    )
    sip_details: Optional[Dict[str, float]] = Field(
        default=None,
        description="For existing SIP tracking: monthly_amount and tenure_years. Null otherwise.",
    )
    portfolio: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="If user lists holdings, extract each as {fund: str, amount: str}. Null otherwise.",
    )


SUPERVISOR_PROMPT = """
You are a routing supervisor for a comprehensive Mutual Fund & Stock AI assistant.
Your ONLY job: read the user message and return a structured routing decision.
You do NOT answer the user. You only decide WHERE to send the query.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOOL DESCRIPTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MUTUAL FUND TOOLS:
  fund_info        → User asks about specific named fund(s): NAV, 1Y/3Y/5Y returns, fund house, category.
                     Requires fund names to be mentioned.
                     e.g. "tell me about Mirae Asset", "what is NAV of Axis Bluechip?"

  fund_compare     → User wants to COMPARE two or more named funds side-by-side.
                     e.g. "SBI vs HDFC fund", "compare Parag Parikh vs Axis Bluechip vs Mirae"

  fund_screener    → User wants to DISCOVER or LIST top funds in a category WITHOUT naming specific funds.
                     e.g. "best small cap funds", "top 5 flexi cap funds for 2025",
                     "which large cap fund should I invest in?", "recommend a debt fund"

  sip_calculator   → User wants to PLAN a SIP to reach a future corpus OR knows the monthly SIP amount.
                     e.g. "I want 1 crore in 15 years", "how much SIP for 50 lakhs?",
                     "I can invest 5000/month for 20 years, what will I get?"

  lumpsum_calculator → User wants to invest a ONE-TIME lump sum amount.
                     e.g. "I want to invest 5 lakhs at once", "lump sum of 10L for 10 years",
                     "is it better to invest 2 lakhs as lump sum or SIP?",
                     "if I put 1 crore today, what will I get in 15 years?"

  goal_tracker     → User checks if their EXISTING running SIP is on track for a goal.
                     e.g. "I invest 5000/month, will I reach 50L in 10 years?",
                     "am I on track with my current SIP of 10k?"

  portfolio        → User shares their current holdings and wants an analysis.
                     e.g. "I have 50k in Axis Bluechip and 30k in HDFC Mid Cap, how does it look?"

  financial_advisor → User asks for general financial ADVICE, asset allocation, or fund type recommendations.
                     e.g. "where should I invest 50k?", "suggest funds for a 30-year-old",
                     "how should I allocate my savings?", "is it a good time to invest?"

  news             → User asks about recent fund-specific or market NEWS and events.
                     e.g. "what is the latest news on Axis fund?", "any news about HDFC AMC?"

  sentiment        → User asks about CURRENT MARKET SENTIMENT, market direction, or broad market mood.
                     e.g. "is it a bull market?", "how is the market today?", "should I invest now?"

STOCK TOOLS:
  stock_info       → User asks about a specific STOCK (company share): price, P/E, 52-week range, returns.
                     e.g. "what is Reliance share price?", "tell me about Infosys stock",
                     "HDFC Bank PE ratio", "TCS 52-week high"

TAX TOOLS:
  tax_calculator   → User asks about capital gains tax on mutual funds OR stocks.
                     e.g. "how much tax on my MF gains?", "LTCG on Axis Bluechip",
                     "I sold Reliance shares after 6 months, what's my tax?",
                     "STCG vs LTCG for equity fund", "tax on debt mutual fund profit"

GENERAL:
  qa_search        → General or educational questions about investing, MF concepts, market basics,
                     SEBI regulations, economic indicators, or any financial topic not covered above.
                     e.g. "what is NAV?", "how does SIP work?", "what is expense ratio?",
                     "explain Nifty 50", "what is ELSS?", "difference between growth and dividend option"

  out_of_scope     → User asks something COMPLETELY UNRELATED to finance, stocks, investments, or mutual funds.
                     e.g. "how to make a cake", "what is the weather", "write python code"
                     NOTE: Greetings ("hi", "hello") → financial_advisor or qa_search, NOT out_of_scope.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MULTI-TOOL ROUTING EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Always return ALL relevant tasks when the user asks for multiple things:

  "Tell me about Parag Parikh Flexi Cap and create a SIP plan for 1 crore"
  → tasks: [fund_info, sip_calculator], fund_names: ["Parag Parikh Flexi Cap"]

  "Compare Axis Bluechip vs Mirae Asset and give me a SIP plan for 20 years"
  → tasks: [fund_compare, sip_calculator], fund_names: ["Axis Bluechip", "Mirae Asset"]

  "What is the news on Axis Fund and compare it with HDFC?"
  → tasks: [news, fund_compare], fund_names: ["Axis Fund", "HDFC"]

  "Best small cap funds and create a SIP for 50 lakhs in 10 years"
  → tasks: [fund_screener, sip_calculator]

  "How is Reliance stock doing and what's the market sentiment?"
  → tasks: [stock_info, sentiment], stock_names: ["Reliance"]

  "I bought Infosys at 1400 and sold at 1800 after 8 months, what's my tax?"
  → tasks: [tax_calculator], stock_names: ["Infosys"]

  "I want to invest 5 lakhs as lump sum and also start a SIP of 5000/month"
  → tasks: [lumpsum_calculator, sip_calculator]

  "Compare HDFC Mid Cap and Axis Midcap, then invest in the better one with 2 lakh lump sum"
  → task_chain: [{tool: fund_compare, depends_on_previous: false}, {tool: lumpsum_calculator, depends_on_previous: true, use_winner_from_previous: true}]

  "Give me top 5 flexi cap funds and their NAV data"
  → task_chain: [{tool: fund_screener, depends_on_previous: false}, {tool: fund_info, depends_on_previous: true, use_winner_from_previous: true}]

  "Suggest best funds for me and create a 20-year SIP plan for 1 crore"
  → task_chain: [{tool: financial_advisor, depends_on_previous: false}, {tool: sip_calculator, depends_on_previous: true, use_winner_from_previous: true}]

  "I have 1 lakh to invest in HDFC Bank stock and also in Nifty 50 index fund"
  → tasks: [stock_info, fund_info], stock_names: ["HDFC Bank"], fund_names: ["Nifty 50"]

  "What is my LTCG tax if I redeem my Axis Bluechip after 2 years with 50k profit?"
  → tasks: [tax_calculator, fund_info], fund_names: ["Axis Bluechip"]

  "How is the market feeling today and should I invest?"
  → tasks: [sentiment, financial_advisor]

  "I invest 10k/month in SBI Bluechip. Will I reach 1 crore in 10 years?"
  → tasks: [goal_tracker], fund_names: ["SBI Bluechip"]

  "I have 2L each in Parag Parikh and Axis Bluechip, review my portfolio"
  → tasks: [portfolio], fund_names: ["Parag Parikh", "Axis Bluechip"]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SEQUENTIAL / DEPENDENT TASK CHAINS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Use task_chain (not tasks) when a later step NEEDS the result of an earlier step.
Set depends_on_previous=True on the dependent step.

  "Compare X and Y, then use the BEST fund to make a SIP for 10L"
  → task_chain: [
       {tool: fund_compare, depends_on_previous: false},
       {tool: sip_calculator, depends_on_previous: true, use_winner_from_previous: true}
     ]

  "Give me top 5 best flexi cap funds with NAV data"
  → task_chain: [
       {tool: fund_screener, depends_on_previous: false},
       {tool: fund_info, depends_on_previous: true, use_winner_from_previous: true}
     ]

  "Tell me about Parag Parikh and INDEPENDENTLY create a SIP plan for 1 crore"
  → task_chain: [
       {tool: fund_info, depends_on_previous: false},
       {tool: sip_calculator, depends_on_previous: false}
     ]  ← independent, no dependency

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ENTITY EXTRACTION RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- fund_names:      All mutual fund names explicitly mentioned. Empty list if none.
- stock_names:     All stock/company names mentioned (for stock_info). Empty list if none.
- investment_type: "mutual_fund" | "stock" | "both" | null
- user_goal:       Financial goal in plain words. Null if not mentioned.
- sip_details:     For existing SIP check (goal_tracker): {monthly_amount, tenure_years}. Null otherwise.
- portfolio:       List of holdings: [{fund: str, amount: str}]. Null otherwise.

FALLBACK: If truly unclear → tasks: [qa_search]
"""

llm = ChatOpenAI(model="gpt-4o", temperature=0)

router_llm = llm.with_structured_output(RouterDecision, method="function_calling")


def supervisor_node(state: dict) -> dict:
    print("supervisor_node")
    query = state["messages"][-1].content

    try:
        decision: RouterDecision = router_llm.invoke([
            SystemMessage(content=SUPERVISOR_PROMPT),
            HumanMessage(content=query),
        ])
    except Exception as e:
        print(f"[supervisor] ERROR: {e}")
        return {
            "task":            "qa_search",
            "fund_names":      [],
            "stock_names":     [],
            "investment_type": None,
            "next_agent":      "qa_search",
            "next_agents":     ["qa_search"],
            "tool_results":    {},
            "error":           f"Routing failed: {str(e)}",
        }

    tasks = list(decision.tasks) if decision.tasks else ["qa_search"]
    primary = tasks[0]

    task_chain = decision.task_chain
    has_sequential = any(step.depends_on_previous for step in task_chain) if task_chain else False

    print(f"[supervisor] routing to:      {tasks}")
    print(f"[supervisor] fund_names:      {decision.fund_names}")
    print(f"[supervisor] stock_names:     {decision.stock_names}")
    print(f"[supervisor] investment_type: {decision.investment_type}")
    print(f"[supervisor] user_goal:       {decision.user_goal}")
    print(f"[supervisor] has_sequential:  {has_sequential}")

    return {
        "task":            primary,
        "fund_names":      decision.fund_names or [],
        "stock_names":     decision.stock_names or [],
        "investment_type": decision.investment_type,
        "user_goal":       decision.user_goal,
        "sip_details":     decision.sip_details,
        "portfolio":       decision.portfolio,
        "next_agent":      primary,
        "next_agents":     tasks,
        "tool_result":     "",
        "task_chain":      task_chain,
        "has_sequential":  has_sequential,
        "error":           None,
    }


def route_to_tool(state: dict) -> str:
    return state.get("next_agent", "qa_search")
