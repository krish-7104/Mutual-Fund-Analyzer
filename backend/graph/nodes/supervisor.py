from pydantic import BaseModel, Field
from typing import Literal, List, Optional, Dict
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
load_dotenv()

class TaskStep(BaseModel):
    tool: Literal[
        "fund_info", "fund_compare", "sip_calculator",
        "qa_search", "news", "financial_advisor",
        "portfolio", "goal_tracker", "sentiment", "out_of_scope"
    ]
    # If True, this step NEEDS output from the previous step
    depends_on_previous: bool = Field(default=False)
    # Override fund_names for this specific step (e.g. "use winner from compare")
    use_winner_from_previous: bool = Field(default=False)

class RouterDecision(BaseModel):
    tasks: List[Literal[
        "fund_info",
        "fund_compare",
        "sip_calculator",
        "qa_search",
        "news",
        "financial_advisor",
        "portfolio",
        "goal_tracker",
        "sentiment",
        "out_of_scope",
    ]] = Field(
        description=(
            "One OR MORE tool nodes to invoke. "
            "Return multiple tools when the user clearly asks for several things at once."
        )
    )


    task_chain: List[TaskStep] = Field(
        default=[],
        description=(
            "Ordered steps. If a step depends on output of the previous step "
            "(e.g. SIP plan using the WINNER of a comparison), set "
            "depends_on_previous=True on that step."
        )
    )

    fund_names: List[str] = Field(
        default=[],
        description="All fund names mentioned by the user. Empty list if none.",
    )

    user_goal: Optional[str] = Field(
        default=None,
        description="Financial goal in plain words. Null if not mentioned.",
    )

    sip_details: Optional[Dict[str, float]] = Field(
        default=None,
        description="If user mentions an EXISTING SIP, extract monthly_amount and tenure_years as numbers. Null otherwise.",
    )

    portfolio: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="If user lists holdings, extract each entry with fund (string) and amount (string). Null otherwise.",
    )


SUPERVISOR_PROMPT = (
    "You are a routing supervisor for a Mutual Fund AI assistant.\n"
    "Your ONLY job is to read the user message and return a structured routing decision.\n"
    "You do NOT answer the user. You only decide WHERE to send the query.\n"
    "You MUST return multiple tasks when the user asks for more than one thing.\n"
    "\n"
    "TOOL DESCRIPTIONS\n"
    "fund_info      -> user asks about ONE specific fund\n"
    "                  e.g. 'tell me about Mirae Asset', 'what is Axis Bluechip?'\n"
    "\n"
    "fund_compare   -> user wants to COMPARE two or more funds\n"
    "                  e.g. 'SBI vs HDFC fund', 'compare X, Y, and Z?'\n"
    "\n"
    "sip_calculator -> user wants to PLAN a SIP to reach a future corpus goal\n"
    "                  e.g. 'I want 1 crore in 15 years', 'how much SIP for 50 lakhs?'\n"
    "                  e.g. 'create a SIP plan for 20 years', 'SIP to make 1 crore'\n"
    "\n"
    "qa_search      -> general or educational question, even if time-sensitive (tax rules, SEBI regulations)\n"
    "\n"
    "news           -> event-based headlines, fund-specific news, market mood TODAY\n"
    "\n"
    "sentiment      -> user asks about CURRENT MARKET SENTIMENT or trend.\n"
    "                  e.g. 'is it a bull market?', 'how is the market today?'\n"
    "\n"
    "financial_advisor -> user asks for financial ADVICE, asset allocation, or fund recommendations.\n"
    "                  e.g. 'suggest a fund for me', 'where should I invest 50k?'\n"
    "\n"
    "portfolio      -> user shares their holdings and wants analysis\n"
    "                  e.g. 'I have 50k in X and 30k in Y, how does it look?'\n"
    "\n"
    "goal_tracker   -> user checks if their EXISTING investment is on track\n"
    "                  e.g. 'I invest 5000/month, will I reach 50L in 10 years?'\n"
    "\n"
    "out_of_scope   -> user asks something COMPLETELY UNRELATED to finance, stocks, investments, or mutual funds.\n"
    "                  e.g. 'how to make a cake', 'what is the weather', 'write python code'\n"
    "                  NOTE: General greetings like 'hi' or 'hello' should map to 'financial_advisor' or 'qa_search', NOT 'out_of_scope'.\n"
    "\n"
    "MULTI-TOOL EXAMPLES — when the query has multiple parts, return ALL relevant tasks:\n"
    "  Query: 'Tell me about Parag Parikh Flexi Cap and create a SIP plan for 1 crore'\n"
    "  -> tasks: [fund_info, sip_calculator]\n"
    "\n"
    "  Query: 'Compare Axis Bluechip vs Mirae Asset and give me a SIP plan for 20 years'\n"
    "  -> tasks: [fund_compare, sip_calculator]\n"
    "\n"
    "  Query: 'What is the news on Axis Fund and compare it with HDFC?'\n"
    "  -> tasks: [news, fund_compare]\n"
    "\n"
    "  Query: 'Details about Parag Parikh Flexi cap fund and also create a SIP Plan for making corpus 1CR'\n"
    "  -> tasks: [fund_info, sip_calculator]\n"
    "\n"
    "FALLBACK: If truly unclear -> tasks: [qa_search]\n"
    "\n"
    "ENTITY EXTRACTION RULES:\n"
    "- fund_names : all fund names mentioned. Empty list if none.\n"
    "- user_goal  : financial goal in plain words. Null if not mentioned.\n"
    "- sip_details: for EXISTING SIP inputs, extract monthly_amount and tenure_years. Null otherwise.\n"
    "- portfolio  : list of holdings each with fund name and amount. Null otherwise.\n"

    "SEQUENTIAL / DEPENDENT TASKS\n"
    "When a later step NEEDS the result of an earlier step, mark it with depends_on_previous=True.\n"
    "\n"
    "  Query: 'Compare X and Y, then use the BEST fund to make a SIP for 10L'\n"
    "  -> task_chain: [\n"
    "       {tool: fund_compare,    depends_on_previous: false, use_winner_from_previous: false},\n"
    "       {tool: sip_calculator,  depends_on_previous: true,  use_winner_from_previous: true}\n"
    "     ]\n"
    "\n"
    "  Query: 'Tell me about Parag Parikh and create a SIP for 1 crore'\n"
    "  -> task_chain: [\n"
    "       {tool: fund_info,       depends_on_previous: false, use_winner_from_previous: false},\n"
    "       {tool: sip_calculator,  depends_on_previous: false, use_winner_from_previous: false}\n"
    "     ]\n"
    "  (These are independent — sip_calculator already knows the fund from fund_names)\n"
)


llm = ChatOpenAI(model="gpt-4o", temperature=0)

router_llm = llm.with_structured_output(
    RouterDecision, method="function_calling")


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
            "task":         "qa_search",
            "fund_names":   [],
            "next_agent":   "qa_search",
            "next_agents":  ["qa_search"],
            "tool_results": {},
            "error":        f"Routing failed: {str(e)}",
        }

    tasks = list(decision.tasks) if decision.tasks else ["qa_search"]
    primary = tasks[0]
    
    task_chain = decision.task_chain
    has_sequential = any(step.depends_on_previous for step in task_chain) if task_chain else False

    print(f"[supervisor] routing to: {tasks}")
    print(f"[supervisor] fund_names: {decision.fund_names}")
    print(f"[supervisor] user_goal:  {decision.user_goal}")

    return {
        "task":         primary,
        "fund_names":   decision.fund_names,
        "user_goal":    decision.user_goal,
        "sip_details":  decision.sip_details,
        "portfolio":    decision.portfolio,
        "next_agent":   primary,
        "next_agents":  tasks,
        "tool_result":  "", # Clear previous turn's synthesized answer
        "task_chain":     task_chain,
        "has_sequential": has_sequential,
        "error":        None,
    }


def route_to_tool(state: dict) -> str:
    return state.get("next_agent", "qa_search")
