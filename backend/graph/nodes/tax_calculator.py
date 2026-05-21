import json
import re
from langchain_openai import ChatOpenAI
from graph.state import AgentState
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0.1)

# Tax rules as per Union Budget 2024 (effective July 23, 2024)
TAX_RULES = """
INDIAN CAPITAL GAINS TAX RULES (Budget 2024, effective July 23 2024):

EQUITY MUTUAL FUNDS & DIRECT STOCKS (equity >= 65% for hybrid):
  Short-Term Capital Gain (STCG): Holding < 12 months → Taxed at 20%
  Long-Term Capital Gain (LTCG): Holding >= 12 months → Taxed at 12.5%
    • First ₹1,25,000 of LTCG per financial year is EXEMPT (grandfathered exemption)
    • Gains above ₹1,25,000 taxed at 12.5% (no indexation)

DEBT MUTUAL FUNDS:
  Units bought AFTER 1 April 2023:
    → All gains taxed at the investor's income tax slab rate (no LTCG benefit regardless of holding)
  Units bought BEFORE 1 April 2023:
    STCG: Holding < 36 months → slab rate
    LTCG: Holding >= 36 months → 20% WITH indexation

GOLD ETFs / Fund of Funds / International Funds:
  STCG: Holding < 24 months → slab rate
  LTCG: Holding >= 24 months → 12.5% (no indexation, post Budget 2024)

SECURITIES TRANSACTION TAX (STT):
  STT is charged by the broker; not included in capital gains calculation.

SURCHARGE & CESS:
  Add 4% Health & Education Cess on the computed tax.
  Surcharge applies if total income > ₹50 lakh.
"""


def _extract_params(query: str) -> dict:
    EXTRACT_PROMPT = f"""
    Extract these fields from the user's message about capital gains tax. Return ONLY valid JSON.

    Fields:
    - asset_type: one of "equity_mf", "debt_mf", "stock", "hybrid_mf", "gold_etf", "index_fund"
    - purchase_price: per unit/share purchase price (number or null)
    - sell_price: per unit/share sell/current price (number or null)
    - units: number of units or shares (number, default 1 if not mentioned)
    - holding_months: holding period in months (number or null)
    - invested_amount: total invested amount if given directly (number or null)
    - current_value: current total value if given directly (number or null)
    - gain_amount: capital gain if given directly (number or null)
    - debt_bought_before_april_2023: true if debt MF bought before April 1, 2023, false otherwise, null if unknown
    - investor_tax_slab_pct: investor's income tax slab % (5, 10, 15, 20, or 30; null if not mentioned)

    User message: {query}
    """
    try:
        raw = llm.invoke(EXTRACT_PROMPT).content.strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
        return json.loads(raw)
    except Exception:
        return {}


def tax_calculator_node(state: AgentState) -> dict:
    print("tax_calculator_node")
    query = state["messages"][-1].content

    p = _extract_params(query)
    print(f"[tax_calculator] extracted params: {p}")

    prev_results = state.get("tool_results", {})
    context_str = ""
    if prev_results:
        context_str = f"Context from previous tools:\n{prev_results}\n\n"

    CALC_PROMPT = f"""
    {TAX_RULES}

    {context_str}
    User query: "{query}"
    Extracted parameters: {json.dumps(p, indent=2)}

    Calculate the capital gains tax step by step:

    Step 1: Identify asset type and applicable tax category.
    Step 2: Determine holding period and whether STCG or LTCG applies.
    Step 3: Calculate the capital gain (sell price - purchase price) × units, or use provided amounts.
    Step 4: Apply the correct tax rate. For equity LTCG, subtract ₹1,25,000 exemption first.
    Step 5: Add 4% Health & Education Cess.
    Step 6: Show Net Gain after tax.

    Also include:
    - Whether it's worth waiting to convert STCG → LTCG (if applicable)
    - Tax-saving tip specific to this situation

    Format as a clear, structured response with Indian number formatting (₹, lakhs, crores).
    End with: "This is an estimate for educational purposes. Consult a CA/tax advisor for filing."
    """

    response = llm.invoke(CALC_PROMPT)
    return {
        "tool_result": response.content,
        "tool_results": {"tax_calculator": response.content},
    }
