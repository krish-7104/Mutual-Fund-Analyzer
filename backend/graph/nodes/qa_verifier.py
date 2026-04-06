import re
from langchain_openai import ChatOpenAI
from graph.state import AgentState
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0.1)

VERIFIER_PROMPT = """
You are a Quality Assurance Verifier for a mutual fund AI assistant.
Your job is to compare the User Query against the Synthesized Answer.

Instructions:
1. Check if the Synthesized Answer perfectly addresses the User Query.
2. Check if there are discrepancies, missing parts (that were meant to be answered based on the context), or formatting issues.
3. If it needs improvement, FIX the Synthesized Answer using ONLY the information already present in the Answer. Do NOT hallucinate novel financial data.
4. If the answer is already perfect, return the original Synthesized Answer EXACTLY as it is.
5. Return ONLY valid HTML—no markdown tags like ```html, no code fences, no explanation.

Return ONLY the final HTML string.
"""

def clean_html_output(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:html)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()

def qa_verifier_node(state: AgentState) -> dict:
    print("qa_verifier_node")

    query = state["messages"][-1].content
    synthesized_answer = state.get("tool_result", "")
    
    if not synthesized_answer or "I could not process that request" in synthesized_answer:
        return {"tool_result": synthesized_answer}

    instruction = (
        f"User Query: '{query}'\n\n"
        f"Synthesized Answer:\n{synthesized_answer}\n"
    )

    response = llm.invoke([
        ("system", VERIFIER_PROMPT),
        ("human", instruction),
    ])

    cleaned = clean_html_output(response.content)

    return {
        "tool_result": cleaned,
    }
