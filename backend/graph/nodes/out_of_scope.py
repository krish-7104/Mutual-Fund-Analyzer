from graph.state import AgentState

def out_of_scope_node(state: AgentState) -> dict:
    print("out_of_scope_node")
    msg = "<p>I am an AI assistant specialized exclusively in finance, mutual funds, investments, and the stock market. I cannot assist with unrelated topics. Please ask me a question related to investing!</p>"
    return {
        "tool_result":  msg,
        "tool_results": {"out_of_scope": msg},
    }
