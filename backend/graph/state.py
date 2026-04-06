from typing import TypedDict, Annotated, List, Optional, Dict
from langchain_core.messages import BaseMessage


def _merge_tool_results(a: Dict, b: Dict) -> Dict:
    merged = dict(a)
    merged.update(b)
    return merged


def _last_wins(a, b):
    return b if b is not None else a


class AgentState(TypedDict):
    messages:     List[BaseMessage]
    task:         str
    fund_names:   List[str]                                    # fund_info, fund_compare, news
    user_goal:    Optional[str]                                # sip_calculator, goal_tracker, recommend
    sip_details:  Optional[dict]                               # goal_tracker
    portfolio:    Optional[List[dict]]                         # portfolio
    tool_result:  Annotated[Optional[str], _last_wins]         # synthesizer output — annotated to allow parallel writes
    tool_results: Annotated[Dict, _merge_tool_results]         # multi-tool: {tool_name: result}, merged across branches
    next_agent:   str                                          # legacy single-tool compat
    next_agents:    List[str]                                    # multi-tool: list of tools to dispatch
    task_chain:     List[dict]                                   # Sequential/complex dispatching states
    has_sequential: bool
    winner_fund:    Optional[str]
    error:          Optional[str]
