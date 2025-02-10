from datetime import datetime
from typing import Literal, cast, Any, TypedDict, Union

from langchain_community.tools import DuckDuckGoSearchResults, OpenWeatherMapQueryRun
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, BaseMessage
from langchain_core.runnables import (
    RunnableConfig,
    RunnableLambda,
    RunnableSerializable,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import ToolNode

from agents.llama_guard import LlamaGuard, LlamaGuardOutput, SafetyAssessment
from agents.tools import calculator
from core import get_model, settings
from agents.researcher_graph.graph import graph as researcher_graph
from shared.utils import format_docs
from shared import prompts

from dataclasses import dataclass, field
from typing import Annotated, Literal, TypedDict
from langchain_core.documents import Document
from shared.state import reduce_docs


class Router(TypedDict):
    """Query router response type."""

    logic: str
    type: Literal["langchain", "more-info", "general"]


class AgentState(MessagesState, total=False):
    """`total=False` is PEP589 specs.

    documentation: https://typing.readthedocs.io/en/latest/spec/typeddict.html#totality
    """

    safety: LlamaGuardOutput
    remaining_steps: RemainingSteps
    router: Router = field(default_factory=lambda: Router(type="general", logic=""))

    steps: list[str] = field(default_factory=list)
    """A list of steps in the research plan."""
    documents: Annotated[list[Document], reduce_docs] = field(default_factory=list)
    """Populated by the retriever. This is a list of documents that the agent can reference."""

    # Feel free to add additional attributes to your state as needed.
    # Common examples include retrieved documents, extracted entities, API connections, etc.


web_search = DuckDuckGoSearchResults(name="WebSearch")
tools = [web_search, calculator]

# Add weather tool if API key is set
# Register for an API key at https://openweathermap.org/api/
if settings.OPENWEATHERMAP_API_KEY:
    wrapper = OpenWeatherMapAPIWrapper(
        openweathermap_api_key=settings.OPENWEATHERMAP_API_KEY.get_secret_value()
    )
    tools.append(OpenWeatherMapQueryRun(name="Weather", api_wrapper=wrapper))

current_date = datetime.now().strftime("%B %d, %Y")
instructions = f"""
    You are a helpful research assistant with the ability to search the web and use other tools.
    Today's date is {current_date}.

    NOTE: THE USER CAN'T SEE THE TOOL RESPONSE.

    A few things to remember:
    - Please include markdown-formatted links to any citations used in your response. Only include one
    or two citations per response unless more are needed. ONLY USE LINKS RETURNED BY THE TOOLS.
    - Use calculator tool with numexpr to answer math questions. The user does not understand numexpr,
      so for the final response, use human readable format - e.g. "300 * 200", not "(300 \\times 200)".
    """


def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    model = model.bind_tools(tools)
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=instructions)] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | model


def format_safety_message(safety: LlamaGuardOutput) -> AIMessage:
    content = f"This conversation was flagged for unsafe content: {', '.join(safety.unsafe_categories)}"
    return AIMessage(content=content)


async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m)
    response = await model_runnable.ainvoke(state, config)

    # Run llama guard check here to avoid returning the message if it's unsafe
    llama_guard = LlamaGuard()
    safety_output = await llama_guard.ainvoke("Agent", state["messages"] + [response])
    if safety_output.safety_assessment == SafetyAssessment.UNSAFE:
        return {
            "messages": [format_safety_message(safety_output)],
            "safety": safety_output,
        }

    if state["remaining_steps"] < 2 and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, need more steps to process this request.",
                )
            ]
        }
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


async def llama_guard_input(state: AgentState, config: RunnableConfig) -> AgentState:
    llama_guard = LlamaGuard()
    safety_output = await llama_guard.ainvoke("User", state["messages"])
    return {"safety": safety_output}


async def block_unsafe_content(state: AgentState, config: RunnableConfig) -> AgentState:
    safety: LlamaGuardOutput = state["safety"]
    return {"messages": [format_safety_message(safety)]}


async def create_research_plan(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[str] | str]:
    """Create a step-by-step research plan for answering a LangChain-related query.

    Args:
        state (AgentState): The current state of the agent, including conversation history.
        config (RunnableConfig): Configuration with the model used to generate the plan.

    Returns:
        dict[str, list[str]]: A dictionary with a 'steps' key containing the list of research steps.
    """

    class Plan(TypedDict):
        """Generate research plan."""

        steps: list[str]

    model = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model = model.with_structured_output(Plan)
    messages = [
        {
            "role": "system",
            "content": prompts.RESEARCH_PLAN_SYSTEM_PROMPT,
        }
    ] + state["messages"]
    response = cast(Plan, await model.ainvoke(messages))
    return {"steps": response["steps"], "documents": "delete"}


async def conduct_research(state: AgentState) -> dict[str, Any]:
    """Execute the first step of the research plan.

    This function takes the first step from the research plan and uses it to conduct research.

    Args:
        state (AgentState): The current state of the agent, including the research plan steps.

    Returns:
        dict[str, list[str]]: A dictionary with 'documents' containing the research results and
                              'steps' containing the remaining research steps.

    Behavior:
        - Invokes the researcher_graph with the first step of the research plan.
        - Updates the state with the retrieved documents and removes the completed step.
    """
    result = await researcher_graph.ainvoke({"question": state["steps"][0]})
    return {"documents": result["documents"], "steps": state["steps"][1:]}


def check_finished(state: AgentState) -> Literal["done", "continue"]:
    """Determine if the research process is complete or if more research is needed.

    This function checks if there are any remaining steps in the research plan:
        - If there are, route back to the `conduct_research` node
        - Otherwise, route to the `respond` node

    Args:
        state (AgentState): The current state of the agent, including the remaining research steps.

    Returns:
        Literal["done", "continue"]: The next step to take based on whether research is complete.
    """
    match len(state["steps"] or []):
        case 0:
            return "done"
        case _:
            return "continue"


async def respond(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Generate a final response to the user's query based on the conducted research.

    This function formulates a comprehensive answer using the conversation history and the documents retrieved by the researcher.

    Args:
        state (AgentState): The current state of the agent, including retrieved documents and conversation history.
        config (RunnableConfig): Configuration with the model used to respond.

    Returns:
        dict[str, list[str]]: A dictionary with a 'messages' key containing the generated response.
    """
    model = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    context = format_docs(state["documents"])
    prompt = prompts.RESPONSE_SYSTEM_PROMPT.format(context=context)
    messages = [{"role": "system", "content": prompt}] + state["messages"]
    response = await model.ainvoke(messages)
    return {"messages": [response]}


async def analyze_and_route_query(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, Router]:
    """Analyze the user's query and determine the appropriate routing.

    Args:
        state (AgentState): The current state of the agent, including conversation history.
        config (RunnableConfig): Configuration with the model used for query analysis.

    Returns:
        dict[str, Router]: A dictionary containing the router classification result.
    """
    model = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    messages = [SystemMessage(content=prompts.ROUTER_SYSTEM_PROMPT)] + state["messages"]

    # Router 타입 검증을 위한 structured output 설정
    router_model = model.with_structured_output(Router)
    try:
        response = await router_model.ainvoke(messages)
        if (
            not isinstance(response, dict)
            or "type" not in response
            or "logic" not in response
        ):
            raise ValueError("Invalid router response structure")
        return {"router": cast(Router, response)}
    except Exception as e:
        # 기본값으로 폴백
        return {
            "router": Router(
                type="general",
                logic="Failed to classify query, falling back to general response",
            )
        }


def route_query(
    state: AgentState,
) -> Literal["create_research_plan", "ask_for_more_info", "respond_to_general_query"]:
    """Determine the next step based on the query classification.

    Args:
        state (AgentState): The current state including the router's classification.

    Returns:
        Literal: The next step to take in the conversation flow.
    """
    try:
        router_type = state["router"]["type"]
        match router_type:
            case "langchain":
                return "create_research_plan"
            case "more-info":
                return "ask_for_more_info"
            case "general":
                return "respond_to_general_query"
            case _:
                # 알 수 없는 타입의 경우 일반 응답으로 폴백
                return "respond_to_general_query"
    except Exception:
        # 상태나 라우터에 문제가 있는 경우 안전하게 일반 응답으로 폴백
        return "respond_to_general_query"


async def ask_for_more_info(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Generate a response asking the user for more information.

    Args:
        state (AgentState): The current state of the agent.
        config (RunnableConfig): Configuration with the model to use.

    Returns:
        dict[str, list[BaseMessage]]: The response message asking for more information.
    """
    model = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    system_prompt = prompts.MORE_INFO_SYSTEM_PROMPT.format(
        logic=state["router"]["logic"]
    )
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = await model.ainvoke(messages)
    return {"messages": [response]}


async def respond_to_general_query(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Generate a response to a general query not related to LangChain.

    Args:
        state (AgentState): The current state of the agent.
        config (RunnableConfig): Configuration with the model to use.

    Returns:
        dict[str, list[BaseMessage]]: The response message for the general query.
    """
    model = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    system_prompt = prompts.GENERAL_SYSTEM_PROMPT.format(logic=state["router"]["logic"])
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = await model.ainvoke(messages)
    return {"messages": [response]}


# After "model", if there are tool calls, run "tools". Otherwise END.
def pending_tool_calls(state: AgentState) -> Literal["tools", "done"]:
    last_message = state["messages"][-1]
    match last_message:
        case AIMessage() as msg if msg.tool_calls:
            return "tools"
        case AIMessage():
            return "done"
        case _:
            raise TypeError(f"Expected AIMessage, got {type(last_message)}")


# Define the graph
agent = StateGraph(AgentState)

# 모든 노드 추가
agent.add_node("guard_input", llama_guard_input)
agent.add_node("block_unsafe_content", block_unsafe_content)
agent.add_node("conduct_research", conduct_research)
agent.add_node("create_research_plan", create_research_plan)
agent.add_node("respond", respond)
agent.add_node("analyze_and_route_query", analyze_and_route_query)
agent.add_node("ask_for_more_info", ask_for_more_info)
# agent.add_node("respond_to_general_query", respond_to_general_query)
agent.add_node("model", acall_model)
agent.add_node("tools", ToolNode(tools))

# 시작점 설정
agent.set_entry_point("guard_input")


# Check for unsafe input and block further processing if found
def check_safety(state: AgentState) -> Literal["unsafe", "safe"]:
    safety: LlamaGuardOutput = state["safety"]
    match safety.safety_assessment:
        case SafetyAssessment.UNSAFE:
            return "unsafe"
        case _:
            return "safe"


agent.add_conditional_edges(
    "guard_input",
    check_safety,
    {"unsafe": "block_unsafe_content", "safe": "analyze_and_route_query"},
)
agent.add_edge("block_unsafe_content", END)


# 3. 쿼리 라우팅 관련 엣지
agent.add_conditional_edges(
    "analyze_and_route_query",
    route_query,
    {
        "ask_for_more_info": "ask_for_more_info",
        "create_research_plan": "create_research_plan",
        # "respond_to_general_query": "respond_to_general_query",
        "respond_to_general_query": "model",
    },
)

# 4. 연구 관련 엣지
agent.add_edge("create_research_plan", "conduct_research")
agent.add_conditional_edges(
    "conduct_research",
    check_finished,
    {"continue": "conduct_research", "done": "respond"},
)

# 4. 도구 사용 관련 엣지
# agent.add_edge("tools", "model")
# agent.add_conditional_edges(
#     "model",
#     pending_tool_calls,
#     {"tools": "tools", "done": END},
# )

# 5. 종료 엣지들
agent.add_edge("block_unsafe_content", END)
agent.add_edge("ask_for_more_info", END)
# agent.add_edge("respond_to_general_query", END)
# 2. 도구 사용 관련 엣지
agent.add_edge("tools", "model")
agent.add_conditional_edges(
    "model",
    pending_tool_calls,
    {"tools": "tools", "done": END},
)
agent.add_edge("respond", END)

# 그래프 컴파일
research_assistant = agent.compile(checkpointer=MemorySaver())
