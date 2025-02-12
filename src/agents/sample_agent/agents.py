import asyncio

from langgraph.graph import END, MessagesState, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

from langchain_community.tools import DuckDuckGoSearchResults

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableSerializable, RunnableLambda
from langchain_core.messages import AIMessage
from langchain_core.runnables import (
    RunnableConfig,
    RunnableLambda,
    RunnableSerializable,
)
from core import get_model, settings

import operator
from typing import Literal, TypedDict, Annotated


# 입출력 스키마 정의
# https://langchain-ai.github.io/langgraph/how-tos/input_output_schema/#setup


# class InputState(TypedDict):
#     text: str
#     file: str


class AgentState(MessagesState, total=False):
    """Agent의 상태를 관리하는 클래스"""

    text: str
    file: str
    aggregate: Annotated[list, operator.add]


# 호출 시 node의 값(node_secret)을 "aggregate" 키에 리스트 값으로 반환하는 클래스
class ReturnNodeValue:
    def __init__(self, node_secret: str):
        self._value = node_secret

    def __call__(self, state: AgentState) -> AgentState:
        print(f"Adding {self._value} to {state['aggregate']}")
        return {"aggregate": [self._value]}


def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    preprocessor = RunnableLambda(
        lambda state: state["messages"],
        name="StateModifier",
    )
    return preprocessor | model


async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m)
    response = await model_runnable.ainvoke(state, config)

    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# 호출 시 node의 값(node_secret)을 "aggregate" 키에 리스트 값으로 반환하는 클래스
class ReturnNodeValue:
    def __init__(self, node_secret: str):
        self._value = node_secret

    async def __call__(self, state: AgentState) -> AgentState:
        await asyncio.sleep(0.5)
        print(f"Adding {self._value} to {state['aggregate']}")
        return {"aggregate": [self._value]}


async def bg_task(state: AgentState, config: RunnableConfig) -> AgentState:
    await asyncio.sleep(0.5)

    return {"messages": []}


class Router(TypedDict):
    """Query router response type."""

    logic: str
    type: Literal["A", "B"]


def route_query(
    state: AgentState,
) -> Literal["A", "B"]:
    """Determine the next step based on the query classification."""
    try:
        router_type = state["router"]["type"]
        match router_type:
            case "A":
                return "A"
            case "B":
                return "B"
            case _:
                # 알 수 없는 타입의 경우 일반 응답으로 폴백
                return "A"
    except Exception:
        # 상태나 라우터에 문제가 있는 경우 안전하게 일반 응답으로 폴백
        return "A"


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


# 파일 분석 에이전트
pdf_analyzer_agent = StateGraph(AgentState)
pdf_analyzer_agent.add_node("pdf_reader", ReturnNodeValue("pdf_reader"))
pdf_analyzer_agent.add_node("pdf_summarizer", ReturnNodeValue("pdf_summarizer"))

pdf_analyzer_agent.set_entry_point("pdf_reader")
pdf_analyzer_agent.add_edge("pdf_reader", "pdf_summarizer")
pdf_analyzer_agent.add_edge("pdf_summarizer", END)
pdf_analyzer_agent = pdf_analyzer_agent.compile(
    checkpointer=MemorySaver(),
    # interrupt_before=["pdf_summarizer"]
)


# 텍스트 분석 에이전트
text_analyzer_agent = StateGraph(AgentState)
text_analyzer_agent.add_node("model", acall_model)
text_analyzer_agent.add_node("text_planner", ReturnNodeValue("text_planner"))
text_analyzer_agent.add_node("text_generator", ReturnNodeValue("text_generator"))

text_analyzer_agent.set_entry_point("model")
text_analyzer_agent.add_edge("model", "text_planner")
text_analyzer_agent.add_edge("text_planner", "text_generator")
text_analyzer_agent.add_edge("text_generator", END)
text_analyzer_agent = text_analyzer_agent.compile(
    checkpointer=MemorySaver(),
    # interrupt_before=["text_generator"],
)


# 웹 검색 에이전트
web_searcher_agent = StateGraph(AgentState)

web_search = DuckDuckGoSearchResults(name="WebSearch")
tools = [web_search]

web_searcher_agent.set_entry_point("model")
web_searcher_agent.add_node("model", acall_model)
web_searcher_agent.add_node("tools", ToolNode(tools))

web_searcher_agent.set_entry_point("model")
web_searcher_agent.add_edge("tools", "model")
web_searcher_agent.add_conditional_edges(
    "model",
    pending_tool_calls,
    {"tools": "tools", "done": END},
)
web_searcher_agent = web_searcher_agent.compile(
    checkpointer=MemorySaver(),
    # interrupt_before=["tools"],
)

# 벡터화 에이전트
vectorizer_agent = StateGraph(AgentState)
vectorizer_agent.add_node("splitter", ReturnNodeValue("splitter"))
vectorizer_agent.add_node("tokenizer", ReturnNodeValue("tokenizer"))
vectorizer_agent.add_node("vectorizer", ReturnNodeValue("vectorizer"))

vectorizer_agent.set_entry_point("splitter")
vectorizer_agent.add_edge("splitter", "tokenizer")
vectorizer_agent.add_edge("tokenizer", "vectorizer")
vectorizer_agent.add_edge("vectorizer", END)
vectorizer_agent = vectorizer_agent.compile(
    checkpointer=MemorySaver(),
    # interrupt_before=["vectorizer"]
)

# retriever 에이전트
retriever_agent = StateGraph(AgentState)
retriever_agent.add_node("model", acall_model)
retriever_agent.add_node("retriever", ReturnNodeValue("retriever"))

retriever_agent.set_entry_point("model")
retriever_agent.add_edge("model", "retriever")
retriever_agent.add_edge("retriever", END)
retriever_agent = retriever_agent.compile(
    checkpointer=MemorySaver(),
    # interrupt_before=["retriever"]
)

# PM 에이전트
pm_agent = StateGraph(AgentState)
pm_agent.add_node("model", acall_model)
pm_agent.add_node("pm_planner", ReturnNodeValue("pm_planner"))
pm_agent.add_node("pm_generator", ReturnNodeValue("pm_generator"))

pm_agent.set_entry_point("model")
pm_agent.add_edge("model", "pm_planner")
pm_agent.add_edge("pm_planner", "pm_generator")
pm_agent.add_edge("pm_generator", END)
pm_agent = pm_agent.compile(
    checkpointer=MemorySaver(),
)

# 사이트 기획 에이전트
site_planner_agent = StateGraph(AgentState)
site_planner_agent.add_node("model", acall_model)
site_planner_agent.add_node("site_planner", ReturnNodeValue("site_planner"))
site_planner_agent.add_node("site_generator", ReturnNodeValue("site_generator"))

site_planner_agent.set_entry_point("model")
site_planner_agent.add_edge("model", "site_planner")
site_planner_agent.add_edge("site_planner", "site_generator")
site_planner_agent.add_edge("site_generator", END)
site_planner_agent = site_planner_agent.compile(
    checkpointer=MemorySaver(),
)


# 페이지 구조 기획 에이전트
page_structure_planner_agent = StateGraph(AgentState)
page_structure_planner_agent.add_node("model", acall_model)
page_structure_planner_agent.add_node(
    "page_structure_planner", ReturnNodeValue("page_structure_planner")
)
page_structure_planner_agent.add_node(
    "page_structure_generator", ReturnNodeValue("page_structure_generator")
)

page_structure_planner_agent.set_entry_point("model")
page_structure_planner_agent.add_edge("model", "page_structure_planner")
page_structure_planner_agent.add_edge(
    "page_structure_planner", "page_structure_generator"
)
page_structure_planner_agent.add_edge("page_structure_generator", END)
page_structure_planner_agent = page_structure_planner_agent.compile(
    checkpointer=MemorySaver(),
)

# 콘텐츠 제작 에이전트
content_creator_agent = StateGraph(AgentState)
content_creator_agent.add_node("model", acall_model)
content_creator_agent.add_node("content_planner", ReturnNodeValue("content_planner"))
content_creator_agent.add_node(
    "content_generator", ReturnNodeValue("content_generator")
)

content_creator_agent.set_entry_point("model")
content_creator_agent.add_edge("model", "content_planner")
content_creator_agent.add_edge("content_planner", "content_generator")
content_creator_agent.add_edge("content_generator", END)
content_creator_agent = content_creator_agent.compile(
    checkpointer=MemorySaver(),
)


# 섹션 제작 에디터 에이전트
section_editor_agent = StateGraph(AgentState)
section_editor_agent.add_node("UI_draft", acall_model)
section_editor_agent.add_node("블록추천", retriever_agent)
section_editor_agent.add_node("CSS_editor", ReturnNodeValue("CSS_editor"))
section_editor_agent.add_node("콘텐츠생성", content_creator_agent)

section_editor_agent.set_entry_point("UI_draft")
section_editor_agent.add_edge("UI_draft", "블록추천")
section_editor_agent.add_edge("블록추천", "콘텐츠생성")
section_editor_agent.add_edge("블록추천", "CSS_editor")
section_editor_agent.add_edge("콘텐츠생성", END)
section_editor_agent.add_edge("CSS_editor", END)
section_editor_agent = section_editor_agent.compile(
    checkpointer=MemorySaver(),
)


# Define the graph
graph = StateGraph(AgentState)

# Add nodes with state management
graph.add_node("사용자입력", ReturnNodeValue("input"))
graph.add_node("pdf_분석_에이전트", pdf_analyzer_agent)
graph.add_node("텍스트_분석_에이전트", text_analyzer_agent)
graph.add_node("벡터화_에이전트", vectorizer_agent)

graph.add_node("PM_에이전트", pm_agent)
graph.add_node("사이트_기획_에이전트", site_planner_agent)
graph.add_node("페이지_구조_기획_에이전트", page_structure_planner_agent)
graph.add_node("섹션_제작_에디터_에이전트", section_editor_agent)

# Set entry point
graph.set_entry_point("사용자입력")

# Add edges to vectorizer_agent and END
graph.add_edge("사용자입력", "pdf_분석_에이전트")
graph.add_edge("사용자입력", "텍스트_분석_에이전트")
graph.add_edge("pdf_분석_에이전트", "벡터화_에이전트")
graph.add_edge("텍스트_분석_에이전트", "벡터화_에이전트")
graph.add_edge("벡터화_에이전트", "PM_에이전트")
graph.add_edge("PM_에이전트", "사이트_기획_에이전트")
graph.add_edge("사이트_기획_에이전트", "페이지_구조_기획_에이전트")
graph.add_edge("페이지_구조_기획_에이전트", "섹션_제작_에디터_에이전트")
graph.add_edge("섹션_제작_에디터_에이전트", END)

# Compile the graph with memory saver and state management
sample_agent = graph.compile(
    checkpointer=MemorySaver(),
)
