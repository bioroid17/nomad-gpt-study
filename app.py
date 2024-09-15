import json
from time import sleep
import openai as client
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.utilities import WikipediaAPIWrapper
from langchain.tools import WikipediaQueryRun, DuckDuckGoSearchResults
from langchain.document_loaders import WebBaseLoader
from typing_extensions import override
from openai import AssistantEventHandler

# # First, we create a EventHandler class to define
# # how we want to handle the events in the response stream.


# class EventHandler(AssistantEventHandler):
#     @override
#     def on_text_created(self, text) -> None:
#         print(f"\nassistant > ", end="", flush=True)

#     @override
#     def on_text_delta(self, delta, snapshot):
#         print(delta.value, end="", flush=True)

#     def on_tool_call_created(self, tool_call):
#         print(f"\nassistant > {tool_call.type}\n", flush=True)

#     def on_tool_call_delta(self, delta, snapshot):
#         if delta.type == "code_interpreter":
#             if delta.code_interpreter.input:
#                 print(delta.code_interpreter.input, end="", flush=True)
#             if delta.code_interpreter.outputs:
#                 print(f"\n\noutput >", flush=True)
#                 for output in delta.code_interpreter.outputs:
#                     if output.type == "logs":
#                         print(f"\n{output.logs}", flush=True)

st.set_page_config(
    page_title="Assistant",
    page_icon="🧰",
)


# OpenAI API key의 유효성을 검사합니다.
# 반환 값은 파일 업로더의 비활성화 여부를 결정합니다.
# disabled가 False일 때 버튼이 활성화되므로, API key가 유효하다면 False를 반환합니다.
def validate_key(api_key):
    try:
        client.api_key = api_key
        client.models.list()
        return False
    except Exception as e:
        print(e)
        return True


with st.sidebar:
    API_KEY = st.text_input("Enter your OpenAI API key", type="password")
    is_invalid = validate_key(API_KEY)
    st.link_button(
        "Github repo",
        "https://github.com/bioroid17/nomad-gpt-study/tree/assistants",
    )

st.title("Assistant")

st.markdown(
    """
Welcome!
            
Use this chatbot to ask questions to an AI!
"""
)


# Tools
def wikipedia_search(inputs):
    query = inputs["query"]
    wp = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    return wp.run(query)


def duckduckgo_search(inputs):
    query = inputs["query"]
    ddg = DuckDuckGoSearchResults()
    result = ddg.run(query)
    # 획득한 결과에서 url을 뽑아내서 리스트로 채우는 코드
    urls = [chunk.split("]")[0] for chunk in result.split("link: ")][1:]
    return urls


def duckduckgo_scrape(inputs):
    urls = inputs["urls"]
    loader = WebBaseLoader(urls)
    docs = loader.load()
    return docs


functions_map = {
    "wikipedia_search": wikipedia_search,
    "duckduckgo_search": duckduckgo_search,
    "duckduckgo_scrape": duckduckgo_scrape,
}

functions = [
    {
        "type": "function",
        "function": {
            "name": "wikipedia_search",
            "description": "Given the query, return the search result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The keyword of the information you want to search.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "duckduckgo_search",
            "description": "Given the query, return the list of urls.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The keyword of the information you want to search.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "duckduckgo_scrape",
            "description": "Given the list of urls, return the list of documents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "urls": {
                        "type": "string",
                        "description": "The keyword of the information you want to search.",
                    }
                },
                "required": ["urls"],
            },
        },
    },
]


#### Utilities
def get_run(run_id, thread_id):
    return client.beta.threads.runs.retrieve(
        run_id=run_id,
        thread_id=thread_id,
    )


def send_message(thread_id, content):
    return client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=content,
    )


def get_messages(thread_id):
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    messages = list(messages)
    return messages


def get_tool_outputs(run_id, thread_id):
    run = get_run(run_id, thread_id)
    outputs = []
    for action in run.required_action.submit_tool_outputs.tool_calls:
        action_id = action.id
        function = action.function
        print(f"Calling function: {function.name} with arg {function.arguments}")
        outputs.append(
            {
                "output": functions_map[function.name](json.loads(function.arguments)),
                "tool_call_id": action_id,
            }
        )
    return outputs


def submit_tool_outputs(run_id, thread_id):
    outputs = get_tool_outputs(run_id, thread_id)
    return client.beta.threads.runs.submit_tool_outputs(
        run_id=run_id,
        thread_id=thread_id,
        tool_outputs=outputs,
    )


def draw_message(message, role):
    with st.chat_message(role):
        st.markdown(message)


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def paint_history():
    for message in st.session_state["messages"]:
        draw_message(
            message["message"],
            message["role"],
        )


if "messages" not in st.session_state:
    st.session_state["messages"] = []

if not is_invalid:
    paint_history()
    if "assistant" not in st.session_state:
        assistant = client.beta.assistants.create(
            name="Search Assistant",
            instructions="You help users do research on the given query using search engines. You give users the summarization of the information you got.",
            model="gpt-4o-mini",
            tools=functions,
        )
        st.session_state["assistant"] = assistant
    else:
        assistant = st.session_state["assistant"]

    if "thread" not in st.session_state:
        thread = client.beta.threads.create()
        st.session_state["thread"] = thread
    else:
        thread = st.session_state["thread"]

    content = st.chat_input("What do you want to search?", disabled=is_invalid)
    if content:
        send_message(thread.id, content)
        save_message(content, "user")
        if "run" not in st.session_state or st.session_state["run"].status == "expired":
            run = client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=assistant.id,
            )
            st.session_state["run"] = run
        else:
            run = st.session_state["run"]

        while run.status != "completed":
            if run.status == "requires_action":
                submit_tool_outputs(run.id, thread.id)
            if run.status == "expired":
                break
            sleep(2)
        else:
            messages = get_messages(thread.id)
            recent_reply = messages[0]
            save_message(recent_reply.content, "assistant")
else:
    st.sidebar.warning("Input OpenAI API Key.")
