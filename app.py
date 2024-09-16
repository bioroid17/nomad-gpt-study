import json
from time import sleep
import openai as client
from openai.types.beta.threads import Text
from openai.types.beta.threads.runs import RunStep, RunStepDelta
import streamlit as st
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.document_loaders import WebBaseLoader
from typing_extensions import override
from openai import AssistantEventHandler

# # First, we create a EventHandler class to define
# # how we want to handle the events in the response stream.


class EventHandler(AssistantEventHandler):

    message = ""
    run_id = ""
    thread_id = ""

    @override
    def on_text_created(self, text) -> None:
        self.message_box = st.empty()

    def on_text_delta(self, delta, snapshot):
        self.message_box.markdown(snapshot.value)

    def on_text_done(self, text):
        save_message(text.value, "assistant")

    def on_event(self, event):
        if event.event == "thread.run.created":
            # self.message_box = st.empty()
            self.run_id = event.data.id
            self.thread_id = event.data.thread_id

        if event.event == "thread.run.requires_action":
            submit_tool_outputs(self.run_id, self.thread_id)

        # if event.event == "thread.message.delta":
        #     self.message += event.data.delta.content[0].text.value
        #     self.message_box.markdown(self.message)

        # if event.event == "thread.message.completed":
        #     save_message(self.message, "assistant")

    def on_end(self):
        messages = get_messages(self.thread_id)
        print(messages)


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
    ddg = DuckDuckGoSearchRun()
    urls = ddg.run(query)
    return urls


# def duckduckgo_scrape(inputs):
#     urls = inputs["urls"]
#     # 획득한 결과에서 url을 뽑아내서 리스트로 채우는 코드
#     urls = [chunk.split("]")[0] for chunk in result.split("link: ")][1:]
#     loader = WebBaseLoader(urls)
#     docs = loader.load()
#     return docs


functions_map = {
    "wikipedia_search": wikipedia_search,
    "duckduckgo_search": duckduckgo_search,
    # "duckduckgo_scrape": duckduckgo_scrape,
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
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "duckduckgo_scrape",
    #         "description": "Given the list of urls, return the list of documents.",
    #         "parameters": {
    #             "type": "array",
    #             "items": {
    #                 "type": "string",
    #             },
    #             "required": ["urls"],
    #         },
    #     },
    # },
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


# def get_messages(thread_id):
#     messages = client.beta.threads.messages.list(thread_id=thread_id)
#     messages = list(messages)
#     messages.reverse()
#     for message in messages:
#         print(f"{message.role}: {message.content[0].text.value}")


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


def insert_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def paint_history():
    for message in st.session_state["messages"]:
        insert_message(
            message["message"],
            message["role"],
            save=False,
        )


if "run" in st.session_state:
    pass

if "messages" not in st.session_state:
    st.session_state["messages"] = []

paint_history()
if not is_invalid:
    if "assistant" not in st.session_state:
        assistant = client.beta.assistants.create(
            name="Search Assistant",
            instructions="You help users do research on the given query using search engines. You give users the summarization of the information you got.",
            model="gpt-4o-mini",
            tools=functions,
        )
        thread = client.beta.threads.create()
        st.session_state["assistant"] = assistant
        st.session_state["thread"] = thread
    else:
        assistant = st.session_state["assistant"]
        thread = st.session_state["thread"]

    content = st.chat_input("What do you want to search?", disabled=is_invalid)
    if content:
        send_message(thread.id, content)
        insert_message(content, "user")

        with st.chat_message("assistant"):
            with client.beta.threads.runs.stream(
                thread_id=thread.id,
                assistant_id=assistant.id,
                event_handler=EventHandler(),
            ) as stream:
                with st.spinner("Processing..."):
                    stream.until_done()
else:
    st.sidebar.warning("Input OpenAI API Key.")
