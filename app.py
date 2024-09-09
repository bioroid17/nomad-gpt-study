import json
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.retrievers import WikipediaRetriever
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema.runnable import RunnablePassthrough
import streamlit as st
import openai, os

if not os.path.exists("./.cache/quizgpt/files/"):
    os.mkdir("./.cache/quizgpt/files/")

st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title("QuizGPT")


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


function = {
    "name": "create_quiz",
    "description": "function that takes a list of questions and answers and returns a quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "difficulty": {
                "type": "string",
            },
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            },
        },
        "required": ["questions"],
    },
}


with st.sidebar:
    API_KEY = st.text_input("Enter your OpenAI API key", type="password")

llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-4o-mini",
    streaming=True,
    callbacks=[
        StreamingStdOutCallbackHandler(),
    ],
    api_key=API_KEY,
).bind(
    function_call={
        "name": "create_quiz",
    },
    functions=[
        function,
    ],
)

prompt = PromptTemplate.from_template(
    """
    You are a helpful assistant that is role playing as a teacher.
         
    Based ONLY on the following context make 10 (TEN) questions to test the user's knowledge about the text.
    
    Each question should have 4 answers, three of them must be incorrect and one should be correct.
         
    Use (o) to signal the correct answer.

    Also, you will create the quiz depending on the difficulty that the user wants.

    The difficulty will be either 'Easy' or 'Hard'.
         
    Question examples:
         
    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue(o)
         
    Question: What is the capital or Georgia?
    Answers: Baku|Tbilisi(o)|Manila|Beirut
         
    Question: When was Avatar released?
    Answers: 2007|2001|2009(o)|1998
         
    Question: Who was Julius Caesar?
    Answers: A Roman Emperor(o)|Painter|Actor|Model
         
    Your turn!

    Difficulty: {difficulty}
         
    Context: {context}
"""
)


# OpenAI API key의 유효성을 검사합니다.
# 반환 값은 파일 업로더의 비활성화 여부를 결정합니다.
# disabled가 False일 때 버튼이 활성화되므로, API key가 유효하다면 False를 반환합니다.
def validate_key(api_key):
    try:
        openai.api_key = api_key
        openai.Model.list()
        return False
    except Exception:
        return True


@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quizgpt/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs, topic, difficulty):
    chain = (
        {
            "context": format_docs,
            "difficulty": get_difficulty,
        }
        | prompt
        | llm
    )
    return chain.invoke(_docs)


@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5)
    docs = retriever.get_relevant_documents(term)
    return docs


def get_difficulty(_):
    return difficulty


with st.sidebar:
    is_invalid = validate_key(API_KEY)
    docs = None
    topic = None
    choice = st.selectbox(
        "Choose what you want to use.",
        (
            "File",
            "Wikipedia Article",
        ),
        disabled=is_invalid,
    )
    if choice == "File":
        file = st.file_uploader(
            "Upload a .docx, .txt or .pdf file",
            type=["pdf", "txt", "docx"],
            disabled=is_invalid,
        )
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input(
            "Name of the article",
            disabled=is_invalid,
        )
        if topic:
            docs = wiki_search(topic)
    difficulty = st.selectbox(
        "Choose difficulty.",
        ("Easy", "Hard"),
        disabled=is_invalid,
    )
    show_answer = st.toggle(
        "Show answers when submitted",
        False,
        disabled=is_invalid,
    )

if not docs:
    st.markdown(
        """
    Welcome to QuizGPT.
                
    I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
                
    Get started by uploading a file or searching on Wikipedia in the sidebar.
    """
    )
else:
    response = run_quiz_chain(docs, topic if topic else file.name, difficulty)
    data_str = response.additional_kwargs["function_call"]["arguments"]
    data = json.loads(data_str)
    questions = data["questions"]
    questions_num = len(questions)
    score = 0
    with st.form("questions_form"):
        for question in questions:
            st.write(question["question"])
            value = st.radio(
                "Select an option",
                [answer["answer"].replace("(o)", "") for answer in question["answers"]],
                index=None,
            )
            if {"answer": value, "correct": True} in question["answers"]:
                st.success("Correct!")
                score += 1
            elif value is not None:
                answer_str = ""
                if show_answer:
                    for index, answer in enumerate(question["answers"]):
                        if "correct" in answer and answer["correct"]:
                            answer_str = f" Answer: {index + 1}"
                            break
                st.error("Wrong!" + answer_str)
        button = st.form_submit_button()
        if score == questions_num:
            st.balloons()
