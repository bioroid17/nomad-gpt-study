from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import streamlit as st
import openai


st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)

st.title("SiteGPT")

st.markdown(
    """
    Ask questions about the content of a website.
            
    Start by writing the URL of the website on the sidebar.
"""
)


llm = ChatOpenAI(
    temperature=0.1,
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


answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                  
    Then, give a score to the answer between 0 and 5.
    If the answer answers the user question the score should be high, else it should be low.
    Make sure to always include the answer's score even if it's 0.
    Context: {context}
                                                  
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                  
    Your turn!
    Question: {question}
"""
)


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
    }


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.
            Use the answers that have the highest score (more helpful) and favor the most recent ones.
            Cite sources and return the sources of the answers as they are, do not change them.
            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nData:{answer['date']}\n"
        for answer in answers
    )

    return choose_chain.invoke({"question": question, "answers": condensed})


def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    nav = soup.find("nav")
    aside = soup.find("aside")
    astro_breadcrumbs = soup.find("astro-breadcrumbs")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    if nav:
        nav.decompose()
    if aside:
        aside.decompose()
    if astro_breadcrumbs:
        astro_breadcrumbs.decompose()
    return (
        str(soup.get_text())
        .replace("\n", " ")
        .replace("\xa0", " ")
        .replace(
            "Cloudflare DashboardDiscordCommunityLearning CenterSupport Portal  Cookie Settings",
            "",
        )
    )


@st.cache_data(show_spinner="Loading website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=200
    )
    try:
        loader = SitemapLoader(
            url,
            filter_urls=[
                r"^(.*\/ai-gateway\/).*",
                r"^(.*\/vectorize\/).*",
                r"^(.*\/workers-ai\/).*",
            ],
            parsing_function=parse_page,
        )
        loader.requests_per_second = 5
        # Set a realistic user agent

        docs = loader.load_and_split(text_splitter=splitter)
        cache_dir = LocalFileStore(f"./.cache/sitegpt/{url.split('/')[2]}")
        embeddings = OpenAIEmbeddings()
        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
            embeddings, cache_dir
        )
        vector_store = FAISS.from_documents(docs, cached_embeddings)
        return vector_store.as_retriever()
    except Exception as e:
        return []


with st.sidebar:
    API_KEY = st.text_input("Enter your OpenAI API key", type="password")
    is_invalid = validate_key(API_KEY)
    url = st.text_input(
        "Write down a URL",
        placeholder="https://example.com",
        disabled=is_invalid,
    )
    st.link_button(
        "Github repo",
        "https://github.com/bioroid17/nomad-gpt-study/tree/sitegpt",
    )
    with st.expander("View source code"):
        st.markdown(
            """
```python
from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import streamlit as st
import openai


st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)

st.title("SiteGPT")

st.markdown(
    '''
    Ask questions about the content of a website.
            
    Start by writing the URL of the website on the sidebar.
'''
)


llm = ChatOpenAI(
    temperature=0.1,
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


answers_prompt = ChatPromptTemplate.from_template(
    '''
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                  
    Then, give a score to the answer between 0 and 5.
    If the answer answers the user question the score should be high, else it should be low.
    Make sure to always include the answer's score even if it's 0.
    Context: {context}
                                                  
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                  
    Your turn!
    Question: {question}
'''
)


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
    }


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            '''
            Use ONLY the following pre-existing answers to answer the user's question.
            Use the answers that have the highest score (more helpful) and favor the most recent ones.
            Cite sources and return the sources of the answers as they are, do not change them.
            Answers: {answers}
            ''',
        ),
        ("human", "{question}"),
    ]
)


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm
    condensed = "\\n\\n".join(
        f"{answer['answer']}\\nSource:{answer['source']}\\nData:{answer['date']}\\n"
        for answer in answers
    )

    return choose_chain.invoke({"question": question, "answers": condensed})


def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    nav = soup.find("nav")
    aside = soup.find("aside")
    astro_breadcrumbs = soup.find("astro-breadcrumbs")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    if nav:
        nav.decompose()
    if aside:
        aside.decompose()
    if astro_breadcrumbs:
        astro_breadcrumbs.decompose()
    return (
        str(soup.get_text())
        .replace("\\n", " ")
        .replace("\\xa0", " ")
        .replace(
            "Cloudflare DashboardDiscordCommunityLearning CenterSupport Portal  Cookie Settings",
            "",
        )
    )


@st.cache_data(show_spinner="Loading website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=200
    )
    try:
        loader = SitemapLoader(
            url,
            filter_urls=[
                r"^(.*\/ai-gateway\/).*",
                r"^(.*\/vectorize\/).*",
                r"^(.*\/workers-ai\/).*",
            ],
            parsing_function=parse_page,
        )
        loader.requests_per_second = 5
        # Set a realistic user agent

        docs = loader.load_and_split(text_splitter=splitter)
        cache_dir = LocalFileStore(f"./.cache/sitegpt/{url.split('/')[2]}")
        embeddings = OpenAIEmbeddings()
        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
            embeddings, cache_dir
        )
        vector_store = FAISS.from_documents(docs, cached_embeddings)
        return vector_store.as_retriever()
    except Exception as e:
        return []


with st.sidebar:
    API_KEY = st.text_input("Enter your OpenAI API key", type="password")
    is_invalid = validate_key(API_KEY)
    url = st.text_input(
        "Write down a URL",
        placeholder="https://example.com",
        disabled=is_invalid,
    )
    st.link_button(
        "Github repo",
        "https://github.com/bioroid17/nomad-gpt-study/tree/streamlit",
    )
    with st.expander("View source code"):
        st.markdown('''
<코드 본문>
''')

if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write down a Sitemap URL")
    else:
        retriever = load_website(url)
        query = st.text_input(
            "Ask a question to the website.",
            disabled=is_invalid,
        )

        if query:
            chain = (
                {
                    "docs": retriever,
                    "question": RunnablePassthrough(),
                }
                | RunnableLambda(get_answers)
                | RunnableLambda(choose_answer)
            )

            result = chain.invoke(query)
            st.markdown(result.content.replace("$", "\$"))

```
"""
        )

if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write down a Sitemap URL")
    else:
        retriever = load_website(url)
        query = st.text_input(
            "Ask a question to the website.",
            disabled=is_invalid,
        )

        if query:
            chain = (
                {
                    "docs": retriever,
                    "question": RunnablePassthrough(),
                }
                | RunnableLambda(get_answers)
                | RunnableLambda(choose_answer)
            )

            result = chain.invoke(query)
            st.markdown(result.content.replace("$", "\$"))
