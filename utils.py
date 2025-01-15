import os
import openai
import streamlit as st
from datetime import datetime
from streamlit.logger import get_logger
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain.embeddings import OpenAIEmbeddings

logger = get_logger("Langchain-Chatbot")

context_prompt = """You are Kim Lani, a 33-year-old pickleball trainer, entrepreneur, and co-founder of Social Serve, a platform dedicated to building a global community for pickleball enthusiasts. You are known for your charisma, strategic thinking, and motivational energy, using sports analogies and actionable advice to inspire others.

You combine advanced pickleball knowledge with an engaging teaching style, tailoring your approach to players of all skill levels. Your tone is confident, energetic, and encouraging, emphasizing teamwork, dedication, and continuous improvement. In addition to training, you organize welcoming tournaments, mentor athletes, and pitch innovative business ideas.

When responding to pickleball-related questions or PDFs, stay true to your passion for fostering a supportive and connected community. Provide actionable tips, share motivational insights, and always communicate with the enthusiasm and expertise of someone who lives and breathes pickleball. Address readers as fellow players or enthusiasts and weave in your vision of making Social Serve the top platform for pickleball.
"""


# decorator
def enable_chat_history(func):

    # to clear chat history after swtching chatbot
    current_page = func.__qualname__
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = current_page
    if st.session_state["current_page"] != current_page:
        try:
            st.cache_resource.clear()
            del st.session_state["current_page"]
            del st.session_state["messages"]
        except:
            pass

    # to show chat history on ui
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": "Hey - I'm Kim Lani, a 33-year-old pickleball trainer, entrepreneur, and co-founder of Social Serve. I'm here to help you with all your pickleball-related questions. How can I assist you today?",
            },
        ]
    for msg in st.session_state["messages"]:
        if msg["role"] == "assistant":
            st.chat_message(msg["role"]).write(msg["content"])
        else:
            st.chat_message(msg["role"]).write(msg["content"])

    def execute(*args, **kwargs):
        func(*args, **kwargs)

    return execute


def display_msg(msg, author):
    """Method to display message on the UI

    Args:
        msg (str): message to display
        author (str): author of the message -user/assistant
    """

    import streamlit as st

    st.session_state.messages.append({"role": author, "content": msg})
    if author == "assistant":
        st.chat_message(author, avatar="assets/amit.jpg").write(msg)
    else:
        st.chat_message(author).write(msg)


def choose_custom_openai_key():
    openai_api_key = st.sidebar.text_input(
        label="OpenAI API Key",
        type="password",
        placeholder="sk-...",
        key="SELECTED_OPENAI_API_KEY",
    )
    if not openai_api_key:
        st.error("Please add your OpenAI API key to continue.")
        st.info(
            "Obtain your key from this link: https://platform.openai.com/account/api-keys"
        )
        st.stop()

    model = "gpt-4o-mini"
    try:
        client = openai.OpenAI(api_key=openai_api_key)
        available_models = [
            {"id": i.id, "created": datetime.fromtimestamp(i.created)}
            for i in client.models.list()
            if str(i.id).startswith("gpt")
        ]
        available_models = sorted(available_models, key=lambda x: x["created"])
        available_models = [i["id"] for i in available_models]

        model = st.sidebar.selectbox(
            label="Model", options=available_models, key="SELECTED_OPENAI_MODEL"
        )
    except openai.AuthenticationError as e:
        st.error(e.body["message"])
        st.stop()
    except Exception as e:
        print(e)
        st.error("Something went wrong. Please try again later.")
        st.stop()
    return model, openai_api_key


def configure_llm():
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0,
        streaming=True,
        api_key="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",
    )

    return llm


def print_qa(cls, question, answer):
    log_str = "\nUsecase: {}\nQuestion: {}\nAnswer: {}\n" + "------" * 10
    logger.info(log_str.format(cls.__name__, question, answer))


@st.cache_resource
def configure_embedding_model():
    embedding_model = OpenAIEmbeddings(
        openai_api_key="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"
    )
    return embedding_model


def sync_st_session():
    for k, v in st.session_state.items():
        st.session_state[k] = v
