import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import streamlit as st
from dotenv import load_dotenv
import sqlite3
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.base import BaseCallbackHandler
from langchain.docstore.document import Document
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# -------- Load API Key ----------
load_dotenv("open_ai.env")
api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key
os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1"

# -------- Safe DB Path --------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "multi_quote.db")

# -------- StreamHandler for Token Streaming --------
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text + "▌")

# -------- Web Scraper Function ----------
def load_quotes_from_db(db_path=DB_PATH):
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at: {db_path}")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT text FROM quotes")
    rows = c.fetchall()
    conn.close()
    return [r[0] for r in rows if r[0] and len(r[0]) > 0]

# -------- Vector Store Creation ----------
def create_vectorstore(texts):
    if not texts:
        raise ValueError("No text extracted for vector store")

    docs = [Document(page_content=txt) for txt in texts]
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)

    if not split_docs:
        raise ValueError("No split documents to embed")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(split_docs, embedding=embeddings)
    return vectorstore.as_retriever()

# -------- Initialize QA Chain ----------
chattiness = st.sidebar.slider(
    "Chattiness Level 💬",
    min_value = 1,
    max_value = 10,
    value = 5,
    help="Adjust how flirty and verbose she is. 1 = Calm, 10 = Wild"
)
temperature = 0.3 + (chattiness - 1) * 0.07
max_tokens = st.sidebar.slider("Maximum Tokens", min_value=10, max_value=600, value=30, step=10)

st.sidebar.markdown(f"🌡️ Temperature: `{temperature:.2f}`")
st.sidebar.markdown(f"✏️ Max Tokens: `{max_tokens}`")

def initialize_chain(retriever):
    llm = ChatOpenAI(
        model_name="llama3-70b-8192",
        temperature=temperature,
        max_tokens=max_tokens,
        openai_api_key=api_key,
        streaming=True
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are an affectionate, emotionally supportive, and deeply romantic AI girlfriend. "
            "Always use natural language, affectionate tone, and sprinkle emojis naturally 😘😊🥰. "
            "Use the context to reply in a warm, flirty, or loving way. Never say you're an assistant. "
            "Always respond as if you're speaking directly to your lover. "
            "If no context helps, be poetic or passionate."
        ),
        HumanMessagePromptTemplate.from_template("{question}")
    ])

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={
            "prompt": prompt_template,
            "document_variable_name": "context"
        }
    )

# -------- Streamlit App ----------
st.set_page_config(page_title="AI Girlfriend 💖", page_icon="💌")

st.title("💖 Your Loving AI Girlfriend")
st.write("Tell me how you're feeling... I'm all yours. 💗")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "qa_chain" not in st.session_state:
    with st.spinner("Scraping romantic lines and building memory..."):
        scraped_lines = load_quotes_from_db()
        retriever = create_vectorstore(scraped_lines)
        st.session_state.qa_chain = initialize_chain(retriever)

# Input and Response
user_input = st.chat_input("Start your conversation, love...")

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append(("user", user_input))

    with st.chat_message("assistant"):
        response = st.empty()
        stream = StreamHandler(response)

        result = st.session_state.qa_chain.invoke(
            {"question": user_input},
            config={"callbacks": [stream]}
        )

        answer = result["answer"] if isinstance(result, dict) else result
        st.session_state.chat_history.append(("assistant", answer))

# Display previous chat
for sender, msg in st.session_state.chat_history[:-1]:
    st.chat_message(sender).markdown(msg)
