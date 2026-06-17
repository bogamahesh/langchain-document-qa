import os
import shutil
from typing import List

import streamlit as st
from dotenv import load_dotenv


load_dotenv()

DOCUMENTS_DIR = "documents"
CHROMA_DIR = "chroma_db"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/embedding-001")


def get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing environment variable: {name}")
    return value


@st.cache_resource
def load_embeddings():
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=get_required_env("GEMINI_API_KEY"),
    )


@st.cache_resource
def load_llm():
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=get_required_env("GEMINI_API_KEY"),
        temperature=0.2,
    )


def load_documents():
    from langchain_community.document_loaders import PyPDFDirectoryLoader

    # RAG step 1: read PDFs from the local knowledge-base folder.
    loader = PyPDFDirectoryLoader(DOCUMENTS_DIR)
    return loader.load()


def split_documents(documents):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # RAG step 2: split large pages into overlapping chunks so retrieval can find focused context.
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return splitter.split_documents(documents)


def build_vector_store(chunks, embeddings):
    from langchain_chroma import Chroma

    # RAG step 3: embed each chunk and persist the vectors locally in ChromaDB.
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    return vector_store


def load_vector_store(embeddings):
    from langchain_chroma import Chroma

    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)


def index_documents() -> tuple[int, int]:
    embeddings = load_embeddings()
    documents = load_documents()
    chunks = split_documents(documents)
    if os.path.isdir(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
    build_vector_store(chunks, embeddings)
    return len(chunks), len(documents)


def retrieve_context(vector_store, question: str, top_k: int = 4):
    # RAG step 4: embed the question and run similarity search against ChromaDB.
    return vector_store.similarity_search_with_score(question, k=top_k)


def answer_question(question: str, matches: List[tuple], llm) -> str:
    context = "\n\n".join(
        f"Source: {doc.metadata.get('source')} page {doc.metadata.get('page')}\n{doc.page_content}"
        for doc, _score in matches
    )

    # RAG step 5: send only retrieved context to Gemini so the answer stays grounded in the PDFs.
    prompt = f"""
You are an AI PDF assistant. Answer the question using only the context below.
If the answer is not present in the context, say that the document does not contain enough information.

Context:
{context}

Question: {question}
Answer:
"""
    return llm.invoke(prompt).content


st.set_page_config(page_title="LangChain Document Q&A", layout="wide")
st.title("LangChain Document Q&A")
st.caption("RAG-based question answering over PDFs using LangChain, ChromaDB, and the Gemini API.")

with st.sidebar:
    st.header("Knowledge Base")
    st.write(f"PDF folder: `{DOCUMENTS_DIR}/`")
    st.write(f"ChromaDB folder: `{CHROMA_DIR}/`")
    if st.button("Index PDFs", type="primary"):
        try:
            with st.spinner("Reading PDFs, chunking text, and creating Gemini embeddings..."):
                chunk_count, page_count = index_documents()
            st.success(f"Indexed {chunk_count} chunks from {page_count} PDF pages.")
        except Exception as exc:
            st.error(str(exc))

question = st.text_input("Ask a question about your PDFs", placeholder="What is the document about?")

if question:
    try:
        if not os.path.isdir(CHROMA_DIR):
            st.warning("Please click `Index PDFs` before asking questions.")
            st.stop()

        with st.spinner("Retrieving relevant chunks and asking Gemini..."):
            embeddings = load_embeddings()
            vector_store = load_vector_store(embeddings)
            matches = retrieve_context(vector_store, question)
            response = answer_question(question, matches, load_llm())

        st.subheader("Answer")
        st.write(response)

        with st.expander("Retrieved sources"):
            for doc, score in matches:
                st.markdown(
                    f"**{doc.metadata.get('source', 'unknown')}**, page "
                    f"{doc.metadata.get('page', 'unknown')} - distance `{score:.3f}`"
                )
                st.write(doc.page_content)
    except Exception as exc:
        st.error(str(exc))
