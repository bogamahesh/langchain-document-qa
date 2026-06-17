import html
import os
import shutil
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

from dotenv import load_dotenv


load_dotenv()

DOCUMENTS_DIR = "documents"
CHROMA_DIR = "chroma_db"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/embedding-001")


def required_env(name):
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing environment variable: {name}")
    return value


def get_embeddings():
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=required_env("GEMINI_API_KEY"),
    )


def get_llm():
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=required_env("GEMINI_API_KEY"),
        temperature=0.2,
    )


def load_documents():
    from langchain_community.document_loaders import PyPDFDirectoryLoader

    # RAG step 1: load PDF pages from the local documents directory.
    return PyPDFDirectoryLoader(DOCUMENTS_DIR).load()


def split_documents(documents):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # RAG step 2: chunk text so each vector represents a focused passage.
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return splitter.split_documents(documents)


def index_pdfs():
    from langchain_chroma import Chroma

    documents = load_documents()
    chunks = split_documents(documents)
    if os.path.isdir(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    # RAG step 3: create Gemini embeddings and store them in a persistent ChromaDB collection.
    Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=CHROMA_DIR,
    )

    return f"Indexed {len(chunks)} chunks from {len(documents)} PDF pages."


def answer_question(question):
    from langchain_chroma import Chroma

    if not os.path.isdir(CHROMA_DIR):
        return "Please index PDFs first.", []

    vector_store = Chroma(persist_directory=CHROMA_DIR, embedding_function=get_embeddings())

    # RAG step 4: retrieve the chunks semantically closest to the user's question.
    matches = vector_store.similarity_search_with_score(question, k=4)
    context = "\n\n".join(
        f"Source: {doc.metadata.get('source')} page {doc.metadata.get('page')}\n{doc.page_content}"
        for doc, _score in matches
    )

    # RAG step 5: Gemini receives the retrieved context and produces the grounded answer.
    prompt = f"""
You are an AI PDF assistant. Answer the question using only the context below.
If the document does not contain enough information, say so clearly.

Context:
{context}

Question: {question}
Answer:
"""
    return get_llm().invoke(prompt).content, matches


def render_page(message="", answer="", sources=None):
    sources = sources or []
    source_html = ""
    for doc, score in sources:
        source_html += f"""
        <details>
            <summary>{html.escape(str(doc.metadata.get("source", "unknown")))}
            page {html.escape(str(doc.metadata.get("page", "unknown")))}
            distance {score:.3f}</summary>
            <pre>{html.escape(doc.page_content)}</pre>
        </details>
        """

    return f"""<!doctype html>
<html>
<head>
    <meta charset="utf-8">
    <title>LangChain Document Q&A</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 980px; margin: 32px auto; padding: 0 18px; line-height: 1.5; }}
        h1 {{ margin-bottom: 4px; }}
        .muted {{ color: #555; margin-top: 0; }}
        form {{ margin: 22px 0; }}
        input[type=text] {{ width: 75%; padding: 10px; font-size: 16px; }}
        button {{ padding: 10px 14px; font-size: 15px; cursor: pointer; }}
        .box {{ border: 1px solid #ddd; padding: 16px; border-radius: 8px; margin-top: 16px; }}
        .success {{ background: #f2fff5; border-color: #b9e7c4; }}
        .answer {{ background: #f8fbff; border-color: #bfd7ff; }}
        pre {{ white-space: pre-wrap; background: #f7f7f7; padding: 12px; border-radius: 6px; }}
    </style>
</head>
<body>
    <h1>LangChain Document Q&A</h1>
    <p class="muted">RAG-based PDF question answering using LangChain, ChromaDB, and the Gemini API.</p>

    <form method="post">
        <input type="hidden" name="action" value="index">
        <button type="submit">Index PDFs</button>
    </form>

    <form method="post">
        <input type="hidden" name="action" value="ask">
        <input type="text" name="question" placeholder="What is the document about?" required>
        <button type="submit">Ask</button>
    </form>

    {f'<div class="box success">{html.escape(message)}</div>' if message else ''}
    {f'<div class="box answer"><h2>Answer</h2><p>{html.escape(answer)}</p></div>' if answer else ''}
    {f'<div class="box"><h2>Retrieved Sources</h2>{source_html}</div>' if source_html else ''}
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.respond(render_page())

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")
        form = parse_qs(body)
        action = form.get("action", [""])[0]

        try:
            if action == "index":
                page = render_page(message=index_pdfs())
            else:
                question = form.get("question", [""])[0]
                answer, sources = answer_question(question)
                page = render_page(answer=answer, sources=sources)
        except Exception as exc:
            page = render_page(message=str(exc))

        self.respond(page)

    def respond(self, content):
        payload = content.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


if __name__ == "__main__":
    server = ThreadingHTTPServer(("localhost", 8000), Handler)
    print("LangChain Document Q&A running at http://localhost:8000")
    server.serve_forever()
