# LangChain Document Q&A

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

A Retrieval-Augmented Generation (RAG) PDF question-answering app that indexes local documents with ChromaDB and answers questions using the Gemini API.

## Architecture

```text
PDF input
   |
   v
Text extraction from documents/
   |
   v
Text chunking
   |
   v
Gemini embedding model
   |
   v
ChromaDB vector store
   |
   v
Similarity retrieval
   |
   v
Gemini API
   |
   v
Answer output with source chunks
```

## Tech Stack

- Python
- LangChain
- Streamlit
- ChromaDB
- Google Gemini API
- Google Gemini embeddings
- PyPDF
- python-dotenv

## Installation

1. Clone the repository.

```bash
git clone https://github.com/bogamahesh/langchain-document-qa.git
cd langchain-document-qa
```

2. Create and activate a virtual environment.

```bash
python -m venv .venv
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

3. Install dependencies.

```bash
pip install -r requirements.txt
```

4. Create a `.env` file.

```bash
copy .env.example .env
```

On macOS or Linux:

```bash
cp .env.example .env
```

5. Add your Gemini API key to `.env`.

```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-1.5-flash
GEMINI_EMBEDDING_MODEL=models/embedding-001
```

6. Place one or more PDF files in the `documents/` folder.

## Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

Then:

1. Click `Index PDFs` in the sidebar.
2. Wait for the app to read PDFs, split text, create embeddings, and store vectors in ChromaDB.
3. Ask a natural-language question about the indexed PDFs.
4. Review the generated answer and the retrieved source chunks.

If Streamlit is not available in your environment, run the plain HTML fallback:

```bash
python simple_app.py
```

Then open:

```text
http://localhost:8000
```

## Example

Example question:

```text
What is the main topic of the document?
```

Example answer:

```text
The document explains how Retrieval-Augmented Generation connects PDF text,
embeddings, vector search, and a Gemini language model to answer questions
using the document as context.
```

Example retrieved source:

```text
documents/sample.pdf, page 2
The application loads PDF pages, splits them into chunks, stores embeddings in
ChromaDB, and retrieves the most relevant chunks for each user question.
```

## Project Structure

```text
.
|-- app.py              # Streamlit RAG application
|-- simple_app.py       # Minimal HTML fallback app
|-- documents/          # Local PDF input folder
|-- requirements.txt    # Python dependencies
|-- .env.example        # Environment variable template
|-- langchain.ipynb     # Notebook version of the RAG flow
|-- LICENSE
`-- README.md
```

## Notes

- The ChromaDB index is created locally in `chroma_db/` after indexing PDFs.
- `.env` and `chroma_db/` are intentionally ignored by Git.
- Answers are grounded in retrieved PDF chunks; if the answer is not present in the retrieved context, the app is instructed to say so.

## License

This project is available under the MIT License.
