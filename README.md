# 📚 AI PDF Assistant (RAG Pipeline)

A complete **Retrieval-Augmented Generation (RAG)** pipeline built with LangChain that allows you to "chat" with your PDF documents. The system reads your PDFs, converts them into vector embeddings, stores them in a Pinecone database, and uses advanced Large Language Models (LLMs) to accurately answer questions based on the document's content.

## 🚀 Features
* **Document Processing:** Automatically loads and splits large PDF documents into manageable chunks.
* **Local Embeddings:** Uses free, local HuggingFace embeddings (`all-MiniLM-L6-v2`) to maintain privacy and eliminate embedding API costs.
* **Vector Database:** Integrates with Pinecone Vector Database for lightning-fast cosine-similarity searches.
* **Dynamic LLM Routing:** Uses OpenRouter to automatically route questions to the fastest available free open-source models (like Llama 3 or Gemma) to prevent rate-limiting.

## 🛠️ Technology Stack
* **Python 3**
* **LangChain** (Framework)
* **Pinecone** (Vector Database)
* **HuggingFace** (Local Embeddings)
* **OpenRouter** (Cloud LLM API)

## ⚙️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/bogamahesh/langchain-document-qa.git
   cd langchain-document-qa
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/Scripts/activate  # On Windows
   ```

3. **Install Dependencies**
   Make sure you install the required packages (you can generate a `requirements.txt` from your environment):
   ```bash
   pip install langchain langchain-classic langchain-openai langchain-community pinecone-client pypdf sentence-transformers python-dotenv requests
   ```

4. **Set up API Keys**
   Create a new file named `.env` in the root folder, using `.env.example` as a template:
   ```env
   PINECONE_API_KEY="your_actual_pinecone_key"
   OPENROUTER_API_KEY="your_actual_openrouter_key"
   ```

5. **Add Documents**
   Place the PDFs you want to analyze inside a folder named `documents/` in the root directory.

## 🧠 Usage
Open the Jupyter Notebook (`langchain.ipynb`) and run the cells sequentially. 
1. The script will read your PDFs.
2. It will embed the text and push it to your Pinecone index.
3. Scroll to the bottom to change the `query` variable and ask your AI anything about the document!

## 📜 License
This project is open-source and available under the MIT License.
