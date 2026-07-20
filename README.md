# LLM Playground - A Star Wars Saga 🚀

This repository is a Streamlit application for exploring generative AI concepts, embeddings, semantic search, and retrieval-augmented generation (RAG). The goal is to demonstrate, in a practical way, how to combine large language models, embeddings, and vector databases to answer questions based on structured documents.

The project uses the Star Wars universe as a friendly and visual way to illustrate these concepts.

## What the application does

The application offers three main experiences:

1. LLM Playground
   - Allows you to send prompts to models hosted on Groq platform.
   - Displays the model response along with technical details such as the final prompt, metadata, and token usage.

2. Semantic search with embeddings
   - Generates embeddings for text and compares the query with a predefined knowledge base.
   - Uses cosine similarity and a reduced-dimensionality visualization with PCA.

3. RAG pipeline
   - Loads Markdown files from the knowledge folder.
   - Splits the documents into chunks.
   - Creates a vector database with FAISS.
   - Retrieves the most relevant passages to answer questions more accurately.

---

## Requirements

Before you start, make sure you have installed:

- Python 3.9+ (3.11 recommended)
- pip
- Git
- A virtual environment (recommended)

---

## Install dependencies

From the project root, run:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If you use a tool such as conda or poetry, you can adapt the environment setup accordingly.

---

## How to start the application

The application is started with:

```bash
streamlit run main.py
```

Run this command from the repository root. Streamlit will open the interface in your browser and you can access the app locally.

If the command is not available, install Streamlit manually:

```bash
pip install streamlit
```

---

## Environment variable configuration

The project uses a `.env` file to configure values such as:

- Groq API key
- embedding model
- knowledge folder
- vector database folder

Example content for the `.env` file:

```env
GROQ_API_KEY=your_key_here
EMBEDDINGS_MODEL=BAAI/bge-m3
KNOWLEDGE_PATH=knowledge
DATABASE_PATH=database
```

Important:

- Never share your real key in public repositories or messages.
- The `.env` file is listed in `.gitignore`, so it should not be versioned.
- The project uses `python-dotenv` to load this file automatically when the app starts.

The `.env` file should not contain real values in commits or screenshots. For safety, use a temporary token or a test key when needed.

---

## Repository structure

```text
.
├── main.py                  # Streamlit app entry point
├── requirements.txt        # Python dependencies
├── knowledge/              # Knowledge files in Markdown
├── database/               # FAISS vector indexes and databases
├── notebooks/              # Experiment notebooks
├── src/                    # Application source code
│   ├── db/                 # Semantic search and vector database logic
│   ├── llm/                # Model integration and response utilities
│   ├── prompts/            # Prompt templates
│   ├── rag/                # RAG pipeline
│   └── ui/                 # Interface and help components
```

---

## Explanation of the directories inside src

### src/db

Responsible for embeddings and semantic search logic.

- `semantic_search.py`
  - Generates embeddings for text and compares the user query with a knowledge base.
  - Uses cosine similarity for ranking.
  - Displays a 2D projection with PCA to visualize the relation between embeddings.

- `vector_database.py`
  - Creates and queries a vector database with FAISS.
  - Converts chunks into embeddings.
  - Stores documents and metadata in a vector index.
  - Supports `L2` and `IP` metrics.

### src/llm

Contains the layer that integrates with language models.

- `client_groq.py`
  - Connects to the Groq API using `ChatGroq`.
  - Loads the selected model.
  - Sends prompts and returns the model response.

- `utils.py`
  - Formats responses for display in the UI.
  - Extracts reasoning text when the model returns tags such as `<think>`.

### src/prompts

Stores the prompts used by the application.

- `prompt_templates.py`
  - Defines system prompts for the playground and the RAG flow.
  - Helps guide the personality and behavior of the model.

### src/rag

Implements the retrieval-augmented generation flow.

- `rag.py`
  - Reads Markdown files from the `knowledge` folder.
  - Splits the documents into chunks with overlap.
  - Creates a vector store from embeddings.
  - Retrieves the most relevant chunks and builds the context for the LLM to answer questions.

- `rag_langchain.py`
  - Alternative implementation based on LangChain components.
  - Demonstrates a more declarative approach for building retrieval chains.

### src/ui

Responsible for the visual layer of the application.

- `help_ui.py`
  - Stores help text used in the Streamlit interface.

- `knowledge_explorer.py`
  - Allows you to view the Markdown files from the knowledge base directly in the app.

---

## How the interface is structured

The application has three main tabs:

### 1. Playground

Lets you experiment directly with a Groq model using custom prompts.

### 2. Semantic Search (Embeddings)

Shows how embeddings can represent meaning and enable search based on similarity.

### 3. RAG

Runs a full retrieval-augmented generation workflow using the files in the knowledge folder to answer questions with context.

---

## AI techniques used

This project combines several common techniques used in modern AI applications:

### Large language models (LLMs)

The application uses models hosted by Groq to generate text responses from prompts.

### Embeddings

Text is converted into numerical vectors by embedding models. This allows semantic representation and efficient comparison of documents.

### Vector search

The vector database is built with FAISS, a high-performance library for similarity search.

### RAG (Retrieval-Augmented Generation)

Instead of relying only on the model’s internal knowledge, the system retrieves relevant passages from local documents and uses them as context for the LLM to answer questions.

### Chunking

Documents are split into smaller pieces to make retrieval, organization, and focused context easier.

### Prompt engineering

Prompts are structured to guide the model’s behavior with clear instructions and specific context.

### Embedding visualization

The app uses PCA to reduce the dimensionality of embeddings to two dimensions for easier visualization.

---

## Application execution flow

1. The user enters a prompt or question.
2. The system loads the selected model.
3. The workflow can:
   - answer directly with the LLM, or
   - retrieve context from documents and compose a response with RAG.
4. The response is formatted and displayed in the Streamlit interface.

---

## Knowledge base

The `knowledge` folder contains Markdown files about Star Wars. These files are used as the basis for the RAG workflow and the knowledge explorer.

Examples of files present:

- `characters.md`
- `organizations.md`
- `planets.md`
- `ships.md`
- `timeline.md`
- `quotes.md`

---

## Usage tips

- Start with the Playground tab to test simple prompts.
- Try the Semantic Search tab to understand how embeddings compare questions and concepts.
- In the RAG tab, ask questions like:
  - Who trained Luke Skywalker?
  - What is the purpose of the Death Star?
  - Which characters appear in the Rebel Alliance?

---

## Security

- Do not expose your API key in code.
- Use a local `.env` file.
- Never share keys in commits, screenshots, or public messages.

If you want, the next step could be to add a `.env.example` file to standardize configuration for other developers.

