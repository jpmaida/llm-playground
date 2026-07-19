import os
from dotenv import load_dotenv

from pathlib import Path
from typing import Any, List, Tuple
from client_groq import load_llm, generate_answer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import prompt_templates
from src.vector_database import VectorDatabase

load_dotenv()

model_path=os.getenv("EMBEDDINGS_MODEL")
def __load_embedding_model__(model: str = model_path):
    """
        Beijing Academy of Artificial Intelligence (BAAI)
    """
    return HuggingFaceEmbeddings(model_name=model)

__embedding_model__ = __load_embedding_model__()

def generate_chunks_and_metadata(chunk_size=100) -> list:
    KNOWLEDGE_PATH = Path("knowledge")
    docs = list(KNOWLEDGE_PATH.glob("*.md"))
    knowledge_as_text = []
    for d in docs:
        if d.is_file:
            content = d.read_text(encoding='utf-8')
            knowledge_as_text.append(content)
    
    # 20% for overlapping
    chunk_overlap = chunk_size * 0.2
    
    text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=chunk_size,
      chunk_overlap=chunk_overlap
    )

    chunks = []
    metadatas = []
    for index, k in enumerate(knowledge_as_text):
        splits = text_splitter.split_text(k)
        chunks.extend(splits)
        doc = docs[index]
        for s in splits:
            metadatas.append({
                "source": doc.name
            })
    
    return chunks, metadatas

def create_context(search_results):
    context = ""

    for r in search_results:
        document = r['document']
        
        context += f"""
        Fonte: {document.metadata['source']}

        {document.page_content}

        ------------------------

        """

    return context

def rag_pipeline(query: str, id_model: str="qwen/qwen3.6-27b", top_k: int = 3, chunk_size: int = 100, temperature: float = 0.7, system_prompt: str = prompt_templates.STAR_WARS_SPECIALIST_RAG.strip()):
    chunks, metadatas = generate_chunks_and_metadata(chunk_size=chunk_size)
    
    db = VectorDatabase(embedding_model=__embedding_model__)
    vectorstore = db.build_vectorstore(chunks=chunks, metadatas=metadatas)
    
    results = db.search(vectorstore, query=query, top_k=top_k)

    context = create_context(results)
    llm = load_llm(id_model=id_model, temperature=temperature)
    response, response_content = generate_answer(llm, prompt=system_prompt.format(context=context, query=query))
    return response, results