import os
from dotenv import load_dotenv

from pathlib import Path
from typing import Any, List, Tuple
from client_groq import load_llm, generate_answer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

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

def embed_knowledge(chunks: list[str]) -> list[list[float]]:
    return __embedding_model__.embed_documents(chunks)

def embed_query(query: str) -> list[float]:
    return __embedding_model__.embed_query(text=query)

def generate_database(chunks: list[str], metadatas: list[str], k: int = 3, fetch_k: int = 4) -> FAISS:
    vectorstore = FAISS.from_texts(
        texts=chunks, 
        embedding=__embedding_model__, 
        metadatas=metadatas
    )

    vectorstore.save_local('database/index_faiss')

    return vectorstore

def search(query: str, vectorstore: FAISS, k: int = 3) -> List[Tuple[Any, float]]:
    query_embedding = embed_query(query=query)
    results = vectorstore.similarity_search_with_score_by_vector(
        embedding=query_embedding,
        k=k
    )
    return results

def create_context(search_results):
    context = ""

    for document, score in search_results:
        context += f"""
        Fonte: {document.metadata['source']}

        {document.page_content}

        ------------------------

        """

    return context

def rag_pipeline(query: str, id_model: str="qwen/qwen3-32b", top_k: int = 3, chunk_size: int = 100, temperature: float = 0.7):
    chunks, metadatas = generate_chunks_and_metadata(chunk_size=chunk_size)
    vectorstore = generate_database(chunks=chunks, metadatas=metadatas)
    results = search(query, vectorstore, k=top_k)
    context = create_context(results)
    
    prompt = """
    Você é um especialista em Star Wars.
    Utilize exclusivamente o contexto abaixo.
    Se não souber responder, informe que a informação não está presente.

    Contexto

    {context}

    Pergunta

    {query}
    """
    
    llm = load_llm(id_model=id_model, temperature=temperature)
    response, response_content = generate_answer(llm, prompt=prompt.format(context=context, query=query))
    return response, results