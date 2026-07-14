from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from client_groq import load_llm

KNOWLEDGE_PATH = Path("knowledge")

def __load_embedding_model__(model: str = "BAAI/bge-m3"):
    """
        Beijing Academy of Artificial Intelligence (BAAI)
    """
    return HuggingFaceEmbeddings(model_name=model)

__embedding_model__ = __load_embedding_model__()

def generate_chunks_and_metadata(chunk_size=100, chunk_overlap=20) -> list:
    docs = list(KNOWLEDGE_PATH.glob("*.md"))
    knowledge_as_text = []
    for d in docs:
        if d.is_file:
            content = d.read_text(encoding='utf-8')
            knowledge_as_text.append(content)
    
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

# def embed_knowledge(chunks: list[str]) -> list[list[float]]:
#     return __embedding_model__.embed_documents(chunks)

def generate_database(chunks: list[str], metadatas: list[str], k: int = 3, fetch_k: int = 4) -> VectorStoreRetriever:
    vectorstore = FAISS.from_texts(texts=chunks, embedding=__embedding_model__, metadatas=metadatas)

    # vectorstore.save_local('index_faiss')

    # Configurando o recuperador de texto / Retriever
    retriever = vectorstore.as_retriever(
        search_type='mmr',
        search_kwargs={'k':3, 'fetch_k':4}
    )

    return retriever

def rag_chain(prompt: str):
    chunks, metadatas = generate_chunks_and_metadata()
    retriever = generate_database(chunks=chunks, metadatas=metadatas)
    llm = load_llm(id_model="qwen/qwen3-32b")
    print(retriever)

    system_prompt = """
        Você é um assistente virtual prestativo e está respondendo perguntas gerais sobre Star Wars. 
        Use os seguintes pedaços de contexto recuperado para responder à pergunta. 
        Se você não sabe a resposta, apenas comente que não sabe dizer com certeza. 
        Mantenha a resposta concisa. 
        Responda em português. \n\n
    """

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Pergunta: {input}\n\n Contexto: {context}")
        ]
    )

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)
    return rag_chain.invoke({"input": prompt})