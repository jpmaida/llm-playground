from enum import Enum
from typing import Any
from langchain_core.documents import Document
from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
from langchain_community.vectorstores import FAISS
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
import uuid

load_dotenv()

class Metric(Enum):
    L2 = "L2"
    IP = "IP"

class VectorDatabase:
    def __init__(self, embedding_model):
        # model=os.getenv("EMBEDDINGS_MODEL")
        self.__embedding_model__ = embedding_model

    def __build_documents__(self, chunks, metadatas):
        return [
            Document(
                page_content=chunk,
                metadata=metadata
            )
            for chunk, metadata in zip(chunks, metadatas)
        ]

    def __build_embeddings__(self, chunks):
        embeddings = self.__embedding_model__.embed_documents(chunks)
    
        return np.asarray(
            embeddings,
            dtype=np.float32
        )

    def __build_index__(self, embeddings, metric=Metric.L2):
        vectors = embeddings.copy()
        dimension = vectors.shape[1]

        if metric == Metric.IP:
            faiss.normalize_L2(vectors)
            index = faiss.IndexFlatIP(dimension)
        else:
            index = faiss.IndexFlatL2(dimension)

        index.add(vectors)

        return index

    def __build_docstore__(self, documents):
        ids = [
            str(uuid.uuid4())
            for _ in documents
        ]

        docstore = InMemoryDocstore(
            {
                id_: document
                for id_, document in zip(ids, documents)
            }
        )

        mapping = {
            i: ids[i]
            for i in range(len(ids))
        }

        return docstore, mapping

    def build_vectorstore(self, chunks: list[str], metadatas: list[str], metric: Metric = Metric.L2) -> FAISS:
        documents = self.__build_documents__(chunks, metadatas)

        embeddings = self.__build_embeddings__(
            chunks
        )

        index = self.__build_index__(
            embeddings,
            metric
        )

        docstore, mapping = self.__build_docstore__(
            documents
        )

        database_path = os.getenv("DATABASE_PATH") + "/" + metric.value

        if os.path.isdir(database_path):
            vectorstore = FAISS.load_local(database_path, self.__embedding_model__, allow_dangerous_deserialization=True)
        else :
            vectorstore = FAISS(
                embedding_function=self.__embedding_model__,
                index=index,
                docstore=docstore,
                index_to_docstore_id=mapping
            )
            vectorstore.save_local(database_path)

        return vectorstore
    
    def search(self, vectorstore: FAISS, query: str, metric=Metric.L2, top_k=3) -> Any:
        query_embedding = self.__embedding_model__.embed_query(query)
        
        query = np.asarray(
            [query_embedding],
            dtype=np.float32
        )

        if metric == Metric.IP:
            faiss.normalize_L2(query)

        scores, indexes = vectorstore.index.search(
            query,
            top_k
        )

        results = []

        for score, idx in zip(scores[0], indexes[0]):
            document_id = vectorstore.index_to_docstore_id[idx]
            document = vectorstore.docstore.search(
                document_id
            )

            results.append({
                "score": float(score),
                "document": document,
                "chunk": document.page_content,
                "metadata": document.metadata
            })

        return results