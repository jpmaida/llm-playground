from enum import Enum
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

    def build_documents(self, chunks, metadatas):
        return [
            Document(
                page_content=chunk,
                metadata=metadata
            )
            for chunk, metadata in zip(chunks, metadatas)
        ]

    def build_embeddings(self, chunks):
        embeddings = self.__embedding_model__.embed_documents(chunks)
    
        return np.asarray(
            embeddings,
            dtype=np.float32
        )

    def build_index(self, embeddings, metric=Metric.L2):
        vectors = embeddings.copy()
        dimension = vectors.shape[1]

        if metric == "IP":
            faiss.normalize_L2(vectors)
            index = faiss.IndexFlatIP(dimension)
        else:
            index = faiss.IndexFlatL2(dimension)

        index.add(vectors)

        return index

    def build_docstore(self, documents):
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
        documents = self.build_documents(chunks, metadatas)

        embeddings = self.build_embeddings(
            chunks
        )

        index = self.build_index(
            embeddings,
            metric
        )

        docstore, mapping = self.build_docstore(
            documents
        )

        vectorstore = FAISS(
            embedding_function=self.__embedding_model__,
            index=index,
            docstore=docstore,
            index_to_docstore_id=mapping
        )

        return vectorstore
    
    def search(self, vectorstore, query: str, metric=Metric.L2, k=3):
        query_embedding = self.__embedding_model__.embed_query(query)
        
        query = np.asarray(
            [query_embedding],
            dtype=np.float32
        )

        if metric == Metric.IP:
            faiss.normalize_L2(query)

        scores, indexes = vectorstore.index.search(
            query,
            k
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