from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import plotly.express as px

def load_embedding_model(model: str = "BAAI/bge-m3"):
    """
        Beijing Academy of Artificial Intelligence (BAAI)
    """
    return HuggingFaceEmbeddings(model_name=model)

embedding_model = load_embedding_model()

def embed(text) -> list[float]:
    return embedding_model.embed_query(text)

knowledge = {
    "Luke Skywalker": embed("Luke Skywalker"),
    "Yoda": embed("Yoda"),
    "Obi-Wan Kenobi": embed("Obi-Wan Kenobi"),
    "Leia Organa": embed("Leia Organa"),
    "Darth Vader": embed("Darth Vader"),
    "Death Star": embed("Death Star"),
    "Millennium Falcon": embed("Millennium Falcon"),
    "Tatooine": embed("Tatooine"),
    "Jedi": embed("Jedi"),
    "Sith": embed("Sith"),
    "Pizza": embed("Pizza"),
    "Linux": embed("Linux"),
    "Kubernetes": embed("Kubernetes"),
    "Red Hat": embed("Red Hat")
}

def search(embedded_query: list[float], top_k: int = 5):
    similarities = {key: cosine_similarity([embedded_query], [value])[0][0] for key, value in knowledge.items()}
    sorted_results = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
    if top_k is None:
        return sorted_results
    return sorted_results[:top_k]

def display_similarities(embedded_query: list[float], query_as_text: str):
    # Embeddings da base
    labels = list(knowledge.keys())
    X = np.array(list(knowledge.values()))

    # Cria e ajusta o PCA
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    # Gera o embedding da consulta
    query_embedding = np.array(embedded_query)

    # Projeta a consulta no MESMO espaço do PCA
    query_2d = pca.transform([query_embedding])

    # Monta um DataFrame para o Plotly
    df_knowledge_base = pd.DataFrame({
        "x": X_2d[:, 0],
        "y": X_2d[:, 1],
        "label": labels,
        "type": "Knowledge Base"
    })

    # Adiciona a consulta
    df_query = pd.DataFrame({
        "x": [query_2d[0][0]],
        "y": [query_2d[0][1]],
        "label": [query_as_text],
        "type": ["Query"]
    })

    df = pd.concat([df_knowledge_base, df_query], ignore_index=True)
    df["size"] = 1
    df.loc[df["type"] == "Query", "size"] = 2

    fig = px.scatter(
        df,
        x="x",
        y="y",
        text="label",
        color="type",
        size="size",
        title="Embedding Space (PCA)",
        color_discrete_map={
            "Knowledge Base": "royalblue",
            "Query": "red"
        }
    )

    fig.update_traces(textposition="top center")

    return fig