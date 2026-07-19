import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import plotly.express as px

load_dotenv()

model_path=os.getenv("EMBEDDINGS_MODEL")
def __load_embedding_model__(model: str = model_path):
    """
        Beijing Academy of Artificial Intelligence (BAAI)
    """
    return HuggingFaceEmbeddings(model_name=model)

embedding_model = __load_embedding_model__()

def embed(text) -> list[float]:
    return embedding_model.embed_query(text)

knowledge = {
    "Luke Skywalker": embed("Luke Skywalker é um Jedi treinado por Obi-Wan Kenobi e Yoda."),
    "Yoda": embed("Yoda é um Mestre Jedi que treinou Luke Skywalker e outros Jedi."),
    "Obi-Wan Kenobi": embed("Obi-Wan Kenobi é um Mestre Jedi que treinou Luke Skywalker e Anakin Skywalker."),
    "Leia Organa": embed("Leia Organa é uma líder da Aliança Rebelde, irmã de Luke Skywalker."),
    "Darth Vader": embed("Darth Vader é um Lorde Sith e pai de Luke Skywalker. Ele era Anakin Skywalker."),
    "Death Star": embed("Death Star é uma estação espacial destrutiva do Império Galáctico."),
    "Millennium Falcon": embed("Millennium Falcon é uma nave espacial usada pela Aliança Rebelde."),
    "Tatooine": embed("Tatooine é um planeta do universo de Star Wars, foi o lar de Luke Skywalker e Anakin Skywalker."),
    "Jedi": embed("Jedi é um membro da ordem jedi."),
    "Sith": embed("Sith é um membro da ordem sith."),
    "Pizza": embed("Pizza é um prato tradicional italiano feito com massa, molho de tomate e queijo."),
    "Linux": embed("Linux é um sistema operacional de código aberto."),
    "Kubernetes": embed("Kubernetes é uma plataforma de orquestração de contêineres."),
    "Red Hat": embed("Red Hat é uma empresa de software livre.")
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

    fig = px.scatter(
        df,
        x="x",
        y="y",
        text="label",
        color="type",
        title="Embedding Space (PCA)",
        color_discrete_map={
            "Knowledge Base": "royalblue",
            "Query": "red"
        }
    )

    fig.update_traces(textposition="top center")
    fig.update_layout(
        xaxis_title="PCA Component 1",
        yaxis_title="PCA Component 2"
    )

    return fig