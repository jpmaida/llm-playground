import os
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

KNOWLEDGE_PATH = Path(os.getenv("KNOWLEDGE_PATH"))

def knowledge_explorer():
    st.markdown("#### 📚 Knowledge Base Explorer")

    files = sorted(KNOWLEDGE_PATH.glob("*.md"))

    if not files:
        st.warning("Nenhum arquivo encontrado.")
        return

    col1, col2 = st.columns([1, 3])

    with col1:
        st.markdown("##### 📁 knowledge")

        selected_file = st.radio(
            "Arquivos",
            files,
            format_func=lambda f: f.name,
            label_visibility="collapsed"
        )

    with col2:
        st.markdown(f"##### 📄 {selected_file.name}")
        with st.container(border=True):
            content = selected_file.read_text(encoding="utf-8")
            st.markdown(content)