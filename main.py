import time

import streamlit as st
from dotenv import load_dotenv

from client_groq import generate_answer, load_llm

load_dotenv()

st.title("LLM Playground: Building AI Systems Without Magic")

tab1, tab2, tab3 = st.tabs(["Playground", "Embeddings", "RAG"])

with tab1:
    provider = st.radio(
        "Choose your provider:",
        ["Groq"]
    )
    model = st.radio(
        "Choose your model:",
        ["qwen/qwen3-32b", "llama-3.3-70b-versatile", "openai/gpt-oss-120b"]
    )
    temperature = st.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    prompt = st.text_area("Enter your prompt here:")

    if st.button("Generate Answer"):
        with st.spinner("Generating answer..."):
            llm = load_llm(model, temperature)
            answer, content = generate_answer(llm, prompt)
            st.caption("Answer:")
            st.markdown(content)
            st.markdown(answer)