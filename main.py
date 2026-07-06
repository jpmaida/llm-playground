import time

import streamlit as st

from dotenv import load_dotenv

from client_groq import generate_answer, load_llm
from utils import format_res

load_dotenv()

st.set_page_config(layout="wide", page_icon="🤖", page_title="LLM Playground")
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
    temperature = st.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.7, step=0.1, width=200)
    prompt = st.text_area("Enter your prompt here:")

    if st.button("Generate Answer"):
        if prompt is not None and prompt.strip() != "":
            with st.spinner("Generating answer..."):
                llm = load_llm(model, temperature)
                answer, content = generate_answer(llm, prompt)
                st.caption("Answer:")
                st.markdown(format_res(content))

                st.divider()

                st.subheader("Developer Tools")

                with st.expander("View Complete Answer", expanded=True):
                    col1_content, col2_content = st.columns([0.1, 0.9])
                    with col1_content:
                        st.caption("Raw Content:")
                    with col2_content:
                        st.markdown(format_res(content, return_thinking=True))
                    
                    col1_kwargs, col2_kwargs = st.columns([0.1, 0.9])
                    with col1_kwargs:
                        st.caption("Additional Kwargs:")
                    with col2_kwargs:
                        st.json(answer.additional_kwargs, expanded=True)

                    col1_response_metadata, col2_response_metadata = st.columns([0.1, 0.9])
                    with col1_response_metadata:
                        st.caption("Response Metadata:")
                    with col2_response_metadata:
                        st.json(answer.response_metadata, expanded=True)

                    col1_id, col2_id = st.columns([0.1, 0.9])    
                    with col1_id:
                        st.caption("Id:")
                    with col2_id:
                        st.markdown(answer.id)

                    col1_tool_calls, col2_tool_calls = st.columns([0.1, 0.9])
                    with col1_tool_calls:
                        st.caption("Tool Calls:")
                    with col2_tool_calls:
                        st.markdown(answer.tool_calls)
                    
                    col1_usage_metadata, col2_usage_metadata = st.columns([0.1, 0.9])
                    with col1_usage_metadata:
                        st.caption("Usage Metadata:")
                    with col2_usage_metadata:
                        st.json(answer.usage_metadata, expanded=True)
        else:
            st.toast("Please enter a prompt before generating an answer.", icon="⚠️")