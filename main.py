import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from client_groq import generate_answer, load_llm
from utils import format_res
from semantic_search import embed, search, display_similarities
from knowledge_explorer import knowledge_explorer
from rag import rag_chain
import prompt_templates

load_dotenv()

st.set_page_config(layout="wide", page_icon="🤖", page_title="LLM Playground")
st.title("LLM Playground: Building AI Systems Without Magic")

tab1, tab2, tab3 = st.tabs(["Playground 🎮", "Semantic Search (Embeddings) 🔍", "RAG"])

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
    
    is_system_prompt = st.checkbox("Add System Prompt?")
    
    system_prompt = ""
    if is_system_prompt:
        system_prompt = st.text_area("Enter your System Prompt here:", value=prompt_templates.YODA_SYSTEM_PROMPT.strip())
    else:
        system_prompt = ""
    
    user_prompt = st.text_area("Enter your prompt here:")

    if st.button("Generate Answer"):
        if user_prompt is not None and user_prompt.strip() != "":
            with st.spinner("Generating answer..."):
                llm = load_llm(model, temperature)
                if system_prompt.strip() == "":
                    answer, content = generate_answer(llm, user_prompt)
                else:
                    answer, content = generate_answer(llm, system_prompt.format(pergunta=user_prompt))
                st.caption("Answer:")
                st.markdown(format_res(content))

                st.divider()

                st.subheader("Developer Tools")

                with st.expander("View Complete Answer", expanded=True):
                    col1_final_prompt, col2_final_prompt = st.columns([0.1, 0.9])
                    with col1_final_prompt:
                        st.caption("Final Prompt:")
                    with col2_final_prompt:
                        if system_prompt.strip() == "":
                            st.markdown("_Your Prompt_: {prompt}".format(prompt=user_prompt))
                        else:
                            col1_system_prompt, col2_system_prompt = st.columns([0.1, 0.9])
                            with col1_system_prompt:
                                st.caption("System Prompt:")
                            with col2_system_prompt:
                                st.markdown(system_prompt)

                            col1_user_prompt, col2_user_prompt = st.columns([0.1, 0.9])
                            with col1_user_prompt:
                                st.caption("User Prompt:")
                            with col2_user_prompt:
                                st.markdown(user_prompt)
                    
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

with tab2:
    st.radio(
        "Choose your model:",
        ["BAAI/bge-m3"]
    )
    query = st.text_input("What are you looking for?", placeholder="Type your query here...")
    if st.button("Search"):
        if query is not None and query.strip() != "":
            embedded_query = embed(query)
            results = search(embedded_query, top_k=None)
            similarity_results = display_similarities(embedded_query, query)

            st.caption("Results:")
            df = pd.DataFrame(results, columns=["Item", "Similarity Score"])
            st.table(data=df, width="content")
            st.plotly_chart(similarity_results, width="stretch")
        else:
            st.toast("Please enter a query before searching.", icon="⚠️")

with tab3:
    pipeline_dot = """
    digraph G {
        rankdir=LR; # Left-to-Right orientation
        node [shape=box, style="filled,rounded", color="#1f77b4", fontcolor=white, fontname="Helvetica", fontsize="12pt"];
        
        # Define Nodes
        A [label="📄 Documents", fillcolor="#1f77b4"];
        B [label="✂️ Chunking", fillcolor="#1f77b4"];
        C [label="🧠 Embeddings", fillcolor="#1f77b4"];
        D [label="🗄️ Vector Store", fillcolor="#1f77b4"];
        E [label="🔍 Retrieval", fillcolor="#1f77b4"];
        F [label="🤖 LLM", fillcolor="#1f77b4"];
        G [label="💬 Answer", fillcolor="#1f77b4"];
        
        # Define Pipeline Connections
        A -> B;
        B -> C;
        C -> D;
        D -> E;
        E -> F;
        F -> G;
    }
    """

    # Render the diagram inside your app
    st.graphviz_chart(pipeline_dot)
    tab_knowledge_base, tab_retrieval, tab_devtools = st.tabs(["Knowledge Base", "Retrieval", "Developer Tools"])
    with tab_knowledge_base:
        knowledge_explorer()
    with tab_retrieval:
        question = st.text_input("Make your question and I will answer:", placeholder="Type your question here...")
        on_off = st.toggle("Dev Tools")
        
        if on_off:
            chunk_size = st.slider("Chunk size:", min_value=100, max_value=1000, value=100, step=50, width=200)
        
        if st.button("Search", key="question_btn"):
            if question is not None and question.strip() != "":
                print("Searching for question:", question)
                with st.spinner("Generating answer..."):
                    response = rag_chain(question)
                    st.caption("Retrieved chunks:")
                    retrieved_chunks = response["context"]
                    for chunk in retrieved_chunks:
                        filename = chunk.metadata.get('source')
                        with st.expander(label=filename, expanded=False):
                            st.markdown(chunk.page_content)
                    st.caption("Answer:")
                    st.markdown(response['answer'])
            else:
                st.toast("Please enter a question before searching.", icon="⚠️")