import streamlit as st
import pandas as pd
from dotenv import load_dotenv

import groq
from client_groq import generate_answer, load_llm
import help_ui
import prompt_templates
from utils import format_res, extract_thinking
from semantic_search import embed, search, display_similarities
from knowledge_explorer import knowledge_explorer
from rag import rag_pipeline

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
        system_prompt = st.text_area("Enter your System Prompt here:", value=prompt_templates.YODA_SYSTEM_PROMPT.strip(), width=500, height=200, help=help_ui.TEXTO_HELP_PLAYGROUND_SYSTEM_PROMPT.strip())
    else:
        system_prompt = ""
    
    user_prompt = st.text_area("Enter your prompt here:")

    if st.button("Generate Answer"):
        if user_prompt is not None and user_prompt.strip() != "":
            with st.spinner("Generating answer..."):
                try:
                    llm = load_llm(model, temperature)
                    if system_prompt.strip() == "":
                        answer, content = generate_answer(llm, user_prompt)
                    else:
                        answer, content = generate_answer(llm, system_prompt.format(pergunta=user_prompt))
                except groq.BadRequestError as e:
                    st.error("""
                             Message: {message}
                             
                             Stacktrace: {body}
                            """.format(message=e.message, body=e.body), icon="🚨")
                    st.stop()
                except Exception as e:
                    st.error(e, icon="🚨")
                    st.stop()
                
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
    tab_knowledge_base, tab_retrieval = st.tabs(["Knowledge Base", "Retrieval"])
    with tab_knowledge_base:
        knowledge_explorer()
    with tab_retrieval:
        question = st.text_input("Make your question and I will answer:", placeholder="Type your question here...")
        is_dev_tools = st.toggle("Dev Tools")
        
        if is_dev_tools:
            with st.container(border=True):
                is_retrieved_chunks = st.toggle("Show retrieved chunks ?")
                is_vectors = st.toggle("Show vectors ?")
                system_prompt = st.text_area(label="System prompt:", value=prompt_templates.STAR_WARS_SPECIALIST_RAG.strip(), width=500, height=300, help=help_ui.TEXTO_HELP_RAG_SYSTEM_PROMPT)
                chunk_size = st.slider("Chunk size:", min_value=100, max_value=1000, value=100, step=50, width=200)
                top_k = st.slider("Top K:", min_value=3, max_value=10, value=3, step=1, width=200)
                temperature = st.slider("Temperature:", min_value=0.1, max_value=1.0, value=0.7, step=0.1, width=200)
                is_reasoning = st.toggle("Show reasoning ?")
        
        if st.button("Search", key="question_btn"):
            st.divider()  # Creates the horizontal line
            if question is not None and question.strip() != "":
                with st.spinner("Generating answer..."):
                    if is_dev_tools:
                        try:
                            response, retrieved_chunks = rag_pipeline(question, top_k=top_k, chunk_size=chunk_size, temperature=temperature, system_prompt=system_prompt)
                        except groq.BadRequestError as e:
                            st.error("""
                                    Message: {message}
                                    
                                    Stacktrace: {body}
                                    """.format(message=e.message, body=e.body), icon="🚨")
                            st.stop()
                        except Exception as e:
                            st.error(e, icon="🚨")
                            st.stop()
                        
                        if is_retrieved_chunks:
                            st.caption("Retrieved chunks:")
                            for chunk in retrieved_chunks:
                                content = chunk[0]
                                score = chunk[1]
                                filename_plus_score = content.metadata.get('source') + " - Score: " + str(score)
                                with st.expander(label=filename_plus_score, expanded=False):
                                    st.markdown(content.page_content)
                                    if is_vectors:
                                        with st.expander(label="Chunk as vector", expanded=False):
                                            st.markdown(embed(content.page_content))

                        col1_chunk, col2_chunk = st.columns([0.1, 0.9])
                        with col1_chunk:
                            st.caption("Chunk size:")
                        with col2_chunk:
                            st.markdown(chunk_size)
                        
                        col1_overlap, col2_overlap = st.columns([0.1, 0.9])
                        with col1_overlap:
                            st.caption("Chunk overlap:")
                        with col2_overlap:
                            st.markdown("20% of " + str(chunk_size) + " = " + str(int(chunk_size*0.2)))

                        if is_reasoning:
                            st.caption("Thinking:")
                            st.markdown(extract_thinking(response.content))

                        st.caption("Answer:")
                        st.markdown(format_res(response.content, return_thinking=False))
                    else:
                        try:
                            response, retrieved_chunks = rag_pipeline(question)
                        except groq.BadRequestError as e:
                            st.error("""
                                    Message: {message}
                                    
                                    Stacktrace: {body}
                                    """.format(message=e.message, body=e.body), icon="🚨")
                            st.stop()
                        except Exception as e:
                            st.error(e, icon="🚨")
                            st.stop()
                        st.caption("Answer:")
                        st.markdown(format_res(response.content, return_thinking=False))
            else:
                st.toast("Please enter a question before searching.", icon="⚠️")