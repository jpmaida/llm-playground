from langchain_groq import ChatGroq

def load_llm(id_model: str, temperature: float = 0.7) -> ChatGroq:
    llm = ChatGroq(
        model=id_model,
        temperature=temperature,
        max_tokens = None,
        timeout = None,
        max_retries = 2
    )
    return llm

def generate_answer(llm: ChatGroq, prompt: str):
    answer = llm.invoke(prompt)
    return answer, answer.content