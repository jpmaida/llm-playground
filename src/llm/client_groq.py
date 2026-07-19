import groq
from langchain_groq import ChatGroq

def load_llm(id_model: str, temperature: float = 0.7) -> ChatGroq:
    try:
        llm = ChatGroq(
            model=id_model,
            temperature=temperature,
            max_tokens = None,
            timeout = None,
            max_retries = 2
        )
        return llm
    except groq.APIStatusError as e:
        print(f"Groq API Error Status: {e.status_code}")
        print(f"Error Message: {e.response}")
        
        # Manually re-raise the exception or raise a new groq error
        raise groq.BadRequestError(
            message="Connection error when creating the client", 
            response=e.response, 
            body=e.body
        )
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise Exception(e)

def generate_answer(llm: ChatGroq, prompt: str):
    try:
        answer = llm.invoke(prompt)
        return answer, answer.content
    except groq.APIStatusError as e:
        print(f"Groq API Error Status: {e.status_code}")
        print(f"Error Message: {e.response}")
        
        # Manually re-raise the exception or raise a new groq error
        raise groq.BadRequestError(
            message="Invoke model error with prompt", 
            response=e.response, 
            body=e.body
        )
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise Exception(e)