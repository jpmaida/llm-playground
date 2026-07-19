from typing import Final

TEXTO_HELP_RAG_SYSTEM_PROMPT: Final[str] = """
Este é o system prompt com área destinada ao contexto, marcada pela palavra-chave context, e a pergunta do usuário, marcada pela palavra-chave query. 
Tome cuidado ao editar esta informação pois se o contexto ou a pergunta do usuário forem apagados a técnica de RAG não funcionará de acordo com o esperado.
Este system prompt também não suporta novas palavras-chave pois este sistema não idealizado com este tipo de funcionalidade.
"""

TEXTO_HELP_PLAYGROUND_SYSTEM_PROMPT: Final[str] = """
Este é o system prompt com área destinada a definir diretivas gerais a LLM e uma personalidade. A pergunta (user prompt) do usuário é marcada pela palavra-chave pergunta. 
Tome cuidado ao editar esta informação pois se a pergunta do usuário for eliminada o modelo não conseguirá responder de forma adequada.
Este system prompt também não suporta novas palavras-chave pois este sistema não idealizado com este tipo de funcionalidade.
"""