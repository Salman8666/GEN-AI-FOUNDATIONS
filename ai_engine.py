from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from typing import List, Generator

class AIEngine:
    def __init__(self, model_name: str = "gemma4:cloud", temperature: float = 0.4, system_prompt: str = "You are a helpful assistant."):
        self.llm = ChatOllama(
            model=model_name,
            temperature=temperature
        )
        self.system_prompt = system_prompt

    def get_response_stream(self, messages: List) -> Generator[str, None, None]:
        """
        Streams the response from the LLM.
        Includes the system prompt at the beginning of the conversation.
        """
        # Ensure the conversation starts with the system message
        full_messages = [SystemMessage(content=self.system_prompt)] + messages

        try:
            for chunk in self.llm.stream(full_messages):
                yield chunk.content
        except Exception as e:
            raise e

    def get_response(self, messages: List) -> str:
        """
        Standard non-streaming response.
        """
        full_messages = [SystemMessage(content=self.system_prompt)] + messages
        response = self.llm.invoke(full_messages)
        return response.content
