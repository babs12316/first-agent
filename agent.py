from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain import agents

load_dotenv()


@tool
def add_number(a: int, b: int) -> int:
    """adds to numbers together"""
    return a + b


@tool
def get_weather(city: str) -> str:
    """Get weather for city (mock)."""
    return f"weather for {city} : sunny 45 degrees  (mock data)"


llm = ChatGroq(model="openai/gpt-oss-120b")

tools = [add_number, get_weather]

agent = agents.create_agent(
    llm,
    tools,
    system_prompt="""You are helpful assistant with calculator and weather tools.
     Use tools when asked for addition of numbers or weather. Be concise. """
)

__all__ = ["agent"]
