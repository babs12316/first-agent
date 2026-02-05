from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain import agents

load_dotenv()


@tool
def get_weather(city: str) -> str:
    """Get weather for city (mock)."""
    return f"weather for {city} : sunny 45 degree Celsius "


llm = ChatGroq(model="openai/gpt-oss-120b")

tools = [get_weather]

agent = agents.create_agent(
    llm,
    tools,
    system_prompt="""You answer ONLY weather-related questions.
If the question is not about weather, say:
"I can only answer weather-related queries." """
)

__all__ = ["agent"]
