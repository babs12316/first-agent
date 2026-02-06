from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain import agents
from langchain.agents.middleware import before_agent, AgentState
from langgraph.runtime import Runtime
from langchain.messages import AIMessage

load_dotenv()

WEATHER_KEYWORDS = [
    'weather', 'temperature', 'rain', 'snow', 'sunny', 'cloudy',
    'forecast', 'climate', 'humidity', 'wind', 'storm', 'thunder',
    'cold', 'hot', 'warm', 'cool', 'precipitation', 'degrees',
    'celsius', 'fahrenheit', 'overcast', 'drizzle', 'hail'
]


@before_agent(can_jump_to=["end"])
def is_weather_related_query(state: AgentState, runtime: Runtime):
    print("hey State message", state["messages"][-1].content)
    is_weather_related = any(keyword in state["messages"][-1].content for keyword in WEATHER_KEYWORDS)
    if is_weather_related is False:
        return {
            "messages": [AIMessage("I can only answer weather-related queries")],
            "jump_to": "end"
        }

    return None


@tool
def get_weather(city: str) -> str:
    """Get weather for city (mock)."""
    return f"weather for {city} : sunny 45 degree Celsius "


llm = ChatGroq(model="openai/gpt-oss-120b")

tools = [get_weather]

agent = agents.create_agent(
    llm,
    tools,
    middleware=[is_weather_related_query],
    system_prompt="""You answer ONLY weather-related questions.
If the question is not about weather, say:
"I can only answer weather-related queries." """
)

__all__ = ["agent"]

if __name__ == "__main__":
    result = agent.invoke({
        "messages": [{"role": "user", "content": "how are you?"}]
    })
    print("✅ Weather result:", result)
