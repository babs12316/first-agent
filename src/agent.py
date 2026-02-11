from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain import agents
from langchain.agents.middleware import before_agent, AgentState
from langgraph.runtime import Runtime
from langchain.messages import AIMessage
import httpx
from pydantic import BaseModel, Field

load_dotenv()

WEATHER_KEYWORDS = [
    'weather', 'temperature', 'rain', 'snow', 'sunny', 'cloudy',
    'forecast', 'climate', 'humidity', 'wind', 'storm', 'thunder',
    'cold', 'hot', 'warm', 'cool', 'precipitation', 'degrees',
    'celsius', 'fahrenheit', 'overcast', 'drizzle', 'hail'
]


class WeatherReport(BaseModel):
    city: str
    temperature: float = Field(..., description="temp in Celsius (e.g. 22.5)")
    windspeed: float = Field(..., description="Wind speed km/h (e.g. 12.3)")
    city: str = Field(..., description="City name")


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
def get_weather(city: str) -> WeatherReport | str:
    """
    Get current weather for ANY city worldwide using Open-Meteo (free API).

    Provides:
    • Temperature (°C)
    • Wind speed (km/h)

    INPUT: City name only (e.g., "London", "New York", "Tokyo")
    OUTPUT: Current conditions summary

    Handles: 10,000+ cities, auto-geocoding, error handling.
    """
    try:
        # 1️⃣ Geocode city → lat/lon
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"
        geo_params = {
            "name": city,
            "count": 1,
            "language": "en",
            "format": "json"
        }

        with httpx.Client(timeout=10) as client:
            geo_resp = client.get(geo_url, params=geo_params)
            geo_resp.raise_for_status()
            geo_data = geo_resp.json()

        if "results" not in geo_data:
            return f"Sorry, I couldn't find weather data for {city}."

        location = geo_data["results"][0]
        lat = location["latitude"]
        lon = location["longitude"]

        # 2️⃣ Get current weather
        weather_url = "https://api.open-meteo.com/v1/forecast"
        weather_params = {
            "latitude": lat,
            "longitude": lon,
            "current_weather": True
        }

        with httpx.Client(timeout=10) as client:
            weather_resp = client.get(weather_url, params=weather_params)
            weather_resp.raise_for_status()
            weather_data = weather_resp.json()

        current = weather_data["current_weather"]

        temperature = current["temperature"]
        windspeed = current["windspeed"]

        return WeatherReport(temperature=temperature, windspeed=windspeed, city=city)

    except Exception as e:
        return f"Failed to fetch weather for {city}: {str(e)}"


llm = ChatGroq(model="openai/gpt-oss-120b")

tools = [get_weather]

agent = agents.create_agent(
    llm,
    tools,
    middleware=[is_weather_related_query],
    system_prompt="""You answers only weather related queries. Use get_weather tool. Final response: PLAIN TEXT like 
    "The temperature in NYC is 22 degrees Celsius and the wind speed is 12 kilometers per hour. ". NO BULLETS. NO 
    LISTS. NO FORMATTING."""
)

__all__ = ["agent"]

if __name__ == "__main__":
    result = agent.invoke({
        "messages": [{"role": "user", "content": "how is weather in Frankfurt?"}]
    })
    print("✅ Weather result:", result)

    for token, metadata in agent.stream(
            {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
            stream_mode="messages",
    ):
        print(f"node: {metadata['langgraph_node']}")
        print(f"content: {token.content_blocks}")
        for block in token.content_blocks:
            if block.get("type") == "text":
                print("text:", block["text"])
        print("\n")
