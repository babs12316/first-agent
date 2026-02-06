from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from src.api.health import router as health_router

load_dotenv()

app = FastAPI()

app.include_router(health_router)

WEATHER_KEYWORDS = [
    'weather', 'temperature', 'rain', 'snow', 'sunny', 'cloudy',
    'forecast', 'climate', 'humidity', 'wind', 'storm', 'thunder',
    'cold', 'hot', 'warm', 'cool', 'precipitation', 'degrees',
    'celsius', 'fahrenheit', 'overcast', 'drizzle', 'hail'
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://first-agent-ui.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatInput(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatInput):
    try:
        is_weather_related = any(keyword in request.message for keyword in WEATHER_KEYWORDS)

        if not is_weather_related:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Invalid query",
                    "reply": "I can only answer weather-related queries"
                }
            )
        from src.agent import agent
        result = agent.invoke(
            {"messages": [{"role": "user", "content": request.message}]}
        )
        messages = result["messages"]
        final_answer = messages[-1].content

        return {"reply": final_answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
