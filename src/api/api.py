from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
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


def stream_response(message: str):
    """Generator function that yields SSE chunks"""
    from src.agent import agent
    first_block = True
    for token, metadata in agent.stream(
            {"messages": [{"role": "user", "content": message}]},
            stream_mode="messages",
    ):

        for block in token.content_blocks:
            if block.get("type") == "text":
                if first_block and block.get('text') != "I can only answer weather-related queries":
                    first_block = False
                    continue
                # Yield each text chunk as SSE
                yield f"data: {block['text']}\n\n"




@app.post("/chat")
def chat(request: ChatInput):
    return StreamingResponse(
        stream_response(request.message),  # Pass the generator
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
