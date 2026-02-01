from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from agent import agent
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI()

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
        result = agent.invoke(
            {"messages": [{"role": "user", "content": request.message}]}
        )
        messages = result["messages"]
        final_answer = messages[-1].content

        return {"reply": final_answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__" :
    uvicorn.run(app, host="0.0.0.0", port=8000)