from fastapi import FastAPI
from pydantic import BaseModel
from graph.graph import graph
from langchain_core.messages import HumanMessage
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message:   str
    thread_id: str = "default"


@app.post("/chat")
async def chat(req: ChatRequest):
    config = {"configurable": {"thread_id": req.thread_id}}
    result = graph.invoke(
        {"messages": [HumanMessage(content=req.message)]},
        config=config
    )
    return {"response": result["tool_result"]}


@app.get("/health")
def health():
    return {"status": "ok"}
