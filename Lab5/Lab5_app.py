"""
Lab5_app — FastAPI service for the Alaska Department of Snow agent.

Serves the static chat UI and exposes:
    GET  /              -> static/index.html
    GET  /static/{path} -> static asset
    POST /chat          -> {question, session_id?} -> agent.answer() dict
    GET  /health        -> {"status": "ok"}
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import Lab5_agent as agent

STATIC_DIR = Path(__file__).parent / "Lab5_static"

app = FastAPI(title="Alaska Department of Snow Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None


@app.get("/")
def root():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse({"message": "ADS agent running. UI not found."})


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    result = agent.answer(req.question, session_id=req.session_id)
    return result


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run("Lab5_app:app", host="0.0.0.0", port=port, reload=False)
