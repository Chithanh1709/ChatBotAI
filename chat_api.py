from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from chatbot import ChatBot
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
bot = ChatBot()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sửa model với default value và validation
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Câu hỏi không được để trống")

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        logger.info(f"📥 Received chat request: '{request.message}'")
        
        # Message đã được validate bởi Pydantic
        answer = bot.get_answer(request.message)
        logger.info("📤 Response generated")
        
        return {"answer": answer}
        
    except Exception as e:
        logger.error(f"💥 Chat endpoint error: {e}", exc_info=True)
        return {"answer": f"Lỗi server: {str(e)}"}

# Endpoint alternative cho trường hợp JSON không đúng format
@app.post("/chat-flexible")
async def chat_flexible(request: dict):
    try:
        logger.info(f"📥 Received flexible request: {request}")
        
        # Xử lý nhiều trường hợp
        message = request.get("message") or request.get("query") or request.get("text") or ""
        
        if not message or not isinstance(message, str):
            return {"answer": "Vui lòng cung cấp câu hỏi dạng văn bản."}
            
        if not message.strip():
            return {"answer": "Câu hỏi không được để trống."}
            
        answer = bot.get_answer(message.strip())
        return {"answer": answer}
        
    except Exception as e:
        logger.error(f"💥 Flexible chat error: {e}")
        return {"answer": f"Lỗi xử lý: {str(e)}"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Food RAG Chatbot"}