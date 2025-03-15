# server.py
import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn
import os
from dotenv import load_dotenv
import datetime
import time
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    DateTime,
    ForeignKey,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

import fitz  # PyMuPDF
import pymupdf4llm

from openai import OpenAI

load_dotenv()

OPEN_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_KEY = os.getenv("DEEPSEEK_KEY")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DB_URL = os.getenv("DATABASE_URL") or "postgresql://postgres:postgres@db:5432/chatapp"

# Database setup
Base = declarative_base()


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, default="New Conversation")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(
        DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow
    )

    messages = relationship(
        "Message", back_populates="conversation", cascade="all, delete-orphan"
    )


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"))
    role = Column(String)  # "user" or "assistant"
    content = Column(Text)
    model = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    conversation = relationship("Conversation", back_populates="messages")


# Database connection with retry logic
def get_db_engine(max_retries=5, retry_interval=5):
    retries = 0
    while retries < max_retries:
        try:
            engine = create_engine(DB_URL)
            # Test connection
            with engine.connect() as conn:
                pass
            return engine
        except Exception as e:
            retries += 1
            print(f"Database connection attempt {retries} failed: {e}")
            if retries < max_retries:
                print(f"Retrying in {retry_interval} seconds...")
                time.sleep(retry_interval)
            else:
                print("Max retries reached. Could not connect to database.")
                raise


# Create database engine and tables with retry
engine = get_db_engine()
Base.metadata.create_all(bind=engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_openai_client():
    if not OPEN_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set in the environment variables.")
    return OpenAI(api_key=OPEN_API_KEY)


def get_deepseek_client():
    if not DEEPSEEK_KEY:
        raise ValueError("DEEPSEEK_KEY is not set in the environment variables.")
    return OpenAI(
        base_url="https://api.deepseek.com",
        api_key=DEEPSEEK_KEY,
    )


def get_claude_client():
    if not CLAUDE_API_KEY:
        raise ValueError("CLAUDE_API_KEY is not set in the environment variables.")

    return OpenAI(
        api_key=CLAUDE_API_KEY,  # Your Anthropic API key
        base_url="https://api.anthropic.com/v1/",  # Anthropic's API endpoint
    )


def get_gemini_client():
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set in the environment variables.")
    return OpenAI(
        api_key=GEMINI_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )


def load_conversation_history(db, conversation_id, max_history=10):
    """Load conversation history from the database"""
    if not conversation_id:
        return []

    messages = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.timestamp)
        .all()
    )

    # Convert to the format expected by the API
    history = [{"role": msg.role, "content": msg.content} for msg in messages]

    # Limit to max_history
    if len(history) > max_history:
        history = history[-max_history:]

    return history


# Initialize FastAPI
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your React app's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Conversation-ID"],  # Expose our custom header
)


OPEN_AI_MODELS = ("o3-mini", "gpt-4o")
DEEPSEEK_MODELS = ("deepseek-chat", "deepseek-reasoner")
CLAUDE_MODELS = ("claude-3-7-sonnet-20250219", "claude-3-5-haiku-20241022")
GEMINI_MODELS = ("gemini-2.0-flash",)

# Supported file types for document processing
CODE_EXTENSIONS = (
    ".txt",
    ".js",
    ".ts",
    ".py",
    ".java",
    ".c",
    ".cpp",
    ".json",
    ".html",
    ".css",
    ".xml",
    ".md",
    ".csv",
    ".yaml",
    ".yml",
    ".sql",
    ".sh",
    ".bash",
    ".go",
    ".php",
    ".rb",
    ".swift",
    ".kt",
    ".dart",
)


# Document processing functions
def extract_text_from_pdf(file_bytes):
    try:
        # Open the PDF from memory
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            # Extract text using pymupdf4llm for better formatting
            md_text = pymupdf4llm.to_markdown(doc)
            return md_text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return f"Error processing PDF: {str(e)}"


def process_text_file(file_bytes):
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        # Try different encodings if utf-8 fails
        try:
            return file_bytes.decode("latin-1")
        except:
            return "Error: Could not decode text file"


@app.post("/process-document")
async def process_document(file: UploadFile = File(...), file_type: str = Form(...)):
    try:
        contents = await file.read()

        if file_type == "application/pdf":
            text = extract_text_from_pdf(contents)
        elif file_type.startswith("text/") or any(
            file.filename.endswith(ext) for ext in CODE_EXTENSIONS
        ):
            text = process_text_file(contents)
        else:
            raise HTTPException(
                status_code=400, detail=f"Unsupported file type: {file_type}"
            )

        return {"filename": file.filename, "text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat(
    model: str = Form(...),
    prompt: str = Form(...),
    max_history: int = Form(10),  # Default limit of 10 messages
    conversation_id: int = Form(None),  # Optional conversation ID
):
    """
    Chat with an AI model.

    This endpoint handles both new conversations and continuing existing ones:
    - If conversation_id is not provided, a new conversation will be created
    - If conversation_id is provided, the conversation will be continued

    The conversation_id is returned in the X-Conversation-ID response header.
    """
    try:
        # Get database session
        db = SessionLocal()

        try:
            # If conversation_id is provided, load history from database
            if conversation_id:
                # Check if conversation exists
                conversation = (
                    db.query(Conversation)
                    .filter(Conversation.id == conversation_id)
                    .first()
                )
                if not conversation:
                    raise HTTPException(
                        status_code=404, detail="Conversation not found"
                    )

                # Load history from database
                message_history = load_conversation_history(
                    db, conversation_id, max_history
                )

                if len(message_history) > max_history:
                    message_history = message_history[-max_history:]
            else:
                message_history = []

        finally:
            db.close()

        # Add the current prompt
        message_history.append({"role": "user", "content": prompt})

        try:
            # Get appropriate client
            if model in OPEN_AI_MODELS:
                client = get_openai_client()
            elif model in DEEPSEEK_MODELS:
                client = get_deepseek_client()
            elif model in CLAUDE_MODELS:
                client = get_claude_client()
            elif model in GEMINI_MODELS:
                client = get_gemini_client()
            else:
                raise HTTPException(status_code=400, detail="Unsupported model")
        except ValueError as e:
            raise HTTPException(status_code=500, detail=str(e))

        # Create completion with history
        openai_response = client.chat.completions.create(
            model=model,
            messages=message_history,
            stream=True,
        )

        # Save to database
        db = SessionLocal()
        try:
            # Get or create conversation
            if conversation_id:
                conversation = (
                    db.query(Conversation)
                    .filter(Conversation.id == conversation_id)
                    .first()
                )
                if not conversation:
                    raise HTTPException(
                        status_code=404, detail="Conversation not found"
                    )
            else:
                # Create new conversation with first few words of prompt as title
                title = prompt[:30] + "..." if len(prompt) > 30 else prompt
                conversation = Conversation(title=title)
                db.add(conversation)
                db.commit()
                db.refresh(conversation)

            # Save user message
            user_message = Message(
                conversation_id=conversation.id,
                role="user",
                content=prompt,
                model=model,
            )
            db.add(user_message)
            db.commit()

            # We'll save the assistant's message after streaming
            conversation_id = conversation.id
        finally:
            db.close()

        # Collect the full response while streaming
        full_response = ""

        async def stream_with_save():
            nonlocal full_response
            for chunk in openai_response:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield f"data: {content}\n\n"

            # Save assistant's response to database
            db = SessionLocal()
            try:
                assistant_message = Message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=full_response,
                    model=model,
                )
                db.add(assistant_message)
                db.commit()
            finally:
                db.close()

            yield "data: [DONE]\n\n"

        # Handle streaming response
        response = StreamingResponse(stream_with_save(), media_type="text/event-stream")

        # Add conversation_id to response headers - this is the primary way for clients to get the conversation ID
        response.headers["X-Conversation-ID"] = str(conversation_id)

        return response
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format for history")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversations")
async def get_conversations():
    """Get a list of all conversations"""
    db = SessionLocal()
    try:
        conversations = (
            db.query(Conversation).order_by(Conversation.updated_at.desc()).all()
        )
        return [
            {
                "id": conv.id,
                "title": conv.title,
                "created_at": conv.created_at,
                "updated_at": conv.updated_at,
            }
            for conv in conversations
        ]
    finally:
        db.close()


@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: int):
    """Get a specific conversation with all its messages"""
    db = SessionLocal()
    try:
        conversation = (
            db.query(Conversation).filter(Conversation.id == conversation_id).first()
        )
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        messages = (
            db.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.timestamp)
            .all()
        )

        return {
            "id": conversation.id,
            "title": conversation.title,
            "created_at": conversation.created_at,
            "updated_at": conversation.updated_at,
            "messages": [
                {
                    "id": msg.id,
                    "role": msg.role,
                    "content": msg.content,
                    "model": msg.model,
                    "timestamp": msg.timestamp,
                }
                for msg in messages
            ],
        }
    finally:
        db.close()


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: int):
    """Delete a conversation and all its messages"""
    db = SessionLocal()
    try:
        conversation = (
            db.query(Conversation).filter(Conversation.id == conversation_id).first()
        )
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        db.delete(conversation)
        db.commit()
        return {"message": "Conversation deleted successfully"}
    finally:
        db.close()


@app.put("/conversations/{conversation_id}")
async def update_conversation_title(conversation_id: int, title: str = Form(...)):
    """Update the title of a conversation"""
    db = SessionLocal()
    try:
        conversation = (
            db.query(Conversation).filter(Conversation.id == conversation_id).first()
        )
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        conversation.title = title
        db.commit()
        return {"message": "Conversation title updated successfully"}
    finally:
        db.close()


@app.get("/chat-history/{conversation_id}")
async def get_chat_history(conversation_id: int, max_history: int = 50):
    """Get chat history in a format suitable for the frontend"""
    db = SessionLocal()
    try:
        # Check if conversation exists
        conversation = (
            db.query(Conversation).filter(Conversation.id == conversation_id).first()
        )
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Get messages
        messages = (
            db.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.timestamp)
            .limit(max_history)
            .all()
        )

        # Format for frontend
        history = [
            {
                "id": msg.id,
                "role": msg.role,
                "content": msg.content,
                "model": msg.model,
                "timestamp": msg.timestamp.isoformat(),
            }
            for msg in messages
        ]

        return {
            "conversation_id": conversation_id,
            "title": conversation.title,
            "history": history,
        }
    finally:
        db.close()


@app.get("/conversation-info/{conversation_id}")
async def get_conversation_info(conversation_id: int):
    """Get basic information about a conversation without its messages"""
    db = SessionLocal()
    try:
        conversation = (
            db.query(Conversation).filter(Conversation.id == conversation_id).first()
        )
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Count messages
        message_count = (
            db.query(Message).filter(Message.conversation_id == conversation_id).count()
        )

        return {
            "conversation_id": conversation.id,
            "title": conversation.title,
            "created_at": conversation.created_at.isoformat(),
            "updated_at": conversation.updated_at.isoformat(),
            "message_count": message_count,
        }
    finally:
        db.close()


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
