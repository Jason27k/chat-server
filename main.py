# server.py
import base64
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
    BigInteger,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import anthropic

import fitz  # PyMuPDF
import pymupdf4llm

from openai import OpenAI

load_dotenv()

OPEN_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_KEY = os.getenv("DEEPSEEK_KEY")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DB_URL = os.getenv("DATABASE_URL") or "postgresql://postgres:postgres@db:5432/chatapp"

OPEN_AI_MODELS = ("o3-mini", "gpt-4o")
DEEPSEEK_MODELS = ("deepseek-chat", "deepseek-reasoner")
CLAUDE_MODELS = ("claude-3-7-sonnet-20250219", "claude-3-5-haiku-20241022")
GEMINI_MODELS = ("gemini-2.0-flash",)

IMAGE_MODELS = CLAUDE_MODELS + GEMINI_MODELS + ("gpt-4o",)

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


class Image(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True, index=True)
    file_key = Column(String, nullable=False)
    filename = Column(String)
    content_type = Column(String)
    size = Column(BigInteger)
    upload_date = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationship with messages
    messages = relationship("Message", back_populates="image")


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"))
    role = Column(String)  # "user" or "assistant"
    content = Column(Text)
    model = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    image_id = Column(Integer, ForeignKey("images.id"), nullable=True)

    conversation = relationship("Conversation", back_populates="messages")
    image = relationship("Image", back_populates="messages")


# Update the Document model to remove Backblaze dependency
class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    file_key = Column(String, nullable=True)  # Changed to nullable=True
    filename = Column(String)
    content_type = Column(String)
    size = Column(BigInteger)
    extracted_text = Column(Text, nullable=False)  # This is now the primary data
    upload_date = Column(DateTime, default=datetime.datetime.utcnow)


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
    return anthropic.Anthropic(
        api_key=CLAUDE_API_KEY,
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
    allow_origins=["http://localhost:5173"],  # Your React app's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Conversation-ID"],  # Expose our custom header
)

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


# Add this constant at the top of your file
MAX_DOCUMENT_TEXT_LENGTH = 100000  # 100K characters


@app.post("/process-document")
async def process_document(file: UploadFile = File(...), file_type: str = Form(...)):
    """
    Process a document by extracting text content and storing it in the database.
    Limits text to MAX_DOCUMENT_TEXT_LENGTH characters.
    """
    try:
        contents = await file.read()
        file_size = len(contents)

        # Extract text based on file type
        extracted_text = ""
        if file_type == "application/pdf":
            extracted_text = extract_text_from_pdf(contents)
        elif file_type.startswith("text/") or any(
            file.filename.endswith(ext) for ext in CODE_EXTENSIONS
        ):
            extracted_text = process_text_file(contents)
        else:
            raise HTTPException(
                status_code=400, detail=f"Unsupported file type: {file_type}"
            )

        # Check if text exceeds the maximum length
        is_truncated = False
        if len(extracted_text) > MAX_DOCUMENT_TEXT_LENGTH:
            # Truncate the text
            extracted_text = extracted_text[:MAX_DOCUMENT_TEXT_LENGTH]
            # Add a note about truncation
            truncation_note = "\n\n[Note: This document was truncated because it exceeds the maximum allowed length.]"
            # Make sure we have room for the truncation note
            extracted_text = (
                extracted_text[: MAX_DOCUMENT_TEXT_LENGTH - len(truncation_note)]
                + truncation_note
            )
            is_truncated = True

        # Save to database
        db = SessionLocal()
        try:
            document = Document(
                filename=file.filename,
                content_type=file_type,
                size=file_size,
                extracted_text=extracted_text,
                # No file_key since we're not storing the file
            )
            db.add(document)
            db.commit()
            db.refresh(document)

            return {
                "document_id": document.id,
                "filename": file.filename,
                "content_type": file_type,
                "size": file_size,
                "is_truncated": is_truncated,
            }
        finally:
            db.close()

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
                    "image_id": msg.image_id,
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
                "image_id": msg.image_id,
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


# Backblaze B2 configuration
B2_BUCKET = os.getenv("B2_BUCKET")
B2_KEY_ID = os.getenv("B2_KEY_ID")
B2_APPLICATION_KEY = os.getenv("B2_APPLICATION_KEY")
B2_ENDPOINT = os.getenv("B2_ENDPOINT")


# Simplified Backblaze storage class
class BackblazeStorage:
    def __init__(self):
        if not B2_KEY_ID or not B2_APPLICATION_KEY:
            raise ValueError(
                "Backblaze B2 credentials not configured. Please set B2_KEY_ID and B2_APPLICATION_KEY environment variables."
            )

        # Initialize S3 client for Backblaze B2
        import boto3

        self.s3 = boto3.client(
            "s3",
            endpoint_url=B2_ENDPOINT,
            aws_access_key_id=B2_KEY_ID,
            aws_secret_access_key=B2_APPLICATION_KEY,
        )
        self.bucket = B2_BUCKET

    async def upload_file(self, file_bytes, filename, content_type):
        """Upload a file to Backblaze B2 and return its file key"""
        # Generate a unique file ID
        file_id = f"{int(time.time())}_{filename}"

        # Upload to B2
        self.s3.put_object(
            Bucket=self.bucket,
            Key=file_id,
            Body=file_bytes,
            ContentType=content_type,
        )

        # Return the file key
        return file_id

    async def download_file(self, file_key):
        """Download a file from Backblaze B2 using authenticated access"""
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=file_key)
            return response["Body"].read()
        except Exception as e:
            print(f"Error downloading file from Backblaze: {e}")
            raise


# Initialize Backblaze storage
try:
    storage = BackblazeStorage()
except ValueError as e:
    print(f"Warning: {e}")
    print("Image upload functionality will not be available.")
    storage = None


# Simple in-memory cache for image base64 data
image_cache = {}


@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    """
    Upload an image to Backblaze B2 and cache its base64 representation on the server.
    """
    if not storage:
        raise HTTPException(
            status_code=500,
            detail="Backblaze B2 storage is not configured properly.",
        )

    try:
        contents = await file.read()
        file_size = len(contents)

        # Convert to base64 for AI models (stored only on server)
        base64_image = base64.b64encode(contents).decode("utf-8")

        # Upload to Backblaze B2 (private access) - returns file key
        file_key = await storage.upload_file(contents, file.filename, file.content_type)

        # Save to database
        db = SessionLocal()
        try:
            image = Image(
                file_key=file_key,
                filename=file.filename,
                content_type=file.content_type,
                size=file_size,
            )
            db.add(image)
            db.commit()
            db.refresh(image)

            # Cache the base64 data with the image ID as key
            image_cache[str(image.id)] = {
                "base64": base64_image,
                "content_type": file.content_type,
                "last_accessed": datetime.datetime.now(),
            }

            # Return metadata to client (no preview URL needed)
            return {
                "image_id": image.id,
                "filename": file.filename,
                "content_type": file.content_type,
                "size": file_size,
            }
        finally:
            db.close()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/text-chat-stream")
async def text_chat_stream(
    model: str,
    prompt: str,
    conversation_id: int = None,
    max_history: int = 10,
):
    """
    Chat with an AI model using text only.
    Simplified endpoint for text-only conversations.
    """
    try:
        # Get database session
        db = SessionLocal()

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

        # Add the current prompt to message history
        message_history.append({"role": "user", "content": prompt})

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

        # Handle different model APIs
        if model in CLAUDE_MODELS:
            # Format messages for Claude API
            claude_messages = []

            # Add all messages including current prompt
            for msg in message_history:
                claude_messages.append({"role": msg["role"], "content": msg["content"]})

            # Create Claude stream
            claude_stream = client.messages.stream(
                model=model,
                messages=claude_messages,
                max_tokens=4000,
            )

            async def stream_claude_with_save():
                nonlocal full_response

                with claude_stream as stream:
                    for text in stream.text_stream:
                        full_response += text

                        # Encode the text as JSON to preserve all whitespace and special characters
                        import json

                        encoded_text = json.dumps(text)

                        print("content", text)
                        yield f"data: {encoded_text}\n\n"

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

            # Return Claude streaming response
            response = StreamingResponse(
                stream_claude_with_save(), media_type="text/event-stream"
            )
            response.headers["X-Conversation-ID"] = str(conversation_id)
            return response

        else:
            # Handle OpenAI-compatible APIs (OpenAI, Deepseek, Gemini)
            messages_for_api = []

            # Add all messages including current prompt
            for msg in message_history:
                messages_for_api.append(
                    {"role": msg["role"], "content": msg["content"]}
                )

            # Create completion with history
            openai_response = client.chat.completions.create(
                model=model,
                messages=messages_for_api,
                stream=True,
            )

            async def stream_with_save():
                nonlocal full_response

                for chunk in openai_response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content

                        # Encode the text as JSON to preserve all whitespace and special characters
                        import json

                        encoded_content = json.dumps(content)

                        print("content", content)
                        yield f"data: {encoded_content}\n\n"

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
            response = StreamingResponse(
                stream_with_save(), media_type="text/event-stream"
            )
            response.headers["X-Conversation-ID"] = str(conversation_id)
            return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/image-chat-stream")
async def image_chat_stream(
    model: str,
    prompt: str,
    image_id: int,
    conversation_id: int = None,
    max_history: int = 10,
):
    """
    Chat with an AI model about an image.
    Dedicated endpoint for image-based conversations.
    """
    try:
        # Get database session
        db = SessionLocal()

        # Check if model supports images
        if model not in IMAGE_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Model {model} does not support images. Supported models: {', '.join(IMAGE_MODELS)}",
            )

        # Get image metadata
        image = db.query(Image).filter(Image.id == image_id).first()
        if not image:
            raise HTTPException(status_code=404, detail="Image not found")

        # Get cached base64 data
        cached_data = image_cache.get(str(image_id))
        if not cached_data:
            # If not in cache, download using authenticated access
            try:
                # Download file using the storage client
                file_bytes = await storage.download_file(image.file_key)

                # Convert to base64
                image_data = base64.b64encode(file_bytes).decode("utf-8")

                # Cache the base64 data
                image_cache[str(image_id)] = {
                    "base64": image_data,
                    "content_type": image.content_type,
                    "last_accessed": datetime.datetime.now(),
                }
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Failed to download image: {str(e)}"
                )
        else:
            # Use cached data
            image_data = cached_data["base64"]
            # Update last accessed time
            image_cache[str(image_id)]["last_accessed"] = datetime.datetime.now()

        image_content_type = image.content_type or "image/jpeg"

        try:
            # Get appropriate client
            if model in OPEN_AI_MODELS:
                client = get_openai_client()
            elif model in CLAUDE_MODELS:
                client = get_claude_client()
            elif model in GEMINI_MODELS:
                client = get_gemini_client()
            else:
                raise HTTPException(
                    status_code=400, detail="Unsupported model for images"
                )
        except ValueError as e:
            raise HTTPException(status_code=500, detail=str(e))

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
                title = f"Image: {image.filename} - {prompt[:20]}..."
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
                image_id=image_id,
            )
            db.add(user_message)
            db.commit()

            # We'll save the assistant's message after streaming
            conversation_id = conversation.id
        finally:
            db.close()

        # Collect the full response while streaming
        full_response = ""

        # Handle different model APIs
        if model in CLAUDE_MODELS:
            # Format messages for Claude API
            claude_messages = []

            # Add previous messages
            for msg in message_history:
                claude_messages.append({"role": msg["role"], "content": msg["content"]})

            # Add current message with image
            claude_messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": image_content_type,
                                "data": image_data,
                            },
                        },
                    ],
                }
            )

            # Create Claude stream
            claude_stream = client.messages.stream(
                model=model,
                messages=claude_messages,
                max_tokens=4000,
            )

            async def stream_claude_with_save():
                nonlocal full_response

                with claude_stream as stream:
                    for text in stream.text_stream:
                        full_response += text

                        # Encode the text as JSON to preserve all whitespace and special characters
                        import json

                        encoded_text = json.dumps(text)

                        print("content", text)
                        yield f"data: {encoded_text}\n\n"

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

            # Return Claude streaming response
            response = StreamingResponse(
                stream_claude_with_save(), media_type="text/event-stream"
            )
            response.headers["X-Conversation-ID"] = str(conversation_id)
            return response

        else:
            # Handle OpenAI-compatible APIs (OpenAI, Gemini)
            messages_for_api = []

            # Add previous messages
            for msg in message_history:
                messages_for_api.append(
                    {"role": msg["role"], "content": msg["content"]}
                )

            # Add current message with image
            img_data = f"data:{image_content_type};base64,{image_data}"
            messages_for_api.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": img_data}},
                    ],
                }
            )

            # Create completion with history
            openai_response = client.chat.completions.create(
                model=model,
                messages=messages_for_api,
                stream=True,
            )

            async def stream_with_save():
                nonlocal full_response

                for chunk in openai_response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content

                        # Encode the text as JSON to preserve all whitespace and special characters
                        import json

                        encoded_content = json.dumps(content)

                        print("content", content)
                        yield f"data: {encoded_content}\n\n"

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
            response = StreamingResponse(
                stream_with_save(), media_type="text/event-stream"
            )
            response.headers["X-Conversation-ID"] = str(conversation_id)
            return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/images/{image_id}")
async def get_image(image_id: int):
    """
    Retrieve an image from Backblaze B2 for display in the frontend.
    This endpoint acts as a secure proxy between the frontend and Backblaze.
    """
    if not storage:
        raise HTTPException(
            status_code=500,
            detail="Backblaze B2 storage is not configured properly.",
        )

    db = SessionLocal()
    try:
        # Get image metadata from database
        image = db.query(Image).filter(Image.id == image_id).first()
        if not image:
            raise HTTPException(status_code=404, detail="Image not found")

        try:
            # Download the image from Backblaze B2
            file_bytes = await storage.download_file(image.file_key)

            # Return the image with the correct content type
            from fastapi.responses import Response

            return Response(
                content=file_bytes, media_type=image.content_type or "image/jpeg"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to retrieve image: {str(e)}"
            )
    finally:
        db.close()


@app.get("/documents/{document_id}")
async def get_document(document_id: int):
    """Get document metadata and extracted text"""
    db = SessionLocal()
    try:
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        return {
            "document_id": document.id,
            "filename": document.filename,
            "content_type": document.content_type,
            "size": document.size,
            "upload_date": document.upload_date.isoformat(),
            "text": document.extracted_text,
        }
    finally:
        db.close()


@app.get("/documents")
async def list_documents():
    """Get a list of all documents"""
    db = SessionLocal()
    try:
        documents = db.query(Document).order_by(Document.upload_date.desc()).all()
        return [
            {
                "id": doc.id,
                "filename": doc.filename,
                "content_type": doc.content_type,
                "size": doc.size,
                "upload_date": doc.upload_date.isoformat(),
            }
            for doc in documents
        ]
    finally:
        db.close()


@app.get("/documents/{document_id}/text")
async def get_document_text(document_id: int):
    """Get just the extracted text from a document"""
    db = SessionLocal()
    try:
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        return {
            "document_id": document.id,
            "filename": document.filename,
            "text": document.extracted_text,
        }
    finally:
        db.close()


@app.get("/chat-with-document")
async def chat_with_document(
    model: str,
    prompt: str,
    document_id: int,
    conversation_id: int = None,
    max_history: int = 10,
):
    """
    Chat with an AI model about a specific document.
    This endpoint prepends the document text to the user's prompt.
    """
    db = SessionLocal()
    try:
        # Get document
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        # Create a new prompt that includes the document text
        document_prompt = f"""Document: {document.filename}
Content:
{document.extracted_text}

User question: {prompt}"""

        # Close the database connection before calling text_chat_stream
        db.close()

        # Call the text_chat_stream endpoint with the modified prompt
        return await text_chat_stream(
            model=model,
            prompt=document_prompt,
            conversation_id=conversation_id,
            max_history=max_history,
        )
    except Exception as e:
        db.close()
        raise HTTPException(status_code=500, detail=str(e))


# Add a document search endpoint
@app.get("/search-documents")
async def search_documents(query: str):
    """
    Search for documents containing the query text.
    This is a simple case-insensitive search in the extracted text.
    """
    db = SessionLocal()
    try:
        # Simple case-insensitive search
        from sqlalchemy import func

        documents = (
            db.query(Document)
            .filter(func.lower(Document.extracted_text).contains(query.lower()))
            .order_by(Document.upload_date.desc())
            .all()
        )

        return [
            {
                "id": doc.id,
                "filename": doc.filename,
                "content_type": doc.content_type,
                "size": doc.size,
                "upload_date": doc.upload_date.isoformat(),
                # Include a snippet of the matching text
                "snippet": get_text_snippet(doc.extracted_text, query.lower(), 200),
            }
            for doc in documents
        ]
    finally:
        db.close()


def get_text_snippet(text, query, max_length=200):
    """
    Extract a snippet of text around the first occurrence of the query.
    """
    query_pos = text.lower().find(query.lower())
    if query_pos == -1:
        # If query not found, return the beginning of the text
        return text[:max_length] + "..." if len(text) > max_length else text

    # Calculate start and end positions for the snippet
    start = max(0, query_pos - max_length // 2)
    end = min(len(text), query_pos + len(query) + max_length // 2)

    # Adjust to not cut words
    if start > 0:
        # Find the first space before the start
        space_before = text.rfind(" ", 0, start)
        if space_before != -1:
            start = space_before + 1

    if end < len(text):
        # Find the first space after the end
        space_after = text.find(" ", end)
        if space_after != -1:
            end = space_after

    snippet = text[start:end]

    # Add ellipsis if we're not at the beginning or end
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""

    return prefix + snippet + suffix


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
