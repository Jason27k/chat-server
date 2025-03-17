# AI Chat Application

A FastAPI-based chat application that supports multiple AI models (OpenAI, DeepSeek, Claude, Gemini) with text, image, and document processing capabilities.

## Features

- Text-based chat with multiple AI models
- Image analysis with vision-capable models
- Document processing and Q&A (PDF, text files, code files)
- Conversation history management
- PostgreSQL database for persistent storage

## Overview

This application provides a backend API for interacting with various AI models through three specialized endpoints:

- Text chat: For standard text-based conversations
- Image chat: For analyzing and discussing images
- Document chat: For querying and discussing document content

## Docker Setup

This application is containerized using Docker and Docker Compose, which includes:

- FastAPI application container
- PostgreSQL database container

### Prerequisites

- Docker and Docker Compose installed on your system
- API keys for at least one of the supported AI services:
  - OpenAI
  - Claude (Anthropic)
  - DeepSeek
  - Gemini

### Getting Started

1. Clone this repository:

   ```bash
   git clone https://github.com/Jason27k/chat-server
   cd <repository-directory>
   ```

2. Create a `.env` file based on the example:

   ```bash
   cp .env.example .env
   ```

3. Edit the `.env` file and add your API keys:

   ```
   OPENAI_API_KEY=your_openai_api_key
   DEEPSEEK_KEY=your_deepseek_api_key
   CLAUDE_API_KEY=your_claude_api_key
   GEMINI_API_KEY=your_gemini_api_key
   ```

4. **Important Note**: The image processing functionality requires Backblaze B2 storage. If you don't need image chat capabilities, you can ignore the Backblaze configuration and still use text and document chat features.

5. Build and start the containers:

   ```bash
   docker-compose up -d
   ```

6. The API will be available at http://localhost:8000

### API Endpoints

#### Core Chat Endpoints

- `/text-chat-stream` - Text-only chat with AI models
- `/image-chat-stream` - Image analysis with vision-capable models (requires Backblaze setup)
- `/chat-with-document` - Chat about document content

#### Document Management

- `/process-document` - Process and store a document
- `/documents` - List all documents
- `/documents/{document_id}` - Get document details
- `/search-documents` - Search document content

#### Conversation Management

- `/conversations` - List all conversations
- `/conversations/{conversation_id}` - Get a specific conversation
- `/chat-history/{conversation_id}` - Get chat history for a conversation

## Development

To make changes to the application:

1. Modify the code as needed
2. Rebuild the containers:
   ```bash
   docker-compose down
   docker-compose up --build -d
   ```

## Troubleshooting

- If the application can't connect to the database, check that the PostgreSQL container is running:

  ```bash
  docker-compose ps
  ```

- To view logs:

  ```bash
  docker-compose logs -f app
  docker-compose logs -f db
  ```

- If you encounter errors related to image processing, but don't need that functionality, you can safely ignore them and continue using the text and document chat features.
