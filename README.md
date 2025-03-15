# AI Chat Application

A FastAPI-based chat application that supports multiple AI models (OpenAI, DeepSeek, Claude, Gemini) and document processing.

## Features

- Chat with multiple AI models
- Document processing (PDF, text files, code files)
- Conversation history management
- PostgreSQL database for persistent storage

## Docker Setup

This application is containerized using Docker and Docker Compose, which includes:

- FastAPI application container
- PostgreSQL database container

### Prerequisites

- Docker and Docker Compose installed on your system
- API keys for the AI services you want to use

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

4. Build and start the containers:

   ```bash
   docker-compose up -d
   ```

5. The API will be available at http://localhost:8000

### API Endpoints

- `/chat` - Chat with an AI model
- `/process-document` - Process a document (PDF, text, code)
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
