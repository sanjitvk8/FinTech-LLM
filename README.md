# FinTech-LLM Document Processing System

**FinTech-LLM** is a comprehensive document processing and analysis platform that leverages Large Language Models (LLMs) to extract insights, answer queries, and provide intelligent analysis of financial and business documents. The system provides both a web interface and API endpoints for document processing, with persistent chat history stored in MongoDB for audit and review purposes.

## Overview

This application enables users to upload various document formats (PDF, DOCX, TXT), process them through state-of-the-art LLMs, and engage in intelligent conversations about the document content. The system features a modern Streamlit web interface, FastAPI backend, and MongoDB database for storing chat conversations and document metadata.

## Key Features

- **Document Processing**: Upload and process various document formats (PDF, DOCX, TXT, EML)
- **Multiple LLM Support**: Integrated support for OpenAI GPT models, Google Gemini, and Groq
- **Intelligent Querying**: Ask questions about your documents with context-aware responses
- **Chat History Storage**: All conversations automatically stored in MongoDB for future reference
- **Web Interface**: Modern Streamlit-based user interface for easy document management
- **API Endpoints**: RESTful API for programmatic access to document processing capabilities  
- **Chat History Viewer**: Built-in tools to view all stored conversations by document or date
- **Database Management**: Command-line tools to view, filter, and manage stored chat data
- **Docker Support**: Full Docker and Docker Compose setup for easy deployment
- **Vector Search**: Efficient similarity search using vector embeddings
- **Pinecone Integration**: Optional Pinecone vector database support for enterprise use

## Technology Stack

- **Backend**: [FastAPI](https://fastapi.tiangolo.com/) - Modern, fast web framework for building APIs with Python
- **Frontend**: [Streamlit](https://streamlit.io/) - Open-source app framework for data science and ML applications
- **Database**: [MongoDB](https://www.mongodb.com/) - NoSQL database for storing chat conversations
- **LLM Providers**: 
  - [OpenAI](https://openai.com/) - GPT models for advanced language understanding
  - [Google Gemini](https://deepmind.google/technologies/gemini/) - Google's state-of-the-art AI models
  - [Groq](https://groq.com/) - High-performance LLM inference platform
- **Vector Database**: [Pinecone](https://www.pinecone.io/) - Managed vector database service (optional)
- **Containerization**: [Docker](https://www.docker.com/) & [Docker Compose](https://docs.docker.com/compose/) - For containerized deployment
- **Embeddings**: [Sentence Transformers](https://www.sbert.net/) - State-of-the-art sentence embeddings
- **Document Processing**: 
  - [PyPDF2](https://pypdf2.readthedocs.io/) - PDF processing library
  - [python-docx](https://python-docx.readthedocs.io/) - DOCX processing library
- **Caching**: [Redis](https://redis.io/) - In-memory data structure store
- **Authentication**: [python-jose](https://python-jose.readthedocs.io/) - JSON Web Token implementation

## Features

- **Document Upload**: Supports uploading various document formats (PDF, TXT, DOCX).
- **Text Processing**: Extracts text content and splits it into manageable chunks.
- **Vector Embeddings**: Generates vector embeddings for text chunks using state-of-the-art models.
- **Vector Store**: Stores and indexes document chunks for fast similarity search.
- **Intelligent Search**: Allows users to query documents using natural language.
- **LLM Integration**: Uses LLMs to provide intelligent answers, summaries, and insights.
- **Dockerized**: Comes with Docker and Docker Compose for easy setup and deployment.

## Project Structure

The project follows a modular structure to separate concerns and facilitate maintainability:

```
FinTech-LLM/
├── app/                      # Main application code
│   ├── __init__.py
│   ├── main.py               # FastAPI application entry point
│   ├── api/                  # API-related modules
│   │   ├── __init__.py
│   │   ├── endpoints.py      # API endpoints
│   │   └── models.py         # Pydantic models for API
│   ├── core/                 # Core application logic
│   │   ├── __init__.py
│   │   ├── config.py         # Configuration settings
│   │   ├── document_processor.py # Document processing logic
│   │   ├── embedding_service.py  # Embedding generation
│   │   ├── llm_service.py        # LLM interaction
│   │   └── query_processor.py    # Query processing logic
│   ├── services/             # Business logic services
│   │   ├── __init__.py
│   │   ├── document_service.py
│   │   ├── retrieval_service.py
│   │   └── decision_service.py
│   └── utils/                # Utility functions and classes
│       ├── __init__.py
│       ├── file_handler.py
│       ├── vector_store.py
│       └── mongodb.py        # MongoDB integration utilities
├── .env.example              # Example environment variables
├── .gitignore                # Git ignore rules
├── docker-compose.yml        # Docker Compose configuration
├── Dockerfile                # Docker configuration
├── LICENSE                   # Project license
├── list_gemini_models.py     # Tool for listing available Gemini models
├── models.txt                # List of supported models
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── run_project.py            # Main application runner
├── setup_mongodb.py          # MongoDB setup and verification script
├── streamlit_app.py          # Streamlit web interface
├── test_api.py               # API testing utilities
├── test_fixes.py             # Testing fixes
├── test_integration.py       # Integration tests
├── test_mongodb_connection.py # MongoDB connection test script
├── test_setup.py             # Setup tests
├── view_mongodb.py           # Command-line MongoDB viewer
└── debug_mongodb_import.py   # MongoDB import debugging script
```

## Contributing

Contributions are welcome! If you have suggestions or find any issues, please open an issue or submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

To contribute:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Update documentation as needed
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a pull request

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Getting Started

### Prerequisites

- [Docker and Docker Compose](https://docs.docker.com/get-docker/) (for containerized deployment)
- [Python 3.9+](https://www.python.org/downloads/) (for direct local execution)
- An API key from one of the supported LLM providers (OpenAI, Google Gemini, or Groq)
- [Git](https://git-scm.com/) for cloning the repository

### Installation Methods

The application can be deployed using either Docker Compose or run directly from the host system.

#### Method 1: Docker Compose (Recommended)

1. **Clone the repository**:

   ```bash
   git clone <repository_url>
   cd FinTech-LLM
   ```

2. **Set up environment variables**:

   Copy the example environment file and configure your API keys:

   ```bash
   cp .env.example .env
   ```

   Edit the `.env` file to add your API keys and other configuration:

   ```bash
   OPENAI_API_KEY="your_openai_api_key"
   GEMINI_API_KEY="your_google_gemini_api_key" 
   GROQ_API_KEY="your_groq_api_key"
   LLM_PROVIDER="openai"  # Choose from: openai, gemini, groq
   ```

3. **Build and run with Docker Compose**:

   ```bash
   docker-compose up --build
   ```

   The application will be available at:
   - Backend API: `http://localhost:8000`
   - Web Interface: `http://localhost:8501` (if using full setup)

#### Method 2: Direct Execution

1. **Clone the repository**:

   ```bash
   git clone <repository_url>
   cd FinTech-LLM
   ```

2. **Set up Python virtual environment**:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:

   Copy the example environment file:

   ```bash
   cp .env.example .env
   ```

   Edit the `.env` file to add your API keys and configure MongoDB:

   ```bash
   OPENAI_API_KEY="your_openai_api_key"
   GEMINI_API_KEY="your_google_gemini_api_key"
   GROQ_API_KEY="your_groq_api_key"
   LLM_PROVIDER="openai"
   MONGODB_URL="mongodb://localhost:27017"
   MONGODB_DB_NAME="fintech_llm"
   ```

5. **Ensure MongoDB is running**:

   If running locally, make sure MongoDB is installed and running on your system:
   - [Install MongoDB Community Server](https://www.mongodb.com/docs/manual/installation/)
   - Start the MongoDB service

6. **Run the setup script** (recommended):

   ```bash
   python setup_mongodb.py
   ```

7. **Run the application**:

   ```bash
   python run_project.py
   ```

   The application will be available at:
   - Backend API: `http://localhost:8000`
   - Web Interface: `http://localhost:8501`

## Configuration

### Environment Variables

The application uses several environment variables for configuration. Create a `.env` file in the root directory to customize settings:

#### LLM Provider Configuration
- `OPENAI_API_KEY`: Your OpenAI API key for GPT model access
- `GEMINI_API_KEY`: Your Google Gemini API key 
- `GROQ_API_KEY`: Your Groq API key
- `LLM_PROVIDER`: The LLM provider to use (`openai`, `gemini`, `groq`)

#### Database Configuration
- `MONGODB_URL`: MongoDB connection string (default: `mongodb://localhost:27017`)
- `MONGODB_DB_NAME`: MongoDB database name (default: `fintech_llm`)
- `CHATS_COLLECTION_NAME`: MongoDB collection name for chats (default: `chats`)
- `DATABASE_URL`: Database connection string for PostgreSQL (default: local SQLite)

#### Vector Database Configuration
- `PINECONE_API_KEY`: Pinecone API key for vector storage (optional)
- `PINECONE_ENVIRONMENT`: Pinecone environment (default: `us-east1-gcp`)
- `PINECONE_INDEX_NAME`: Pinecone index name (default: `document-retrieval`)

#### Application Configuration
- `EMBEDDING_MODEL`: Embedding model to use (default: `text-embedding-3-small`)
- `CHUNK_SIZE`: Size of text chunks for processing (default: 1000)
- `CHUNK_OVERLAP`: Overlap between text chunks (default: 200)
- `MAX_FILE_SIZE`: Maximum file size in bytes (default: 50MB)
- `SIMILARITY_THRESHOLD`: Threshold for similarity matching (default: 0.3)
- `MAX_RETRIEVED_CHUNKS`: Maximum chunks to retrieve per query (default: 10)

#### Security Configuration
- `SECRET_KEY`: Secret key for authentication (randomly generated if not set)
- `ACCESS_TOKEN_EXPIRE_MINUTES`: Token expiration time (default: 30)

#### Cache Configuration
- `REDIS_URL`: Redis connection string (default: local Redis)
- `CACHE_TTL`: Cache time-to-live in seconds (default: 3600)

### API Usage

The API provides the following endpoints:

- **`POST /api/v1/documents/upload`**: Upload a document for processing.
- **`POST /api/v1/query`**: Query the processed documents.

For detailed API documentation, visit `http://localhost:8000/docs` after running the application.

## How It Works

1. **Document Upload**: When a document is uploaded, it is saved to a temporary location.
2. **Processing**: The document's text is extracted and divided into smaller chunks.
3. **Embedding**: Each chunk is converted into a numerical vector (embedding) using an embedding model.
4. **Storage**: The chunks and their embeddings are stored in a vector store for efficient search.
5. **Querying**: When a user submits a query, it is also converted into an embedding.
6. **Retrieval**: The vector store finds the document chunks with embeddings most similar to the query embedding.
7. **Response Generation**: The most relevant chunks are used as context for an LLM to generate a final answer.
8. **Chat Storage**: All queries and responses are automatically stored in MongoDB with timestamps and metadata.

## Usage

### Web Interface

The application provides a user-friendly Streamlit interface accessible at `http://localhost:8501`.

#### Main Features:
- **Home**: Overview of the system and status information
- **Upload Documents**: Upload PDF, DOCX, or TXT files for processing
- **Ask Questions**: Query documents without uploading new files
- **Chat History**: View all stored conversations with the AI
- **About**: Documentation and system information

#### Document Processing Workflow:
1. Navigate to "Upload Documents" section
2. Upload your financial or business document
3. Optionally provide an initial query (e.g., "Summarize this document")
4. Wait for processing to complete
5. Ask follow-up questions about the document
6. All conversations are automatically saved to MongoDB

#### Chat History Access:
1. Navigate to "Chat History" section
2. View all stored conversations 
3. Filter by document ID if needed
4. Access detailed conversation data including context and sources

### API Usage

For programmatic access, the API provides comprehensive endpoints at `http://localhost:8000`. Detailed API documentation is available at `http://localhost:8000/docs`.

#### Key Endpoints:
- `POST /api/v1/documents/upload`: Upload a document for processing
- `POST /api/v1/documents/process`: Process a document with an initial query
- `POST /api/v1/query`: Query documents with follow-up questions
- `GET /api/v1/chats`: Retrieve all stored chat conversations
- `GET /api/v1/chats/document/{document_id}`: Retrieve chats for a specific document
- `GET /api/v1/mongodb-health`: Check MongoDB connection status
- `GET /api/v1/models/gemini`: List available Gemini models
- `GET /api/v1/health`: System health check

#### Example API Usage:
```bash
# Upload a document
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf"

# Query documents
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the total revenue?", "document_id": "your_document_id"}'

# Retrieve chat history
curl "http://localhost:8000/api/v1/chats"
```

## Database Schema

### MongoDB Collection: `chats`

The application stores all conversation history in MongoDB with the following schema:

```javascript
{
  "_id": ObjectId("..."),                    // MongoDB document ID
  "user_query": "string",                    // The question asked by the user
  "response": "string",                      // The AI's response 
  "document_id": "string",                   // Optional document ID for context
  "sources": ["string"],                     // List of source documents used
  "context": ["string"],                     // List of context chunks used
  "timestamp": ISODate("...")                // When the conversation occurred
}
```

#### Indexes:
- `timestamp`: Sorted index for time-based queries
- `document_id`: Index for document-specific queries  
- `user_query`, `response`: Text index for content searching

This schema enables efficient retrieval of conversation history by time, document, or content.

## Database Management and Viewer Tools

The application includes several tools for managing and viewing the stored MongoDB data:

### 1. Web-based Chat History Viewer
Access chat history directly through the web interface:
- Navigate to the "Chat History" section in the Streamlit app
- View all conversations with filtering by document ID
- Access detailed information including context, sources, and timestamps

### 2. Command-Line Database Viewer (`view_mongodb.py`)
A comprehensive command-line tool for viewing stored conversations:

```bash
# View the most recent 10 conversations (default)
python view_mongodb.py

# View all stored conversations
python view_mongodb.py --all

# View conversations for a specific document
python view_mongodb.py --document <document_id>

# View database statistics
python view_mongodb.py --stats

# Limit results (default is 100)
python view_mongodb.py --all --limit 50
```

### 3. Setup and Verification Script (`setup_mongodb.py`)
Ensures proper MongoDB setup and verifies connectivity:

```bash
python setup_mongodb.py
```

This script will:
- Check and install required dependencies (pymongo, PyPDF2)
- Verify MongoDB connection
- Create necessary indexes
- Validate configuration settings

### 4. API Endpoints for Database Access
Programmatic access to stored conversations via API:

- `GET /api/v1/chats` - Retrieve all chat conversations
- `GET /api/v1/chats/document/{document_id}` - Retrieve chats for specific document
- `GET /api/v1/mongodb-health` - Check MongoDB connection status

### 5. MongoDB GUI Clients
Connect to your MongoDB instance using popular GUI tools:
- [MongoDB Compass](https://www.mongodb.com/products/compass) (official MongoDB GUI)
- [Studio 3T](https://studio3t.com/)
- VS Code with MongoDB extension

Use connection string: `mongodb://localhost:27017` (or your configured MongoDB URL)

## Troubleshooting

### Common Issues and Solutions

#### MongoDB Connection Issues
**Problem**: "Failed to connect to MongoDB" error
**Solutions**:
- Ensure MongoDB is running on your system: `sudo systemctl start mongod`
- Check that your MongoDB URL is correct in the environment variables
- Verify authentication credentials if using a secured MongoDB instance
- Run the setup script: `python setup_mongodb.py`

#### API Key Issues
**Problem**: "Invalid API key" or "Authentication failed"
**Solutions**:
- Verify your API keys are correctly set in the `.env` file
- Ensure there are no extra spaces or quotes around the API keys
- Check that your selected LLM provider is enabled (e.g., if using Gemini, make sure `GEMINI_API_KEY` is set and `LLM_PROVIDER="gemini"`)

#### Document Processing Issues
**Problem**: "Error processing document" or "File processing failed"
**Solutions**:
- Check that the file format is supported (PDF, DOCX, TXT, EML)
- Verify the file size is below the configured limit (default 50MB)
- Ensure the document isn't password-protected or corrupted
- Check the application logs for detailed error messages

#### Docker Setup Issues
**Problem**: "Port already in use" or "Container won't start"
**Solutions**:
- Stop any running containers: `docker-compose down`
- Check for conflicting processes using the required ports (8000, 8501, 27017, 5432, 6379)
- Use a different port configuration in docker-compose.yml if necessary

#### Dependency Installation Issues
**Problem**: "Module not found" or "Import errors"
**Solutions**:
- Ensure you're using the correct Python virtual environment
- Install dependencies again: `pip install -r requirements.txt`
- Run the setup script: `python setup_mongodb.py`

#### Application Won't Start
**Problem**: "Application crashes" or "Failed to initialize"
**Solutions**:
- Check the logs in `app.log` for error details
- Verify all required environment variables are set
- Ensure MongoDB and other services are running before starting the application
- Run the setup script to verify dependencies: `python setup_mongodb.py`

#### Chat History Not Saving
**Problem**: Conversations aren't appearing in MongoDB
**Solutions**:
- Verify MongoDB connection by checking the health endpoint: `http://localhost:8000/api/v1/mongodb-health`
- Check application logs for MongoDB save errors
- Run the test script: `python test_mongodb_connection.py`
- Verify your MongoDB configuration in environment variables

### Getting Help

If you encounter issues not covered here:
1. Check the application logs in `app.log`
2. Run the database connection test: `python test_mongodb_connection.py`
3. Verify your configuration: `python debug_mongodb_import.py`
4. Consult the API documentation at `http://localhost:8000/docs`

## Customization

- **LLM and Embedding Models**: You can change the models used for LLM and embedding by modifying the `config.py` file.
- **Vector Store**: The default vector store is a simple JSON-based implementation. For production use, consider replacing it with a more robust solution like FAISS, Milvus, or a cloud-based vector database.
- **Document Loaders**: To support more document types, you can extend the `DocumentProcessor` with new loaders (e.g., for CSV, HTML, etc.).

## Contributing

Contributions are welcome! If you have suggestions or find any issues, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

