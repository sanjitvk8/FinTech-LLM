# LLM Document Processing

This project is a FastAPI-based application for processing and querying documents using Large Language Models (LLMs). It allows users to upload documents, which are then chunked, vectorized, and stored for efficient retrieval and analysis.

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
llm-document-processing/
├── app/                  # Main application code
│   ├── __init__.py
│   ├── main.py           # FastAPI application entry point
│   ├── api/              # API-related modules
│   │   ├── __init__.py
│   │   ├── endpoints.py  # API endpoints
│   │   └── models.py     # Pydantic models for API
│   ├── core/             # Core application logic
│   │   ├── __init__.py
│   │   ├── config.py     # Configuration settings
│   │   ├── document_processor.py # Document processing logic
│   │   ├── embedding_service.py # Embedding generation
│   │   ├── llm_service.py    # LLM interaction
│   │   └── query_processor.py # Query processing logic
│   ├── services/           # Business logic services
│   │   ├── __init__.py
│   │   ├── document_service.py
│   │   ├── retrieval_service.py
│   │   └── decision_service.py
│   └── utils/              # Utility functions and classes
│       ├── __init__.py
│       ├── file_handler.py
│       └── vector_store.py
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose configuration
└── README.md             # This file
```

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Python 3.9+
- An OpenAI API key

### Installation

1. **Clone the repository**:

   ```bash
   git clone <repository_url>
   cd llm-document-processing
   ```

2. **Set up environment variables**:

   Create a `.env` file in the root directory and add your OpenAI API key:

   ```
   OPENAI_API_KEY="your_openai_api_key"
   ```

3. **Build and run with Docker**:

   ```bash
   docker-compose up --build
   ```

   The application will be available at `http://localhost:8000`.

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

## Customization

- **LLM and Embedding Models**: You can change the models used for LLM and embedding by modifying the `config.py` file.
- **Vector Store**: The default vector store is a simple JSON-based implementation. For production use, consider replacing it with a more robust solution like FAISS, Milvus, or a cloud-based vector database.
- **Document Loaders**: To support more document types, you can extend the `DocumentProcessor` with new loaders (e.g., for CSV, HTML, etc.).

## Contributing

Contributions are welcome! If you have suggestions or find any issues, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

