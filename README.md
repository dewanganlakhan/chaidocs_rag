# ChaiDocs RAG Application with Qdrant

## Overview

This application implements a Retrieval-Augmented Generation (RAG) system for the ChaiDocs documentation. It dynamically parses the documentation structure, creates vector collections in Qdrant for each course ("Chai aur" series), and provides intelligent question answering with source attribution.

## Features

- **Dynamic Documentation Parsing**: Automatically extracts course structure from the sidebar navigation
- **Intelligent Routing**: Uses LLM to determine the most relevant course for each query
- **Source Attribution**: Provides links back to the original documentation
- **Future-Proof**: Automatically adapts to new courses added to the documentation
- **Modular Design**: Easy to extend for other documentation sites

## Prerequisites

- Python 3.8+
- Qdrant database (local or cloud)
- OpenAI API key

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/dewanganlakhan/chaidocs_rag.git
   cd chaidocs-rag
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   echo "OPENAI_API_KEY=your_openai_api_key" > .env
   echo "QDRANT_URL=http://localhost:6333" >> .env
   # For Qdrant Cloud:
   # echo "QDRANT_API_KEY=your_qdrant_api_key" >> .env
   ```

## Configuration

Edit the following variables in `config.py` if needed:

```python
DOCS_BASE_URL = "https://chaidocs.vercel.app"
COLLECTION_PREFIX = "chaidocs_"
CHUNK_SIZE = 1000  # Size of document chunks
CHUNK_OVERLAP = 200  # Overlap between chunks
```

## Usage

### Indexing Documents

Run the indexing process (only needs to be done once or when docs update):

```bash
python main.py --index
```

### Querying the System

Start the interactive query interface:

```bash
python main.py
```

Example queries:
- "How do I use Emmet in HTML?"
- "Explain Git branches"
- "What are PostgreSQL best practices?"

### Command Line Options

```
usage: main.py [-h] [--index] [--query QUERY]

ChaiDocs RAG System

options:
  -h, --help   show this help message and exit
  --index      Index documents into Qdrant
  --query QUERY  Run a single query non-interactively
```

## Architecture

```
ChaiDocs RAG System
├── Document Parser
│   ├── Sidebar Structure Extraction
│   └── Content Crawler
├── Vector Database (Qdrant)
│   ├── Course Collections
│   └── Document Chunks
├── Query Processor
│   ├── Course Router (LLM)
│   └── Answer Generator (LLM)
└── Output Formatter
    ├── Answer Generation
    └── Source Attribution
```

## Customization

To adapt this for other documentation sites:

1. Modify the `parse_sidebar_structure()` method to match your site's navigation
2. Adjust the CSS selectors in the BeautifulSoup parsing
3. Update the base URL and collection naming conventions

## Troubleshooting

- **Parsing Errors**: Check if the sidebar HTML structure has changed
- **Connection Issues**: Verify Qdrant and OpenAI API connectivity
- **Empty Results**: Ensure documents were properly indexed

## License

MIT License

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
