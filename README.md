# INDEX.PY
# Cohere Embeddings and Pinecone Indexing

This file contains a script for leveraging Cohere embeddings to embed text data and indexing the embeddings in Pinecone for efficient similarity searches.

## Prerequisites

Before running the code, ensure you have the necessary dependencies installed. You can install them using the following:

```bash
pip install pinecone-client langchain cohere
pip install openai==0.27.1
```

make sure to set the environmental variables. You will find them defined in the prompt.py file section


# PROMPT.PY
## Conversational AI Interface with Gradio

This file contains a simple conversational AI interface using Gradio, integrating OpenAI's language model, Cohere embeddings, and Pinecone vector storage for conversational retrieval.

## Prerequisites

Before running the code, ensure you have the necessary dependencies installed. You can install them using the following:

```bash
pip install gradio langchain python-dotenv
```
Make sure to set the required environment variables by creating a .env file and adding the necessary keys:
> - OPEN_API_KEY=your_openai_api_key
> - PINECONE_API_KEY=your_pinecone_api_key
> - COHERE_API_KEY=your_cohere_api_key

# RAG_1.IPYNB 
## JUPYTER NOTEBOOK 
This notebook contains the code for embedding data using cohere embeddings and also using gpt model. It also contains code for upserting data to the pinecone database.
