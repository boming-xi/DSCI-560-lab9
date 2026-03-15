# PDFs Chatbot using Langchain, GPT 3.5 and Llama 2
This is a Python gui application that demonstrates how to build a custom PDF chatbot using LangChain and GPT 3.5 / Llama 2. 


## How it works (GPT 3.5)
1. The application gui is built using streamlit
2. The application reads text from PDF files, splits it into chunks
3. Uses OpenAI Embedding API to generate embedding vectors used to find the most relevant content to a user's question 
4. Build a conversational retrieval chain using Langchain
5. Use OpenAI GPT API to generate respond based on content in PDF


## Requirements
1. Install dependencies:
```
pip install -r requirements.txt
```

For `App_p2.py` (open-source local models), install extra packages:
```
pip install transformers sentence-transformers
```

2. Create a `.env` file in the root directory of the project and add the following environment variables:
```
OPENAI_API_KEY= # Your OpenAI API key
```


## Code Structure

The code is structured as follows:

- `app.py`: The main application file that defines the Streamlit gui app and the user interface.
    * get_pdf_text function: reads text from PDF files
    * get_text_chunks function: splits text into chunks
    * get_vectorstore function: creates a FAISS vectorstore from text chunks and their embeddings
    * get_conversation_chain function: creates a retrieval chain from vectorstore
    * handle_userinput function: generates response from OpenAI GPT API
- `htmlTemplates.py`: A module that defines HTML templates for the user interface.


## How to run
```
streamlit run app.py
```

## Lab 9 quick setup (recommended)
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Then edit `.env` and set:
```bash
OPENAI_API_KEY=your_real_openai_api_key
```

## Lab 9 (Part 1 + Part 2) script
This repo also includes `App_p1.py` and `App_p2.py`.

### `App_p1.py` (CLI version, OpenAI API)
Implements the full lab pipeline:
1. Iterate over PDFs in a folder and extract text.
2. Store extracted text and chunks in SQLite tables.
3. Split text with `CharacterTextSplitter` (`chunk_size=500` by default).
4. Create OpenAI embeddings and store vectors in FAISS.
5. Build a conversational retrieval chain with `ConversationBufferMemory`.
6. Run an interactive driver loop (type `exit` to quit).

Run:
```bash
python App_p1.py --data-dir . --db-path lab9_documents.db --index-dir lab9_faiss_index
```

Part 1 only (data prep):
```bash
python App_p1.py --prepare-only
```

Notes:
- Requires `OPENAI_API_KEY` in `.env` or environment variables.
- Default LLM model: `gpt-4o-mini`
- Default embedding model: `text-embedding-3-small`

### `App_p2.py` (Streamlit version, open-source local models)
This version runs a local-model chatbot UI in Streamlit:
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- LLM: `google/flan-t5-base`
- No OpenAI API key required for this script.

Run:
```bash
streamlit run App_p2.py
```

Usage:
1. Open the local Streamlit URL in browser.
2. Upload one or more PDFs in the sidebar.
3. Click `Process`.
4. Ask questions in the input box.


## Update to use Llama 2 running locally
1. Install Python bindings for llama.cpp library
```
pip install llama-cpp-python
```
2. Download the llama 2 7B GGML model from https://huggingface.co/TheBloke/LLaMa-7B-GGML/blob/main/llama-7b.ggmlv3.q4_1.bin and place it in the models folder
3. Switch language model to use Llama 2 loaded by LlamaCpp
4. Switch embedding model to MiniLM-L6-v2 using HuggingFaceEmbeddings
