from __future__ import annotations

import argparse
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

# Mitigate common macOS OpenMP duplicate runtime crashes from mixed environments.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter

try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except ImportError:
    from langchain.chat_models import ChatOpenAI  # type: ignore
    from langchain.embeddings import OpenAIEmbeddings  # type: ignore

try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    from langchain.vectorstores import FAISS  # type: ignore

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document  # type: ignore


@dataclass
class ExtractedDocument:
    doc_id: int
    file_path: str
    content: str


def initialize_database(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT NOT NULL UNIQUE,
            content TEXT NOT NULL,
            extracted_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            FOREIGN KEY(document_id) REFERENCES documents(id)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id)"
    )
    conn.commit()


def list_pdf_files(data_dir: Path) -> list[Path]:
    return sorted(path for path in data_dir.rglob("*.pdf") if path.is_file())


def extract_text_from_pdf(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages.append(page_text.strip())
    return "\n".join(pages).strip()


def collect_pdf_texts(data_dir: Path, conn: sqlite3.Connection) -> list[ExtractedDocument]:
    pdf_files = list_pdf_files(data_dir)
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {data_dir}")

    conn.execute("DELETE FROM chunks")
    conn.execute("DELETE FROM documents")
    conn.commit()

    collected_docs: list[ExtractedDocument] = []
    extracted_at = datetime.now(timezone.utc).isoformat()

    for pdf_file in pdf_files:
        text = extract_text_from_pdf(pdf_file)
        if not text:
            continue

        cursor = conn.execute(
            "INSERT INTO documents(file_path, content, extracted_at) VALUES (?, ?, ?)",
            (str(pdf_file.resolve()), text, extracted_at),
        )
        collected_docs.append(
            ExtractedDocument(
                doc_id=cursor.lastrowid,
                file_path=str(pdf_file.resolve()),
                content=text,
            )
        )

    conn.commit()

    if not collected_docs:
        raise ValueError("PDF files were found, but no extractable text was produced.")

    return collected_docs


def get_text_chunks(
    extracted_docs: Iterable[ExtractedDocument],
    conn: sqlite3.Connection,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[Document]:
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    conn.execute("DELETE FROM chunks")
    all_chunks: list[Document] = []

    for doc in extracted_docs:
        split_chunks = splitter.split_text(doc.content)
        for idx, chunk_text in enumerate(split_chunks):
            clean_chunk = chunk_text.strip()
            if not clean_chunk:
                continue
            conn.execute(
                "INSERT INTO chunks(document_id, chunk_index, chunk_text) VALUES (?, ?, ?)",
                (doc.doc_id, idx, clean_chunk),
            )
            all_chunks.append(
                Document(
                    page_content=clean_chunk,
                    metadata={
                        "source": doc.file_path,
                        "document_id": doc.doc_id,
                        "chunk_index": idx,
                    },
                )
            )

    conn.commit()

    if not all_chunks:
        raise ValueError("No text chunks were produced from extracted document text.")

    return all_chunks


def build_embeddings(embedding_model: str) -> OpenAIEmbeddings:
    for kwargs in ({"model": embedding_model}, {"model_name": embedding_model}, {}):
        try:
            return OpenAIEmbeddings(**kwargs)
        except TypeError:
            continue
    return OpenAIEmbeddings()


def create_vectorstore(
    chunks: list[Document],
    index_dir: Path,
    embedding_model: str = "text-embedding-3-small",
) -> FAISS:
    embeddings = build_embeddings(embedding_model)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))
    return vectorstore


def build_chat_model(model_name: str, temperature: float) -> ChatOpenAI:
    for kwargs in (
        {"model": model_name, "temperature": temperature},
        {"model_name": model_name, "temperature": temperature},
    ):
        try:
            return ChatOpenAI(**kwargs)
        except TypeError:
            continue
    return ChatOpenAI(temperature=temperature)


def create_conversation_chain(
    vectorstore: FAISS,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    top_k: int = 4,
) -> ConversationalRetrievalChain:
    llm = build_chat_model(model_name=model_name, temperature=temperature)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": top_k}
        ),
        memory=memory,
        return_source_documents=True,
    )


def prepare_search_data(
    data_dir: Path,
    db_path: Path,
    index_dir: Path,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    embedding_model: str = "text-embedding-3-small",
) -> tuple[FAISS, int, int]:
    conn = sqlite3.connect(str(db_path))
    try:
        initialize_database(conn)
        extracted_docs = collect_pdf_texts(data_dir, conn)
        chunks = get_text_chunks(
            extracted_docs,
            conn=conn,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        vectorstore = create_vectorstore(
            chunks=chunks,
            index_dir=index_dir,
            embedding_model=embedding_model,
        )
    finally:
        conn.close()

    return vectorstore, len(extracted_docs), len(chunks)


def ask_chain(
    chain: ConversationalRetrievalChain, question: str
) -> tuple[str, list[str]]:
    payload = {"question": question}
    if hasattr(chain, "invoke"):
        response = chain.invoke(payload)
    else:
        response = chain(payload)

    answer = response.get("answer") or response.get("result") or ""

    sources: list[str] = []
    for source_doc in response.get("source_documents", []):
        source = source_doc.metadata.get("source", "unknown")
        chunk_index = source_doc.metadata.get("chunk_index")
        label = f"{source}#chunk{chunk_index}" if chunk_index is not None else source
        if label not in sources:
            sources.append(label)

    return answer, sources


def run_driver(chain: ConversationalRetrievalChain) -> None:
    print("\nChatbot is ready. Type your question and press Enter.")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            user_question = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting chatbot.")
            break

        if user_question.lower() == "exit":
            print("Exiting chatbot.")
            break
        if not user_question:
            continue

        answer, sources = ask_chain(chain, user_question)
        print(f"Bot> {answer}")
        if sources:
            print("Sources>")
            for source in sources[:5]:
                print(f" - {source}")
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DSCI-560 Lab 9 Part 1+2: PDF extraction, vector store, and QA chatbot."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("."),
        help="Directory containing PDF files to ingest.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("lab9_documents.db"),
        help="SQLite database path for extracted text and chunks.",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("lab9_faiss_index"),
        help="Directory to save the FAISS vector index.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Chunk size for CharacterTextSplitter.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        help="Chunk overlap for CharacterTextSplitter.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="text-embedding-3-small",
        help="OpenAI embedding model name.",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI chat model name.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM temperature.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Retriever top-k similar chunks.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Run Part 1 (data prep) only and skip the interactive chat loop.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is missing. Set it in your environment or .env.")

    args = parse_args()
    vectorstore, doc_count, chunk_count = prepare_search_data(
        data_dir=args.data_dir,
        db_path=args.db_path,
        index_dir=args.index_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_model=args.embedding_model,
    )

    print("Part 1 complete:")
    print(f" - documents extracted: {doc_count}")
    print(f" - chunks created: {chunk_count}")
    print(f" - sqlite db: {args.db_path.resolve()}")
    print(f" - vector index: {args.index_dir.resolve()}")

    if args.prepare_only:
        return

    chain = create_conversation_chain(
        vectorstore=vectorstore,
        model_name=args.llm_model,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print("\nPart 2 conversation chain ready.")
    run_driver(chain)


if __name__ == "__main__":
    main()
