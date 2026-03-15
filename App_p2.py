from PyPDF2 import PdfReader

from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline

from transformers import pipeline


def get_pdf_text(pdf_docs):

    text = ""

    for pdf in pdf_docs:

        pdf_reader = PdfReader(pdf)

        for page in pdf_reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text

    return text


def get_text_chunks(text):

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )

    chunks = text_splitter.split_text(text)

    return chunks


def get_vectorstore(text_chunks):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings
    )

    return vectorstore


def load_llm():

    pipe = pipeline(
        "text-generation",
        model="google/flan-t5-base",
        max_new_tokens=256,
        temperature=0.1
    )

    return HuggingFacePipeline(pipeline=pipe)


def get_conversation_chain(vectorstore):

    llm = load_llm()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        ),
        memory=memory
    )

    return conversation_chain