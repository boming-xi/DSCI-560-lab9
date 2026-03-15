import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import os

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template

# extract text from uploaded PDFs
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            content=page.extract_text()
            if content:
                text+=content
    return text


# split text into chunks
def get_text_chunks(text):
    text_splitter=CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    chunks=text_splitter.split_text(text)
    return chunks

# create vector database
def get_vectorstore(text_chunks):
    embeddings=OpenAIEmbeddings()
    vectorstore=FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings
    )
    return vectorstore


# create conversation chain
def get_conversation_chain(vectorstore):
    llm=ChatOpenAI(
        temperature=0,
        model_name="gpt-3.5-turbo"
    )
    memory=ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    conversation_chain=ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k":4}
        ),
        memory=memory
    )
    return conversation_chain


# handle user questions
def handle_userinput(user_question):
    response=st.session_state.conversation(
        {"question":user_question}
    )
    st.session_state.chat_history=response["chat_history"]
    for i,message in enumerate(st.session_state.chat_history):
        if i%2==0:
            st.write(
                user_template.replace(
                    "{{MSG}}",
                    message.content
                ),
                unsafe_allow_html=True
            )

        else:
            st.write(
                bot_template.replace(
                    "{{MSG}}",
                    message.content
                ),
                unsafe_allow_html=True
            )

# main streamlit app
def main():
    load_dotenv()
    st.set_page_config(
        page_title="Chat with PDFs",
        layout="wide"
    )
    st.write(css, unsafe_allow_html=True)
    st.markdown("""
    <style>

    /* expand main container */
    section.main > div {
        max-width: 95% !important;
        padding-left: 3rem !important;
        padding-right: 3rem !important;
    }

    /* make input box wider */
    .stTextInput > div > div > input {
        width: 100% !important;
    }

    /* chat message width */
    .chat-message {
        width: 100%;
    }

    </style>
    """, unsafe_allow_html=True)

    # initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation=None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history=None
    st.header("Chat with PDFs ")
    user_question=st.text_input(
        "Ask questions about your documents:"
    )
    if user_question:
        if st.session_state.conversation is None:
            st.warning("Please upload and process PDFs first.")
        else:
            handle_userinput(user_question)

    # sidebar
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs=st.file_uploader(
            "Upload PDF files",
            accept_multiple_files=True
        )
        if st.button("Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF.")
                return
            with st.spinner("Processing PDFs..."):
                raw_text=get_pdf_text(pdf_docs)
                text_chunks=get_text_chunks(raw_text)
                vectorstore=get_vectorstore(text_chunks)
                st.session_state.conversation=\
                    get_conversation_chain(vectorstore)
            st.success("Documents processed successfully!")


if __name__=="__main__":
    main()