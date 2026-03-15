import streamlit as st
from htmlTemplates import css, bot_template, user_template
from App_p2 import (
    get_pdf_text,
    get_text_chunks,
    get_vectorstore,
    get_conversation_chain
)


def handle_userinput(user_question):
    response = st.session_state.conversation(
        {"question": user_question}
    )
    st.session_state.chat_history = response["chat_history"]
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(
                user_template.replace(
                    "{{MSG}}", message.content
                ),
                unsafe_allow_html=True
            )
        else:
            st.write(
                bot_template.replace(
                    "{{MSG}}", message.content
                ),
                unsafe_allow_html=True
            )


def main():
    st.set_page_config(
        page_title="Chat with PDFs",
        layout="wide"
    )
    st.write(css, unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    st.header("Chat with PDFs")
    user_question = st.text_input(
        "Ask questions about your documents:"
    )

    if user_question:
        if st.session_state.conversation is None:
            st.warning("Please upload and process a PDF first.")
        else:
            handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs",
            accept_multiple_files=True
        )

        if st.button("Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF.")
            else:
                with st.spinner("Processing PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(
                        vectorstore
                    )
                st.success("Documents processed!")

if __name__ == "__main__":
    main()