import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

st.set_page_config(page_title="RAG Book Assistant")

st.title("📚 CourseMate")
st.write("Upload a PDF and ask questions from the document")

uploaded_file = st.file_uploader("Upload a PDF book", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    st.success("PDF uploaded successfully!")

    if st.button("Create Vector Database"):
        with st.spinner("Processing document..."):
            loader = PyPDFLoader(file_path)
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )

            chunks = splitter.split_documents(docs)

            embeddings = OpenAIEmbeddings()

            # Creates the database and automatically persists to "chroma_db"
            vectorstore = Chroma.from_documents(
                documents=chunks, embedding=embeddings, persist_directory="chroma_db"
            )

        st.success("Vector database created!")


if os.path.exists("chroma_db"):
    st.divider()
    st.subheader("Ask Questions From the Book")

    query = st.text_input("Enter your question")

    # Connect to the database ONLY when a query is submitted to prevent locking
    if query:
        embeddings = OpenAIEmbeddings()

        vectorstore = Chroma(
            persist_directory="chroma_db", embedding_function=embeddings
        )

        retriever = vectorstore.as_retriever(
            search_type="mmr", search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5}
        )

        llm = ChatMistralAI(model="mistral-small-2506")

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
""",
                ),
                (
                    "human",
                    """Context:
{context}

Question:
{question}
""",
                ),
            ]
        )

        docs = retriever.invoke(query)

        context = "\n\n".join([doc.page_content for doc in docs])

        final_prompt = prompt.invoke({"context": context, "question": query})

        response = llm.invoke(final_prompt)

        st.write("### AI Answer")
        st.write(response.content)
