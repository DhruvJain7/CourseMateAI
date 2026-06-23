# Load the pdf
# Split into chunks
# create the embeddings
# Store into chromaDB
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

data = PyPDFLoader("document_loaders/deeplearning.pdf")
docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks = splitter.split_documents(docs)

embedding_model = OpenAIEmbeddings()

vectorstore = Chroma.from_documents(
    documents=chunks, embedding=embedding_model, persist_directory="chroma_DB"
)
