from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_openai import OpenAIEmbeddings

load_dotenv()

embedding_model = OpenAIEmbeddings()
# Retrieving the already created db
vectorstore = Chroma(persist_directory="chroma_DB", embedding_function=embedding_model)

retriever = vectorstore.as_retriever(
    search_type = "mmr",
    search_kwargs={
        "k":4,
        "fetch_k" : 10,
        "lambda_mult":

    }
)
