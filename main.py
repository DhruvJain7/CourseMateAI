from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


template = ChatPromptTemplate.from_messages(
    [
        ("system", "you are an AI that summarize the text"),
        ("human", "{data}"),
    ]
)

model = ChatMistralAI(model="mistral-small-2506")
