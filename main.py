from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

load_dotenv()
data = PyPDFLoader("document_loaders/GRU.pdf")
docs = data.load()
template = ChatPromptTemplate.from_messages(
    [
        ("system", "you are an AI that summarize the text"),
        ("human", "{data}"),
    ]
)
model = ChatMistralAI(model="mistral-small-2506")
prompt = template.format_messages(data=docs[0].page_content)
result = model.invoke(prompt)
print(result.content)
