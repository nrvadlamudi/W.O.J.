from dotenv import load_dotenv
load_dotenv()
import os

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
teamsLoader = WebBaseLoader("https://www.nba.com/news/nba-offseason-every-deal-2024")
tradesLoader = WebBaseLoader("https://www.nba.com/news/2024-offseason-trade-tracker")

trade_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 20,
    length_function = len,
    is_separator_regex=False
)

key = os.getenv("OPENAI_API_KEY")

from langchain.agents.agent_toolkits import create_conversational_retrieval_agent, create_retriever_tool
from langchain_openai.chat_models import ChatOpenAI

trade_page = tradesLoader.load()
trades = trade_splitter.create_documents([trade_page[0].page_content])

sign_page = teamsLoader.load()
signings = trade_splitter.create_documents([sign_page[0].page_content])

content = [trades, signings]

embeddings = OpenAIEmbeddings()

db = FAISS.from_documents(signings, embeddings)

retriever = db.as_retriever()

tool = create_retriever_tool(
    retriever,
    "search_nba_transactions",
    "Search for signings, extensions, and trades made by each team"
)

tools = [tool]

llm = ChatOpenAI(temperature=0)
agent_executor= create_conversational_retrieval_agent(llm,tools,verbose=True)

input = "What free agents have the Bulls signed this offseason?"
result = agent_executor.invoke({"input":input})
print(result)