import os
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv


load_dotenv()

def main(return_chain_only=False):
    
    llm = ChatOpenAI(temperature=0.7, openai_api_key=os.getenv("OPENAI_API_KEY"))
    memory = ConversationBufferMemory()
    chain = ConversationChain(llm=llm, memory=memory)
    
    if return_chain_only:
        return chain

# chain 객체 만들어서 views.py에서 import해서 사용
chain = main(return_chain_only=True)
