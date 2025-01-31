from langchain_google_genai import ChatGoogleGenerativeAI
from browser_use import Agent
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
import asyncio

llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        google_api_key=api_key
    )

async def main():
    agent = Agent(
        task="Compare the price of gpt-4o, gemini and DeepSeek-V3", #Compare the price of gpt-4o and DeepSeek-V3
        llm=llm,
    )
    result = await agent.run()
    print(result)

asyncio.run(main())
