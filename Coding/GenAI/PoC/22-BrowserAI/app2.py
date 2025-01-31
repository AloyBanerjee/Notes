import streamlit as st
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from browser_use import Agent
from dotenv import load_dotenv
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import (
    BrowserContextConfig,
    BrowserContextWindowSize,
)
import os

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    google_api_key=api_key
)
browser = Browser(
    config=BrowserConfig(
        headless=False,
        disable_security=True,
        chrome_instance_path=chrome_path,
        extra_chromium_args=[f"--window-size={window_w},{window_h}"],
    )
)
# Define async function
async def run_agent():
    agent = Agent(
        task="Compare the price of gpt-4o, gemini and DeepSeek-V3",
        llm=llm,
        
    )
    return await agent.run()

# Wrapper to call async function inside Streamlit
def call_async_agent():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(run_agent())

# Streamlit UI
st.title("AI Price Comparison")

if st.button("Compare Prices"):
    with st.spinner("Fetching prices..."):
        result = call_async_agent()
        st.write("### Price Comparison Result:")
        st.write(result)
