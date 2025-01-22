import os 
import openai
from dotenv import load_dotenv

from phi.agent import Agent 
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from phi.model.openai import OpenAIChat
from phi.tools.arxiv_toolkit import ArxivToolkit

import phi.api # Converting the app as api
import phi
from phi.playground import Playground, serve_playground_app

load_dotenv()

phi.api = os.getenv('PHI_API_KEY')



## Agent Creation 

web_serach_agent = Agent(
    name="Web Search Agent",
    role = "Search in the web for the information",
    model = Groq(id = 'llama3-groq-70b-8192-tool-use-preview'), 
    tools = [DuckDuckGo(search = True, 
                        news = True)],
    instructions=['Always include the source'], 
    show_tool_calls=True,
    markdown=True,
)

research_paper_agent = Agent(
    name = "Research Paper Search",
    role = "Search in the Arxiv repository and fetch the relevant information",
    model = Groq(id = 'llama3-groq-70b-8192-tool-use-preview'),
    tools=[ArxivToolkit(search_arxiv = True,
                        read_arxiv_papers = True, 
                        download_dir = 'download')], 
    instructions=['Always include the source'], 
    show_tool_calls=True,
    markdown=True,
)


financial_agent = Agent(
    name="Finance Agent",    
    model = Groq(id = 'llama3-groq-70b-8192-tool-use-preview'), 
    tools = [YFinanceTools(stock_price=True,
                           company_info = True, 
                           income_statements = True,
                           key_financial_ratios = True,
                           analyst_recommendations=True,
                           company_news = True,
                           technical_indicators = True,
                           historical_prices = True,
                           stock_fundamentals=True)],
    instructions=['Use tabular format to show all the details'],
    show_tool_calls=True,
    markdown=True,
)

app_details = Playground(agents = [web_serach_agent,financial_agent, research_paper_agent]).get_app()

if __name__ == '__main__':
    serve_playground_app('playground:app_details', reload=True) # reload is used for debugging
