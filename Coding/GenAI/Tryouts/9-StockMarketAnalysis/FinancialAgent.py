from phi.agent import Agent 
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.arxiv_toolkit import ArxivToolkit
import openai
from dotenv import load_dotenv
import os 

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')


## Agent Creation 

web_serach_agent = Agent(
    name="Web Search Agent",
    role = "Search in the web for the information",
    model = Groq(id = 'llama-3.3-70b-versatile'), 
    tools = [DuckDuckGo(search = True, 
                        news = True)],
    instructions=['Always include the source'], 
    show_tool_calls=True,
    markdown=True,
)

research_paper_agent = Agent(
    name = "Research Paper Search",
    role = "Search in the Arxiv repository and fetch the relevant information",
    model = Groq(id = 'llama-3.3-70b-versatile'),
    tools=[ArxivToolkit(search_arxiv = True,
                        read_arxiv_papers = True, 
                        download_dir = 'download')], 
    instructions=['Always include the source'], 
    show_tool_calls=True,
    markdown=True,
) #  Search arxiv for Attention all you need paper

financial_agent = Agent(
    name="Finance Agent",    
    model = Groq(id = 'llama-3.3-70b-versatile'), 
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

multi_ai_agent = Agent(
    team = [web_serach_agent, financial_agent, research_paper_agent],
    instructions=['Always include the source', 
                  'Use tabular format to show all the details'],
    show_tool_calls=True,
    markdown=True,
)


## Calling the agent for getting the answer

#multi_ai_agent.print_response('Summarize analyst recommendation and share the latest news for Nvidia and Intel stoks and share the final recommendation between the stokcs', stream=True)
multi_ai_agent.print_response('Is there any paper related to Solar power prediction', stream=True)
# setx OPENAI_API_KEY sk-proj-cUt1IVJk51PA4UmJcao7y06Oab31qT-SsFISaxn0JmywGLIQepPLhtVxkzmfoBANESDg6VZxxlT3BlbTTlDNmI_lNs1ftD-JydUBD3WYicqItcf9pmPvPBHUVQqAm3gA
# setx GROQ_API_KEY gsk_uvusndMMt0WVJhP3hAnFWGdyb3FYgec9ixvRSzFkGVq
# setx PHIDATA_API_KEY phi-Y2cwXLtfZawnZCIcokeOFO5TFVy4zRg
