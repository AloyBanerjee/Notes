from crewai import Agent
from crewai import LLM
from langchain_google_genai import ChatGoogleGenerativeAI
from Tools import file_tool, search_tool, web_rag_tool
import os
from dotenv import load_dotenv
load_dotenv()


## Set up the LLM model with google Gemini

google_api_key = os.getenv('GEMINI_API_KEY')

# gemini_llm =  ChatGoogleGenerativeAI(
#                                      provider="google",
#                                      model="gemini/gemini-1.5-flash",
#                                      verbose=True,
#                                      temperature=0.3,
#                                      google_api_key = google_api_key)

gemini_llm = LLM(
    provider="google",
    model="gemini/gemini-1.5-pro-latest",
    verbose=True,
    temperature=0.3,
    google_api_key = google_api_key
)


# Creating Agent
news_researcher = Agent(
    role = "Senior Researcher", 
    goal = 'Uncover the ground breaking technolgies in {topic}',
    verbose=True, 
    memory = True,
    backstory=(
        "Driven by curosity, you are at the forefron of innovation"
        "eager to explore and share the knowldge that could change the world."
    ), 
    tools=[search_tool,web_rag_tool], 
    llm=gemini_llm, 
    allow_delegation=True
)

news_writer = Agent(
    role = "Senior Writer", 
    goal = 'Narrate complelling tech stories about {topic}',
    verbose=True, 
    memory = True,
    backstory=(
        "With a flair for simplifying complex topics, you craft enggng narratives that captivate"
        "and educate, bringing new disciveries to light in an accessible manner."
    ), 
    tools=[search_tool,web_rag_tool], 
    llm=gemini_llm, 
    allow_delegation=False
)