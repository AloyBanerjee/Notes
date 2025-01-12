from phi.agent import Agent 
from phi.model.groq import Groq
from phi.model.openai import OpenAIChat
from phi.embedder.ollama import OllamaEmbedder
from phi.embedder.openai import OpenAIEmbedder
#from phi.embedder.google import GeminiEmbedder
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.lancedb import LanceDb, SearchType
import os 
from dotenv import load_dotenv
#from google.generativeai import model_types

## Working with Open AI Embeddings 

load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing. Check your .env file.")

# Create a knowledge base from a PDF
knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    # Use LanceDB as the vector database
    vector_db=LanceDb(
        table_name="recipes",
        uri="tmp/lancedb",
        search_type= SearchType.vector,
        embedder= OpenAIEmbedder(model="text-embedding-3-small"),#https://docs.phidata.com/embedder/ollama#OllamaEmbedder(),GeminiEmbedder(api_key=GEMINI_API_KEY,model='models/text-embedding-004')
    ),
)
# Comment out after first run as the knowledge base is loaded
knowledge_base.load()

agent = Agent(
    model = OpenAIChat(id="gpt-4o"),#Groq(id ='llama-3.2-3b-preview'),
    # Add the knowledge base to the agent
    knowledge=knowledge_base,
    show_tool_calls=True,
    markdown=True,
)
agent.print_response("What are the ingredients required for Tom Kha Gai", stream=True)