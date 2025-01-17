# import numpy as np
# import pandas as pd

# from phi.knowledge.csv import CSVKnowledgeBase
# from phi.vectordb.chroma import ChromaDb

# from phi.agent import Agent



# maude_fda_data = pd.read_excel('fda_device_data.xlsx')


# maude_knowledge_base = CSVKnowledgeBase(
#     path="fda_device_data.xlsx",
#     vector_db=ChromaDb(collection="maude"),
# )
# # Comment out after first run
# maude_knowledge_base.load(recreate=False)


# agent = Agent(
#     knowledge=maude_knowledge_base,
#     search_knowledge=True,
# )
# agent.knowledge.load(recreate=False)

# agent.print_response("What are the potential hazards present from the knowledge base?")


from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.embedder.openai import OpenAIEmbedder
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.knowledge.csv import CSVKnowledgeBase
from phi.vectordb.lancedb import LanceDb, SearchType

# Create a knowledge base from a PDF
knowledge_base = CSVKnowledgeBase(
    path="H:\Interview Preparation\Coding\GenAI\Tryouts\6-Maude DB Analysis\fda_device_data.xlsx",
    # Use LanceDB as the vector database
    vector_db=LanceDb(
        table_name="maude",
        uri="tmp/lancedb",
        search_type=SearchType.vector,
        embedder=OpenAIEmbedder(model="text-embedding-3-small"),
    ),
)
# Comment out after first run as the knowledge base is loaded
knowledge_base.load(recreate=False)

agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    # Add the knowledge base to the agent
    knowledge=knowledge_base,
    show_tool_calls=True,
    markdown=True,
)
agent.print_response("What are the potential hazards present from the knowledge base?", stream=True)
