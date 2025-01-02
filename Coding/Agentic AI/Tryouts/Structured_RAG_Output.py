from typing import List
from pydantic import BaseModel, Field
from phi.agent import Agent, RunResponse
from rich.pretty import pprint
from phi.model.openai import OpenAIChat
from phi.model.groq import Groq


# Define a Pydantic model to enforce the structure of the output
class MovieScript(BaseModel):
    setting: str = Field(..., description="Provide a nice setting for a blockbuster movie.")
    ending: str = Field(..., description="Ending of the movie. If not available, provide a happy ending.")
    genre: str = Field(..., description="Genre of the movie. If not available, select action, thriller or romantic comedy.")
    name: str = Field(..., description="Give a name to this movie")
    characters: List[str] = Field(..., description="Name of characters for this movie.")
    storyline: str = Field(..., description="3 sentence storyline for the movie. Make it exciting!")

# Agent that uses JSON mode
json_mode_agent = Agent(
    name="Structure Agent",
    model= Groq(id = 'llama3-groq-70b-8192-tool-use-preview'),
    description="You write movie scripts in structured format.",
    response_model=MovieScript,
)

# Agent that uses structured outputs
structured_output_agent = Agent(
    name="Structure Agent",
    #model= Groq(id = 'llama3-groq-70b-8192-tool-use-preview'),
    model=OpenAIChat(id="gpt-4o-2024-08-06"),
    description="You write movie scripts in JSON format.",
    response_model=MovieScript,
    structured_outputs=True,
)

json_mode_agent.print_response("New York")
message = "Provide details about New York in a structured JSON format."
structured_output_agent.print_response("New York")

