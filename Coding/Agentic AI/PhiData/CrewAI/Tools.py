import os
from dotenv import load_dotenv

from crewai_tools import (
    DirectoryReadTool,
    FileReadTool,
    SerperDevTool,
    WebsiteSearchTool
)
from crewai.tools import BaseTool, tool



load_dotenv()

os.environ['SERPER_API_KEY'] = os.getenv('SERPER_API_KEY')

file_tool = FileReadTool()
search_tool = SerperDevTool()
web_rag_tool = WebsiteSearchTool()


# Defining Custom Tool

#### There are two main ways for one to create a CrewAI tool: 
#### Subclassing BaseTool
#### Utilizing the tool Decorator

class MyCustomTool(BaseTool):
    name: str = "Name of my tool"
    description: str = "Clear description for what this tool is useful for, your agent will need this information to use it."

    def _run(self, argument: str) -> str:
        # Implementation goes here
        return "Result from custom tool"

#Better way of defining custom rools
@tool("QnA")
def my_tool(question: str) -> str:
    """Clear description for what this tool is useful for, your agent will need this information to use it."""
    # Function logic here
    return "Result from your custom tool"

@tool("Multiple")
def multiplication_tool(first_number: int, second_number: int) -> str:
    """Useful for when you need to multiply two numbers together."""
    return first_number * second_number

