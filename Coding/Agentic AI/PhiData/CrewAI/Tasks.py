from crewai import Task
from Agent import news_researcher,news_writer
from Tools import search_tool,web_rag_tool
# Creating task for each agent

researcher_task = Task(
    description=(
        "Identify the next big trend in {topic}."
        "Focus on idenifying pros and cons and the overall narrative."
        "Your final report should clearly articulate the key points, its market"
        "opportunities and pootential risks."
    ),
    expected_output="A comprehensive 3 paragraphs long report on the latest AI trends.",
    tools=[search_tool,web_rag_tool], 
    agent=news_researcher
)


writer_task = Task(
    description=(
        "Compose an insightful article on {topic}."
        "focus on the latest trends and how its impacting the industry."
        "This article should be easy to understand, engaging and positive."
    ),
    expected_output="A comprehensive 4 paragraphs long rarticle on {topic} advancements formatted as markdown.",
    tools=[search_tool,web_rag_tool],  
    agent=news_writer,
    async_execution=False, 
    output_file="blog-post.md"
)