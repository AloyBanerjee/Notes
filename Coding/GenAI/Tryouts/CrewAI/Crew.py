from crewai import Crew, Process
from Agent import news_researcher, news_writer
from Tasks import researcher_task, writer_task

## Creating the tech focused crew with cofguration 

crew = Crew(
            agents=[news_researcher, news_writer],
            tasks=[researcher_task, writer_task],
            process=Process.sequential
            )

## Staring the task execution process

result = crew.kickoff(inputs={'topic': 'AI in Medtech industry'})