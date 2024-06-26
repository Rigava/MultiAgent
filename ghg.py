# To install required packages:
# pip install crewai==0.22.5 streamlit==1.32.2

# How to Implement a Simple UI for CrewAI applications
# https://www.youtube.com/watch?v=gWrqfnTGtl8&t=72s

import streamlit as st

from crewai import Crew, Process, Agent, Task
from langchain_core.callbacks import BaseCallbackHandler
from typing import TYPE_CHECKING, Any, Dict, Optional

from langchain_google_genai import GoogleGenerativeAI
key =st.secrets.API_KEY
llm = GoogleGenerativeAI(model='gemini-pro',google_api_key=key)

avators = {"Writer":"https://cdn-icons-png.flaticon.com/512/320/320336.png",
            "Reviewer":"https://cdn-icons-png.freepik.com/512/9408/9408201.png"}

class MyCustomHandler(BaseCallbackHandler):

    
    def __init__(self, agent_name: str) -> None:
        self.agent_name = agent_name

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Print out that we are entering a chain."""
        st.session_state.messages.append({"role": "assistant", "content": inputs['input']})
        st.chat_message("assistant").write(inputs['input'])
   
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Print out that we finished a chain."""
        st.session_state.messages.append({"role": self.agent_name, "content": outputs['output']})
        st.chat_message(self.agent_name, avatar=avators[self.agent_name]).write(outputs['output'])

researcher = Agent(
    role='Senior Researcher',
    backstory='''You work at a leading shipping company as a decarbonisation expert.
                Your expertise lies in developing methodology to calculated GHG intensity.
                You have a knack for dissecting complex data and presenting actionable insights.
                ''',
    goal="Identify the methodology for green house gas intensity.",
    # tools=[]  # This can be optionally specified; defaults to an empty list
    llm=llm,
    callbacks=[MyCustomHandler("Writer")],
)
writer = Agent(
    role='Senior Writer',
    backstory = '''You're a meticulous analyst with a keen eye for detail. You're known for
                your ability to turn complex data into clear and concise reports, making
                it easy for others to understand and act on the information you provide..''',
    goal="Build a compelling and engaging business case about the impact of regulation",
    # tools=[]  # Optionally specify tools; defaults to an empty list
    llm=llm,
    callbacks=[MyCustomHandler("Reviewer")],
)

st.title("💬 GHG Research Studio") 

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "What is the regulation name?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    task1 = Task(
      description=f"""Write a methodology to calculate the impact of {prompt}. """,
      agent=researcher,
      expected_output="Prepare a framework to calculate the impact under 1000 words."
    )

    task2 = Task(
      description=(
        "1. Use the content plan to craft a compelling "
            "blog post on {topic}.\n"
        "2. Incorporate SEO keywords naturally.\n"
		"3. Sections/Subtitles are properly named "
            "in an engaging manner.\n"
        "4. Ensure the post is structured with an "
            "engaging introduction, insightful body, "
            "and a summarizing conclusion.\n"
        "5. Proofread for grammatical errors and "
            "alignment with the brand's voice.\n"
    ),
      agent=writer,
      expected_output="A well-written blog post "
        "in markdown format, ready for publication, "
        "each section should have 2 or 3 paragraphs.",
    )
    # Establishing the crew with a hierarchical process
    project_crew = Crew(
        tasks=[task1, task2],  # Tasks to be delegated and executed under the manager's supervision
        agents=[researcher, writer],
        manager_llm=llm,
        process=Process.hierarchical  # Specifies the hierarchical management approach
    )
    final = project_crew.kickoff()

    result = f"## Here is the Final Result \n\n {final}"
    st.session_state.messages.append({"role": "assistant", "content": result})
    st.chat_message("assistant").write(result)