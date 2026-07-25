
import os
import time
import streamlit__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from crewai import Agent, Crew, Process, Task, LLM
# ... rest of your imports and code below ... as st
from crewai import Agent, Crew, Process, Task, LLM
from crewai_tools import SerperDevTool
from pypdf import PdfReader

# Page Configuration
st.set_page_config(page_title="Agentic Resume Analyzer", page_icon="🤖", layout="wide")

st.title("🤖 Autonomous Agentic Resume & Skill-Gap Analyzer")
st.caption("IBM SkillsBuild Internship Submission | Powered by Multi-Agent AI")

# Sidebar Configuration
with st.sidebar:
    st.header("🔑 API Credentials")
    google_api_key = st.text_input("Gemini API Key", type="password", help="Get a free key from Google AI Studio")
    serper_api_key = st.text_input("Serper API Key (Optional)", type="password", help="Optional: Enables live web search")
    
    st.markdown("---")
    st.markdown("### How it works")
    st.write("1. **Agent 1:** Parses resume context")
    st.write("2. **Agent 2:** Researches live market skills")
    st.write("3. **Agent 3:** Builds 4-week action plan")

# Main Interface
col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
with col2:
    target_role = st.text_input("Target Role", value="AI & Machine Learning Engineer")

if st.button("🚀 Analyze Profile with AI Agents", use_container_width=True):
    if not uploaded_file:
        st.error("Please upload a PDF resume.")
    elif not google_api_key:
        st.error("Please enter your Gemini API key in the sidebar.")
    else:
        try:
            # Set Environment Variables
            os.environ["GEMINI_API_KEY"] = google_api_key
            if serper_api_key:
                os.environ["SERPER_API_KEY"] = serper_api_key

            # Initialize Gemini LLM using 2.5-flash or 1.5-flash
            llm = LLM(
                model="gemini/gemini-2.5-flash",
                api_key=google_api_key
            )

            # Extract Resume Text
            reader = PdfReader(uploaded_file)
            resume_text = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    resume_text += text

            # Truncate text to stay well within free tier input limits
            truncated_resume = resume_text[:1500]

            st.info("⚡ Agents initialized! Processing pipeline...")

            # Define CrewAI Agents
            parser_agent = Agent(
                role="Resume Technical Extractor",
                goal="Extract key technical skills, projects, and background from resumes.",
                backstory="You are an expert technical recruiter analyzing core competencies.",
                llm=llm,
                verbose=True
            )

            research_agent = Agent(
                role="Tech Market Research Specialist",
                goal=f"Determine key industry requirements for {target_role}.",
                backstory="You are a tech analyst tracking software engineering hiring trends.",
                tools=[SerperDevTool()] if serper_api_key else [],
                llm=llm,
                verbose=True
            )

            advisor_agent = Agent(
                role="Career Strategy Agent",
                goal="Identify skill gaps and generate a realistic 4-week roadmap.",
                backstory="You are a tech mentor guiding candidates to bridge skill gaps.",
                llm=llm,
                verbose=True
            )

            # Define Tasks
         # Define Tasks with Context Pipeline
            task1 = Task(
                description=f"Analyze the following resume text:\n\n{truncated_resume}\n\nSummarize key technical skills, programming languages, and projects.",
                expected_output="A list of candidate's technical skills and projects extracted from the resume.",
                agent=parser_agent
            )

            task2 = Task(
                description=f"List top essential technical skills, frameworks, and methodologies required for a {target_role} in today's market.",
                expected_output="A list of core market skill requirements for the target role.",
                agent=research_agent
            )

            task3 = Task(
                description="Review the candidate's parsed resume skills and compare them against the market requirements for the target role. "
                            "Produce a Markdown report containing: 1. Matched Skills, 2. Skill Gaps, 3. A detailed 4-Week Action Plan.",
                expected_output="A markdown report with Skill Gap Analysis and a 4-Week Roadmap.",
                agent=advisor_agent,
                context=[task1, task2]  # <--- THIS PASSES THE OUTPUT OF TASK 1 AND 2 TO AGENT 3
            )

            # Execute Crew Sequentially
            crew = Crew(
                agents=[parser_agent, research_agent, advisor_agent],
                tasks=[task1, task2, task3],
                process=Process.sequential
            )

            with st.spinner("🤖 Autonomous agents are working..."):
                result = crew.kickoff()
                st.success("Analysis Complete!")
                st.markdown("---")
                st.markdown("### 📊 Generated Career & Gap Report")
                st.markdown(result)

        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                st.error("⏳ Free tier rate limit reached! Please wait 60 seconds and try again, or create a new API Key in Google AI Studio.")
            else:
                st.error(f"An error occurred: {str(e)}")
