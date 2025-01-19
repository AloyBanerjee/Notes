import streamlit as st
import numpy as np 
import pandas as pd
import json
import os 
import time
from dotenv import load_dotenv  
from langchain.schema import BaseOutputParser, AIMessage 
from phi.agent import Agent
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_groq import ChatGroq
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from tqdm import tqdm
from langchain.schema import BaseOutputParser, AIMessage
from streamlit_extras.let_it_rain import rain 
from phi.tools.serpapi_tools import SerpApiTools
import pygwalker as pyg
from pygwalker.api.streamlit import StreamlitRenderer

load_dotenv()


st.session_state.prompt_info = '''You are an expert in identifying and categorizing technical & medical device related risks and issues in text. 
You are tasked with extracting structured information from text related to medical device failures. 
Your goal is to extract and organize the following fields from the provided content:

Hazard: A single-word or concise description of the unique hazard (e.g., "Battery Depletion"). This should not be a sentence.
Hazardous Situation: Detailed description of the situation, capturing the context from complaint text, follow-up, and manufacturer narrative.
Harm/Potential Harm: A paragraph explaining the potential harm or adverse outcomes that could result from the issue in the given context.
Manufacturer Name: A list of all unique manufacturer names mentioned in the provided data.
Format the output as follows:

Hazard: [Single word or short phrase]
Hazardous Situation: [Detailed description]
Harm/Potential Harm: [Paragraph explaining potential harm]
Manufacturer Name: [Unique manufacturer names as a list]
'''

st.session_state.prompt_info_new = '''

You are a highly skilled medical device safety expert with extensive knowledge in risk assessment and failure analysis. Analyze the provided data and structure your response as follows:

### 1. Hazard
- Identify the unique hazard using a single word or a concise phrase.
- Avoid full sentences and ensure it is a precise descriptor.

### 2. Hazardous Situation
- Provide a detailed description of the situation, including context from complaint text, follow-up, and manufacturer narrative.
- Focus on capturing the specific conditions leading to the hazard.

### 3. Harm/Potential Harm
- Write a paragraph explaining the potential harm or adverse outcomes that could result from the identified hazard in the given context.
- Be thorough in describing the potential risks and their impacts.

### 4. Manufacturer Name
- Extract and list all unique manufacturer names mentioned in the data.
- Ensure the list includes only distinct names with no duplicates.

### 5. Contextual Research (Optional)
- If requested, use external sources to:
  - Provide examples of similar hazards in medical devices.
  - Search for risk mitigation strategies.
  - Cite recent articles or research supporting the analysis.

Format your response using clear markdown headers and bullet points. Be precise, structured, and thorough in your analysis.'''

### Configuring API Keys ###
if "GEMINI_API_KEY" not in st.session_state:
    st.session_state.GEMINI_API_KEY = None
if "OPENAI_API_KEY" not in st.session_state:
    st.session_state.OPENAI_API_KEY = None
if "GROQ_API_KEY" not in st.session_state:
    st.session_state.GROQ_API_KEY = None
if 'SERP_API_KEY' not in st.session_state:
    st.session_state.SERP_API_KEY = None

st.session_state.GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
st.session_state.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
st.session_state.GROQ_API_KEY = os.getenv('GROQ_API_KEY')
st.session_state.SERP_API_KEY = os.getenv('SERP_API_KEY')

### Variable Declaration & Common Function ###

independent_schema = [
    ResponseSchema(name="Hazard", description="A unique descriptor of the hazard, typically a single word or concise phrase."),
    ResponseSchema(name="Hazardous Situation", description="A detailed description of the situation derived from the complaint text, follow-up, and manufacturer narrative."),
    ResponseSchema(name="Harm/Potential Harm", description="A paragraph explaining the potential adverse outcomes or harm caused by the identified hazard in the given context."),
    ResponseSchema(name="Manufacturer Name", description="A list of unique manufacturer names mentioned in the provided data.")
]
response_schemas = [
    ResponseSchema(
        name="MaudeRiskList",
        description="A list of all failure mode and hazard information extracted from the maude data",
        type="array",
        items={"type": "object", "properties": independent_schema},
    )
]
# Create an output parser
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()
class InfoListParser(BaseOutputParser):
    def parse(self, text: dict) -> dict:
        return text

# Define the prompt template
st.session_state.prompt_one = PromptTemplate(
    template= st.session_state.prompt_info + "\n{format_instructions}\n\nContext: {paragraph}",
    input_variables=["paragraph"],
    partial_variables={"format_instructions": format_instructions},
)


### LLM Configuration 
llama_vers_llm = ChatGroq(
    api_key = os.getenv('GROQ_API_KEY'),
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

llama_spectdoc_llm = ChatGroq(
    api_key = os.getenv('GROQ_API_KEY'),
    model_name="llama-3.3-70b-specdec",
    temperature=0
)

llama_8192_llm = ChatGroq(
    api_key = os.getenv('GROQ_API_KEY'),
    model_name="llama3-70b-8192",
    temperature=0
)

mixtral_8x7b_llm = ChatGroq(
    api_key = os.getenv('GROQ_API_KEY'),
    model_name="mixtral-8x7b-32768",
    temperature=0
)

gemma_llm = ChatGroq(
    api_key = os.getenv('GROQ_API_KEY'),
    model_name="gemma2-9b-it", 
    temperature=0
)

gemini_model=Gemini(
        api_key = os.getenv('GEMINI_API_KEY'),
        id="gemini-2.0-flash-exp"
    )

llms_selection = {
    "Llama Versatile": ChatGroq(
        api_key=os.getenv('GROQ_API_KEY'),
        model_name="llama-3.3-70b-versatile",
        temperature=0
    ),
    "Llama Spectdoc": ChatGroq(
        api_key=os.getenv('GROQ_API_KEY'),
        model_name="llama-3.3-70b-specdec",
        temperature=0
    ),
    "Llama 8192": ChatGroq(
        api_key=os.getenv('GROQ_API_KEY'),
        model_name="llama3-70b-8192",
        temperature=0
    ),
    "Mixtral 8x7b": ChatGroq(
        api_key=os.getenv('GROQ_API_KEY'),
        model_name="mixtral-8x7b-32768",
        temperature=0
    ),
    "Gemma 9b": ChatGroq(
        api_key=os.getenv('GROQ_API_KEY'),
        model_name="gemma2-9b-it",
        temperature=0
    )
    # ),
    # "Gemini": Gemini(
    #     api_key=os.getenv('GEMINI_API_KEY'),
    #     id="gemini-2.0-flash-exp"
    # )
}

# Initialize modal state in session state
if "show_reference_popup" not in st.session_state:
    st.session_state.show_reference_popup = False


### Agent Configuration
maude_google_agent = Agent(
    model=Gemini(
        api_key = os.environ['GEMINI_API_KEY'],
        id="gemini-2.0-flash-exp"
    ),
    tools=[DuckDuckGo(), SerpApiTools()],
    markdown=True
)

#### Page Design ####

st.set_page_config(page_title='Maude Data Analysis', page_icon='📚', layout='wide')

# Define CSS for light blue theme

# CSS for a blue-themed table
table_css = """
<style>
    .stTable tbody tr:nth-child(even) {
        background-color: #e8f4ff;  /* Light blue for even rows */
    }
    .stTable tbody tr:nth-child(odd) {
        background-color: #f2f9ff;  /* Very light blue for odd rows */
    }
    .stTable thead th {
        background-color: #80bfff;  /* Light blue header */
        color: white;               /* White text for the header */
        font-weight: bold;
    }
    .stTable tbody td {
        color: #004085;             /* Dark blue text for table body */
        font-weight: bold;
    }
    .stTable tbody th {
        background-color: #d6ecff;  /* Lighter blue for index column */
        color: #004085;             /* Dark blue text for index column */
        font-weight: bold;
    }
    </style>
"""
dropdown_upload_css = """
    <style>
    /* Customize file uploader     
    [data-testid="stFileUploader"] > label {
        background-color: #e8f4ff; 
        color: #004085;           
        font-weight: bold;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #80bfff; 
        cursor: pointer;
    }*/
    [data-testid="stFileUploader"] button {
        
    }

    [data-testid="stFileUploader"] button:hover {
        background-color: #0056b3; /* Darker blue on hover */
        color: white;
        border-color: #0056b3; /* Darker blue border on hover */
    }


    [data-testid="stFileUploader"] > div {
        background-color: #f2f9ff; /* Very light blue background for file area */
        border: 1px solid #80bfff; /* Light blue border */
        border-radius: 5px;
    }

    /* Customize select box */
    [data-baseweb="select"] > div {
        background-color: #e8f4ff;  /* Light blue background */
        color: #004085;             /* Dark blue text */
        font-weight: bold;
        border: 1px solid #80bfff;  /* Light blue border */
        border-radius: 5px;
    }

    [data-baseweb="select"] .css-1hb7zxy-Input {
        color: #004085;             /* Dark blue text inside the dropdown */
    }

    [data-baseweb="select"] [role="option"] {
        background-color: #f2f9ff;  /* Very light blue for dropdown options */
        color: #004085;             /* Dark blue text */
    }

    [data-baseweb="select"] [role="option"]:hover {
        background-color: #cce7ff;  /* Slightly darker blue on hover */
    }
    </style>
"""
btncss = """
    <style>
    div.stButton > button {
        background-color: #004085; /* Blue background */
        color: white; /* White text */
        border: 2px solid #004085; /* Blue border */
        padding: 0.5em 1em; /* Padding inside the button */
        border-radius: 4px; /* Rounded corners */
        font-size: 16px; /* Font size */
        font-weight: bold; /* Bold text */
        cursor: pointer; /* Pointer cursor on hover */
    }
    div.stButton > button:hover {
        background-color: #0056b3; /* Darker blue on hover */
        color: white;
    }
    </style>
    """
scroll_css = """
<style>
.scrollable-container {
    max-height: 700px; /* Set the height of the scrollable area */
    overflow-y: auto; /* Enable vertical scrolling */
    padding: 10px;
    border: 1px solid #ddd;
    border-radius: 5px;
    background-color: #f2f9ff; /* Optional: Add a background color */
    font-size: 12px; /* Smaller font size */
    line-height: 1.4; /* Optional: Adjust line spacing for readability */
}
</style>
"""

st.markdown(btncss,
    unsafe_allow_html=True
)
st.markdown(dropdown_upload_css,
    unsafe_allow_html=True
)

st.markdown(scroll_css, 
            unsafe_allow_html=True
)

main_col1, main_col2 = st.columns([2, 1])
with main_col1:
    st.title("🤖 Maude Data Analysis Agent🚀")
with main_col2:
    selected_llm = st.selectbox(
        "Choose an LLM for processing:",
        list(llms_selection.keys())  # Dropdown options from the LLM dictionary
    )
    st.session_state.selected_llm = llms_selection[selected_llm]
    st.session_state.selected_model_name = st.session_state.selected_llm.model_name if hasattr(st.session_state.selected_llm, 'model_name') else "Unknown Model"
    st.session_state.selected_llm = llms_selection[selected_llm]
    st.write(f"📝 **Selected LLM: {st.session_state.selected_model_name}**")


st.write("🚀Upload Maude Data CSV file to analyze the medical device failures")

# Create containers for better organization
upload_container = st.container()
data_container = st.container()
analysis_container = st.container()

with upload_container:
    uploaded_file = st.file_uploader(
        "Upload Maude Data File",
        type=["csv", "xls", "xlsx"],
        help="Supported formats: CSV, XLS, XLSX",
    )


with data_container:
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write("📄 **File Details**")
            st.info(f"File Name: {uploaded_file.name}")
            st.info(f"File Type: {uploaded_file.type}")

            with st.spinner("Processing file..."):
                if uploaded_file.name.endswith('.csv'):
                    df_maude_data = pd.read_csv(uploaded_file)             
                elif uploaded_file.name.endswith('.xlsx'):
                    df_maude_data = pd.read_excel(uploaded_file, engine='openpyxl') 

                
                # Subset the necessary columns
                maude_subset = df_maude_data[['product_problems', 'complaint_txt', 'follow_up', 'manufacturer_narrative']]    

                # Add a new 'event_info' column
                for index, item in maude_subset.iterrows():
                    maude_subset.loc[index, 'event_info'] = (
                        'product_problems - ' + ' ' + str(item.product_problems) + ', ' +
                        'complaint_text - ' + ' ' + str(item.complaint_txt) + ', ' +
                        'complaint_follow_up - ' + ' ' + str(item.follow_up) + ', ' +
                        'manufacturer_narrative - ' + ' ' + str(item.manufacturer_narrative)
                    )
                st.write("📄 **Data Preview**")
                st.dataframe(maude_subset.head())

                # Concatenate all event information into a single string
                st.session_state.eventinfo_all = ''
                for _, row in maude_subset.iterrows():
                    st.session_state.eventinfo_all += row['event_info'] + ' '

            st.success('Data loaded successfully, and ready for analysis.')
        with col2:
            st.write("📄 **Data Summary**")
            #gwalker = pyg.walk(maude_subset) # for jupyter notebook
            #pyg_app = StreamlitRenderer(maude_subset)            
            #pyg_app.explorer(default_tab="data")

            @st.cache_resource
            def get_pyg_renderer() -> StreamlitRenderer:
                return StreamlitRenderer(maude_subset, spec="./gw_config.json", spec_io_mode="rw")
            
            renderer = get_pyg_renderer()
            
            tab1, tab2 = st.tabs(["Explorer", "Data Profiling"])
            
            with tab1:
                renderer.explorer(key="explorer_tab")
            
            with tab2:
                renderer.explorer(default_tab="data", key="data_tab")


with analysis_container:
    col1, col2 = st.columns([1, 2])
    with col2:
        if uploaded_file is not None:
            analyze_button = st.button(
                    "🔍 Analyze Data",
                    type="primary",
                    use_container_width=True
                )
            with analysis_container:
                if analyze_button:
                    with st.spinner("🔄 Analyzing data... Please wait."):
                        try:
                            ### Chaining with LCEL
                            parser=StrOutputParser()

                            chain= st.session_state.prompt_one | st.session_state.selected_llm | parser

                            result = chain.invoke({"paragraph": st.session_state.eventinfo_all})

                            st.session_state.result = result

                            parser = InfoListParser()

                            parsed_output = parser.parse(output_parser.parse(result))

                            st.session_state.parsed_output = parsed_output

                            st.markdown("### 📋Medical Device Risk Analysis Results📝")

                            table_data = []
                            for risk in parsed_output['MaudeRiskList']:
                                table_data.append({
                                    "Hazard": risk["Hazard"],
                                    "Hazardous Situation": risk["Hazardous Situation"],
                                    "Harm/Potential Harm": risk["Harm/Potential Harm"],
                                    "Manufacturer Name(s)": ", ".join(risk["Manufacturer Name"])
                                })      
                            
                                                                        
                            st.session_state.eventinfo_combined = f"""
                                                                    ### Event Information:
                                                                    {st.session_state.eventinfo_all}

                                                                    ### Result Information:
                                                                    {st.session_state.parsed_output}
                                                                    """
                            st.session_state.prompt_info_new = f"""
                                                                You are a highly skilled medical device safety expert with extensive knowledge in risk assessment and failure analysis. Analyze the provided data and structure your response as follows 
                                                                and do not include any additonal Hazard, Hazardous Situation, Harm/Potential Harm, Manufacturer Name which is not available in provided data:

                                                                ### ☣️ Hazard
                                                                IMPORTANT: Use the DuckDuckGo & Serp API search tool to:
                                                                - Identify the unique hazard using a single word or a concise phrase.
                                                                - Avoid full sentences and ensure it is a precise descriptor.

                                                                ### ☣️ Hazardous Situation
                                                                IMPORTANT: Use the DuckDuckGo & Serp API search tool to:
                                                                - Provide a detailed description of the situation, including context from complaint text, follow-up, and manufacturer narrative.
                                                                - Focus on capturing the specific conditions leading to the hazard.

                                                                ### ⚠️ Harm/Potential Harm
                                                                IMPORTANT: Use the DuckDuckGo & Serp API search tool to:
                                                                - Write a paragraph explaining the potential harm or adverse outcomes that could result from the identified hazard in the given context.
                                                                - Be thorough in describing the potential risks and their impacts.

                                                                ### 🏢🏭 Manufacturer Name
                                                                IMPORTANT: Use the DuckDuckGo & Serp API search tool to:
                                                                - Extract and list all unique manufacturer names mentioned in the data.
                                                                - Ensure the list includes only distinct names with no duplicates.

                                                                ### 🤔 Root Cause Analysis
                                                                IMPORTANT: Use the DuckDuckGo & Serp API search tool to:
                                                                - Identify the potential root cause of the hazard based on the provided context.
                                                                - Provide evidence or reasoning from the data to support your assessment.
                                                                - If the root cause cannot be determined directly from the data, state it as "Root cause unclear based on the provided information."

                                                                ### ☢️ Risk Mitigation Strategies
                                                                IMPORTANT: Use the DuckDuckGo & Serp API search tool to:
                                                                - Propose potential mitigation strategies that could reduce or eliminate the risk
                                                                - Include examples of preventive measures, design changes, or monitoring mechanisms relevant to the identified hazard.

                                                                ### 📃 Likelihood and Severity Assessment
                                                                IMPORTANT: Use the DuckDuckGo & Serp API search tool to:
                                                                - Assess the likelihood of the hazard occurring (e.g., rare, likely, frequent).
                                                                - Evaluate the severity of potential harm if the hazard occurs (e.g., minor, moderate, severe, catastrophic).
                                                                - Use qualitative terms and explain the rationale for your assessment.

                                                                ### 📑 Regulatory Implications
                                                                IMPORTANT: Use the DuckDuckGo & Serp API search tool to:
                                                                - Highlight any regulatory implications related to the identified hazard.
                                                                - Specify whether the issue might require notification to regulatory bodies, recalls, or modifications to labeling and instructions for use (IFU).

                                                                ### 📑 Contextual Insights
                                                                IMPORTANT: Use the DuckDuckGo & Serp API search tool to:
                                                                - If applicable, provide insights from similar cases or known hazards in medical devices.
                                                                - Include references to published studies, research, or industry examples to support your analysis.                                                                
                                                                - Find recent medical literature about similar cases 
                                                                - Provide a list of relevant  links of them too
                                                                - Research any relevant technological advances
                                                                - Include 2-3 key references to support your analysis

                                                                ### Provided Data:
                                                                {st.session_state.eventinfo_combined}
                                                                """
                            response_gemini = maude_google_agent.run(st.session_state.prompt_info_new)

                            

                            # Create a DataFrame
                            df = pd.DataFrame(table_data)
                            #df = df.reset_index(drop=True)
                            df.index.name = "ID"
                            #st.write(df.columns)
                            # Display the DataFrame as a table   

                            

                            data_col1, data_col2 = st.columns([3, 1])
                            with data_col1:
                                st.write("### Risk Analysis Table")   
                                st.markdown(table_css, unsafe_allow_html=True)                      
                                st.table(df)  
                                 
                            with data_col2:                                                 
                                st.markdown("#### 🔍 Gen AI Based Insight 📖")
                                st.markdown(f'<div class="scrollable-container">{response_gemini.content}</div>', unsafe_allow_html=True)
                               
                           

                            st.session_state.summary_prompt = f'''You are a highly skilled medical device safety expert with extensive knowledge in risk analysis and report summarization. Your task is to create a detailed summary based on the provided data. Structure the summary using the following format:

                                                                Conclusion:

                                                                Provide a high-level summary of the findings and their implications. Mention if there is any direct or indirect harm identified and the general resolution or recommendation.
                                                                Patient Outcome:

                                                                Describe the potential impact on patients due to the identified risks. Include specific scenarios, such as the need for emergency replacement surgeries, increased healthcare costs, or patient burden.
                                                                Manufacturer Outcome:

                                                                Highlight the implications for the manufacturer, such as liability claims, product recalls, and the potential impact on trust or reputation among healthcare providers (HCPs).
                                                                
                                                                ### Provided Data:
                                                                {st.session_state.eventinfo_combined}
                                                                '''


                            summary_response = maude_google_agent.run(st.session_state.summary_prompt)
                            st.markdown("### 📚 Summary 📖")
                            # st.markdown("---")
                            st.markdown(summary_response.content)

                            st.markdown(
                                """
                                <p style="font-weight: bold; color: red; font-size: 14px;">
                                    ⚠️ Note: This analysis is generated by AI and should be reviewed by a qualified risk analyst.
                                </p>
                                """,
                                unsafe_allow_html=True
                            )


                        except Exception as e:
                            st.error(f"Analysis error: {e}")
                        finally:
                             rain(
                                    emoji="🎊",
                                    font_size=25,
                                    falling_speed=10,
                                    animation_length=1,
                                )

                               
# Footer with HTML and CSS
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #004085; /* Change this value for a new background color */
        color: white;
        text-align: center;
        padding: 10px 0;
        font-family: Arial, sans-serif;
        font-size: 14px;
        border-top: 2px solid #007bff; /* Optional border color */
        z-index: 1000;
    }

    .footer a {
        color: #ffdd00; /* Link color */
        text-decoration: none;
        margin: 0 10px;
    }

    .footer a:hover {
        text-decoration: underline;
    }
    </style>

    <div class="footer">
        <p>
            Copyright &copy; 2025 <strong>Life Science - GEN AI CoE</strong> |
            <a href="https://tcs.com" target="_blank">Visit Our Website</a> |
            <a href="mailto:support@example.com">Contact Us</a>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
