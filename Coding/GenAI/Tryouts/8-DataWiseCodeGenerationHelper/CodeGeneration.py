import os
import json
import re
import sys
import io
import contextlib
import warnings
from typing import Optional, List, Any, Tuple
from PIL import Image
import streamlit as st
import pandas as pd
import base64
from io import BytesIO
#from together import Together
from groq import Groq
from e2b_code_interpreter import Sandbox
from dotenv import load_dotenv

load_dotenv()

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

st.set_page_config(page_title='Python Code Generator', page_icon=':emoji:', layout='wide')

GROQ_API_KEY = os.getenv('GROQ_API_KEY')

pattern = re.compile(r"```python\n(.*?)\n```", re.DOTALL)

def code_interpret(e2b_code_interpreter: Sandbox, code: str) -> Optional[List[Any]]:
    with st.spinner('Executing code in E2B sandbox...'):
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                #st.sidebar.info('Coming inside the code interpret!!!')
                exec = e2b_code_interpreter.run_code(code)
                #st.sidebar.info(exec)

        # if stderr_capture.getvalue():
        #     st.sidebar.error("[Code Interpreter Warnings/Errors]", file=sys.stderr)
        #     st.sidebar.error(stderr_capture.getvalue(), file=sys.stderr)

        # if stdout_capture.getvalue():
        #     st.sidebar.error("[Code Interpreter Output]", file=sys.stdout)
        #     st.sidebar.error(stdout_capture.getvalue(), file=sys.stdout)

        # if exec.error:
        #     st.sidebar.error(f"[Code Interpreter ERROR] {exec.error}", file=sys.stderr)
        #     return None
        if stderr_capture.getvalue():
            st.sidebar.error("[Code Interpreter Warnings/Errors]")
            st.sidebar.error(stderr_capture.getvalue())  # No `file=sys.stderr` argument here.

        if stdout_capture.getvalue():
            st.sidebar.error("[Code Interpreter Output]")
            st.sidebar.error(stdout_capture.getvalue())  # No `file=sys.stdout` argument here.

        if exec.error:
            st.sidebar.error(f"[Code Interpreter ERROR] {exec.error}")
            return None
        st.sidebar.info(exec)
        return exec

def match_code_blocks(llm_response: str) -> str:
    match = pattern.search(llm_response)
    if match:
        code = match.group(1)
        return code
    return ""

def chat_with_llm(e2b_code_interpreter: Sandbox, user_message: str, dataset_path: str) -> Tuple[Optional[List[Any]], str]:
    # Update system prompt to include dataset path information
    system_prompt = f"""You're a Python data scientist and data visualization expert. You are given a dataset at path '{dataset_path}' and also the user's query.
You need to analyze the dataset and answer the user's query with a response and you run Python code to solve them.
IMPORTANT: Always use the dataset path variable '{dataset_path}' in your code when reading the CSV file."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    with st.spinner('Getting response from GROQ LLM model...'):
        #client = Together(api_key=st.session_state.together_api_key)
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"),)
        response = client.chat.completions.create(
            model=st.session_state.model_name,
            messages=messages,
        )

        response_message = response.choices[0].message
        python_code = match_code_blocks(response_message.content)
        
        #st.sidebar.info(python_code)
        
        if python_code:
            #st.sidebar.info('Coming Here!!')
            code_interpreter_results = code_interpret(e2b_code_interpreter, python_code)
            #st.sidebar.info(code_interpreter_results)
            return code_interpreter_results, response_message.content
        else:
            st.warning(f"Failed to match any Python code in model's response")
            return None, response_message.content

def upload_dataset(code_interpreter: Sandbox, uploaded_file) -> str:
    dataset_path = f"./{uploaded_file.name}"
    
    try:
        code_interpreter.files.write(dataset_path, uploaded_file)
        return dataset_path
    except Exception as error:
        st.error(f"Error during file upload: {error}")
        raise error


def main():
    """Main Streamlit application."""
    st.title("📊 AI Code Generation based on uploaded data")
    st.write("Upload your dataset and ask questions about it!")

    # Initialize session state variables
    # if 'together_api_key' not in st.session_state:
    #     st.session_state.together_api_key = ''
    if 'groq_api_key' not in st.session_state:
        st.session_state.groq_api_key = ''
    if 'e2b_api_key' not in st.session_state:
        st.session_state.e2b_api_key = ''
    if 'model_name' not in st.session_state:
        st.session_state.model_name = ''

    with st.sidebar:
        st.header("API Keys and Model Configuration")

        st.session_state.together_api_key = st.sidebar.text_input("Together AI API Key", type="password")
        st.sidebar.info("💡 Everyone gets a free $1 credit by Together AI - AI Acceleration Cloud platform")

        st.session_state.groq_api_key = st.sidebar.text_input("GROQ API Key", type="password")
        st.sidebar.info("💡 Everyone gets a free GROQ API Key")
        st.sidebar.markdown("[GROQ API Key](https://console.groq.com/keys)")
        
        st.session_state.e2b_api_key = st.sidebar.text_input("Enter E2B API Key", type="password")
        st.sidebar.markdown("[Get E2B API Key](https://e2b.dev/docs/legacy/getting-started/api-key)")
        
        # Add model selection dropdown
        model_options = {
            "Gemma 2 - 9b": "gemma2-9b-it",
            "Llama 3.3 Versatile - 70b": "llama-3.3-70b-versatile",
            "Llama 3.3 SpecDec - 70b": "llama-3.3-70b-specdec",
            "Mixtral 8 - 7b": "mixtral-8x7b-32768"
        }
        # model_options_together = {
        #     "Meta-Llama 3.1 405B": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        #     "DeepSeek V3": "deepseek-ai/DeepSeek-V3",
        #     "Qwen 2.5 7B": "Qwen/Qwen2.5-7B-Instruct-Turbo",
        #     "Meta-Llama 3.3 70B": "meta-llama/Llama-3.3-70B-Instruct-Turbo"
        # }
        st.session_state.model_name = st.selectbox(
            "Select Model",
            options=list(model_options.keys()),
            index=0  # Default to first option
        )
        st.session_state.model_name = model_options[st.session_state.model_name]

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Display dataset with toggle
        df = pd.read_csv(uploaded_file)
        st.write("Dataset:")
        show_full = st.checkbox("Show full dataset")
        if show_full:
            st.dataframe(df)
        else:
            st.write("Preview (first 5 rows):")
            st.dataframe(df.head())
        # Query input
        query = st.text_area("What would you like to know about your data?",
                            "Can you compare the average cost for two people between different categories?")
        
        if st.button("Analyze"):
            if not st.session_state.groq_api_key or not st.session_state.e2b_api_key:
                st.error("Please enter both API keys in the sidebar.")
            else:
                with Sandbox(api_key=st.session_state.e2b_api_key) as code_interpreter:
                    # Upload the dataset
                    dataset_path = upload_dataset(code_interpreter, uploaded_file)
                    
                    # Pass dataset_path to chat_with_llm
                    execution, llm_response = chat_with_llm(code_interpreter, query, dataset_path)
                    
                    # Display LLM's text response
                    st.write("AI Response:")
                    st.write(llm_response)
                    st.write(execution.results)
                    # Display results/visualizations

                    for result in execution.results: 
                        st.write('Coming!!!')
                        st.write(result)


                    # if code_results:
                    #     st.write('Coming!!!')
                    #     # Access the 'Results' attribute which contains the list of Result objects
                    #     #results = code_results.results  # This is where the list of Result objects is stored
                    #     st.write(code_results.Logs.stdout)
                    #     #st.dataframe(results[0].logs.stdout)  
                    #     # # Iterate over each result in the list
                    #     # for result in results:
                    #     #     if hasattr(result, 'png') and result.png:  # Check if PNG data is available
                    #     #         # Decode the base64-encoded PNG data
                    #     #         png_data = base64.b64decode(result.png)
                    #     #         with open('chart.png', 'wb') as f:
                    #     #             f.write(base64.b64decode(png_data.png))
                    #     #         # Convert PNG data to an image and display it
                    #     #         image = Image.open(BytesIO(png_data))
                    #     #         st.image(image, caption="Generated Visualization", use_container_width=False)
                    #     #     if hasattr(result, 'figure'):  # For Matplotlib figures
                    #     #         fig = result.figure  # Extract the Matplotlib figure
                    #     #         if fig:  # Ensure the figure exists
                    #     #             st.pyplot(fig)  # Display using st.pyplot
                    #     #     if hasattr(result, 'show'):  # For Plotly figures
                    #     #         st.plotly_chart(result)
                    #     #     if isinstance(result.logs.stdout, (pd.DataFrame, pd.Series)):
                    #     #         st.write('Coming Here!!')
                    #     #         st.dataframe(result.logs.stdout)                          
                    #     #     st.dataframe(result.logs.stdout)  

                    # # if code_results:
                    # #     for result in code_results:
                    # #         if hasattr(result, 'png') and result.png:  # Check if PNG data is available
                    # #             # Decode the base64-encoded PNG data
                    # #             png_data = base64.b64decode(result.png)                                
                    # #             # Convert PNG data to an image and display it
                    # #             image = Image.open(BytesIO(png_data))
                    # #             st.image(image, caption="Generated Visualization", use_container_width=False)
                    # #         elif hasattr(result, 'figure'):  # For matplotlib figures
                    # #             fig = result.figure  # Extract the matplotlib figure
                    # #             if fig:  # Ensure the figure exists
                    # #                 st.pyplot(fig)  # Display using st.pyplot
                    # #         elif hasattr(result, 'show'):  # For plotly figures
                    # #             st.plotly_chart(result)
                    # #         elif isinstance(result, (pd.DataFrame, pd.Series)):
                    # #             st.dataframe(result.logs.stdout)
                    # #         else:
                    # #             st.dataframe(result.logs.stdout)  

if __name__ == "__main__":
    main()