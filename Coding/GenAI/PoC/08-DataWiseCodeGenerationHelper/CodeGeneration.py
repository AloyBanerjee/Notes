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
#pip install streamlit-code-editor
from code_editor import code_editor
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
        - the python code runs in jupyter notebook as well as in a streamlit app.
        - every time you call `execute_python` tool, the python code is executed in a separate cell. it's okay to multiple calls to `execute_python`.
        - display visualizations using matplotlib or any other visualization library directly in the notebook. don't worry about saving the visualizations to a file.
        - you have access to the internet and can make api requests.
        - you also have access to the filesystem and can read/write files.
        - you can install any pip package (if it exists) if you need to but the usual packages for data analysis are already preinstalled.
        - you can run any python code you want, everything is running in a secure sandbox environment
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
            return python_code, response_message.content
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

def code_executor(e2b_code_interpreter, python_code, dataset_path, MODEL_NAME):
  
  MODEL_NAME = MODEL_NAME#"llama3-70b-8192"
  client = Groq(api_key=st.session_state.groq_api_key) 
  SYSTEM_PROMPT = """you are a python data scientist. you are given tasks to complete and you run python code to solve them.
        - the python code runs in jupyter notebook and in streamlit app.
        - every time you call `execute_python` tool, the python code is executed in a separate cell. it's okay to multiple calls to `execute_python`.
        - display visualizations using matplotlib or any other visualization library directly in the notebook. don't worry about saving the visualizations to a file.
        - you have access to the internet and can make api requests.
        - you also have access to the filesystem and can read/write files.
        - you can install any pip package (if it exists) if you need to but the usual packages for data analysis are already preinstalled.
        - you can run any python code you want, everything is running in a secure sandbox environment"""
  messages = [
      {"role": "system", "content": SYSTEM_PROMPT}
      
  ] 
  tools = [
        {
            "type": "function",
            "function": {
                "name": "execute_python",
                "description": "Execute Python code generated by the 'explanation_problem' tool in a Streamlit app cell and return any result, stdout, stderr, display_data, and error.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The Python code to execute in a single cell."
                        }
                    },
                    "required": ["code"]
                }
            }
        }
    ]

  def code_interpret(e2b_code_interpreter, code):
    st.write("Running code interpreter...")
    exec = e2b_code_interpreter.run_code(code,
    on_stderr=lambda stderr: print("[Code Interpreter]", stderr),
    on_stdout=lambda stdout: print("[Code Interpreter]", stdout))
    if exec.error:
      print("[Code Interpreter ERROR]", exec.error)
    else:
      return exec.results
  
  
  response = client.chat.completions.create(
      model=MODEL_NAME,
      messages=messages,
      tools=tools,
      tool_choice="auto",
      max_tokens=4096,
  )
  response_message = response.choices[0].message
  tool_calls = response_message.tool_calls
  
  if tool_calls:
    for tool_call in tool_calls:
      st.info(tool_call.function.name)
      function_name = tool_call.function.name
      function_args = json.loads(tool_call.function.arguments)
      if function_name == "execute_python":
        code = function_args["code"]
        code_interpreter_results = code_interpret(e2b_code_interpreter, code)
        return code_interpreter_results
      else:
        raise Exception(f"Unknown tool {function_name}")
  else:
    print(f"(No tool call in model's response) {response_message}")
    return None



def main():
    """Main Streamlit application."""
    st.title("📊 Python Code Generation based on uploaded data")
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
            "Llama 3 8192 - 70b": "llama3-70b-8192",
            "Mixtral 8 - 7b": "mixtral-8x7b-32768"
        }
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
        query = st.text_area("What would you like to know about your data?")
        
        if st.button("Analyze"):
            if not st.session_state.groq_api_key or not st.session_state.e2b_api_key:
                st.error("Please enter both API keys in the sidebar.")
            else:
                with Sandbox(api_key=st.session_state.e2b_api_key) as code_interpreter:
                    # Upload the dataset
                    dataset_path = upload_dataset(code_interpreter, uploaded_file)
                    python_code, llm_response = chat_with_llm(code_interpreter, query, dataset_path)                   
                    
                st.header("AI Response:")
                st.write(llm_response)


                # with Sandbox(api_key=st.session_state.e2b_api_key) as code_interpreter:
                #     code_results = code_executor(code_interpreter, python_code, dataset_path, st.session_state.model_name)
                #     st.write('Execution completed')
                #     st.write(code_results)
                #     if code_results:
                #         first_result = code_results[0]
                #         st.sidebar.info(first_result)
                #         #If there is an image
                #         if hasattr(first_result, 'png') and first_result.png:  # Check if PNG data is available
                #             # Decode the base64-encoded PNG data
                #             png_data = base64.b64decode(first_result.png)
                #             st.image(png_data, caption="Generated Visualization", use_container_width=False)  
                #         #If there is a figure
                #         if hasattr(first_result, 'figure'):  # For Matplotlib figures
                #             fig = first_result.figure  # Extract the Matplotlib figure
                #             if fig:  # Ensure the figure exists
                #                 st.pyplot(fig)  # Display using st.pyplot
                #         if hasattr(first_result, 'show'):  # For Plotly figures
                #             st.plotly_chart(first_result)
                #         #if there is a dataframe or series
                #         if isinstance(first_result, (pd.DataFrame, pd.Series)):
                #             st.dataframe(first_result)  
                        
                #     else:
                #         st.info("No code results")
                #         exit(0)
                     

if __name__ == "__main__":
    main()