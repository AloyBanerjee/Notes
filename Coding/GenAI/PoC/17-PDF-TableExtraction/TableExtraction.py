import os
from PIL import Image
from phi.agent import Agent
from phi.model.google import Gemini
import streamlit as st
from phi.tools.duckduckgo import DuckDuckGo
import fitz
import camelot

if "GOOGLE_API_KEY" not in st.session_state:
    st.session_state.GOOGLE_API_KEY = None

st.set_page_config(page_title='PDF Page Wise Information Extraction', page_icon=':emoji:', layout='wide')

with st.sidebar:
    st.title("ℹ️ Configuration")
    
    if not st.session_state.GOOGLE_API_KEY:
        api_key = st.text_input(
            "Enter your Google API Key:",
            type="password"
        )
        st.caption(
            "Get your API key from [Google AI Studio]"
            "(https://aistudio.google.com/apikey) 🔑"
        )
        if api_key:
            st.session_state.GOOGLE_API_KEY = api_key
            st.success("API Key saved!")
            st.rerun()
    else:
        st.success("API Key is configured")
        if st.button("🔄 Reset API Key"):
            st.session_state.GOOGLE_API_KEY = None
            st.rerun()
    
    st.info(
        "This tool provides AI-powered information extraction from uploaded PDF data using "
        "advanced AI & Gen AI technology."
    )
    st.warning(
        "⚠DISCLAIMER: This tool is only for informational purposes only. "
        
    )
pdf_agent = Agent(
    model=Gemini(
        api_key=st.session_state.GOOGLE_API_KEY,
        id="gemini-2.0-flash-exp"
    ),
    tools=[DuckDuckGo()],
    markdown=True
) if st.session_state.GOOGLE_API_KEY else None

if not pdf_agent:
    st.warning("Please configure your API key in the sidebar to continue")

# Medical Analysis Query
prompt_question = """
Prompt for Table Data Extraction from Images
You are a highly skilled data extraction and OCR (Optical Character Recognition) expert specializing in analyzing and extracting tabular information from images. Your task is to analyze the provided image containing tabular data and produce a structured and detailed response. Follow this format:

### 1. Table Identification
- Identify and describe the type of table (e.g., financial, inventory, scientific data, survey results).
- Specify the source context if visible (e.g., invoice, academic report, business dashboard, etc.).
- Comment on the image quality, including resolution, text clarity, alignment, or any skewness issues.
### 2. Table Structure
- Identify and describe the structure of the table:
- Number of rows and columns.
- Presence of headers or labels.
- Types of data in the table (e.g., numeric, textual, dates, mixed formats).
- Note any merged or spanned cells.
- Suggest any transformations or assumptions needed for clean extraction.
### 3. Data Extraction
- Extract and present the tabular data:
- Provide the text or numeric values of each cell.
- Mention any units or symbols in the data (e.g., currency, percentages, metrics).
- Handle abbreviations, symbols, and unclear entries:
- Provide the full forms for abbreviations.
- Flag unclear entries and propose possible interpretations.
### 4. Reconstructed Table
- Format the extracted data into a clean Markdown table:
sql
Copy
Edit
| Column Header 1 | Column Header 2 | Column Header 3 |
|-----------------|-----------------|-----------------|
| Value 1         | Value 2         | Value 3         |
| Value 4         | Value 5         | Value 6         |
Ensure alignment and consistency in presentation.
5. Handling Imperfections
- Highlight any issues with the image:
- Missing or illegible data.
- Misaligned table sections.
- Inconsistencies in layout or formatting.
- Provide recommendations for addressing these issues (e.g., assumptions for missing data, manual verification for unclear entries).
6. Insights and Contextual Analysis
- Analyze the table content:
- Identify any patterns, trends, or anomalies in the data.
- Provide meaningful insights or interpretations based on the table’s context (e.g., financial trends, inventory shortages, experimental results).
- Offer recommendations or next steps based on the data analysis.
7. Research and References
Use online research tools (e.g., DuckDuckGo or Google) to:

- Find information about similar types of tables or datasets.
- Provide references to standards or methodologies for interpreting the data.
- Include 2-3 links to reliable resources that support your analysis.
8. Simplified Explanation (Optional for Non-Experts)
- If needed, explain the data in plain, jargon-free language.
- Use analogies or visual examples to make the explanation more accessible to non-technical audiences.
- Address common questions or concerns about the table data.
IMPORTANT:

Be precise and systematic in your response.
Use Markdown formatting for clarity.
Ensure all extracted data is clean, consistent, and ready for further analysis. If any information is missing or unclear, explain the issue and make reasonable assumptions.
"""

#def extract_table(pdf_path):

    # tables = camelot.read_pdf(pdf_path, pages="all", flavor="lattice")#stream

    # if len(tables) == 0:
    #     st.write("No tables found in the PDF")
    # else:
    #     for i, table in enumerate(tables):
    #         st.write(f"### Table {i + 1}")
    #         st.dataframe(table.df)  



st.title("🏥 PDF Page Wise Information Extraction Agent")
st.write("Upload a PDF image to extract tabular data")

# Create containers for better organization
upload_container = st.container()
image_container = st.container()
analysis_container = st.container()

with upload_container:
    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        help="Supported formats: PDF"
    )

if uploaded_file is not None:
    with image_container:
        # Extract images from PDF
        pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        images = []
        for page_number in range(len(pdf_document)):
            page = pdf_document[page_number]
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)

        pdf_document.close()

        with open("temp_uploaded_file.pdf", "wb") as temp_file:
            temp_file.write(uploaded_file.read())  # Write the file to disk

        if images:
            st.sidebar.markdown("### 🖼️ PDF Preview")
            st.sidebar.markdown(
                "The uploaded PDF has been converted into images for analysis. "
                "You can view the extracted images below:"
            )
            for page_number, image in enumerate(images):
                st.sidebar.image(
                    image.resize((250, int(250 * image.height / image.width))),
                    caption=f"Page {page_number + 1}",
                    use_container_width=False
                )
        analyze_button = st.button(
            "🔍 Analyze PDF",
            type="primary",
            use_container_width=True
        )    
    with analysis_container:
        if images:
            for page_number, image in enumerate(images):            
                if analyze_button:
                    image_path = f"temp_page_{page_number + 1}.png"
                    image.save(image_path)

                    with st.spinner(f"🔄 Analyzing Page {page_number + 1}... Please wait."):
                        try:
                            st.write(f"##### 📄 Page {page_number + 1}")
                            #extract_table('temp_uploaded_file.pdf')
                            response = pdf_agent.run(prompt_question, images=[image_path])
                            st.markdown("### 📋 Analysis Results")
                            st.markdown("---")
                            st.markdown(response.content)
                            st.markdown("---")
                            st.caption(
                                "Note: This analysis is generated by AI and should be reviewed by "
                                "a qualified healthcare professional."
                            )
                        except Exception as e:
                            st.error(f"Analysis error on page {page_number + 1}: {e}")
                        finally:
                            if os.path.exists(image_path):
                                os.remove(image_path)

            st.caption(
                "Note: This analysis is generated by AI and should be reviewed by "
                "a qualified professional."
            )
else:
    st.info("👆 Please upload a PDF to begin analysis")
