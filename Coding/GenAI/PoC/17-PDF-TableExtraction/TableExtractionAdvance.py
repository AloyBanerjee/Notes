import os
from PIL import Image
from phi.agent import Agent
from phi.model.google import Gemini
import streamlit as st
from phi.tools.duckduckgo import DuckDuckGo
import fitz
import camelot
import io
from PIL import Image
import tempfile

#https://sebastian-petrus.medium.com/build-a-local-ollama-ocr-application-using-llama-3-2-vision-bfc3014e3ad6

if "GOOGLE_API_KEY" not in st.session_state:
    st.session_state.GOOGLE_API_KEY = None

st.set_page_config(page_title='PDF Page Wise Information Extraction', page_icon='💡', layout='wide')

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

overall_prompt = """
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

# Define prompts for each tab
prompt_table = """
### Table Extraction
Prompt for Table Data Extraction from Images
You are a highly skilled data extraction and OCR (Optical Character Recognition) expert specializing in analyzing and extracting tabular information from images. Your task is to analyze the provided image containing tabular data and produce a structured and detailed response. Follow this format:
1. **Table Identification**:
   - Identify and describe the type of table (e.g., financial, inventory, scientific data, survey results).
   - Specify the source context if visible (e.g., invoice, academic report, business dashboard, etc.).
   - Comment on the image quality, including resolution, text clarity, alignment, or any skewness issues.

2. **Table Structure**:
   - Identify and describe the structure of the table:
     - Number of rows and columns.
     - Presence of headers or labels.
     - Types of data in the table (e.g., numeric, textual, dates, mixed formats).
     - Note any merged or spanned cells.
     - Suggest any transformations or assumptions needed for clean extraction.

3. **Data Extraction**:
   - Extract and present the tabular data:
     - Provide the text or numeric values of each cell.
     - Mention any units or symbols in the data (e.g., currency, percentages, metrics).
     - Handle abbreviations, symbols, and unclear entries:
       - Provide the full forms for abbreviations.
       - Flag unclear entries and propose possible interpretations.
"""

prompt_image = """
### Image Extraction
Prompt for Image Data Extraction from Images
You are a highly skilled data extraction and OCR (Optical Character Recognition) expert specializing in analyzing and extracting images present in the page wise screenshot of the uploaded pdf. Your task is to analyze the provided image containing multiple diagram or other images and produce a structured and detailed response and display the same. Follow this format:
- Extract and display all images from the provided PDF, page by page.
- For each image:
  - Comment on the image quality, including resolution, text clarity, alignment, or any skewness issues.
  - Identify if the image contains any tabular data, graphs, or charts.
  - Provide any insights or relevant observations based on the image content.
"""

prompt_image_new = """ 
### Image Extraction
Prompt for Image Data Extraction from Images
You are a highly skilled data extraction and OCR (Optical Character Recognition) expert specializing in analyzing and extracting images present in the page-wise screenshot of the uploaded PDF. Your task is to analyze the provided image containing multiple diagrams or other images and produce a structured response that displays only the extracted images. Follow this format:

- Extract and display all images from the provided PDF, page by page using Markdown.
  
For each image, include:
  - ![Image Description](image_url)  <!-- Replace 'image_url' with the actual URL or path of the extracted image -->
    - Comment on the image quality, including resolution, text clarity, alignment, or any skewness issues.
"""

prompt_graph = """
### Graph Extraction
Prompt for graph and plot Data Extraction from Images
You are a highly skilled data extraction and OCR (Optical Character Recognition) expert specializing in analyzing and extracting graphs and plots present in the page wise screenshot of the uploaded pdf. Your task is to analyze the provided image containing multiple graphs and plots and produce a display the same. Follow this format:
- Identify and extract graphs from the PDF, segregated page-wise.
- For each graph:
  - Describe the graph type (e.g., bar chart, line graph, scatter plot).
  - Extract and summarize key data points, trends, or patterns from the graph.
  - Mention any potential issues in the graph (e.g., missing labels, unclear axes).
  - Provide a clean and reconstructed version of the graph, if necessary.
"""

prompt_summary = """
### Page wise Summary
Prompt for page wise summary creation
You are a highly skilled data extraction and OCR (Optical Character Recognition) expert specializing in creating summary from the extracted content, graphs and plots present in the page wise screenshot of the uploaded pdf. Your task is to analyze the extracted information includign text, graph , plot and table data and generate a detailed response and display the same. Follow this format:
- Generate a detailed summary of the given PDF given as a page wise image:
  - Highlight key topics or sections covered in the document.
  - Provide an overview of the types of data and content (e.g., tables, graphs, images, text).
  - Mention any high-level insights or observations about the document structure and content.
"""

prompt_insights = """
### Insights and Contextual Analysis
Prompt for Insights and Contextual Analysis creation
You are a highly skilled data extraction and OCR (Optical Character Recognition) expert specializing in insightful contextiual analyzing and extracting content, graphs and plots present in the page wise screenshot of the uploaded pdf. Your task is to analyze the extracted information includign text, graph , plot and table data and generate a contextual insight. Follow this format:
- Perform an in-depth analysis of the PDF given as a page wise image content:
  - Identify any patterns, trends, or anomalies across the document.
  - Highlight important findings or conclusions based on the data presented (e.g., financial trends, inventory shortages, experimental results).
  - Offer actionable recommendations or next steps based on the analysis.
"""

prompt_references = """
### Research and References
Prompt for research information and referece link creation
You are a highly skilled data extraction and OCR (Optical Character Recognition) expert specializing in doing research and refrence collection in line with extracted content, graphs and plots present in the page wise screenshot of the uploaded pdf. Your task is to analyze the extracted information includign text, graph , plot and table data and generate a research reference and furtehr study valid link. Follow this format:
Use online research tools (e.g., DuckDuckGo or Google) to:
- Conduct research based on the content of the PDF given as a page wise image:
  - Find information about similar types of tables, graphs, or datasets.
  - Provide references to standards or methodologies for interpreting the data.
  - Include 2-3 links to reliable resources that support your analysis.
"""

prompt_literature_review = """
### Literature Review
Prompt for Literature Review creation
You are a highly skilled data extraction and OCR (Optical Character Recognition) expert specializing in creating literature review document from the collection in line with extracted content, graphs and plots present in the page wise screenshot of the uploaded pdf. Your task is to analyze the extracted information includign text, graph , plot and table data and generate a research paper like literature review content. Follow this format:
Use online research tools (e.g., DuckDuckGo or Google) to:
- Provide a detailed literature review related to the PDF given as a page wise image:
  - Summarize related studies, research, or findings in the same domain.
  - Highlight any similarities, differences, or gaps in the information provided in the images compared to existing literature.
"""

prompt_simplified = """
### Simplified Explanation
Prompt for simplified explanation creation
You are a highly skilled data extraction and OCR (Optical Character Recognition) expert specializing in creating a simplified information from the collection in line with extracted content, graphs and plots present in the page wise screenshot of the uploaded pdf. Your task is to analyze the extracted information includign text, graph , plot and table data and generate a simplifed content for a non-technical audiences. Follow this format:
- Create a simplified explanation of the image content for non-technical audiences:
  - Avoid technical jargon and use plain, accessible language.
  - Use analogies or visual examples to make the explanation more accessible to non-technical audiences.
  - Address common questions or misconceptions about the given content.
  - If needed, explain the data in plain, jargon-free language.
"""

def extract_images_from_pdf(pdf_path, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the PDF file
    pdf_document = fitz.open(pdf_path)

    # Loop through each page
    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)
        images = page.get_images(full=True)  # Get all images from the page

        # Extract each image
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image = Image.open(io.BytesIO(image_bytes))

            # Save image as pageX_Y.ext (X = page number, Y = image index)
            image_filename = f"page{page_number + 1}_{img_index + 1}.{image_ext}"
            image_path = os.path.join(output_folder, image_filename)
            image.save(image_path)

            print(f"Saved: {image_path}")

    print("Image extraction completed.")
st.title("💡 PDF Page Wise Information Extraction Agent")
st.write("Upload a PDF image to extract tabular data")

# Create containers for better organization
upload_container = st.container()
file_container = st.container()
analysis_container = st.container()

with upload_container:
    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        help="Supported formats: PDF"
    )
tabs = st.tabs(["📋 Tables", "🖼️ Images & Graphs 📊", "📝 Summary", "🔍 Insights", "📚 References", "📖 Literature Review", "💡 Simplified Explanation"])

if uploaded_file is not None:
    with file_container:
        # Extract images from PDF
        pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        images = []
        extracted_images = []
        for page_number in range(len(pdf_document)):
            page = pdf_document[page_number]
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)   
        pdf_document.close()

        
        temp_dir = tempfile.mkdtemp()
        path = os.path.join(temp_dir, uploaded_file.name)
        with open(path, "wb") as f:
            f.write(uploaded_file.getvalue())
        extract_images_from_pdf(path, f"{uploaded_file.name}\extracted_image")
                
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
            image_files = []
            for page_number, image in enumerate(images):            
                if analyze_button:
                    image_path = f"temp_page_{page_number + 1}.png"
                    image.save(image_path)

                    with st.spinner(f"🔄 Analyzing Page {page_number + 1}... Please wait."):
                        with tabs[0]:  # Tables Tab
                            try:
                                st.write(f"##### 📄 Page {page_number + 1}")
                                #extract_table('temp_uploaded_file.pdf')
                                table_response = pdf_agent.run(prompt_table, images=[image_path])
                                st.markdown("### 📋 Extracted Tables")
                                st.markdown("---")
                                st.markdown(table_response.content)
                                st.markdown("---")
                                st.caption(
                                    "Note: This analysis is generated by AI and should be reviewed by "
                                    "a qualified healthcare professional."
                                )
                            except Exception as e:
                                st.error(f"Analysis error on page {page_number + 1}: {e}")
                        with tabs[1]:  # Images Tab                            
                            try:
                                extracted_images_folder = f"{uploaded_file.name}\extracted_image"                                  
                                
                                if not image_files:
                                    
                                    image_files = sorted(
                                        [f for f in os.listdir(extracted_images_folder) if f.endswith((".png", ".jpg", ".jpeg"))],
                                        key=lambda x: int(x.split('_')[0][4:])  # Sort by page index (e.g., "page3_1.png")
                                    )
                                    if image_files:
                                        st.markdown("### 🖼️ Extracted Images & Graphs")
                                    for image_file in image_files:
                                        # Extract page number from the file name (e.g., "page3_1.png")
                                        page_number = image_file.split('_')[0][4:]  # Extract "3" from "page3_1.png"

                                        # Full path to the image
                                        image_path = os.path.join(extracted_images_folder, image_file)
                                        image = Image.open(image_path)
                                        width, height = image.size
                                        aspect_ratio = width / height
                                        new_width = 400
                                        new_height = int(new_width / aspect_ratio)
                                        resized_image = image.resize((new_width, new_height))

                                        # Display the page heading
                                        st.write(f"##### 📄 Page {page_number}")
                                        
                                        # Display the image
                                        st.image(resized_image, caption=f"Page {page_number} Extracted Image", use_container_width=False)
                                            
                                        st.markdown("---")                                  
                                    st.caption(
                                        "Note: This analysis is generated by AI and should be reviewed by "
                                        "a qualified healthcare professional."
                                    )
                            except Exception as e:
                                st.error(f"Analysis error on page {page_number}: {e}")                        
                        with tabs[2]:  # Summary Tab
                            try:
                                st.write(f"##### 📄 Page {page_number + 1}")
                                #extract_table('temp_uploaded_file.pdf')
                                summary_response = pdf_agent.run(prompt_summary, images=[image_path])
                                st.markdown("### 🔍 Summary")
                                st.markdown("---")
                                st.markdown(summary_response.content)
                                st.markdown("---")
                                st.caption(
                                    "Note: This analysis is generated by AI and should be reviewed by "
                                    "a qualified healthcare professional."
                                )
                            except Exception as e:
                                st.error(f"Analysis error on page {page_number + 1}: {e}")
                        with tabs[3]:  # Insights Tab
                            try:
                                st.write(f"##### 📄 Page {page_number + 1}")
                                #extract_table('temp_uploaded_file.pdf')
                                insight_response = pdf_agent.run(prompt_insights, images=[image_path])
                                st.markdown("### 🔍 Insights and Contextual Analysis")
                                st.markdown("---")
                                st.markdown(insight_response.content)
                                st.markdown("---")
                                st.caption(
                                    "Note: This analysis is generated by AI and should be reviewed by "
                                    "a qualified healthcare professional."
                                )
                            except Exception as e:
                                st.error(f"Analysis error on page {page_number + 1}: {e}")
                        with tabs[4]:  # References Tab
                            try:
                                st.write(f"##### 📄 Page {page_number + 1}")
                                #extract_table('temp_uploaded_file.pdf')
                                reference_response = pdf_agent.run(prompt_references, images=[image_path])
                                st.markdown("### 📚 References")
                                st.markdown("---")
                                st.markdown(reference_response.content)
                                st.markdown("---")
                                st.caption(
                                    "Note: This analysis is generated by AI and should be reviewed by "
                                    "a qualified healthcare professional."
                                )
                            except Exception as e:
                                st.error(f"Analysis error on page {page_number + 1}: {e}")
                        with tabs[5]:  # Literature Review Tab
                            try:
                                st.write(f"##### 📄 Page {page_number + 1}")
                                #extract_table('temp_uploaded_file.pdf')
                                literature_review_response = pdf_agent.run(prompt_literature_review, images=[image_path])
                                st.markdown("### 📖 Literature Review")
                                st.markdown("---")
                                st.markdown(literature_review_response.content)
                                st.markdown("---")
                                st.caption(
                                    "Note: This analysis is generated by AI and should be reviewed by "
                                    "a qualified healthcare professional."
                                )
                            except Exception as e:
                                st.error(f"Analysis error on page {page_number + 1}: {e}")                                               
                        with tabs[6]:  # Simplified Explanation Tab
                            try:
                                st.write(f"##### 📄 Page {page_number + 1}")
                                #extract_table('temp_uploaded_file.pdf')
                                simplified_response = pdf_agent.run(prompt_simplified, images=[image_path])
                                st.markdown("### 💡 Simplified Explanation")
                                st.markdown("---")
                                st.markdown(simplified_response.content)
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
        
else:
    st.info("👆 Please upload a PDF to begin analysis")
