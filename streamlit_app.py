import streamlit as st
from PyPDF2 import PdfReader
import io
from openai_rag import get_answer

def read_pdf(file_path):
    pdf_reader = PdfReader(file_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def main():
    st.title("Document RAG")
    
    # Create a file uploader widget for PDF files
    st.markdown("""
    ## How To Use This App
    1. **Upload a PDF file**: Use the file uploader to select a PDF file. Please note the file size limit is 200 MB.
    2. **Extract Text & creating vectorstore**: Please note that this is a demo application, and processing larger files may take some time to generate answers
    3. **Ask a Question**: Enter a question related to the content of the PDF in the text input box.
    4. **Get an Answer**: Click the 'Get Answer' button to receive an answer generated by OpenAI model based on the content of the PDF and your question.
    """)
    uploaded_file = st.file_uploader("Choose a PDF file...", type=['pdf'])
    
    if uploaded_file is not None:
        # Read the text from the PDF
        
        pdf_text = read_pdf(uploaded_file)
        
        # Display the PDF text
        #st.text_area("Extracted Text from PDF:", pdf_text, height=300)
        
        # Ask a question to the PDF input
        user_input = st.text_input("Ask a question", "")
        
        # Create a button to trigger the processing
        if st.button("Get Answer"):
            if user_input:  # Ensure there's some text entered
                # Call the function with the text from the PDF and the user's question
                answer = get_answer(question=user_input, pdf_content=pdf_text)
                
                # Display the result
                st.text_area("This is the answer:", answer, height=300)
            else:
                st.error("Please enter a question to get an answer.")
    else:
        st.warning("Please upload a PDF file to get started.")

if __name__ == "__main__":
    main()
