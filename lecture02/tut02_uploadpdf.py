from google import genai
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import get_key

# This application allows users to upload a PDF file and ask questions about its content using the Gemini API.
# The AI will answer questions based ONLY on the content of the provided PDF document.
# If information is not in the document, the AI will say so clearly.

def initialize_client():
    """Initialize and return the Gemini client."""
    api_key = get_key()
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable not found.\n")
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    print(">>> Successfully initialized the Google Gemini client.")
    return client

def get_pdf_path():
    """Get and validate the PDF path from user input."""
    pdf_path = input(">>> Please enter the path to the PDF file: ").strip()
    if not pdf_path:
        print("Error: PDF path cannot be empty.", file=sys.stderr)
        sys.exit(1)
    return pdf_path

def upload_pdf(client, pdf_path):
    """Upload PDF file using the File API."""
    print(">>> Uploading PDF file...")
    pdf_file = client.files.upload(file=pdf_path)
    print(">>> PDF uploaded successfully!")
    return pdf_file

def get_initial_response(client, pdf_file):
    """Get initial response from Gemini with the PDF."""
    initial_prompt = """You are an AI assistant. Your task is to answer questions based ONLY on the content of the provided PDF document.
    Do not use any external knowledge. If information is not in the document, say so clearly.
    Please confirm you understand and are ready to answer questions about this document."""

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=[pdf_file, initial_prompt]
    )
    return response

def ask_question(client, pdf_file, question):
    """Ask a question about the PDF and get Gemini's response."""
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=[pdf_file, question]
    )
    return response

def run_qa_loop(client, pdf_file):
    """Run the question-answering loop."""
    print(">>> Gemini is ready! Start asking questions.")
    print("--- (Type 'quit' or 'exit' to stop) ---")

    while True:
        try:
            question = input("You: ").strip()
            if question.lower() in ['quit', 'exit']:
                break
                
            response = ask_question(client, pdf_file, question)
            print(f"Gemini: {response.text}")
            
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again or type 'quit' to exit.")

def main():
    try:
        # Initialize client
        client = initialize_client()
        
        # Get and upload PDF
        pdf_path = get_pdf_path()
        pdf_file = upload_pdf(client, pdf_path)
        
        # Get initial response
        get_initial_response(client, pdf_file)
        
        # Run Q&A loop
        run_qa_loop(client, pdf_file)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
