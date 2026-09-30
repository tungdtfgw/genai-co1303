from google import genai
from PyPDF2 import PdfReader, errors as pypdf_errors
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import get_key

from google.genai.types import GenerateContentConfig

# This application extracts text from a PDF file, send the content to AI and allows users to ask questions about its content using the Gemini API.

def extract_text(pdf_path):
    print(f">>> Reading PDF file: '{pdf_path}'...")
    extracted_text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            num_pages = len(reader.pages)

            if num_pages == 0:
                print(f"Warning: PDF file '{pdf_path}' contains no pages.")
                return ""

            print(f">>> Total pages found: {num_pages}")
            for i, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        extracted_text += page_text + "\n"
                    print(f"\r>>> Processed page {i + 1}/{num_pages}", end="")
                except Exception as page_error:
                    print(f"\nError processing page {i + 1}: {page_error}", file=sys.stderr)
                    continue
            print("\n>>> PDF text extraction complete (using PyPDF2).")

        if not extracted_text.strip():
            print(f"Warning: No text could be extracted from '{pdf_path}' using PyPDF2. The file might be image-based or use an unsupported format.")
        return extracted_text

    except FileNotFoundError:
        print(f"Error: PDF file not found at path '{pdf_path}'", file=sys.stderr)
        return None
    except pypdf_errors.PdfReadError as pdf_error:
        print(f"Error: PyPDF2 failed to read PDF '{pdf_path}'. It might be corrupted or encrypted: {pdf_error}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error reading or processing PDF file '{pdf_path}' with PyPDF2: {e}", file=sys.stderr)
        return None

def main():
    # Initialize Gemini client
    api_key = get_key()
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable not found.\n")
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    print(">>> Successfully initialized the Google Gemini client.")

    # Get PDF path and extract text
    pdf_path = input(">>> Please enter the path to the PDF file: ").strip()
    if not pdf_path:
        print("Error: PDF path cannot be empty.", file=sys.stderr)
        sys.exit(1)

    pdf_text = extract_text(pdf_path)
    if pdf_text is None:
        sys.exit(1)

    # Create a system prompt
    system_prompt = f"""You are an AI assistant. Your task is to answer questions based ONLY on the following document content. 
    Do not use any external knowledge. If information is not in the document, say so clearly.

    Document content:
    {pdf_text}

    Please confirm you understand and are ready to answer questions about this document."""

    # Create chat session with a system prompt
    config = GenerateContentConfig(system_instruction=system_prompt)
    chat = client.chats.create(model='gemini-3-flash-preview', config=config)
    
    
    print(">>> Gemini is ready! Start asking questions.")
    print("--- (Type 'quit' or 'exit' to stop) ---")

    # Main Q&A loop
    while True:
        try:
            question = input("You: ").strip()
            if question.lower() in ['quit', 'exit']:
                break
                
            response = chat.send_message(message=question)
            print(f"Gemini: {response.text}")
            
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again or type 'quit' to exit.")

if __name__ == "__main__":
    main()