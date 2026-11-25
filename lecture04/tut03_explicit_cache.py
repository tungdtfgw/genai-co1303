from google import genai
from google.genai import types
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import get_key

# This application uploads a PDF file, creates an explicit cache using the Gemini API,
# and allows users to ask questions about the PDF content using the cached data.
# Note: Gemini now don't support explicit caching in Gemini, must be paid tier.
# File to upload should large enough to see token savings.

def display_usage_info(response, title="Token Usage"):
    """Display token usage information from the response"""
    if hasattr(response, 'usage_metadata') and response.usage_metadata:
        print(f"\n--- {title} ---")
        metadata = response.usage_metadata
        
        print(f"Prompt tokens: {metadata.prompt_token_count}")
        cached_tokens = 0
        if hasattr(metadata, 'cached_content_token_count') and metadata.cached_content_token_count:
            cached_tokens = metadata.cached_content_token_count
            print(f"Cached tokens: {cached_tokens}")
        else:
            print(f"Cached tokens: 0")
        print(f"Output tokens: {metadata.candidates_token_count}")
        print("------------------------------")


def upload_pdf_to_cache(client, pdf_path):
    """Upload PDF file and create explicit cache"""
    try:
        print(f"Uploading PDF: {pdf_path}")
        
        # Upload the PDF file (correct syntax from tut02)
        pdf_file = client.files.upload(file=pdf_path)
        print(f"File uploaded successfully: {pdf_file.name}")
        
        # Wait for file processing if needed
        while pdf_file.state.name == 'PROCESSING':
            print('Waiting for PDF to be processed.')
            time.sleep(2)
            pdf_file = client.files.get(name=pdf_file.name)
        
        print(f'PDF processing complete: {pdf_file.uri}')
        
        
        
        # Create explicit cache with Gemini 2.0 Flash 
        cache = client.caches.create(
            model='gemini-2.0-flash',
            config=types.CreateCachedContentConfig(
                display_name='PDF document cache',
                system_instruction=(
                    'You are an expert document analyzer. Answer questions '
                    'based only on the content of the provided PDF document. '
                ),
                contents=[pdf_file],
                ttl="3600s",  # Cache for 1 hour
            )
        )
        
        print(f"Cache created successfully: {cache.name}")
        print(f"Cache TTL: {cache.ttl}")
        return cache, pdf_file
        
    except Exception as e:
        print(f"Error uploading PDF or creating cache: {e}")
        return None, None


def ask_question_with_cache(client, cache, question):
    """Ask a question using the cached content"""
    try:
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=question,
            config=types.GenerateContentConfig(
                cached_content=cache.name
            )
        )
        
        return response
        
    except Exception as e:
        print(f"Error asking question: {e}")
        return None


def cleanup_cache(client, cache, pdf_file):
    """Clean up cache and uploaded file"""
    try:
        if cache:
            client.caches.delete(name=cache.name)
            print(f"Cache deleted: {cache.name}")
        
        if pdf_file:
            client.files.delete(name=pdf_file.name)
            print(f"File deleted: {pdf_file.name}")
            
    except Exception as e:
        print(f"Error during cleanup: {e}")


def test_explicit_cache():
    """Test explicit caching with PDF file"""
    print("=== GEMINI EXPLICIT CACHE TEST ===")
    print("Cache a PDF file and ask questions about it\n")
    
    client = genai.Client(api_key=get_key())
    
    # Get PDF filename from user
    pdf_filename = input("Enter PDF filename (in current directory): ").strip()
    
    # Check if file exists
    if not os.path.exists(pdf_filename):
        print(f"Error: File '{pdf_filename}' not found in current directory")
        return
    
    # Upload PDF and create cache
    cache, pdf_file = upload_pdf_to_cache(client, pdf_filename)
    
    if not cache:
        print("Failed to create cache. Exiting...")
        return
    
    print(f"\nPDF cached successfully! You can now ask questions about: {pdf_filename}")
    print("Type 'quit' to exit\n")
    
    question_count = 0
    
    try:
        while True:
            # Get question from user
            question = input("Your question: ").strip()
            
            if question.lower() == 'quit':
                break
            
            if not question:
                print("Please enter a question or 'quit' to exit")
                continue
            
            question_count += 1
            print(f"\nQuestion {question_count}: {question}")
            
            # Ask question using cached content
            response = ask_question_with_cache(client, cache, question)
            
            if response:
                print(f"Answer: {response.text}")
                display_usage_info(response, f"Question {question_count}")
            else:
                print("Failed to get response")
    
    finally:
        # Always cleanup
        print("\nCleaning up...")
        cleanup_cache(client, cache, pdf_file)
        print("Done!")


def main():
    """Main function to test explicit caching"""
    try:
        test_explicit_cache()
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have:")
        print("- GEMINI_API_KEY in your environment")
        print("- A PDF file in the current directory")


if __name__ == "__main__":
    main()
