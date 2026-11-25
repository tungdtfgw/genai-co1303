from google import genai
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import get_key, get_long_prompt

# This application tests implicit caching in Gemini 2.5 models by asking literary analysis questions.
# It uses a long prompt to see if the model automatically caches the context for subsequent questions.


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


def test_implicit_caching():
    """Test Gemini implicit caching with literary analysis questions"""
    print("=== GEMINI IMPLICIT CACHING TEST ===")
    print("Testing automatic caching in Gemini 2.5 models\n")
    
    client = genai.Client(api_key=get_key())
    long_prompt = get_long_prompt()
    
    print(f"Long prompt length: {len(long_prompt)} characters\n")
    
    # Question 1: Long prompt + Charles Dickens
    print("Question 1: Long prompt + Analyse Charles Dickens")
    question1 = "Analyse the writing style of Charles Dickens."
    full_message1 = f"{long_prompt}\n\n{question1}"
    
    response1 = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=full_message1
    )
    
    print(f"Response: {response1.text[:100]}...")
    display_usage_info(response1, "Question 1")
    
    # Question 2: Long prompt + William Shakespeare
    print("\nQuestion 2: Long prompt + Analyse William Shakespeare")
    question2 = "Analyse the writing style of William Shakespeare."
    full_message2 = f"{long_prompt}\n\n{question2}"
    
    response2 = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=full_message2
    )
    
    print(f"Response: {response2.text[:100]}...")
    display_usage_info(response2, "Question 2")
    
    # Question 3: Just analyse another author
    print("\nQuestion 3: Analyse Jane Austen")
    question3 = "Analyse the writing style of Jane Austen."
    
    response3 = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=question3
    )
    
    print(f"Response: {response3.text[:100]}...")
    display_usage_info(response3, "Question 3")


def main():
    """Main function to test implicit caching"""
    try:
        test_implicit_caching()
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have:")
        print("- GEMINI_API_KEY in your environment")
        print("- long_prompt.txt file in the utils directory")


if __name__ == "__main__":
    main()