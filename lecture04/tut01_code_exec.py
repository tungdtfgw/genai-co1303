from google import genai
from google.genai import types

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import get_key


def display_code_execution_result(response):
    # Check if any code execution occurred
    has_code_execution = any(
        part.executable_code is not None or part.code_execution_result is not None
        for part in response.candidates[0].content.parts
    )
    
    if not has_code_execution:
        return  # Don't print anything if no code execution
    
    for part in response.candidates[0].content.parts:
        if part.executable_code is not None:
            print("\nCode:")
            print(part.executable_code.code)
        if part.code_execution_result is not None:
            print("\nExecution Result:")
            print(part.code_execution_result.output)

    print("---")

def init_chat():
    client = genai.Client(api_key=get_key())
    system_prompt = "You are an AI assistant. You will use Code execution feature to answer questions that requires calculation if needed, for example counting number of characters in a word, number of words in a sentence, comparing numbers, etc. You will also use Code execution feature to answer questions that requires code execution, for example writing a simple python code to calculate the sum of two numbers, writing a simple python code to calculate the factorial of a number, etc."
    
    code_config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=[types.Tool(
                code_execution=types.ToolCodeExecution
            )])
    chat = client.chats.create(model='gemini-2.5-flash', config=code_config)
    
    return client, chat # return client so it does not get garbage collected

def main():
    client, chat = init_chat()
    while True:
        question = input('[User]: ')
        
        response = chat.send_message(message=question)
        # Only show the text response without warnings
        if hasattr(response, 'candidates') and len(response.candidates) > 0:
            text_parts = [part.text for part in response.candidates[0].content.parts if hasattr(part, 'text') and part.text is not None]
            print('[GenAI]:', ' '.join(text_parts))
        else:
            print('[GenAI]:', response.text)
        
        # Uncomment the following line to display code execution results
        display_code_execution_result(response)

        if question.strip() == 'exit' or question.strip() == 'quit':
            break
if __name__ == "__main__":
    main()