from google import genai
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import get_key

# Example of using systemp prompt. 
# AI refers to the system prompt to generate the response
# AI has no memory, so it will not remember the context of the conversation
client = genai.Client(api_key=get_key())
system_prompt="You are a cat. You will answer questions as a cat. Convert words to meow, meoow, meooww, etc. based on the length of the word, minimum is meow. ONLY MEOWS ALLOWED."

# Create a config with system_instruction to use when model is generating content
config = genai.types.GenerateContentConfig(system_instruction=system_prompt)
while True:
    question = input('[User]: ')
    if question.strip() == 'exit' or question.strip() == 'quit':
        break
    response = client.models.generate_content(
        model='gemini-3-flash-preview',
        contents=[question],
        config=config   # config with system_instruction
    )
    print('[GenAI]:', response.text)
