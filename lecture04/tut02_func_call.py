from google import genai
from google.genai.types import GenerateContentConfig
from google.genai import types
import requests

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import get_key

client = genai.Client(api_key=get_key())

# Function declarations for APIs
cat_function = {
    "name": "get_cat_fact",
    "description": "Retrieves a random fact about cats from the catfact.ninja API",
    "parameters": {
        "type": "object",
        "properties": {},    # No arguments are required for this function
        "required": []       
    }
}

gender_function = {
    "name": "get_gender",
    "description": "Predicts the gender of a given name using the genderize.io API",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The name to predict gender for"
            }
        },
        "required": ["name"]
    }
}

holidays_function = {
    "name": "get_holidays",
    "description": "Get holidays for a country",
    "parameters": {
        "type": "object",
        "properties": {
            "country": {
                "type": "string",
                "description": "The country abbreviations (i.e us for united state) to get holidays"
            }
        },
        "required": ["country"]
    }
}


# Generation Config
config = GenerateContentConfig(
    system_instruction="You are a helpful assistant, you can answer everything. However, when user asks about cats or cat facts, use the get_cat_fact function. When user asks about any question that requires gender prediction for a name, you must use the get_gender function then based on the probability returned to answer. When user asks about holidays in a country, use get_holidays function. Otherwise, provide a general response from your general knowledge.",
    tools=[{"function_declarations": [cat_function, gender_function, holidays_function]}]
)

# Function to get a random cat fact from a cat facts API
def get_cat_fact():
    print("[Tool called!!!]: Fetching a random cat fact from catfact.ninja")
    try:
        response = requests.get("https://catfact.ninja/fact")
        data = response.json()
        return data["fact"]
    except Exception as e:
        return {"error": str(e)}

# Function to predict gender from a name
def get_gender(name):
    print(f"[Tool called!!!]: Predicting name from genderize.io")
    try:
        response = requests.get(f"https://api.genderize.io/?name={name}")
        data = response.json()
        return {
            "name": data["name"],
            "gender": data["gender"],
            "probability": data["probability"],
            "count": data["count"]
        }
    except Exception as e:
        return {"error": str(e)}

def get_holidays(country):
    print(f"[Tool called!!!]: Get holidays from caldays")
    try:
        response = requests.get(f"https://caldays.com/api/long-weekends/{country}")
        data = response.json()
        return {
            "code": data["code"],
            "country": data["country"],
            "year": data["year"],
            "longWeekends": data["longWeekends"]
        }
    except Exception as e:
        return {"error": str(e)}

# Function dictionary to map the function name to the function
functions = {
    "get_cat_fact": get_cat_fact,
    "get_gender": get_gender,
    "get_holidays": get_holidays
}

# Helper function to call the function
def call_function(function_name, **kwargs):
    return functions[function_name](**kwargs)

# Agentic loop to handle the function call
def answer_with_function_call(chat, prompt):
    # Send the message to the existing chat session
    response = chat.send_message(message=prompt)
    
    # Process the response and handle function calls if needed
    for part in response.candidates[0].content.parts:
        if part.function_call:
            function_call = part.function_call
            # Call the tool with arguments
            tool_result = call_function(function_call.name, **function_call.args)
            # Build the function response
            function_response_part = types.Part.from_function_response(
                name=function_call.name,
                response={"result": tool_result},
            )
            # Send the function result back to the model, the model will use the function result to generate a response
            response = chat.send_message(function_response_part)
    
    # Return the final response
    if response and response.candidates and response.candidates[0].content.parts:
        return response.candidates[0].content.parts[0].text.strip()
    return "Sorry, I couldn't generate a response."

# Main chat loop
# Create a chat session
chat = client.chats.create(model='gemini-2.5-flash', config=config)
print("Welcome to the AI Chat! Type 'exit' to quit.")
while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        print("Goodbye!")
        break
    response = answer_with_function_call(chat, user_input)
    print("AI:", response)