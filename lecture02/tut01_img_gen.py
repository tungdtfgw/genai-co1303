from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import tkinter as tk
from tkinter import ttk
from PIL import ImageTk
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import get_key

from tkinter import messagebox

# This application allows users to input a text prompt and generates an image using the Gemini API.
# The generated image is displayed in the GUI and saved locally as 'gemini-native-image.png'.

class ImageGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Gemini Image Generator")
        self.root.resizable(False, False)
        self.root.geometry("800x600")
        
        self.client = genai.Client(api_key=get_key())
        
        # Create main frame
        my_frame = ttk.Frame(root)
        my_frame.pack(padx=20, pady=20)
        
        # Create and pack widgets
        self.prompt_label = ttk.Label(my_frame, text="Enter your prompt:")
        self.prompt_label.pack(pady=5)
        
        self.prompt_entry = ttk.Entry(my_frame, width=50)
        self.prompt_entry.pack(pady=5)
        
        self.generate_btn = ttk.Button(my_frame, text="Generate Image", command=self.generate_image)
        self.generate_btn.pack(pady=5)
        
        # Create fixed size frame for image
        self.image_frame = ttk.Frame(my_frame, width=512, height=512)
        self.image_frame.pack_propagate(False)
        self.image_frame.pack(pady=10)
        
        self.image_label = ttk.Label(self.image_frame)
        self.image_label.pack(expand=True)
    
    def generate_image(self):
        prompt = self.prompt_entry.get()
        if not prompt:
            return
        
        self.generate_btn.config(state='disabled')
        
        def process_image():
            try:
                response = self.client.models.generate_content(
                    model="gemini-3.1-flash-image-preview", # Use the image generation model
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=['Text', 'Image']       # Request image in response
                    )
                )
                
                for part in response.candidates[0].content.parts:
                    if part.inline_data is not None:
                        # Convert to PIL Image
                        image = Image.open(BytesIO((part.inline_data.data)))
                        
                        # Resize image to fit the frame
                        image.thumbnail((512, 512))
                        
                        # Convert PIL image to PhotoImage
                        photo = ImageTk.PhotoImage(image)
                        
                        # Update label with new image
                        self.image_label.config(image=photo)
                        self.image_label.image = photo
                        
                        # Save image
                        image.save('gemini-native-image.png')
            except Exception as e:
                messagebox.showerror('Error creating image', str(e))
            finally:
                self.generate_btn.config(state='normal')
        
        # Schedule the image processing to run after GUI updates
        self.root.after(100, process_image)

if __name__ == "__main__":
    # Create the main window
    root = tk.Tk()
    app = ImageGeneratorApp(root)
    root.mainloop()