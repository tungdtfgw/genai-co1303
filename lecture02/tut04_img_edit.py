from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import tkinter as tk
from tkinter import ttk
from PIL import ImageTk
import os
import sys
# open file dialog
from tkinter import filedialog

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import get_key


from tkinter import messagebox

# This application allows users to input a text prompt and generates an image using the Gemini API.
# The generated image is displayed in the GUI and saved locally as 'gemini-native-image.png'.

class ImageEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Gemini Image Editor")
        self.root.resizable(False, False)
        self.root.geometry("800x600")
        
        self.client = genai.Client(api_key=get_key())

        self.create_widgets()

    def create_widgets(self):
        # open dialog to select image
        self.open_btn = ttk.Button(self.root, text="Open Image", command=self.open_image)
        self.open_btn.grid(row=0, column=0, padx=10, pady=10)

        # image display area
        self.image_frame = ttk.Frame(self.root, width=512, height=512)
        self.image_frame.grid_propagate(False)
        self.image_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10)
        self.image_label = ttk.Label(self.image_frame)
        self.image_label.pack(expand=True)

        # edit prompt
        self.entry_prompt = ttk.Entry(self.root, width=50)
        self.entry_prompt.grid(row=0, column=1, padx=10, pady=10)
        self.edit_btn = ttk.Button(self.root, text="Edit Image", command=self.edit_image)
        self.edit_btn.grid(row=0, column=2, padx=10, pady=10)

    def edit_image(self):
        prompt = self.entry_prompt.get()
        image = self.original_image
        if not prompt or image is None:
            messagebox.showwarning("Input Error", "Please select an image and enter a prompt.")
            return
        
        response = self.client.models.generate_content(
            model="gemini-3.1-flash-image-preview",
            contents=[prompt, image])

        for part in response.candidates[0].content.parts:
            if part.text is not None:
                print(part.text)
            elif part.inline_data is not None:
                image = Image.open(BytesIO(part.inline_data.data))
                image.save("generated_image.png")
                self.display_image(image)
                messagebox.showinfo("Success", "Image edited and saved as 'generated_image.png'.")

    def open_image(self):
        try:
            img_file = filedialog.askopenfilename(
                
                filetypes=[("Image Files", "*.png"), ("All Files", "*.*")])
            if not img_file:
                messagebox.showwarning("No file selected", "Please select an image file.")
                return
            self.original_image = Image.open(img_file)
            self.display_image(self.original_image)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open image: {e}")

    def display_image(self, pil_image):
        self.tk_image = ImageTk.PhotoImage(pil_image)
        # resize image to fit in label
        self.tk_image = ImageTk.PhotoImage(pil_image.resize((512, 512)))

        self.image_label.config(image=self.tk_image) 

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageEditorApp(root)
    app.run()