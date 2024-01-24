from datetime import datetime
startTime = datetime.now()
from dotenv import load_dotenv
import os
import google.generativeai as genai

# Image import
import PIL.Image
img = PIL.Image.open('../food/beef/curry.jpg')

# Load key
load_dotenv()

key = os.getenv("API_KEY_GEMINI")
genai.configure(api_key=key)
# model for text
# model = genai.GenerativeModel('gemini-pro')
# model for image
model = genai.GenerativeModel('gemini-pro-vision')
# Text
# response = model.generate_content("I have chciken, tomato and carrot! Give me receipe for dinner of 400 calorie. Give two options of dish")
# Image only
# response = model.generate_content(img)

# # Image and text
response = model.generate_content(["What are the objects in picture?(name only) and give me recipe of 400 calorie with these", img], stream=True)
# response
response.resolve()
# response text
generated_text = response.candidates[0].content.parts[0].text
print(generated_text)
print(datetime.now() - startTime)