from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
import google.generativeai as genai

from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to use Gemini API for code conversion
def convert_code_with_gemini(code, language):
    model = genai.GenerativeModel('gemini-pro')
    prompt = f"Convert the following Python code to {language}:\n{code}"
    
    # Send the request to Gemini API
    response = model.generate_content(prompt)
    
    if response:
        return response.text.strip()
    else:
        return "// Conversion failed"

@app.route('/convert', methods=['POST'])
def convert():
    data = request.get_json()
    code = data.get('code')
    language = data.get('language')
    
    # Call the Gemini conversion function
    converted_code = convert_code_with_gemini(code, language)
    
    return jsonify({'convertedCode': converted_code})

if __name__ == '__main__':
    app.run(debug=True)