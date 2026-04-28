import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=API_KEY)

# Use Gemini 1.5 Flash (ideal for fast tasks and text/code processing)
model = genai.GenerativeModel('gemini-3-flash-preview')

def get_data_insights(df_info, df_head, query):
    """
    Sends a prompt containing data info and user query to the LLM.
    """
    prompt = f"""
    You are an expert Data Analyst AI.
    Below is the information about the CSV dataset the user just uploaded:
    - Data Structure (Data types): \n{df_info}
    - First 5 rows of data: \n{df_head}

    User's request: "{query}"

    RESPONSE GUIDELINES:
    1. If the user asks general analysis questions, respond with clear, concise text.
    2. IF the user requests VISUALIZATION (drawing a chart/graph), you MUST return a PYTHON CODE SNIPPET using `matplotlib` or `seaborn` and `pandas`.
       - The code must be enclosed in a ```python ... ``` block.
       - Assume that the dataframe is already loaded into a variable named `df`.
       - Create the plot and save the figure object to a variable named `fig` (e.g., `fig, ax = plt.subplots(...)`).
       - Do not include commands like `plt.show()`.
    """
    
    response = model.generate_content(prompt)
    return response.text