import streamlit as st
import pandas as pd
import io
import re
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from services.gemini_service import get_data_insights

st.set_page_config(page_title="AI Data Assistant", page_icon="🤖", layout="wide")

st.title("Data Analysis AI Assistant")
st.markdown("Upload your CSV file and ask questions for analysis or request data visualizations.")

# 1. CSV File Upload Feature
uploaded_file = st.file_uploader("Choose a CSV file...", type=["csv"])

if uploaded_file is not None:
    # Read the data
    df = pd.read_csv(uploaded_file)
    st.success(f"Successfully loaded: {uploaded_file.name}")
    
    with st.expander("Data Preview"):
        st.dataframe(df.head())

    # Extract dataframe info to use as context for the LLM
    buffer = io.StringIO()
    df.info(buf=buffer)
    df_info = buffer.getvalue()
    df_head = df.head().to_string()

    # 2. Initialize Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "fig" in message:
                st.pyplot(message["fig"])

    # 3. Handle User Queries
    if prompt := st.chat_input("Enter your query (e.g., Plot the distribution of age)"):
        # Save user query to session
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                # Call the API via the service
                response_text = get_data_insights(df_info, df_head, prompt)
                
                # Helper function to extract Python code if the LLM returns a chart request
                def extract_python_code(text):
                    match = re.search(r'```python\n(.*?)\n```', text, re.DOTALL)
                    return match.group(1) if match else None

                code_block = extract_python_code(response_text)
                
                if code_block:
                    st.markdown("Generated chart based on your request:")
                    try:
                        # Execute the LLM-generated code to draw the chart
                        # Note: In a production environment, use exec() with caution
                        local_vars = {"df": df, "plt": plt, "sns": sns, "pd": pd}
                        exec(code_block, globals(), local_vars)
                        
                        if "fig" in local_vars:
                            fig = local_vars["fig"]
                            st.pyplot(fig)
                            # Save to history to re-render on reload
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": "Here is your chart:", 
                                "fig": fig
                            })
                        else:
                            st.warning("The AI did not return the `fig` variable as requested.")
                    except Exception as e:
                        st.error(f"Error generating chart: {e}")
                        st.code(code_block, language='python')
                else:
                    # If it's not a visualization request, print the text response
                    st.markdown(response_text)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})