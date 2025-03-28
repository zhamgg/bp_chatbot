import streamlit as st

# Set page configuration - MUST be the first Streamlit command
st.set_page_config(page_title="BoardingPass Chatbot", layout="wide")

import pandas as pd
import os
import io
import time
import requests
import json

# Application title
st.title("BoardingPass Data Assistant")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "df" not in st.session_state:
    st.session_state.df = None


# Use a fixed model instead of selection
MODEL = "claude-3-sonnet-20240229"

# Get API key from Streamlit secrets
# Get API key from Streamlit secrets or allow user input as fallback
try:
    api_key = st.secrets["anthropic_api_key"]
    api_key_source = "from secrets"
except Exception:
    # If secret isn't available, show input field
    with st.sidebar:
        api_key = st.text_input("Enter your Anthropic API Key:", type="password")
        api_key_source = "from user input"
    
# Show API key status indicator
with st.sidebar:
    if api_key:
        st.success(f"API Key loaded {api_key_source}")
    else:
        st.warning("API Key required to use the chat functionality")

# Sidebar for file upload and settings
with st.sidebar:
    st.header("Setup")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload Boardingpass PA CSV file", type=["csv"])
    
    # Reset button
    if st.button("Reset Chat History"):
        st.session_state.messages = []
        st.success("Chat history has been reset!")
    
    # Instructions and information
    st.markdown("### About")
    st.markdown("""
    This app allows you to chat with your BoardingPass PA data.
    
    Simply upload your CSV file and start asking questions about the data.
    
    **Example questions:**
    - How many plan requests are there in total?
    - What are the top 5 advisors by number of plans?
    - Show me all plans with funding amounts over $1M
    - What's the average funding amount by plan type?
    - List the most common recordkeepers
    """)

# Load data when file is uploaded
if uploaded_file is not None:
    try:
        # Cache the dataframe to avoid reloading on each interaction
        if st.session_state.df is None:
            data_load_state = st.info("Loading data...")
            st.session_state.df = pd.read_csv(uploaded_file, encoding='cp1252', low_memory=False)
            data_load_state.success("Data loaded successfully!")
            
            # Show basic statistics
            st.subheader("Dataset Overview")
            st.write(f"Total rows: {len(st.session_state.df)}")
            st.write(f"Total columns: {len(st.session_state.df.columns)}")
            
            # Display sample of the dataframe
            with st.expander("Preview Data"):
                st.dataframe(st.session_state.df.head())
    except Exception as e:
        st.error(f"Error loading the file: {e}")

# Function to call Claude API directly
def call_claude_api(system_prompt, messages, api_key, model=MODEL):
    headers = {
        "x-api-key": api_key,
        "content-type": "application/json",
        "anthropic-version": "2023-06-01"
    }
    
    # Create the message array for the API
    api_messages = []
    for msg in messages:
        if msg["role"] != "system" and msg["content"] and msg["content"].strip():  # Skip system messages and ensure content is not empty
            api_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
    
    # Ensure there's at least one user message
    if not any(msg["role"] == "user" for msg in api_messages):
        # Add a default user message if none exists
        api_messages.append({
            "role": "user",
            "content": "Hello"
        })
    
    data = {
        "model": model,
        "system": system_prompt,  # Use as top-level system parameter
        "max_tokens": 2000,
        "temperature": 0,
        "messages": api_messages
    }

    # Debug information
    debug_data = {
        "url": "https://api.anthropic.com/v1/messages",
        "headers": {
            "x-api-key": "sk-***" + api_key[-4:],  # Show only last 4 chars
            "content-type": "application/json",
            "anthropic-version": "2023-06-01"
        },
        "data": {
            "model": MODEL,
            "system": system_prompt[:100] + "...",  # Truncate for display
            "max_tokens": 2000,
            "temperature": 0,
            "messages": [
                {
                    "role": msg["role"],
                    "content": msg["content"][:20] + "..." if len(msg["content"]) > 20 else msg["content"]
                }
                for msg in api_messages
            ]
        }
    }
    st.expander("Request Debug Info").json(debug_data)

    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers=headers,
        json=data
    )
    
    if response.status_code != 200:
        error_info = response.json()
        st.expander("API Response Error").json(error_info)
        raise Exception(f"Error code: {response.status_code} - {error_info}")
    
    return response.json()


# Chat interface
st.header("Chat with Your Data")

# Only show chat interface if data is loaded and API key is provided
if st.session_state.df is not None:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Get user input
    if prompt := st.chat_input("Ask a question about your BoardingPass data"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            # Get data stats for system prompt
            df = st.session_state.df
            
            # Create description of dataframe
            num_rows = len(df)
            num_cols = len(df.columns)
            column_info = "\n".join([f"- {col}: {df[col].dtype}" for col in df.columns[:10]])
            if len(df.columns) > 10:
                column_info += f"\n- ... and {len(df.columns) - 10} more columns"
            
            # Create sample rows
            sample_rows = df.head(3).to_string()
            
            # Get basic stats
            try:
                total_funding = df["Estimated Funding Amount"].sum()
                avg_funding = df["Estimated Funding Amount"].mean()
                top_recordkeepers = df["Recordkeeper Name"].value_counts().head(5).to_dict()
                top_rk_str = "\n".join([f"- {k}: {v} plans" for k, v in top_recordkeepers.items()])
                plan_types = df["Plan Type"].value_counts().to_dict()
                plan_types_str = "\n".join([f"- {k}: {v} plans" for k, v in plan_types.items()])
                
                stats_summary = f"""
                Total Funding Amount: ${total_funding:,.2f}
                Average Funding Amount: ${avg_funding:,.2f}
                
                Top Recordkeepers:
                {top_rk_str}
                
                Plan Types:
                {plan_types_str}
                """
            except:
                stats_summary = "Unable to calculate all statistics due to data format issues."
            
            try:
                # Prepare system prompt with data context
                system_prompt = f"""
                You are a helpful assistant that answers questions about Boardingpass PA data.
                
                Here's information about the dataset:
                - The dataset contains {num_rows} rows and {num_cols} columns
                - Key columns include: Request ID, Plan Name, Fund Name, Estimated Funding Amount, 
                  Advisor Name, Recordkeeper Name, Plan Type, etc.
                
                Basic statistics:
                {stats_summary}
                
                When asked questions about the data:
                1. Be specific and precise in your answers
                2. When appropriate, mention the number of records that match the criteria
                3. For numerical questions, provide the calculation method used
                4. If you don't have enough information, ask for clarification
                5. Keep your answers concise and focused on the data
                
                DO NOT make up information that is not in the data.
                """
                
                # Create chat history for the API call
                api_messages = []
                
                # Add previous conversation context (limited to last 4 messages)
                for msg in st.session_state.messages[-6:]:
                    # Ensure content is not empty
                    if msg["content"] and msg["content"].strip():
                        api_messages.append({
                            "role": msg["role"],
                            "content": msg["content"]
                        })
                
                # Call Claude API directly
                with st.spinner("Thinking..."):
                    full_response = ""
                    try:
                        response = call_claude_api(
                            system_prompt=system_prompt,
                            messages=api_messages,
                            api_key=api_key
                        )

                        # Get the content from the response
                        if "content" in response:
                            # New API format with content blocks
                            if isinstance(response["content"], list):
                                for block in response["content"]:
                                    if isinstance(block, dict) and "text" in block:
                                        full_response += block["text"]
                            # String content (fallback)
                            elif isinstance(response["content"], str):
                                full_response = response["content"]
                        # Legacy format
                        elif "completion" in response:
                            full_response = response["completion"]
                    except Exception as e:
                        st.error(f"Error with Claude API: {str(e)}")
                        full_response = "I encountered an error and couldn't process your question."
                
                message_placeholder.markdown(full_response)
            
            except Exception as e:
                st.error(f"Error processing request: {str(e)}")
                full_response = f"I encountered an error: {str(e)}"
                message_placeholder.markdown(full_response)
        
        # Add assistant response to chat history
        if full_response and full_response.strip():
            st.session_state.messages.append({"role": "assistant", "content": full_response})
else:
    st.info("Please upload a CSV file to start chatting.")

