import streamlit as st
import pandas as pd
import anthropic
import os
import io
import time
from anthropic import Anthropic

# Set page configuration
st.set_page_config(page_title="BoardingPass PA Chatbot", layout="wide")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "df" not in st.session_state:
    st.session_state.df = None

# Application title
st.title("BoardingPass PA Data Assistant")

# Sidebar for file upload and settings
with st.sidebar:
    st.header("Setup")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload Boardingpass PA CSV file", type=["csv"])
    
    # Instructions and information
    st.markdown("### About")
    st.markdown("""
    This app allows you to chat with your BoardingPass PA data.
    
    
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

# Chat interface
st.header("Chat with Your Data")

# Initialize Anthropic client (using API key from secrets)
@st.cache_resource
def get_anthropic_client():
    api_key = st.secrets["ANTHROPIC_API_KEY"]
    client = Anthropic(api_key=api_key)
    return client

# Only show chat interface if data is loaded
if st.session_state.df is not None:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Get user input
    if prompt := st.chat_input("Ask a question about your BoardingPass PA data"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
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
                
                # Get the anthropic client
                client = get_anthropic_client()
                
                # Create chat history for the API call
                messages = [
                    {"role": "system", "content": system_prompt},
                ]
                
                # Add previous conversation context (limited to last 4 messages)
                for msg in st.session_state.messages[-6:]:
                    messages.append({"role": msg["role"], "content": msg["content"]})
                
                # Stream the response
                with st.spinner("Thinking..."):
                    response = client.messages.create(
                        model="claude-3-5-sonnet-20240620",
                        max_tokens=2000,
                        temperature=0,
                        messages=messages,
                        stream=True
                    )
                    
                    for chunk in response:
                        if chunk.type == "content_block_delta":
                            content_delta = chunk.delta.text
                            full_response += content_delta
                            message_placeholder.markdown(full_response + "▌")
                            time.sleep(0.01)
                    
                    message_placeholder.markdown(full_response)
            
            except Exception as e:
                st.error(f"Error communicating with Claude: {e}")
                full_response = f"I encountered an error: {str(e)}"
                message_placeholder.markdown(full_response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
else:
    st.info("Please upload a CSV file to start chatting.")

