import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get API key from environment variable
API_KEY = os.getenv('GEMINI_API_KEY')

if not API_KEY:
    st.error("API key not found. Please check your .env file.")
    st.stop()

# Configure the API
genai.configure(api_key=API_KEY)

# Set up the model configuration
generation_config = {
    "temperature": 0.7,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

def get_chat_response(prompt, chat_history):
    """Get response from the Gemini model."""
    try:
        # Create a new chat for each conversation
        chat = genai.GenerativeModel(model_name='gemini-pro',
                                   generation_config=generation_config,
                                   safety_settings=safety_settings)
        
        # Initialize chat
        chat = chat.start_chat(history=[])
        
        # Get response
        response = chat.send_message(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

def create_chat_interface():
    """Create the chat interface in Streamlit."""
    st.sidebar.title("AI Fitness Assistant")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about fitness and exercise..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get and display assistant response
        with st.chat_message("assistant"):
            response = get_chat_response(prompt, st.session_state.messages)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Clear chat button
    if st.sidebar.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun() 