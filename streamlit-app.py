import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from chat_interface import ChatInterface
from model_handler import ModelHandler

# Set page configuration
st.set_page_config(page_title="Inspection Engineer Chat", page_icon="🔍")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are an experienced senior inspection engineer. Your task is to analyze the scope provided in the input and determine the class item as an output."}
    ]

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("amiguel/classItem-FT-llama-3-1-8b-instruct")
    model = AutoModelForCausalLM.from_pretrained("amiguel/classItem-FT-llama-3-1-8b-instruct")
    return ModelHandler(model, tokenizer)

def main():
    st.title("Inspection Engineer Assistant")

    # Load model
    model_handler = load_model()

    # Initialize chat interface
    chat_interface = ChatInterface()

    # Display chat messages
    for message in st.session_state.messages[1:]:  # Skip the system message
        chat_interface.display_message(message["role"], message["content"])

    # Chat input
    user_input = chat_interface.get_user_input()

    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        chat_interface.display_message("user", user_input)

        # Prepare the full conversation context
        conversation = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])

        # Generate response
        with st.spinner("Analyzing..."):
            response = model_handler.generate_response(conversation)

        # Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        chat_interface.display_message("assistant", response)

if __name__ == "__main__":
    main()
