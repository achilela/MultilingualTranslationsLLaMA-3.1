import streamlit as st

class ChatInterface:
    def __init__(self):
        self.chat_input_key = "chat_input"

    def get_user_input(self):
        return st.chat_input("Type your message here...", key=self.chat_input_key)

    def display_message(self, role, content):
        with st.chat_message(role):
            st.write(content)
