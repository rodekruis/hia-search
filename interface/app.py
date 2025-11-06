import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv("../.env")
load_dotenv(".env")

with st.sidebar:
    api_key = st.text_input("Aidly API Key", key="chatbot_api_key", type="password")
    google_sheet_id = st.text_input(
        "HIA Google Sheet ID", key="google_sheet_id", type="password"
    )

st.title("ℹ️️ Aidly")
st.caption("Aids you with helpful information.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input():
    if not api_key:
        st.info("Please add the Aidly API key to continue.")
        st.stop()
    if not google_sheet_id:
        st.info("Please add the HIA Google Sheet ID to continue.")
        st.stop()

    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # make a POST request to chat API
    answer = requests.post(
        # "http://127.0.0.1:8000/chat",
        "https://hia-search-dev.azurewebsites.net/chat",
        headers={"Authorization": api_key},
        json={"question": prompt, "googleSheetId": google_sheet_id},
    )

    response = answer.content.decode("utf-8")
    response = response[1:-1]  # remove quotes from response
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
