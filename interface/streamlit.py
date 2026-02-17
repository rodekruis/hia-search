import os
import requests
import streamlit as st
from streamlit.runtime import get_instance
from streamlit.runtime.scriptrunner import get_script_run_ctx
from dotenv import load_dotenv

load_dotenv("../.env")

def _get_session() -> str:
    "Get the current Streamlit session ID, which is used as a thread ID for the chat API."
    runtime = get_instance()
    session_id = get_script_run_ctx().session_id
    session_info = runtime._session_mgr.get_session_info(session_id)
    if session_info is None:
        raise RuntimeError("Couldn't get your Streamlit Session object.")
    return str(session_info.session.id)


with st.sidebar:
    google_sheet_id = st.text_input("Google Sheet ID", key="google_sheet_id", type="default")

st.title("ℹ️ Aidly")
st.caption("Ask him questions about any HIA instance.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input():
    if not google_sheet_id:
        st.info("Please add a valid Google Sheet ID to continue.")
        st.stop()

    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # make a POST request to chat API
    answer = requests.post(
        # "http://127.0.0.1:8000/chat-dummy",
        "https://hia-search-dev.azurewebsites.net/chat-dummy",
        headers={"Authorization": os.environ["API_KEY"]},
        params={"googleSheetId": google_sheet_id, "threadId": _get_session()},
        json={"message": prompt},
    )

    response = answer.json()['response'].strip()  # remove quotes from response
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})