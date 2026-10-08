import os
import uuid

import requests
import streamlit as st
from dotenv import load_dotenv


load_dotenv()
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

st.title("AI Knowledge Assistant")


# Conversation ID
if "conversation_id" not in st.query_params:
    conversation_id = str(uuid.uuid4())
    st.query_params["conversation_id"] = conversation_id
else:
    conversation_id = st.query_params["conversation_id"]


# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None


# PDF upload
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    if uploaded_file.name != st.session_state.uploaded_file_name:
        try:
            response = requests.post(
                f"{API_BASE_URL}/upload",
                files={
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        "application/pdf",
                    )
                },
                timeout=120,
            )

            if response.status_code == 200:
                data = response.json()
                st.success(data["message"])
                st.session_state.uploaded_file_name = uploaded_file.name
            else:
                st.error(f"Upload failed: {response.text}")

        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the FastAPI server.")
        except requests.exceptions.RequestException:
            st.error("The PDF upload request failed.")


# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            st.markdown("**Sources:**")
            for source in message["sources"]:
                st.write(f"- {source}")


# Chat
message = st.chat_input("Ask me something...")

if message:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": message,
        }
    )

    with st.chat_message("user"):
        st.write(message)

    with st.spinner("Thinking..."):
        try:
            response = requests.post(
                f"{API_BASE_URL}/chat",
                json={
                    "session_id": conversation_id,
                    "message": message,
                },
                timeout=120,
            )

            if response.status_code == 200:
                data = response.json()
                answer = data["answer"]
                sources = data["sources"]

                with st.chat_message("assistant"):
                    st.write(answer)

                    if sources:
                        st.markdown("**Sources:**")
                        for source in sources:
                            st.write(f"- {source}")

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )
            else:
                st.error("Failed to get a response from the AI assistant.")

        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the FastAPI server.")
        except requests.exceptions.RequestException:
            st.error("The chat request failed.")
