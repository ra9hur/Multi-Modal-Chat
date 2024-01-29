import os
import yaml

import streamlit as st
from langchain.memory import StreamlitChatMessageHistory
from streamlit_mic_recorder import mic_recorder

from llm_chains import load_normal_chain, load_pdf_chat_chain
from utils import save_chat_history_json, load_chat_history_json, get_timestamp
from image_handler import handle_image
from pdf_handler import add_documents_to_db
from html_templates import get_bot_template, get_user_template, css


with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

def load_chain(chat_history):
    if st.session_state.pdf_chat:
        print("loading pdf chat chain")
        return load_pdf_chat_chain(chat_history)
    return load_normal_chain(chat_history)


def clear_input_field():
    st.session_state["user_question"] = st.session_state["user_input"]
    st.session_state["user_input"] = ""

def set_send_input():
    st.session_state["send_input"] = True
    clear_input_field()

def toggle_pdf_chat():
    st.session_state.pdf_chat = True

# https://discuss.streamlit.io/t/how-to-create-a-chat-history-on-the-side-bar-just-like-chatgpt/59492/2
def save_chat_history():
    if st.session_state.history != []:
        if st.session_state.session_key == "new_session":
            st.session_state.new_session_key = get_timestamp() + ".json"
            save_chat_history_json(st.session_state.history, 
                                   config["chat_history_path"] + st.session_state.new_session_key)
        else:
            save_chat_history_json(st.session_state.history, 
                                   config["chat_history_path"] + st.session_state.session_key)


def main():
    st.sidebar.title("Chat Sessions")
    st.title("Multimodal Local Chat App")
    st.write(css, unsafe_allow_html=True)
    st.sidebar.write(css, unsafe_allow_html=True)

    chat_container = st.container()

    chat_sessions = ['new_session'] + os.listdir(config["chat_history_path"])

    if 'send_input' not in st.session_state:
        st.session_state.session_key = "new_session"
        st.session_state.send_input = False
        st.session_state.user_question = ""
        st.session_state.new_session_key = None
        st.session_state.session_index_tracker = "new_session"
    if st.session_state.session_key == "new_session" and st.session_state.new_session_key != None:
        st.session_state.session_index_tracker = st.session_state.new_session_key
        st.session_state.new_session_key = None

    index = chat_sessions.index(st.session_state.session_index_tracker)
    st.sidebar.selectbox("Select a chat session", chat_sessions, key="session_key", index=index)

    if st.session_state.session_key != "new_session":
        st.session_state.history = load_chat_history_json(config["chat_history_path"] + st.session_state.session_key)
    else:
        st.session_state.history = []

    chat_history = StreamlitChatMessageHistory(key="history")
    #print(chat_history)

    user_input = st.text_input("Type your message here", key="user_input", on_change = set_send_input)

    voice_recording_col, send_button_col = st.columns(2)

    with voice_recording_col:
        voice_recording = mic_recorder(start_prompt="Start recording", stop_prompt="Stop recording", just_once=True)

    with send_button_col:
        send_button = st.button("Send", key="send_button")


    uploaded_image = st.sidebar.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])

    st.sidebar.toggle("PDF Chat", key="pdf_chat", value=False)
    uploaded_pdf = st.sidebar.file_uploader("Upload a pdf file", accept_multiple_files=True, key="pdf_upload", type=["pdf"], on_change=toggle_pdf_chat)

    if uploaded_pdf:
        with st.spinner("Processing pdf..."):
            add_documents_to_db(uploaded_pdf)

    if send_button or st.session_state.send_input:
        if uploaded_image:
            with st.spinner("Processing image..."):
                user_message = "Describe this image in detail please."
                if st.session_state.user_question != "":
                    user_message = st.session_state.user_question
                    st.session_state.user_question = ""
                llm_response = handle_image(uploaded_image.getvalue(), user_message)
                chat_history.add_user_message(user_message)
                chat_history.add_ai_message(llm_response)
        
        if st.session_state.user_question != "":

            llm_chain = load_chain(chat_history)

            # https://bijukunjummen.medium.com/chat-application-using-streamlit-and-text-bison-05024f939827
            # https://github.com/langchain-ai/streamlit-agent/blob/main/streamlit_agent/basic_memory.py
            with st.spinner("Wait for AI response ..."):
                llm_response = llm_chain.run(st.session_state.user_question)
            st.session_state.user_question = ''

        st.session_state.send_input = False


    if chat_history.messages != 0:
        with chat_container:
            # Render current messages from StreamlitChatMessageHistory
            st.write("Chat History")
            for msg in chat_history.messages:
                if msg.type == "human":
                    st.write(get_user_template(msg.content), unsafe_allow_html=True)
                else:
                    st.write(get_bot_template(msg.content), unsafe_allow_html=True)



    save_chat_history()


if __name__ == "__main__":
    main()
