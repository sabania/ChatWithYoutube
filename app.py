import os
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound

# Custom Imports
from youtube_helper import extract_video_id, create_metadata
from transcript_processing import split_transcript
from chat_ui import create_chat_area
from answer_parsing import parse_answer

# Replace the following imports with your own implementations
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader

from chat_gpt import chat
from yt_templates.templates import get_follow_up_template
from yt_templates.templates import get_initial_template
import openai

# Streamlit Setup
st.markdown(
    """<style>.block-container{max-width: 66rem !important;}</style>""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; align-items: left; border: 3px solid #d35353; padding: 25px">
    <div style="margin-right: 20px">
        <strong>Arben Sabani</strong><br>
        <small>Developer <br> NLP Enthusiast <br> Language Technologist</small><br><a style="text-decoration:none;" href="https://www.supertext.ch/" target="_blank">@Supertext AG </a>              
    </div>
    <div>
        <a href="https://www.linkedin.com/in/arben-sabani-339074128/" target="_blank">
            <img src="https://img.icons8.com/fluent/48/000000/linkedin.png" style="height: 58px;">
        </a>   <br>
        <a href="https://www.supertext.ch/" target="_blank">
            <img src="https://media.licdn.com/dms/image/C4D0BAQEJ4VK0Yhvf8Q/company-logo_200_200/0/1625125787308?e=1703116800&v=beta&t=A4z331QbMsGct2sBcNNusL60IqvOPJnRkmAQlQKqv0M" style="height: 48px;margin-left: 6px; margin-top:15px">
        </a>
    </div>
</div>

        """,
        unsafe_allow_html=True
    )

st.title("Chat with your YouTube Video")
st.markdown('---')

# Initialize session state variables
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'chat_history_view' not in st.session_state:
    st.session_state.chat_history_view = []
if 'expanded_preprocessing' not in st.session_state:
    st.session_state.expanded_preprocessing = True

openai_key = os.environ.get("OPENAI_API_KEY")
if openai_key is None:
    with st.sidebar:
        st.subheader("Settings")
        openai_key = st.text_input("Enter your OpenAI key:", type="password")
        print("after set:", openai_key)
        
if openai_key:
    # Preprocess Video
    openai.api_key = openai_key
    st.subheader("Preprocess Video")
    with st.status("", expanded=st.session_state.expanded_preprocessing) as status:
        url = st.text_input("Enter YouTube Video URL:")
        if url:
            video_id = extract_video_id(url)
            if not video_id:
                st.error("Invalid YouTube URL!")
            else:
                try:
                    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                    available_transcripts = {
                        transcript.language_code: transcript for transcript in transcript_list
                    }
                    selected_language = st.selectbox(
                        "Choose video language:",
                        list(available_transcripts.keys())
                    )
                    if st.button("Preprocess Video"):
                        status.update(label="Processing...", state="running", expanded=True)
                        transcript = available_transcripts[selected_language]
                        st.session_state.transcript = transcript
                        st.session_state.transcript_parts = split_transcript(transcript, 100)
                        embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
                        st.session_state.vector_store = FAISS.from_documents(st.session_state.transcript_parts, embeddings)
                        status.update(label="Processing completed! You can start chatting with the video!", state="complete", expanded=False)
                        st.session_state.expanded_preprocessing = False
                except NoTranscriptFound:
                    st.error("No transcript found for the given video!")
                    status.update(label="No transcript found for the given video!", state="error", expanded=True)

    # Chat Interface
    if 'vector_store' in st.session_state:
        st.markdown('---')
        st.subheader("Chat Interface")
        create_chat_area(st.session_state.chat_history_view)
        clear_button = st.button("Clear Chat History") if len(st.session_state.chat_history_view) > 0 else None
        user_input = st.chat_input("Ask something about the video")

        if clear_button:
            st.session_state.chat_history = []
            st.session_state.chat_history_view = []
            st.experimental_rerun()

        if user_input:
            search_result = st.session_state.vector_store.search(user_input, search_type="similarity", k=25)
            st.session_state.current_search_result = search_result
            current_msg_txt = get_initial_template(search_result, user_input) if len(st.session_state.chat_history) == 0 else get_follow_up_template(search_result, user_input)
            st.session_state.chat_history.append({"role": "user", "content": current_msg_txt})
            st.session_state.chat_history_view.append({"role": "user", "content": user_input})

            gpt_answer = chat(st.session_state.chat_history, 1000, model="gpt-3.5-turbo-16k")
            st.session_state.chat_history.append({"role": "assistant", "content": gpt_answer})

            text, vid_content = parse_answer(gpt_answer, video_id)
            st.session_state.chat_history_view.append({"role": "assistant", "content": text, "vid_content": vid_content})
            st.experimental_rerun()
else:
    st.error("Please enter your OpenAI key in the sidebar to start chatting with the video!")
