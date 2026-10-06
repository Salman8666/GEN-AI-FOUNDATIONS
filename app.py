import streamlit as st
from ai_engine import AIEngine
from langchain_core.messages import HumanMessage, AIMessage

# Page Configuration
st.set_page_config(
    page_title="AI Assistant",
    page_icon="🤖",
    layout="centered"
)

# Custom CSS for polished UI
st.markdown("""
    <style>
    .stChatMessage {
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .main {
        background-color: #f8f9fa;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize AI Engine in Session State
if "ai_engine" not in st.session_state:
    st.session_state.ai_engine = AIEngine()

# Initialize Chat History in Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar Configuration
with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("---")

    system_prompt = st.text_area(
        "System Prompt",
        value=st.session_state.ai_engine.system_prompt,
        help="Define how the AI should behave."
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.4,
        step=0.1,
        help="Higher values make the output more random."
    )

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.info("Powered by LangChain & Ollama (gemma4:cloud)")

# Update engine settings based on sidebar
st.session_state.ai_engine.system_prompt = system_prompt
# Note: temperature is set at init, updating it would require re-init if we wanted dynamic changes.
# For a production app, we'd handle re-initialization of the LLM object here.

# Main UI
st.title("🤖 Modern AI Assistant")
st.caption("A professional chatbot interface built with Streamlit and LangChain")

# Display Chat History
for message in st.session_state.messages:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# Chat Input
if prompt := st.chat_input("Type your message here..."):
    # User Message
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant Response (Streaming)
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        try:
            with st.spinner("Thinking..."):
                # We pass the session messages to the engine
                for chunk in st.session_state.ai_engine.get_response_stream(st.session_state.messages):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")

                response_placeholder.markdown(full_response)

            # Save Assistant Response to History
            st.session_state.messages.append(AIMessage(content=full_response))

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            # Optionally remove the last human message if the AI failed completely
            # st.session_state.messages.pop()
