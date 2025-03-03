import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as gen_ai
from PIL import Image

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

gen_ai.configure(api_key=GOOGLE_API_KEY)
model = gen_ai.GenerativeModel('gemini-2.0-pro-exp')

# Configure Streamlit with Moai emoji as page icon
st.set_page_config(
    page_title="AI Assistant by Aayushmaan!",
    page_icon="🗿",  # Moai emoji as the page icon
    layout="centered",
)

# Display the logo in the header (if you want to add any logo as well)
st.sidebar.title("⚙️ Settings")
mode = st.sidebar.radio("Select AI Personality:", ["Friendly", "Professional", "Fun", "Optimistic", "Motivational"])
language = st.sidebar.selectbox("Select Language:", ["English", "Spanish", "French", "German", "Hindi"])

def get_ai_prompt(prompt):
    """Modify prompt based on personality mode and selected language."""
    # Adjust the tone based on the selected personality mode
    if mode == "Friendly":
        prompt = f"You are a friendly assistant. {prompt}"
    elif mode == "Professional":
        prompt = f"You are a highly professional AI. Answer concisely: {prompt}"
    elif mode == "Fun":
        prompt = f"Crack a joke while answering: {prompt}"
    elif mode == "Optimistic":
        prompt = f"You are an optimistic assistant. Always provide positive and encouraging answers: {prompt}"
    elif mode == "Motivational":
        prompt = f"You are a motivational assistant. Encourage and uplift the user with your responses: {prompt}"

    # Add language-based adjustments (for now, you can change the language of the AI's response)
    if language == "Spanish":
        prompt = f"Por favor responde en español: {prompt}"
    elif language == "French":
        prompt = f"Veuillez répondre en français : {prompt}"
    elif language == "German":
        prompt = f"Bitte antworten Sie auf Deutsch: {prompt}"
    elif language == "Hindi":
        prompt = f"कृपया हिंदी में उत्तर दें: {prompt}"

    return prompt

# Initialize chat session
if "chat_session" not in st.session_state:
    st.session_state.chat_session = model.start_chat(history=[])

st.title("🗿 I am Aayushmaan's AI 🗿")

# Display chat history
for message in st.session_state.chat_session.history:
    with st.chat_message("assistant" if message.role == "model" else "user"):
        st.markdown(message.parts[0].text)

# Accept user input (Text)
user_prompt = st.chat_input("Type your message...")

if user_prompt:
    user_prompt = get_ai_prompt(user_prompt)
    st.chat_message("user").markdown(user_prompt)

    response = st.session_state.chat_session.send_message(user_prompt)
    with st.chat_message("assistant"):
        st.markdown(response.text)
