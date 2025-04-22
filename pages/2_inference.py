import streamlit as st

from src.model import NGram

def load_model():
    try:
        import pickle
        # print("trying to load model")
        with open("trained_model.pkl", "rb") as f:
            model = pickle.load(f)
        # print("model loaded")
        return model
    except Exception as e:
        # print("failed to load model")
        st.error(f"Error loading model: {e}")
        raise

model = load_model()

st.set_page_config(page_title="Inference", page_icon="💬", layout="wide")
st.title("🧠 Chat with Language Model")

col1, col2 = st.columns(2)

with col1:
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

with col2:
    max_tokens = st.slider("Maximum Token to generate", min_value=1, max_value=100, value=50, step=5)

# Initialize session state if not already
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chat_metadata" not in st.session_state:
    st.session_state.chat_metadata = []
if "last_user_input" not in st.session_state:
    st.session_state.last_user_input = ""

# Input box
user_input = st.text_input("You:", placeholder="Type a message and press Enter")

# Only process if new input is given (avoids rerun duplicate)
if user_input and user_input != st.session_state.last_user_input:
    st.session_state.last_user_input = user_input  # Track last input to avoid duplicates

    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_metadata.append({})  # No metadata for user

    with st.spinner("Generating response..."):
        response, perplexity_dict = model.forward(user_input, max_tokens, temp=temperature)

    st.session_state.chat_history.append(("Bot", str(response)))  # Ensure string
    st.session_state.chat_metadata.append(perplexity_dict)

# Display chat
for idx, (sender, message) in enumerate(st.session_state.chat_history):
    if sender == "You":
        st.markdown(f"**🧍‍♂️ {sender}:** {message}")
    else:
        st.markdown(f"**🤖 {sender}:** {message}")
        metadata = st.session_state.chat_metadata[idx]
        if metadata:
            with st.expander("check perplexity"):
                for key, value in metadata.items():
                    st.markdown(f"<span style='color:red; font-size: 10px;'>... + {key}: {value}</span>",
                                unsafe_allow_html=True)
