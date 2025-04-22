import streamlit as st
import os
import tempfile
import pickle
from src.model import NGram


st.set_page_config(page_title="Train", page_icon="📊", layout="wide")
st.title("📊 Train Your Cascade N-Gram Model")

# Upload section
uploaded_file = st.file_uploader("📄 Upload a text file for training", type=["txt"])

# Show context window input only if file is uploaded
if uploaded_file is not None:
    st.success("✅ File uploaded successfully!")

    # Context window input
    context_window = st.number_input("🧠 Set Context Window Size (Max: 50)", min_value=1, value=10)

    if context_window > 50:
        st.warning("⚠️ Context window cannot be greater than 50. Using default value: 50")
        context_window = 50
    ngram = NGram(context_window)

    # Train button
    if st.button("🚀 Start Training"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
            temp_path = temp_file.name
            temp_file.write(uploaded_file.getvalue())

        with st.spinner("🛠️ Training in progress..."):
            # print(temp_path, context_window)  # Pass context window to your function
            ngram.train(temp_path)

        os.remove(temp_path)  # Clean up
        with open("trained_model.pkl", "wb") as f:
            pickle.dump(ngram, f)
        st.success("✅ Training complete! Your model is now live.")
