import streamlit as st
import pandas as pd

st.set_page_config(page_title="AG News Classifier", layout="wide")
st.title("📰 AG News Headline Classifier")

# Load and preview the dataset
@st.cache_data
def load_agnews_csv():
    url = "https://drive.google.com/uc?id=1xr-eyagU6GeZlYpn8qGIuMSdK5WFUV5x"
    return pd.read_csv(url)

df = load_agnews_csv()
df.dropna(inplace=True)

st.subheader("📄 Preview the Dataset")
num_rows = st.slider("Select number of rows to display", 5, 100, 10)
st.dataframe(df.head(num_rows))

# Divider
st.markdown("---")

# User Input Interface for Prediction
st.subheader("🤖 Try It Yourself: Headline Classification")

headline = st.text_input("Enter a news headline or snippet:")

user_guess = st.radio(
    "What category do you think it belongs to?",
    ["Select", "World", "Sports", "Business", "Sci/Tech"],
    index=0
)

user_confidence = st.slider("How confident are you in your guess?", 1, 5, 3)

# Placeholder for future model prediction
if headline:
    st.write("### 🔍 Prediction Result:")
    st.info("Model prediction will be shown here once connected.")

    # Optional feedback if user made a guess
    if user_guess != "Select":
        st.write(f"Your guess: **{user_guess}** with confidence level **{user_confidence}/5**")

