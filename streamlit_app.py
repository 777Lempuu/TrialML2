import streamlit as st
import pandas as pd
import torch
from joblib import load
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(page_title="AG News Classifier", layout="wide")
st.title("📰 AG News Headline Classifier")

# Load model and tokenizer
@st.cache_resource
def load_model():
    model_data = load('ag_news_model.pkl')
    model = model_data['model']
    tokenizer = model_data['tokenizer']
    return model, tokenizer

model, tokenizer = load_model()
model.eval()  # Set to evaluation mode

# Class labels mapping
class_labels = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech"
}

# Prediction function
def predict(text):
    inputs = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(**inputs).logits
    probs = torch.nn.functional.softmax(outputs, dim=1)
    pred_class = torch.argmax(probs).item()
    confidence = torch.max(probs).item()
    return class_labels[pred_class], confidence

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

# Prediction section
if headline:
    st.write("### 🔍 Prediction Result:")
    
    # Get model prediction
    pred_class, confidence = predict(headline)
    
    # Display prediction
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted Category", pred_class)
    with col2:
        st.metric("Confidence", f"{confidence:.1%}")
    
    # Optional feedback if user made a guess
    if user_guess != "Select":
        st.write(f"Your guess: **{user_guess}** with confidence level **{user_confidence}/5**")
        if user_guess == pred_class:
            st.success("🎯 You matched the model's prediction!")
        else:
            st.warning("🤔 Your guess didn't match the model's prediction")

