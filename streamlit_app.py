import streamlit as st
import pandas as pd
import torch
from joblib import load
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(page_title="AG News Classifier [ReFixMatch method]", layout="wide")
st.title("📰 AG News Headline Classifier")

# Load model and tokenizer
@st.cache_resource
def load_model():
    try:
        device = torch.device('cpu')
        
        # Use context manager to allow the tokenizer class
        with torch.serialization.safe_globals([type(AutoTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2"))]):
            model_data = torch.load('ag_news_model.pt', 
                                  map_location=device,
                                  weights_only=False)  # Required for custom classes
            
        # Recreate model architecture
        model = AutoModelForSequenceClassification.from_pretrained(
            "google/bert_uncased_L-2_H-128_A-2",
            num_labels=4
        )
        model.load_state_dict(model_data['model_state_dict'])
        model.to(device)
        
        tokenizer = model_data['tokenizer']
        st.success("✅ Model loaded successfully!")
        return model, tokenizer
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None

model, tokenizer = load_model()
if model:
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
    try:
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
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None

# [Keep your existing dataset loading code...]

# User Input Interface for Prediction
st.subheader("🤖 Try It Yourself: Headline Classification")

headline = st.text_input("Enter a news headline or snippet:")

if headline and model:  # Only show if we have both input and model
    st.write("### 🔍 Prediction Result:")
    
    # Get model prediction
    pred_class, confidence = predict(headline)
    
    if pred_class:  # Only show if prediction succeeded
        # Display prediction
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Category", pred_class)
        with col2:
            st.metric("Confidence", f"{confidence:.1%}")
