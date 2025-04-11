import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np

# Set up the app title
st.title("AG News Classifier with BERT-Tiny")
st.write("This app classifies news articles into 4 categories using a fine-tuned BERT model")

# Load model and tokenizer (with caching)
@st.cache_resource
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained("google/bert_uncased_L-2_H-128_A-2", num_labels=4)
    model.load_state_dict(torch.load('best_model1.pth', map_location=torch.device('cpu')))
    return model

@st.cache_resource
def load_tokenizer():
    return AutoTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2")

model = load_model()
tokenizer = load_tokenizer()

# Define class labels
class_labels = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech"
}

# Create text input area
user_input = st.text_area("Enter a news article to classify:", "The stock market reached record highs today...")

# Add a button to trigger classification
if st.button("Classify"):
    if user_input:
        # Preprocess and tokenize the input
        inputs = tokenizer(
            user_input,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors="pt"
        )
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get probabilities and predicted class
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        top_prob, top_class = torch.max(probs, dim=1)
        
        # Display results
        st.subheader("Prediction Results")
        st.write(f"**Predicted Class:** {class_labels[top_class.item()]}")
        st.write(f"**Confidence:** {top_prob.item():.2%}")
        
        # Show probabilities for all classes
        st.subheader("Class Probabilities")
        for i, prob in enumerate(probs.squeeze()):
            st.write(f"{class_labels[i]}: {prob:.2%}")
    else:
        st.warning("Please enter some text to classify.")
