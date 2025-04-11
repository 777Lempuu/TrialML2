import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Set up the page
st.set_page_config(page_title="AG News Classifier", layout="wide")
st.title("📰 AG News Dataset Viewer & Predictor")

# Load the dataset
@st.cache_data
def load_agnews_csv():
    url = "https://drive.google.com/uc?id=1xr-eyagU6GeZlYpn8qGIuMSdK5WFUV5x"
    return pd.read_csv(url)

df = load_agnews_csv()
df.dropna(inplace=True)


# Data preview
st.subheader("📄 Dataset Preview")
num_rows = st.slider("Select number of rows to display", 5, 100, 10)
st.dataframe(df.head(num_rows))

# Visualization
st.subheader("📊 Class Distribution")
fig, ax = plt.subplots()
df['Category'].value_counts().plot(kind='bar', color='lightgreen', ax=ax)
ax.set_xlabel("Category")
ax.set_ylabel("Number of Articles")
ax.set_title("Distribution of News Categories")
st.pyplot(fig)

# Divider
st.markdown("---")

# User Input Section
st.subheader("🤖 Try It Yourself: Headline Classification")
headline = st.text_input("Enter a news headline or short snippet for prediction:")

# Optional user guess
user_guess = st.radio(
    "What do you think the category is?",
    ["Select", "World", "Sports", "Business", "Sci/Tech"],
    index=0
)

# Optional confidence
confidence = st.slider("How confident are you in your guess?", 1, 5, 3)

# Placeholder prediction section
if headline:
    st.write("### 🔍 Prediction Result:")
    
    # When you load your ReFixMatch model later, replace the following:
    # ---- Dummy Example (Replace this with real model prediction) ----
    predicted_label = "Business"
    prediction_confidence = {
        "World": 0.05,
        "Sports": 0.10,
        "Business": 0.70,
        "Sci/Tech": 0.15
    }
    # ------------------------------------------------------------------

    st.success(f"Predicted Category: **{predicted_label}**")
    
    # Confidence bar chart
    st.write("Model Confidence:")
    st.bar_chart(pd.Series(prediction_confidence))

    # Feedback to user
    if user_guess != "Select":
        if user_guess == predicted_label:
            st.info("✅ Your guess matched the model!")
        else:
            st.warning(f"❌ Your guess was **{user_guess}**, but the model predicted **{predicted_label}**.")
