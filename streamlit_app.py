import streamlit as st
import pandas as pd

st.title("AG News Dataset Viewer")

@st.cache_data
def load_agnews_csv():
    url = "https://drive.google.com/uc?id=1xr-eyagU6GeZlYpn8qGIuMSdK5WFUV5x"
    return pd.read_csv(url)

df = load_agnews_csv()

st.write("### AG News Dataset (Train Split)")
st.write("Showing a preview of the data:")
num_rows = st.slider("Number of rows to display", 5, 100, 10)
st.dataframe(df.head(num_rows))
