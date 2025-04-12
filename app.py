import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
import os
import plotly.express as px

# ---- MODEL DEFINITION ----
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# ---- MODEL LOADING ----
@st.cache_resource
def load_model():
    if not os.path.exists("model.pth"):
        st.error("Model file not found. Please make sure `model.pth` is in the same directory.")
        st.stop()

    input_size = 30
    hidden_size = 64
    output_size = 1
    model = NeuralNet(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# ---- SCALER ----
data = load_breast_cancer()
scaler = StandardScaler()
scaler.fit(data.data)

# ---- STREAMLIT UI ----
st.set_page_config(page_title="Breast Cancer Classifier", layout="centered")
st.title("🩺 Breast Cancer Detection")

# --- PASSWORD CHECK ---
PASSWORD = "abhay2310"  
password = st.text_input("🔒 Enter access password", type="password")

if password != PASSWORD:
    st.warning("Please enter the correct password to access the app.")
    st.stop()

# ---- SIDEBAR EXTRAS ----
# Feature explanation
with st.sidebar.expander("📚 Feature Info"):
    st.write("Each column represents a feature from the breast cancer dataset. For example:")
    st.markdown("""
    - `mean radius`: average size of the nucleus
    - `mean texture`: standard deviation of gray-scale values
    - `mean smoothness`: smoothness of cell borders
    - ...
    """)

# Sample CSV download
sample_df = pd.DataFrame(data.data[:5], columns=data.feature_names)
csv_template = sample_df.to_csv(index=False).encode("utf-8")
with st.sidebar:
    st.download_button("📄 Download Sample CSV", data=csv_template, file_name="sample_template.csv", mime="text/csv")

# ---- FILE UPLOAD ----
st.write("Upload a **CSV** file with 30 numerical features (from breast cancer dataset) to predict if samples are **Malignant** or **Benign**.")

uploaded_file = st.file_uploader("📁 Upload CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        if df.shape[1] != 30:
            st.error(f"Expected 30 features, but got {df.shape[1]}. Please check your CSV.")
        else:
            st.success("✅ File uploaded successfully!")
            st.dataframe(df.head())

            # Preprocess and predict
            inputs = scaler.transform(df.values)
            inputs_tensor = torch.tensor(inputs, dtype=torch.float32)

            with torch.no_grad():
                outputs = model(inputs_tensor)
                predictions = outputs.round().numpy().astype(int).flatten()

            df["Prediction"] = ["Malignant" if p == 1 else "Benign" for p in predictions]

            # Pie chart of results
            counts = df["Prediction"].value_counts().reset_index()
            counts.columns = ["Label", "Count"]

            fig = px.pie(
                counts,
                names="Label",
                values="Count",
                hole=0.3,
                hover_data=["Count"],
                title="🔬 Prediction Distribution",
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            st.plotly_chart(fig)

            # Results table
            st.subheader("🧾 Prediction Results")
            st.dataframe(df[["Prediction"]])

            # Download results
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Results as CSV",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Awaiting file upload...")
