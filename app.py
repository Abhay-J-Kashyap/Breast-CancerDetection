
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

device = torch.device("cpu")

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

input_size = 30
hidden_size = 64
output_size = 1

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = data.data
scaler = StandardScaler()
scaler.fit(X)

st.title("🧠 Breast Cancer Detection")
st.write("Upload a CSV file with 30 features to predict benign (0) or malignant (1).")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if df.shape[1] != 30:
        st.error("CSV file must have exactly 30 features (no target column).")
    else:
        st.write("Data preview:")
        st.dataframe(df.head())

        inputs = scaler.transform(df.values)
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32)

        with torch.no_grad():
            outputs = model(inputs_tensor)
            predictions = outputs.round().numpy().astype(int).flatten()

        result_df = df.copy()
        result_df["Prediction"] = predictions
        result_df["Prediction"] = result_df["Prediction"].map({0: "Benign", 1: "Malignant"})
        st.subheader("Prediction Results")
        st.write(result_df)
