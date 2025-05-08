import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Simulate data
np.random.seed(42)
n = 500
df = pd.DataFrame({
    "LegalOrder_Spatial": np.random.rand(n),
    "LegalOrder_Temporal": np.random.rand(n),
    "Memory": np.random.rand(n),
    "Outrage": np.random.rand(n),
    "Democracy": np.random.rand(n)
})
df["RevisionFunction"] = 0.4 * df["Memory"] + 0.3 * df["Outrage"] + 0.3 * df["Democracy"] + np.random.normal(0, 0.05, n)
df["LegalOrderFunction"] = 0.6 * df["LegalOrder_Spatial"] + 0.4 * df["LegalOrder_Temporal"] + np.random.normal(0, 0.05, n)
df["PeaceScore"] = (
    0.5 * df["LegalOrderFunction"] +
    0.5 * df["RevisionFunction"] +
    0.2 * df["LegalOrderFunction"] * df["RevisionFunction"]
).clip(0, 1)

# Train simple NN
features = ["LegalOrder_Spatial", "LegalOrder_Temporal", "Memory", "Outrage", "Democracy"]
X = df[features].values
y = df["PeaceScore"].values.reshape(-1, 1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

class PeaceNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

model = PeaceNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()
for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    loss = loss_fn(model(X_tensor), y_tensor)
    loss.backward()
    optimizer.step()

# Streamlit interface
st.title("🕊️ Peace Score Simulator")
st.sidebar.markdown("Adjust the input values to see the predicted Peace Score.")

inputs = {
    "LegalOrder_Spatial": st.sidebar.slider("Legal Order (Spatial)", 0.0, 1.0, 0.5, 0.01),
    "LegalOrder_Temporal": st.sidebar.slider("Legal Order (Temporal)", 0.0, 1.0, 0.5, 0.01),
    "Memory": st.sidebar.slider("Collective Memory", 0.0, 1.0, 0.5, 0.01),
    "Outrage": st.sidebar.slider("Moral Outrage", 0.0, 1.0, 0.5, 0.01),
    "Democracy": st.sidebar.slider("Democratic Struggle", 0.0, 1.0, 0.5, 0.01),
}

input_array = np.array([list(inputs.values())])
input_scaled = scaler.transform(input_array)
input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

model.eval()
with torch.no_grad():
    score = model(input_tensor).item()

st.metric("Predicted Peace Score", f"{score:.3f}")
st.bar_chart(pd.DataFrame({"Peace Score": [score]}, index=["Predicted"]))
