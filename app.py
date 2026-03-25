import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap

st.title("ML Project with XAI")

# Load data
df = pd.read_csv("data.csv")

# EDA
st.subheader("EDA")
st.write(df.describe())

# Model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predictions
pred = model.predict(X_test)
st.write("Predictions:", pred)

# Feature Importance
st.subheader("Feature Importance")
importances = model.feature_importances_
st.bar_chart(importances)

# SHAP
st.subheader("SHAP Values")
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

fig = shap.plots.bar(shap_values, show=False)
st.pyplot(fig)