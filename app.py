import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("bissell.csv") 
    return df

data = load_data()
st.title("ML Model Trainer")
st.write("Dataset preview:", data.head())

# --- Select Features and Target ---
all_columns = data.columns.tolist()
target_column = st.selectbox("Select target column", all_columns)
feature_columns = st.multiselect("Select feature columns", [c for c in all_columns if c != target_column])

if len(feature_columns) == 0:
    st.warning("Please select at least one feature column.")
    st.stop()
# Remove date column (important)
if "Date and time" in data.columns:
    data = data.drop(columns=["Date and time"])

# Keep only numbers
data = data.select_dtypes(include=['number'])

# Remove empty values
data = data.dropna()
X = data[feature_columns]
y = data[target_column]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Choose Model ---
model_choice = st.selectbox("Choose ML model", ["Linear Regression", "Random Forest", "Decision Tree"])

if model_choice == "Linear Regression":
    model = LinearRegression()
elif model_choice == "Random Forest":
    model = RandomForestRegressor(n_estimators=100, random_state=42)
else:
    model = DecisionTreeRegressor(random_state=42)

# --- Train Model ---
if st.button("Train Model"):
    model.fit(X_train, y_train)
    st.success(f"{model_choice} trained successfully!")

    # --- Evaluate Model ---
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write(f"**Mean Squared Error:** {mse:.2f}")
    st.write(f"**R2 Score:** {r2:.2f}")
