import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

st.title("⚡ Solar Energy ML Predictor")

# Upload file
uploaded_file = st.file_uploader("Upload CSV file")

if uploaded_file is not None:
    # Read data
    data = pd.read_csv(uploaded_file)

    # CLEAN DATA
    if "Date and time" in data.columns:
        data = data.drop(columns=["Date and time"])

    # remove unnamed columns
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

    # remove empty rows
    data = data.dropna()

    # Show dataset
    st.write("### Dataset Preview")
    st.write(data.head())

    # Select columns
    all_columns = data.columns.tolist()

    target_column = st.selectbox("Select target column", all_columns)

    feature_columns = st.multiselect(
        "Select feature columns",
        [c for c in all_columns if c != target_column]
    )

    # Validation
    if len(feature_columns) == 0:
        st.warning("Please select at least one feature column.")
        st.stop()

    # Define X and y
    X = data[feature_columns]
    y = data[target_column]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model selection
    model_choice = st.selectbox(
        "Choose ML model",
        ["Linear Regression", "Random Forest", "Decision Tree"]
    )

    if model_choice == "Linear Regression":
        model = LinearRegression()
    elif model_choice == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model = DecisionTreeRegressor(random_state=42)

    # Train model
    if st.button("Train Model"):
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.success("Model trained successfully!")

        st.write("### Model Performance")
        st.write(f"MSE: {mse:.2f}")
        st.write(f"R2 Score: {r2:.2f}")

        # Prediction
        st.write("### Make Prediction")

        input_data = []
        for col in feature_columns:
            val = st.number_input(f"Enter {col}", value=0.0)
            input_data.append(val)

        if st.button("Predict"):
            prediction = model.predict([input_data])
            st.success(f"Prediction: {prediction[0]:.2f}")