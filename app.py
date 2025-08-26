import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("logistic_model.pkl")

st.title("🚢 Titanic Survival Prediction App")

st.write("Enter passenger details to predict survival:")

# Inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0, max_value=100, value=30)
sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=32.0)
embarked = st.selectbox("Embarked", ["C", "Q", "S"])

# Convert inputs to match training encoding
sex_num = 1 if sex == "female" else 0
embarked_map = {"C": 0, "Q": 1, "S": 2}
embarked_num = embarked_map[embarked]

input_df = pd.DataFrame({
    "Pclass": [pclass],
    "Sex": [sex_num],
    "Age": [age],
    "SibSp": [sibsp],
    "Parch": [parch],
    "Fare": [fare],
    "Embarked": [embarked_num]
})

# Prediction
if st.button("Predict"):
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    st.subheader("Result")
    if pred == 1:
        st.success(f"✅ Passenger likely SURVIVES (Probability: {proba:.2%})")
    else:
        st.error(f"❌ Passenger likely does NOT survive (Probability: {proba:.2%})")
