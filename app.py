import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# ========== Train the model inside app ==========
# Load dataset
train_df = pd.read_csv("Titanic_train.csv")

# Fill missing values
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)
train_df['Fare'].fillna(train_df['Fare'].median(), inplace=True)
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)

# Drop unused columns
train_df.drop(columns=['Cabin', 'Name', 'Ticket', 'PassengerId'], inplace=True, errors="ignore")

# Encode categorical
train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})
train_df = pd.get_dummies(train_df, columns=['Embarked'], drop_first=True)

# Features & target
X = train_df.drop("Survived", axis=1)
y = train_df["Survived"]

# Handle missing values
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

# Train model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_imputed, y)

# ========== Streamlit UI ==========
st.title("🚢 Titanic Survival Prediction App")
st.write("Enter passenger details to predict survival:")

# User inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0, max_value=100, value=30)
sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=32.0)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Convert categorical inputs
sex_num = 0 if sex == "male" else 1
embarked_map = {"C": [1,0], "Q": [0,1], "S": [0,0]}  
embarked_C, embarked_Q = embarked_map[embarked]

# Build input dataframe (must match training columns!)
input_df = pd.DataFrame([[
    pclass, sex_num, age, sibsp, parch, fare, embarked_C, embarked_Q
]], columns=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked_C", "Embarked_Q"])

# Prediction
if st.button("Predict"):
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    st.subheader("Result")
    if pred == 1:
        st.success(f"✅ Passenger likely SURVIVES (Probability: {proba:.2%})")
    else:
        st.error(f"❌ Passenger likely does NOT survive (Probability: {proba:.2%})")
