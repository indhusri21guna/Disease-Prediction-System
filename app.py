import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import streamlit as st
import matplotlib.pyplot as plt

# Load dataset
dataset = pd.read_csv("health_care.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# Initialize and train the Decision Tree Classifier
classifier = DecisionTreeClassifier(random_state=0)
classifier.fit(x_train, y_train)

# Evaluate model accuracy
y_pred = classifier.predict(x_test)
accuracy = metrics.accuracy_score(y_test, y_pred)

# Streamlit User Interface
st.title("Healthcare Diagnostic Tool")
st.write("This tool predicts the health condition of a patient based on medical parameters.")

# Show model accuracy
st.subheader("Model Accuracy")
st.write(f"The Decision Tree Classifier achieved an accuracy of **{accuracy:.2f}** on the test data.")

# Feature importance visualization
st.subheader("Feature Importance")
feature_importances = classifier.feature_importances_
plt.figure(figsize=(10, 6))
plt.bar(dataset.columns[:-1], feature_importances, color='skyblue')
plt.xticks(rotation=45, ha="right")
plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance Score")
st.pyplot(plt)

# Input fields
st.subheader("Enter Patient Parameters")
user_inputs = []
feature_names = dataset.columns[:-1]
for feature in feature_names:
    value = st.number_input(f"Enter {feature}", value=0.0)
    user_inputs.append(value)

# Prediction button
if st.button("Predict Health Condition"):
    try:
        # Predict health condition
        prediction = classifier.predict([user_inputs])[0]
        st.success(f"The predicted health condition is: **{prediction}**")
        
        # Show additional information
        if prediction == "Anemia":
            st.write("🔴 You may be an Anemia patient. Consider consulting a doctor.")
        elif prediction == "Diabetes":
            st.write("⚠️ You may be diabetic. Please monitor your blood sugar levels.")
        elif prediction == "Heart Disease":
            st.write("💔 You may have heart-related issues. Immediate medical attention is advised.")
        elif prediction == "Thalassemia":
            st.write("🩸 You may be Thalassemic. Consider seeking specialized treatment.")
        elif prediction == "Thrombocytopenia":
            st.write("🔴 You may have low platelet count. Seek medical advice.")
        elif prediction == "Healthy":
            st.write("✅ You seem to be healthy. Keep up with regular checkups!")
    except Exception as e:
        st.error(f"An error occurred: {e}")
