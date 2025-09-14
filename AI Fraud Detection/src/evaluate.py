# src/evaluate.py

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from preprocess import clean_text  # make sure this function exists in preprocess.py

# -------- load dataset --------
data_path = "../data/spam.csv"
df = pd.read_csv(data_path)
text_df = df[df['channel'] == 'text'].copy()
# df['channel'] == 'text' filter the rows where column chanel = 'text' and .copy is used to avoid SettitngWithCopyWarning in pandas
# Clean the messages
text_df['clean_content'] = text_df['content'].apply(clean_text)

# Features and Labels
X = text_df['clean_content']
y = text_df['label']

# ---------------------------
# Step 2: Load Trained Model
# ---------------------------
model_path = "../models/scam_classifier.pkl"
model = joblib.load(model_path)

# ---------------------------
# Step 3: Evaluate Model
# ---------------------------
# Predict on the dataset
y_pred = model.predict(X)

# Accuracy
accuracy = accuracy_score(y, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Detailed Classification Report
print("\nClassification Report:")
print(classification_report(y, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y, y_pred))
