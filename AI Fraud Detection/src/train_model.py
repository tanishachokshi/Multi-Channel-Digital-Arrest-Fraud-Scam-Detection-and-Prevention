import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
from preprocess import clean_text

# Load dataset
df = pd.read_csv('../data/spam.csv')

# Use only text channel for initial model
text_df = df[df['channel'] == 'text'].copy()
text_df['cleaned'] = text_df['content'].apply(clean_text)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    text_df['cleaned'], text_df['label'], test_size=0.2, random_state=42
)

# Vectorize text
vectorizer = CountVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)

# Train Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_vect, y_train)

# Save model and vectorizer
joblib.dump(model, '../models/scam_classifier.pkl')
joblib.dump(vectorizer, '../models/vectorizer.pkl')

print("Model and vectorizer saved in 'models/' folder")