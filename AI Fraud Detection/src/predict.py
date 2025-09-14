import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

#  ------ Load Data ------
print("Starting predict.py...")
df = pd.read_csv("../data/spam.csv")
print("Dataset loaded. Number of rows:", len(df))
x = df["content"] # message text
y = df["label"] # the label eg: spam or ham

# ------ Split Data ------
X_train, X_test ,y_train, y_test = train_test_split(x,y, test_size =0.2, random_state = 42) # 80% training and 20% test


# ------ Vectorization ------
vectorizer = TfidfVectorizer(stop_words="english") # removes stop words like a,an,the etc.

X_train_tfidf = vectorizer.fit_transform(X_train) # 
X_test_tfidf = vectorizer.transform(X_test)  # applies same transformation to test data.

# ------ Train Classifier -------

model = LogisticRegression(max_iter=1000)  #gives the probablities for each class and then assigns the class with higher probablity
model.fit(X_train_tfidf,y_train) # model learn from data

# ------ Make Predictions ------
y_pred = model.predict(X_test_tfidf)  #This function uses the trained Logistic Regression model to predict labels for new data. X_test_tfidf is the TF-IDF transformed feature matrix for the test dataset. (messages converted into vectors). 
print("Accuracy:", accuracy_score(y_test,y_pred)) # y_test = true labes for the set and y_pred = predicted labels for the set.
print("\n Classification Report:\n", classification_report(y_test,y_pred)) # gives precision, recall, f1-score for each class

# ------ Save Model and Vectorizer ------
joblib.dump(model, "spam_classifier_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl") #saves the vectorizer object to a file named "vectorizer.pkl" using joblib.dump. This allows you to reuse the same vectorization process later without needing to retrain or redefine the vectorizer.

# ------ predict new message ------
def predict_message(message):
    loaded_model = joblib.load("spam_classifier_model.pkl")
    loaded_vectorizer = joblib.load("vectorizer.pkl")
    
    message_tfidf = loaded_vectorizer.transform([message]) # transform the message to tfidf vector
    prediction = loaded_model.predict(message_tfidf) [0]# predict the label
    return prediction

if __name__ == "__main__":
    print("Running sample prediction...")
    sample = "Your account is blocked. Click here to unlock."
    print(f"Message : {sample}")
    print('Prediction :',predict_message("Congratulations! You won a free iPhone. Click here to claim your prize."))