import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def clean_text(text):
    """
    Cleans the input text by removing URLs, special characters, and stop words.Args:
        text (str): The input text to be cleaned."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)