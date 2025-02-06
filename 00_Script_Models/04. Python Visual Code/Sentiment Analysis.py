positive_words = ["happy", "good", "amazing"]
negative_words = ["sad", "bad", "terrible"]

def sentiment_score(text):
    positive_count = sum(text.count(word) for word in positive_words)
    negative_count = sum(text.count(word) for word in negative_words)
    return positive_count - negative_count

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

# Assuming a labeled dataset is available
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

sentiment_score = model.predict(["happy day"])[0]

from transformers import pipeline

sentiment_analysis = pipeline("sentiment-analysis")
sentiment_score = sentiment_analysis("happy day")[0]["label"]