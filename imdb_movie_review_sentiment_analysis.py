"""
IMDB Sentiment Classifier

Trains Logistic Regression and Multinomial Naive Bayes models on IMDB movie reviews
to classify sentiments as positive or negative. Includes data cleaning, TF-IDF
vectorization, training, and evaluation with accuracy, classification report,
and confusion matrix outputs.
"""

#Import RE
import re
#Import Pandas
import pandas as pd
#Import Numpy
import numpy as np
#Import split
from sklearn.model_selection import train_test_split
#Import model
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
#Import vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
#Import metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


#Import data
DATA_PATH = "data/IMDB Dataset.csv"
imdb_df = pd.read_csv(DATA_PATH, encoding="latin-1")
#Convert Positive to 1 & Negative to 0
imdb_df["sentiment"] = imdb_df["sentiment"].map({"negative":0, "positive":1})
#Clean review

def clean_review(text):
    '''Lowercase, remove HTML tags and extra punctuation.'''
    text = text.strip()
    text = re.sub(r"<.*?>", "", text)  # removes any HTML tag, not just <br />
    text = re.sub(r"[^a-z0-9'\s]", "", text)  # removes punctuation and symbols
    return text

imdb_df["clean_review"] = imdb_df["review"].apply(clean_review)
#Split data
X_train, X_test, y_train, y_test = train_test_split(
    imdb_df["clean_review"],
    imdb_df["sentiment"],
    test_size=0.2,
    random_state=1,
    stratify=imdb_df["sentiment"]
)
#Vectorize
vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1,2),
    min_df=2,
    max_df=0.9,
    stop_words="english",
    lowercase=True,
    sublinear_tf=True
)
vectorizer.fit(X_train)
X_train_tfidf = vectorizer.transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
#Logistic Regression model
logreg = LogisticRegression(max_iter=2000, random_state=1, solver="liblinear", C=1.0)
logreg.fit(X_train_tfidf, y_train)
logreg_prediction = logreg.predict(X_test_tfidf)
#Multinomial Naive Bayes model
mnnb_model = MultinomialNB(alpha=0.5, fit_prior=True)
mnnb_model.fit(X_train_tfidf, y_train)
mnnb_model_prediction = mnnb_model.predict(X_test_tfidf)

#Print emoji vs sentiment
PRINT_SAMPLE = True
N = 3
if PRINT_SAMPLE:
    sample = imdb_df.sample(N, random_state=1)
    for review, sentiment in zip(sample["review"], sample["sentiment"]):
        SENTIMENT_EMOJI = "😊 Positive" if sentiment == 1 else "😠 Negative"
        print(f"{SENTIMENT_EMOJI} -> {review}")

#Evaluate models
logreg_accuracy = accuracy_score(y_test, logreg_prediction)
logreg_report = classification_report(y_test, logreg_prediction)
logreg_confusion  = confusion_matrix(y_test, logreg_prediction)

mnnb_accuracy = accuracy_score(y_test, mnnb_model_prediction)
mnnb_report = classification_report(y_test, mnnb_model_prediction)
mnnb_confusion  = confusion_matrix(y_test, mnnb_model_prediction)

#Save text report
with open("reports/evaluation.txt", "w", encoding="utf-8") as f:
    f.write("=== Logistic Regression ===\n")
    f.write(f"Accuracy: {logreg_accuracy:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(logreg_report + "\n")
    f.write("Confusion Matrix:\n")
    f.write(np.array2string(logreg_confusion) + "\n\n")

    f.write("=== Multinomial Naive Bayes ===\n")
    f.write(f"Accuracy: {mnnb_accuracy:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(mnnb_report + "\n")
    f.write("Confusion Matrix:\n")
    f.write(np.array2string(mnnb_confusion) + "\n")