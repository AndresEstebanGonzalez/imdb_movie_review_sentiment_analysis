# 🎬 IMDB Sentiment Classifier

Trains **Logistic Regression** and **Multinomial Naive Bayes** models on IMDB movie reviews to classify sentiments as *positive* or *negative*.  
Includes text cleaning, TF-IDF vectorization, and model evaluation metrics.

---

## 📚 Overview
This script performs binary sentiment classification using two models:
- **Logistic Regression**
- **Multinomial Naive Bayes**

It demonstrates how text preprocessing and feature extraction can be combined for quick experimentation in NLP classification tasks.

---

## ⚙️ Requirements
pip install pandas scikit-learn

---

## 🧠 Features
- Cleans HTML tags and punctuation  
- Applies TF-IDF vectorization (unigrams + bigrams, 20k features)  
- Trains both Logistic Regression and MultinomialNB  
- Reports accuracy, classification report, and confusion matrix  
- Prints a few random samples with sentiment emoji  

---

## 🚀 Usage
1. Place the IMDB dataset under `data/IMDB Dataset.csv`
2. Run:
   python imdb_sentiment_classifier.py

---

## 📊 Example Output
😊 Positive -> Loved the performances!
😠 Negative -> The movie dragged forever...

Accuracy Logistic Regression: 0.8842
Classification report:
              precision    recall  f1-score   support
0             0.88        0.89      0.88      12500
1             0.89        0.88      0.88      12500
Confusion matrix: [[11100, 1400], [1550, 10950]]

---

## 🧩 Next Steps
- Add hyperparameter tuning
- Integrate grid search or random search
- Try other models (SVM, RandomForest)
