🎬 IMDB Sentiment Classifier

Trains Logistic Regression and Multinomial Naive Bayes models on IMDB movie reviews to classify sentiments as positive or negative.
Performs text cleaning, TF-IDF vectorization, model training, and evaluation with both console output and a saved report.

⸻

📚 Overview

This script performs binary sentiment classification using two classic NLP models:
	•	Logistic Regression — strong linear baseline for text classification
	•	Multinomial Naive Bayes — fast probabilistic model well-suited for word frequency features

It demonstrates how text preprocessing, TF-IDF feature extraction, and model evaluation can be integrated into a reproducible and documented workflow.

⸻

⚙️ Requirements

pip install pandas numpy scikit-learn


⸻

🧠 Features
	•	✅ Text cleaning (HTML & punctuation removal)
	•	✅ TF-IDF vectorization (20k features, unigrams + bigrams)
	•	✅ Trains Logistic Regression and MultinomialNB models
	•	✅ Evaluates models with accuracy, precision, recall, F1, and confusion matrix
	•	✅ Prints random sample reviews with sentiment emojis
	•	✅ Saves evaluation results to a text report (reports/evaluation.txt)

⸻

🗂️ Project Structure

.
├── data/
│   └── IMDB Dataset.csv
├── reports/
│   └── evaluation.txt
└── imdb_sentiment_classifier.py


⸻

🚀 Usage
	1.	Download the IMDB dataset → data/IMDB Dataset.csv
	2.	Run the script:

python imdb_sentiment_classifier.py


	3.	After running:
	•	The script will print random reviews with their predicted sentiment
	•	Evaluation results for both models will be saved under
reports/evaluation.txt

⸻

📊 Example Output

😊 Positive -> Loved the performances!
😠 Negative -> The movie dragged forever...

=== Logistic Regression ===
Accuracy: 0.8842

Classification Report:
              precision    recall  f1-score   support
0             0.88        0.89      0.88      12500
1             0.89        0.88      0.88      12500

Confusion Matrix:
[[11100 1400]
 [1550 10950]]

=== Multinomial Naive Bayes ===
Accuracy: 0.8715

Classification Report:
              precision    recall  f1-score   support
0             0.87        0.87      0.87      12500
1             0.87        0.87      0.87      12500

Confusion Matrix:
[[10875 1625]
 [1610 10890]]


⸻

🧩 Next Steps
	•	Add cross-validation and hyperparameter tuning
	•	Automate model comparison with saved summaries
	•	Extend evaluation metrics (ROC-AUC, precision-recall curve)
	•	Explore alternative models (SVM, RandomForest, or deep learning approaches)
