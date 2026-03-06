🍽️ Restaurant Review Sentiment Analysis

A Natural Language Processing (NLP) project that classifies restaurant reviews as positive or negative using machine learning.
The project applies text preprocessing, TF-IDF feature extraction, and Logistic Regression to predict sentiment from customer reviews.

This implementation uses scikit-learn and NLTK for text processing and model training.

📌 Project Overview

Customer reviews contain valuable insights about service and food quality.
This project builds a sentiment classification model that:

Cleans and preprocesses raw text reviews

Converts text into numerical features

Trains a machine learning model

Evaluates performance

Predicts sentiment for new reviews

The system automatically determines whether a review is Positive or Negative.

⚙️ Features

✔ Text preprocessing using lemmatization
✔ Stopword removal using NLTK
✔ TF-IDF vectorization with bi-grams
✔ Hyperparameter tuning for Logistic Regression
✔ Performance evaluation using:

Accuracy

Precision

Recall

Confusion Matrix

✔ Custom function to predict sentiment for new reviews

🧠 Machine Learning Pipeline
1️⃣ Data Loading

The dataset is loaded from:

Restaurant_Reviews.tsv

The dataset contains:

Column	Description
Review	Customer review text
Liked	Sentiment label (1 = Positive, 0 = Negative)
2️⃣ Text Preprocessing

The text is cleaned using:

Lowercasing

Removing punctuation

Removing stopwords

Lemmatization

Example:

Original:
"The food was absolutely wonderful!"

Processed:
food absolutely wonderful
3️⃣ Feature Extraction

Text is converted to numerical vectors using:

TF-IDF Vectorization

Key settings:

max_features = 2500
ngram_range = (1,2)

This captures:

individual words (unigrams)

word pairs (bigrams)

Example:

very good
not good
bad service
4️⃣ Model Training

The model used:

Logistic Regression

Parameters used:

class_weight = balanced
max_iter = 1000

Multiple values of C (regularization strength) are tested:

0.1
1.0
10.0

The best performing model is selected automatically.

5️⃣ Model Evaluation

Metrics used:

Accuracy

Precision

Recall

Confusion Matrix

Example confusion matrix:

[[90 10]
 [15 85]]
📂 Project Structure
Restaurant-Sentiment-Analysis
│
├── Restaurant_Reviews.tsv
├── sentiment_analysis.py
├── README.md
🚀 Installation

Clone the repository:

git clone https://github.com/Sriramreddymusukula/Restraunt-sentimental-analysis/tree/main?tab=readme-ov-file
cd restaurant-sentiment-analysis

Install dependencies:

pip install pandas nltk scikit-learn

Download required NLTK resources:

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
▶️ Running the Project

Run the script:

python sentiment_analysis.py

The program will:

Load and preprocess the dataset

Train the model

Evaluate performance

Test predictions with sample reviews

🔍 Example Predictions

Example outputs:

'The food is really bad.'
→ Negative review

'Food was pretty bad and the service was very slow.'
→ Negative review

'The food was absolutely wonderful.'
→ Positive review
📊 Example Output
Accuracy Score: 87.5%
Precision Score: 88.2%
Recall Score: 86.9%

Confusion Matrix:
[[92  8]
 [12 88]]
🛠️ Technologies Used

Python

scikit-learn

NLTK

Pandas

Regular Expressions

🔮 Future Improvements

Possible enhancements:

Deep learning models (LSTM / BERT)

Web interface for predictions

Real-time sentiment analysis API

Support for multiple languages

Larger training datasets
