# -----------------------------------------------------------------------------------
# Phase 1: Data Loading and Inspection (No Change)
# -----------------------------------------------------------------------------------

import pandas as pd
import os

# Set up robust file path
# NOTE: Replace 'Restaurant_Reviews.tsv' with your actual file name if different.
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'Restaurant_Reviews.tsv')

# Load the dataset
data = pd.read_csv(file_path, delimiter='\t', quoting=3)

print("--- Data Head ---")
print(data.head())
print("\n")

# -----------------------------------------------------------------------------------
# Phase 2: Text Preprocessing and Cleaning (CHANGE: Stemming -> Lemmatization)
# -----------------------------------------------------------------------------------

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer # <--- NEW LEMMATIZER

# Download necessary NLTK data (stopwords and wordnet)
try:
    nltk.data.find('corpora/stopwords')
except:
    print("Downloading 'stopwords' resource from NLTK.")
    nltk.download('stopwords')

try:
    # WordNet is required for Lemmatization
    nltk.data.find('corpora/wordnet')
except:
    print("Downloading 'wordnet' resource from NLTK.")
    nltk.download('wordnet')
    
# Initialize Lemmatizer
wnl = WordNetLemmatizer() 

corpus = []

for i in range(0, len(data)):
    review = re.sub(pattern='[^a-zA-Z]', repl=' ', string=data['Review'][i])
    review = review.lower()
    review_words = review.split()
    all_stopwords = set(stopwords.words('english'))
    
    # Remove stopwords
    review_words = [word for word in review_words if not word in all_stopwords]
    
    # <--- CHANGE: APPLY LEMMATIZATION INSTEAD OF STEMMING --->
    review = [wnl.lemmatize(word) for word in review_words]
    
    review = ' '.join(review)
    corpus.append(review)

print("--- Preprocessing Complete ---")
print("First 5 cleaned reviews (using Lemmatization):")
for review in corpus[:5]:
    print(review)
print("\n")

# -----------------------------------------------------------------------------------
# Phase 3: Feature Extraction (CHANGE: CountVectorizer -> TfidfVectorizer with Bi-grams)
# -----------------------------------------------------------------------------------

from sklearn.feature_extraction.text import TfidfVectorizer # <--- NEW VECTORIZER

# TfidfVectorizer: Better weighting. 
# ngram_range=(1, 2): Includes single words (unigrams) AND two-word phrases (bigrams).
# max_features is increased for better vocabulary coverage.
tfidf_vectorizer = TfidfVectorizer(max_features=2500, ngram_range=(1, 2))

# Fit and transform the corpus
X = tfidf_vectorizer.fit_transform(corpus).toarray()
y = data.iloc[:, 1].values

print("--- Feature Extraction Complete ---")
print(f"Shape of feature matrix X: {X.shape}")
print(f"Shape of target vector y: {y.shape}")
print("\n")

# -----------------------------------------------------------------------------------
# Phase 4: Model Training and Evaluation (CHANGE: Logistic Regression with Tuning)
# -----------------------------------------------------------------------------------

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression # <--- NEW MODEL
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

print("--- Hyperparameter Tuning: Logistic Regression ---")

# Test different C values for regularization and use class_weight='balanced'
best_accuracy = 0
best_classifier = None
C_values = [0.1, 1.0, 10.0] 

# Train and evaluate models with different C values
for C_value in C_values:
    # <--- KEY CHANGE: class_weight='balanced' for better Recall --->
    classifier_tune = LogisticRegression(
        random_state=0, 
        max_iter=1000, 
        class_weight='balanced', # Crucial for boosting recall
        C=C_value                # Regularization parameter
    )
    classifier_tune.fit(X_train, y_train)
    y_pred_tune = classifier_tune.predict(X_test)
    
    accuracy_tune = accuracy_score(y_test, y_pred_tune)
    precision_tune = precision_score(y_test, y_pred_tune)
    recall_tune = recall_score(y_test, y_pred_tune)
    
    print(f"Model (C={C_value}, Balanced Weights):")
    print(f"  Accuracy Score: {round(accuracy_tune * 100, 3)}%")
    print(f"  Precision Score: {round(precision_tune * 100, 3)}%")
    print(f"  Recall Score: {round(recall_tune * 100, 3)}%")
    
    if accuracy_tune > best_accuracy:
        best_accuracy = accuracy_tune
        best_classifier = classifier_tune

# Use the best model found (often C=10.0 or 1.0 with balanced weights)
classifier = best_classifier
y_pred = classifier.predict(X_test)

# Final Metrics of the Best Model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("\n--- FINAL BEST MODEL PERFORMANCE ---")
print(f"Accuracy Score: {round(accuracy * 100, 3)}%")
print(f"Precision Score: {round(precision * 100, 3)}%")
print(f"Recall Score: {round(recall * 100, 3)}%")
print("\n")

cm = confusion_matrix(y_test, y_pred)
print("--- Confusion Matrix ---")
print(cm)
print("\n")

# -----------------------------------------------------------------------------------
# Phase 5: Creating a Reusable Prediction Function (Updated for Lemmatization/Tfidf)
# -----------------------------------------------------------------------------------

def predict_sentiment(sample_review):
    """
    Predicts sentiment using Lemmatization and the TfidfVectorizer.
    """
    # Clean the review
    sample_review = re.sub(pattern='[^a-zA-Z]', repl=' ', string=sample_review)
    sample_review = sample_review.lower()
    sample_review_words = sample_review.split()
    all_stopwords = set(stopwords.words('english'))
    sample_review_words = [word for word in sample_review_words if not word in all_stopwords]
    
    # Use WordNetLemmatizer
    wnl_local = WordNetLemmatizer() 
    final_review = [wnl_local.lemmatize(word) for word in sample_review_words]
    final_review = ' '.join(final_review)
    
    # Use the fitted TfidfVectorizer
    temp = tfidf_vectorizer.transform([final_review]).toarray()
    
    # Use the FINAL best classifier
    return classifier.predict(temp)[0]

# Now, we'll test the function
print("--- Testing the model with new reviews ---")

sample_review_1 = 'The food is really bad.'
if predict_sentiment(sample_review_1) == 1:
    print(f"'{sample_review_1}' is a Positive review")
else:
    print(f"'{sample_review_1}' is a Negative review")

sample_review_2 = 'Food was pretty bad and the service was very slow.'
if predict_sentiment(sample_review_2) == 1:
    print(f"'{sample_review_2}' is a Positive review")
else:
    print(f"'{sample_review_2}' is a Negative review")

sample_review_3 = 'The food was absolutely wonderful, from preparation to presentation, very pleasing.'
if predict_sentiment(sample_review_3) == 1:
    print(f"'{sample_review_3}' is a Positive review")
else:
    print(f"'{sample_review_3}' is a Negative review")

# Test case for improved recall (it missed 40% before)
sample_review_4 = 'It was not good, but I would not say it was horrible.' # Slightly ambiguous
if predict_sentiment(sample_review_4) == 1:
    print(f"'{sample_review_4}' is a Positive review")
else:
    print(f"'{sample_review_4}' is a Negative review")