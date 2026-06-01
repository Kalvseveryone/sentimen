import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import os

# Define paths
data_path = os.path.join(os.path.dirname(__file__), 'data', 'amazon_cells_labelled.txt')
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')

vectorizer_path = os.path.join(os.path.dirname(__file__), 'vectorizer.pkl')

print("Loading data...")
# Load the dataset
# Data is formatted as: text \t label
df = pd.read_csv(data_path, sep='\t', header=None, names=['text', 'label'])

# Prepare data
X = df['text']
y = df['label']

print("Vectorizing data...")
# Vectorize text data
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

print("Training model (Multinomial Naive Bayes)...")
# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)

import json

# Extract params for manual prediction
vocab = vectorizer.vocabulary_
class_log_prior = model.class_log_prior_.tolist()
feature_log_prob = model.feature_log_prob_.tolist()
classes = model.classes_.tolist()

model_data = {
    'vocab': vocab,
    'class_log_prior': class_log_prior,
    'feature_log_prob': feature_log_prob,
    'classes': classes
}

json_path = os.path.join(os.path.dirname(__file__), 'model_data.json')
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(model_data, f)

print(f"Model data exported to {json_path}")


print(f"Model saved to {model_path}")
print(f"Vectorizer saved to {vectorizer_path}")
