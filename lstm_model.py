# -*- coding: utf-8 -*-
"""LSTM_model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/ThisarikaTNA/sentiment_analysis-CM3604/blob/main/LSTM_model.ipynb

Import necessary Libraries
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.optimizers import Adam

from google.colab import drive
drive.mount('/content/drive')

review_datasetpath = '/content/drive/MyDrive/yelp_dataset/yelp_academic_dataset_review.json'

"""Load ans Print sample review dataset data"""

# Load review data
review_df = pd.read_json(review_datasetpath, lines=True, nrows=1000000)

print("Review DataFrame Info:")
print(review_df.info())

print("Sample of Review DataFrame:")
print(review_df.head())

# Create a new column "length" with the word length of the review
review_df['length'] = review_df['text'].apply(lambda x: len(x.split()))

# Visualize the correlation between stars and the length of the review
sns.scatterplot(x='stars', y='length', data=review_df)
plt.title('Correlation between Stars and Review Length')
plt.show()

# getting mean value of the vote columns
mean_votes = review_df[['useful', 'funny', 'cool']].mean()
print('Mean Value of the Vote columns:')
print(mean_votes)

# Correlation between the voting columns
correlation_matrix = review_df[['useful', 'funny', 'cool']].corr()
print('\nCorrelation between the voting columns:')
print(correlation_matrix)

"""Preprocess the Dataset"""

def preprocess_text(text):
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in ENGLISH_STOP_WORDS])
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])
    return text


# Apply text preprocessing to the 'text' column in the review DataFrame
review_df['text'] = review_df['text'].apply(preprocess_text)

# Handle missing values if any
review_df.dropna(inplace=True)

# Remove duplicate rows based on the 'text' column
review_df.drop_duplicates(subset=['text'], inplace=True)

# Assuming sentiment is positive for stars greater than or equal to 4, and negative otherwise
review_df['label'] = (review_df['stars'] >= 4).astype(int)

# Display the preprocessed DataFrame
print("\nSample of Preprocessed Review DataFrame:")
print(review_df.head())

# Save the preprocessed review DataFrame to a CSV file
preprocessed_csv_path = '/content/drive/MyDrive/yelp_dataset/preprocessed_reviews.csv'
review_df.to_csv(preprocessed_csv_path, index=False)
print(f"Preprocessed dataset saved to {preprocessed_csv_path}")

"""Split the dataset into training (80%) and test (20%) sets"""

train_data, test_data = train_test_split(review_df, test_size=0.2, random_state=42)

# Load  preprocessed Yelp Reviews dataset
data = pd.read_csv('/content/drive/MyDrive/yelp_dataset/preprocessed_reviews.csv')

# getting data to list from the lable column for sentiment (0 or 1)
labels = data['label'].tolist()

"""Model 1: LSTM"""

max_len = 128
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(train_data['text'])

train_sequences = tokenizer.texts_to_sequences(train_data['text'])
test_sequences = tokenizer.texts_to_sequences(test_data['text'])

train_padded = pad_sequences(train_sequences, maxlen=max_len, padding='post', truncating='post')
test_padded = pad_sequences(test_sequences, maxlen=max_len, padding='post', truncating='post')

lstm_model = Sequential()
lstm_model.add(Embedding(len(tokenizer.word_index) + 1, 128, input_length=max_len))
lstm_model.add(Bidirectional(LSTM(64)))
lstm_model.add(Dense(1, activation='sigmoid'))

lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
lstm_model.fit(train_padded, train_data['label'], epochs=10, batch_size=32, validation_data=(test_padded, test_data['label']))

"""Evaluate and Calculate metrics for LSTM model"""

lstm_predictions = lstm_model.predict(test_padded)
lstm_predictions = np.round(lstm_predictions).flatten()
lstm_accuracy = accuracy_score(test_data['label'], lstm_predictions)
print(f'LSTM Model Accuracy: {lstm_accuracy * 100:.2f}%')

lstm_precision = precision_score(test_data['label'], lstm_predictions)
lstm_recall = recall_score(test_data['label'], lstm_predictions)
lstm_f1 = f1_score(test_data['label'], lstm_predictions)
lstm_conf_matrix = confusion_matrix(test_data['label'], lstm_predictions)

print(f'LSTM Model Precision: {lstm_precision:.2f}')
print(f'LSTM Model Recall: {lstm_recall:.2f}')
print(f'LSTM Model F1 Score: {lstm_f1:.2f}')
print('LSTM Model Confusion Matrix:')
print(lstm_conf_matrix)