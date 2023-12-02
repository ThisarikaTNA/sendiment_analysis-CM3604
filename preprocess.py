import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

review_path = 'yelp_dataset/yelp_academic_dataset_review.json'

# Load review data
review_df = pd.read_json(review_path, lines=True, nrows=100000)

print("\nReview DataFrame Info:")
print(review_df.info())

# Sample of Review DataFrame
print("\nSample of Review DataFrame:")
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
preprocessed_csv_path = 'yelp_dataset/preprocessed_reviews.csv'
review_df.to_csv(preprocessed_csv_path, index=False)
print(f"Preprocessed dataset saved to {preprocessed_csv_path}")