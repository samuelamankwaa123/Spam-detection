# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Step 1: Load the Dataset
# Update the path to your Downloads folder
data_path = r'C:\Users\Kasutaja\Downloads\SMSSpamCollection'
df = pd.read_csv(data_path, sep='\t', header=None, names=['label', 'message'], encoding='latin-1')

# View first few rows to understand the data
print(df.head())

# Step 2: Data Preprocessing
# Convert labels to binary format: 'spam' -> 1, 'ham' -> 0
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Check for any missing values
print("Missing values:\n", df.isnull().sum())

# Step 3: Feature Extraction
# Use TF-IDF to convert text to numerical features
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
X = tfidf.fit_transform(df['message'])
y = df['label']

# Step 4: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Model Building
# We'll use a Naive Bayes classifier, which is commonly used for text classification tasks
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 6: Predictions
y_pred = model.predict(X_test)

# Step 7: Evaluation
# Confusion Matrix, Accuracy, Precision, Recall, F1 Score
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))

# Optional: Testing a sample message
sample_message = ["Congratulations! You've won a $1000 Walmart gift card. Call now to claim your prize."]
sample_vector = tfidf.transform(sample_message)
prediction = model.predict(sample_vector)
print("\nPrediction for sample message:", "Spam" if prediction[0] == 1 else "Ham")
