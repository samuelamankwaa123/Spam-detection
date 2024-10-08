# Spam-detection
Spam Detection using Naive Bayes

This project is an implementation of a Spam Detection System for SMS messages using Natural Language Processing (NLP) and Machine Learning. The main objective is to classify SMS messages as either spam or ham (not spam).

Project Overview

The project demonstrates how to use text classification techniques to build a Spam Detection model. We use the SMS Spam Collection dataset, a well-known dataset for text classification tasks.

Key Components:

Dataset Preprocessing: Cleaning and transforming text data.

Feature Extraction: Using TF-IDF Vectorizer to convert text into numerical features.

Model Training: Training a Naive Bayes classifier to distinguish between spam and ham messages.

Evaluation: Assessing the model's performance using metrics like accuracy, precision, recall, and F1-score.

Dataset

The dataset used is the SMS Spam Collection dataset from the UCI ML Repository.

Each row in the dataset represents an SMS message labeled as either:

ham: Not spam (legitimate message)

spam: Unsolicited message often containing promotional or fraudulent content

Dataset URL: SMS Spam Collection Dataset

Dataset Columns:

Label: Specifies whether the message is ham or spam.

Message: The actual text content of the SMS.

Dependencies

The project requires the following Python libraries:

pandas: For data manipulation and analysis.

numpy: For numerical operations.

matplotlib: For data visualization (optional).

scikit-learn: For machine learning, data preprocessing, and evaluation.

Install the required dependencies using:

How to Run

Clone the repository:

Ensure you have the dataset:

Place the SMSSpamCollection file in the project directory or update the file path accordingly.

Run the Python script:

Prediction:

After training, you can test the model with sample messages using the provided script.

Project Structure

spam_detection.py: Main script containing the full implementation, from data preprocessing to model evaluation.

SMSSpamCollection: The dataset used for training and testing the model.

README.md: Documentation on the project.

Steps Followed in the Project

1. Data Preprocessing

Loaded and cleaned the dataset.

Converted the labels (ham, spam) into binary values (0 for ham, 1 for spam).

2. Feature Extraction

TF-IDF Vectorizer: Used to convert the SMS messages into a numeric form suitable for model training. Stop words (common words like "the", "is") were removed.

3. Model Building

The Multinomial Naive Bayes classifier was used, which is particularly effective for text classification tasks involving frequency-based features.

4. Evaluation

Evaluated the model using a Confusion Matrix, Accuracy Score, Precision, Recall, and F1-Score.

The model achieved an accuracy of 97.9%, indicating its high reliability for spam detection.

5. Sample Prediction

The model was tested with a sample spam message and successfully classified it as spam.

Evaluation Metrics

Accuracy: The overall accuracy of the model was 97.9%, meaning it made correct predictions for nearly all SMS messages in the test set.

Precision and Recall: The precision for detecting spam messages was 1.00, meaning there were no false positives, while the recall was 0.85, showing that the model missed a small number of spam messages.

F1-Score: The F1-score for spam detection was 0.92, indicating a good balance between precision and recall.

Results

The model was able to effectively classify SMS messages with high accuracy. The confusion matrix showed that the model made very few mistakes, mostly involving some spam messages that were misclassified as legitimate (false negatives). The final model is capable of accurately distinguishing between spam and ham messages, providing a valuable tool for detecting unsolicited or harmful content.

Future Improvements

Hyperparameter Tuning: Use GridSearchCV to find the optimal parameters for the Naive Bayes model or other models like SVM.

Different Algorithms: Experiment with other machine learning models such as Logistic Regression or Random Forest for improved performance.

Deep Learning: Explore the use of Recurrent Neural Networks (RNNs) or LSTMs to capture sequential relationships in the text, especially if the dataset size is increased.

