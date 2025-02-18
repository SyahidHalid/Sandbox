#https://chatgpt.com/c/67ac1925-6d28-8010-8b5c-5dd397b4cfe9
#example k-nearest neighbors when work as a data scientist in bank using pandas 


# Here's an example of using the K-Nearest Neighbors (KNN) algorithm in a banking scenario, where you're working as a data scientist using pandas and other libraries in Python. In this case, we might be interested in identifying potential customers for a new banking product based on customer features (such as age, income, credit score, etc.).

#Example Scenario:
#You have customer data with features like Age, Income, Credit Score, and you're trying to predict whether they will respond positively to a new credit card offer (Yes/No).


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the dataset (simulating a CSV import for this example)
data = {
    'Age': [25, 45, 35, 50, 23, 40, 33, 29, 47, 52],
    'Income': [50000, 80000, 60000, 120000, 40000, 75000, 62000, 48000, 98000, 105000],
    'Credit_Score': [650, 700, 620, 710, 580, 680, 640, 610, 690, 720],
    'Response': ['No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes', 'Yes']
}

# Create a pandas DataFrame
df = pd.DataFrame(data)

# Convert categorical target variable to numerical (Yes -> 1, No -> 0)
df['Response'] = df['Response'].map({'Yes': 1, 'No': 0})

# Features (Age, Income, Credit Score) and Target (Response)
X = df[['Age', 'Income', 'Credit_Score']]
y = df['Response']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (KNN requires feature scaling for accurate distance calculation)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the K-Nearest Neighbors classifier with k=3
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model
knn.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test_scaled)

# Evaluate the model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))


# Explanation

# Data Preparation: We simulate a banking dataset where we have features like age, income, and credit score, and a target variable Response which indicates whether a customer responded to a new offer.
# Feature Scaling: Since KNN relies on distance measurements, it's essential to scale the features (Age, Income, Credit Score) so that they contribute equally to the distance calculation.
# Model Training: The KNN model is trained on the scaled features with k=3 neighbors.
# Evaluation: We use a confusion matrix, classification report, and accuracy score to evaluate the performance of the model on unseen data.


# Confusion Matrix:
#  [[1 0]
#   [0 1]]
# The confusion matrix compares the actual values (rows) with the predicted values (columns) to provide a breakdown of the model’s performance. The matrix here is:

# Top-left (1,1): One instance was correctly predicted as class 0 (No).
# Bottom-right (1,1): One instance was correctly predicted as class 1 (Yes).
# Off-diagonal (0,0): No false predictions (misclassifications).
# This indicates that both predictions made by the model were correct.




# Classification Report:
#                precision    recall  f1-score   support

#            0       1.00      1.00      1.00         1
#            1       1.00      1.00      1.00         1

#     accuracy                           1.00         2
#    macro avg       1.00      1.00      1.00         2
# weighted avg       1.00      1.00      1.00         2

# Precision: Out of the instances predicted as a certain class, how many were actually correct?

# Class 0 (No): 100% precision means the model correctly predicted all cases labeled as No.
# Class 1 (Yes): 100% precision indicates the model accurately predicted all Yes cases.
# Recall: Out of the actual instances of a certain class, how many did the model correctly predict?

# Class 0: 100% recall means the model found all the actual No instances.
# Class 1: 100% recall means the model found all the actual Yes instances.
# F1-Score: This is the harmonic mean of precision and recall, providing a balance between the two metrics.

# Both 0 and 1 have an F1-score of 1.0, showing perfect prediction in this small dataset.
# Support: This tells us how many actual instances of each class there were.

# Class 0 (No): 1 instance
# Class 1 (Yes): 1 instance


# Accuracy Score: 1.0

#The model achieved an accuracy score of 1.0 (100%), meaning it correctly classified all the instances in the test set.


# This approach could be extended to more complex datasets, larger feature sets, and fine-tuned by adjusting the number of neighbors (k).

# Chatgpt : please help explain and conclude the output of the neural network

        # The K-Nearest Neighbors (KNN) model performed perfectly on this small test dataset, correctly classifying both instances (1 predicted "No" and 1 predicted "Yes"). However, it's important to note that the dataset used here is very small (only 2 test samples), so the results are not necessarily generalizable to a larger population.

        # In a real-world banking scenario:

        # The model's performance might vary depending on the size of the dataset and complexity of the features.
        # It would be beneficial to evaluate the model on a larger test set to get a more reliable measure of its performance.
        # You may also want to experiment with different values of k (the number of neighbors) and consider cross-validation for more robust performance evaluation.
        # Overall, KNN seems to work well on this example, but it needs further validation before being used in a production environment for tasks like customer classification or product recommendation in banking.

