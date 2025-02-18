# https://chatgpt.com/c/67a5c9c4-5128-8010-9351-7718823511cf
# chatgpt : example Decision trees when work as a data scientist in bank using pandas

# In a banking context, decision trees are commonly used for tasks like credit scoring, fraud detection, customer segmentation, and loan approval predictions. I'll walk you through an example using decision trees in Python with pandas and scikit-learn. In this example, we'll predict whether a customer will default on their loan using a dataset.
# Here’s a step-by-step example of how you can implement a decision tree model in Python for a bank-related use case, such as predicting loan default.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


# chatgpt : please help me crate the loan_data.csv using pandas

# Sample dataset
data = {
    'age': [25, 45, 35, 50, 23, 40, 60, 48, 33, 39],
    'income': [50000, 100000, 75000, 120000, 45000, 90000, 130000, 105000, 68000, 80000],
    'loan_amount': [10000, 20000, 15000, 25000, 9000, 18000, 22000, 21000, 14000, 16000],
    'credit_score': [700, 800, 750, 820, 680, 770, 830, 790, 740, 760],
    'default': [1, 1, 0, 1, 0, 1, 1, 1, 0, 1]
}

# Load the dataset into a DataFrame
df = pd.DataFrame(data)

# Features and target
X = df[['age', 'income', 'loan_amount', 'credit_score']]  # Features
y = df['default']  # Target variable

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Decision Tree model
clf = DecisionTreeClassifier(random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Classification report
print(classification_report(y_test, y_pred))

from sklearn import tree
import matplotlib.pyplot as plt

# Plot the decision tree
plt.figure(figsize=(12,8))
tree.plot_tree(clf, feature_names=X.columns, class_names=['No Default', 'Default'], filled=True)
plt.show()

#Output

#You’ll get an accuracy score, a classification report, and a visualization of the decision tree that can help explain how decisions are made in predicting whether a customer will default on their loan.

# chatgpt : please help explain and conclude the output of the decision tree




        # In the decision tree diagram you provided, we can break down the nodes and branches to explain the output and draw some conclusions about the model’s decision-making process.

        # Here is a detailed explanation:

        # Key Terms:
        # Gini: Gini impurity measures the degree of probability of a particular feature being misclassified. A Gini value of 0 means perfect classification, and higher values indicate more uncertainty.
        # Samples: The number of observations at each node.
        # Value: This shows how many samples at this node belong to each class. [x, y] means there are x samples of class "No Default" and y samples of class "Default."
        # Class: The predicted class at that node, based on the majority of samples.
        # Root Node:
        # Age <= 37.0: The first split is made on age. If the customer's age is less than or equal to 37, it moves to the left subtree; otherwise, it moves to the right subtree.
        # Gini = 0.408: There is some impurity here, meaning the model is not perfectly certain about whether a customer will default.
        # Samples = 7: There are 7 total samples at this node.
        # Value = [2, 5]: Out of the 7 samples, 2 belong to the "No Default" class, and 5 belong to the "Default" class.
        # Class = Default: Since there are more "Default" samples, the root node classifies a customer as "Default."
        # Left Subtree (Age <= 37):
        # Age <= 24.0:
        # Gini = 0.444: There’s a bit of uncertainty in this split.
        # Samples = 3: Three samples reached this node.
        # Value = [2, 1]: Two samples are "No Default" and one is "Default."
        # Class = No Default: Since the majority class is "No Default," it predicts "No Default."
        # Moving to the leaves:
        # If Age <= 24, the leftmost leaf has gini = 0.0, with one "No Default" sample, making a confident prediction of "No Default."
        # If Age > 24, the right leaf has one "Default" sample with gini = 0.0, leading to a "Default" classification.
        # Right Subtree (Age > 37):
        # Gini = 0.0: This node is pure, meaning there is no uncertainty.
        # Samples = 4: Four samples reached this node.
        # Value = [0, 4]: All four samples are "Default."
        # Class = Default: The model classifies customers in this group as "Default" since all samples are from that class.
        # Middle Node (Credit Score <= 725 for Age > 24 and <= 37):
        # Credit Score <= 725.0: For those between 24 and 37 years old, the next split is based on credit score.
        # Gini = 0.5: This indicates maximum uncertainty, as there is an equal split between the classes.

        # Samples = 2: Two samples reached this node.

        # Value = [1, 1]: One sample is "No Default" and the other is "Default."

        # If Credit Score <= 725, the left leaf has one "No Default" sample, with gini = 0.0, predicting "No Default."

        # If Credit Score > 725, the right leaf has one "Default" sample, with gini = 0.0, predicting "Default."

        # Conclusions:
        # Age is an important feature in this dataset, and the model uses it first to split the data. Customers under 37 tend to have more uncertainty (mix of classes), while those over 37 are consistently predicted to default.
        # Credit score also plays a role, particularly for customers in the middle age range (24 to 37). If their credit score is less than or equal to 725, they are more likely to not default.
        # The gini index shows how confident the model is at each step. Lower gini values indicate more confident decisions.
        # Pure nodes (gini = 0): When the gini index reaches 0 at a node, the model is very certain of its classification, meaning that all samples at that node belong to the same class.
        # This decision tree provides insight into how age and credit score impact loan default predictions. Generally:

        # Younger customers (age <= 24) are more likely to not default.
        # Older customers (age > 37) are more likely to default.
        # Customers between 24 and 37 with lower credit scores (<= 725) are less likely to default, while those with higher credit scores tend to default.
        # This is a small and simplified tree, but it gives a clear view of how the model makes predictions based on customer characteristics.



