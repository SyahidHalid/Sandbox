#https://chatgpt.com/c/67d0e99b-3a1c-8010-beb4-e97967a1764a

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Step 1: Generate simulated data
np.random.seed(42)

# Simulate 1000 borrowers with features
n_samples = 1000
credit_score = np.random.normal(600, 50, n_samples)  # Mean credit score of 600
income = np.random.normal(50000, 15000, n_samples)   # Mean income of 50k
debt_to_income = np.random.uniform(0, 1, n_samples)  # Debt-to-income ratio (0 to 1)
age = np.random.randint(21, 65, n_samples)           # Age of borrower

# Simulate the target variable: 1 if default, 0 if not
default = (0.03 * (700 - credit_score)) + (0.02 * debt_to_income * income / 1000) - (0.005 * age)
probability_default = 1 / (1 + np.exp(-default))  # Logistic function
default_binary = np.random.binomial(1, probability_default)

# Create a DataFrame for analysis
data = pd.DataFrame({
    'credit_score': credit_score,
    'income': income,
    'debt_to_income': debt_to_income,
    'age': age,
    'default': default_binary
})

#======================================================================================================

# Step 2: Train a logistic regression model
X = data[['credit_score', 'income', 'debt_to_income', 'age']]
y = data['default']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 3: Estimate Probability of Default (PD) for test data
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of default (class 1)

#======================================================================================================

# Print out the first few PD estimates
pd_estimates = pd.DataFrame({
    'credit_score': X_test['credit_score'],
    'income': X_test['income'],
    'debt_to_income': X_test['debt_to_income'],
    'age': X_test['age'],
    'predicted_PD': y_pred_proba
})

# Print classification report for model performance
print(classification_report(y_test, model.predict(X_test)))

# Display the first 10 borrowers and their PD estimates
print(pd_estimates.head(10))
