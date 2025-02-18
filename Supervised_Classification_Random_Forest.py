# https://chatgpt.com/c/67a4694e-8db0-8010-9054-654f644a018b
# chatgpt : example random forest when work as a data scientist in bank using pandas 

# In a banking context, you might use a Random Forest for tasks like predicting loan default, credit scoring, or customer segmentation. Here’s an example using Random Forest to predict loan defaults using customer data with Pandas and scikit-learn.

# Scenario:
# You are working as a data scientist at a bank, and you want to predict whether a customer will default on their loan based on features like income, loan amount, age, etc.

                    # # Import necessary libraries
                    # import pandas as pd
                    # from sklearn.model_selection import train_test_split
                    # from sklearn.ensemble import RandomForestClassifier
                    # from sklearn.metrics import classification_report, accuracy_score

                    # # Sample data (you can replace this with actual bank data)
                    # data = {
                    #     'age': [25, 45, 35, 50, 23, 40],
                    #     'income': [50000, 80000, 60000, 100000, 40000, 90000],
                    #     'loan_amount': [20000, 50000, 30000, 60000, 10000, 40000],
                    #     'loan_duration': [36, 60, 48, 60, 24, 36],
                    #     'defaulted': [0, 1, 0, 1, 0, 1]  # 0: No Default, 1: Default
                    # }

                    # # Load data into a pandas DataFrame
                    # df = pd.DataFrame(data)

                    # # Separate features (X) and target (y)
                    # X = df[['age', 'income', 'loan_amount', 'loan_duration']]
                    # y = df['defaulted']

                    # # Split the data into training and testing sets
                    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                    # # Create a RandomForestClassifier
                    # rf = RandomForestClassifier(n_estimators=100, random_state=42)

                    # # Train the model
                    # rf.fit(X_train, y_train)

                    # # Make predictions
                    # y_pred = rf.predict(X_test)

                    # # Evaluate the model
                    # print("Accuracy:", accuracy_score(y_test, y_pred))
                    # print("\nClassification Report:\n", classification_report(y_test, y_pred))

                    # # Feature importance
                    # feature_importances = pd.DataFrame(rf.feature_importances_, index=X.columns, columns=['importance'])
                    # print("\nFeature Importances:\n", feature_importances)

# Explanation:
# Data Preparation: We're simulating customer loan data where the target variable (defaulted) is a binary indicator of whether a customer defaulted on a loan.

# Train-Test Split: We split the data into training and testing sets using train_test_split to evaluate the model's performance.

# Random Forest Model: We initialize a RandomForest classifier and train it on the training data (X_train, y_train).

# Model Evaluation: After predicting the outcomes on the test data (y_test), we use accuracy_score and classification_report to evaluate model performance.

# Feature Importance: Random Forest provides a way to measure the importance of each feature, which can help the bank understand the key drivers of loan defaults (e.g., loan amount, income, etc.).

# Bank Use Cases for Random Forest:
# Credit Scoring: To assess a customer’s creditworthiness.
# Fraud Detection: Identify potentially fraudulent transactions.
# Loan Default Prediction: Predict the likelihood of a customer defaulting on a loan.

    # Great! Let's expand the loan default prediction into a more comprehensive use case and add a dashboard for visualization.

    # Full Use Case:
    # We will create a solution where:

    # A Random Forest model predicts whether a customer will default on a loan.
    # A dashboard visualizes important insights:
    # Prediction results (default vs. no default).
    # Model accuracy and performance.
    # Key feature importance.
    # Interactive inputs to simulate customer loan data and predict outcomes.
    # We will build the dashboard using Streamlit, a popular tool for creating quick and interactive web apps, and Plotly for visualization.

    # Step 1: Set Up the Machine Learning Model
    # The machine learning part will predict loan defaults, which we already started in the code provided. Let's enhance the dataset to include more features and a larger sample size to simulate real-world banking data.

    # Step 2: Create a Dashboard
    # We'll use Streamlit to make an interactive interface where bank analysts can:

    # Input customer data.
    # Predict the likelihood of defaulting.
    # Visualize feature importance and model performance.
    # Code for End-to-End Solution

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import streamlit as st
import plotly.express as px

# Simulate a larger banking dataset
np.random.seed(42)
n_samples = 1000

data = {
    'age': np.random.randint(20, 60, size=n_samples),
    'income': np.random.randint(30000, 150000, size=n_samples),
    'loan_amount': np.random.randint(5000, 60000, size=n_samples),
    'loan_duration': np.random.randint(12, 72, size=n_samples),  # loan duration in months
    'credit_score': np.random.randint(300, 850, size=n_samples),  # FICO credit score
    'employment_years': np.random.randint(1, 30, size=n_samples),
    'defaulted': np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])  # 30% default rate
}

# Create DataFrame
df = pd.DataFrame(data)

# Define features and target
X = df[['age', 'income', 'loan_amount', 'loan_duration', 'credit_score', 'employment_years']]
y = df['defaulted']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict on the test set
y_pred = rf.predict(X_test)

# Calculate accuracy and classification report
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

# Feature importance
feature_importances = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values(by='importance', ascending=False)

# --------------------- Streamlit Dashboard --------------------- #
st.title("Loan Default Prediction Dashboard")

# Sidebar for user input
st.sidebar.header("Customer Input Parameters")

def user_input_features():
    age = st.sidebar.slider("Age", 20, 60, 35)
    income = st.sidebar.slider("Income", 30000, 150000, 60000)
    loan_amount = st.sidebar.slider("Loan Amount", 5000, 60000, 30000)
    loan_duration = st.sidebar.slider("Loan Duration (Months)", 12, 72, 36)
    credit_score = st.sidebar.slider("Credit Score", 300, 850, 700)
    employment_years = st.sidebar.slider("Years of Employment", 1, 30, 10)
    
    data = {
        'age': age,
        'income': income,
        'loan_amount': loan_amount,
        'loan_duration': loan_duration,
        'credit_score': credit_score,
        'employment_years': employment_years
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Prediction
prediction = rf.predict(input_df)
prediction_proba = rf.predict_proba(input_df)

# Display input data
st.subheader("Customer Input Data")
st.write(input_df)

# Display prediction result
st.subheader("Prediction")
result = "Default" if prediction[0] == 1 else "No Default"
st.write(f"Prediction: **{result}**")
st.write(f"Probability of Default: **{prediction_proba[0][1]:.2f}**")

# Display model accuracy
st.subheader("Model Accuracy")
st.write(f"Accuracy: **{accuracy * 100:.2f}%**")

# Display classification report
st.subheader("Classification Report")
st.write(pd.DataFrame(report).transpose())

# Plot feature importance
st.subheader("Feature Importance")
fig = px.bar(feature_importances, x='importance', y='feature', orientation='h', title="Feature Importances")
st.plotly_chart(fig)

# Explanation of Key Sections:
# Simulated Banking Data:

# The dataset now includes more features like credit_score, employment_years, in addition to basic attributes like age, income, and loan_amount.
# The target variable (defaulted) simulates whether a customer has defaulted on their loan, with a 30% default rate.
# Random Forest Model:

# We use a RandomForestClassifier to train the model and predict loan defaults.
# The model provides feature importances to show which factors are driving the default prediction (e.g., credit score, loan amount).
# Streamlit Dashboard:

# Input Features: Bank analysts can simulate a new customer’s profile using sliders for features like age, income, and credit_score.
# Prediction: After entering customer data, the model predicts whether the customer is likely to default and displays the probability.
# Model Performance: The dashboard shows model accuracy and a detailed classification report.
# Feature Importance: A bar chart visualizes which features are most important in predicting loan defaults.
# Step 3: Run the Dashboard
# To run this dashboard, save the code in a Python file (e.g., loan_default_dashboard.py) and run the following command in your terminal:

# Step 3: Run the Dashboard
# To run this dashboard, save the code in a Python file (e.g., loan_default_dashboard.py) and run the following command in your terminal:

# bash
# Copy
# Edit
# streamlit run loan_default_dashboard.py
# This will open the dashboard in your browser, allowing you to interactively explore the loan default prediction.

# Future Enhancements:
# Model Tuning: Improve the Random Forest model by tuning hyperparameters and cross-validation.
# Additional Insights: Add more detailed reports, such as ROC curves, confusion matrices, or trend analysis.
# Customer Segmentation: Combine this with your customer segmentation project for deeper insights into customer behavior.
# Would you like to enhance the dashboard with more features or work on the data preprocessing step?


# Chatgpt : please help explain and conclude the output of the random forest

        # Understanding and Concluding the Random Forest Output
        # When using the Random Forest model to predict loan defaults in a banking context, you obtain several key outputs, including model predictions, accuracy, classification report, and feature importance. Let’s break down each part:

        # 1. Predictions:
        # After the Random Forest model is trained on historical customer data, you can input new customer data (e.g., age, income, loan amount) and receive a prediction on whether the customer will default or not default on a loan.

        # In the Streamlit dashboard, we can see the prediction as:

        # Prediction: Default or No Default – A simple binary classification of the customer’s likelihood of defaulting.
        # Probability of Default – The model also outputs the probability of the customer defaulting, which helps gauge the risk level. For example, if a customer has a default probability of 0.85 (85%), they are considered high risk.
        # 2. Model Accuracy:
        # The accuracy score tells you how well the Random Forest model performs on the test data. It is calculated as the percentage of correct predictions out of all predictions made.

        # For example:

        # Accuracy: 85% – This means the model correctly predicted whether a customer would default or not for 85% of the test cases.
        # A good accuracy score indicates the model is reasonably accurate, but it’s important to evaluate it alongside other metrics like precision, recall, and F1-score for a more holistic view.

        # 3. Classification Report:
        # The classification report gives a detailed breakdown of the model's performance using key metrics like precision, recall, and F1-score for each class (Default and No Default).

        # For example:

        # Precision: Measures how many of the customers predicted to default actually did default. A precision of 0.80 (80%) for the "Default" class means that 80% of the predicted defaults were true defaults.

        # Recall: Measures how many of the actual defaulters were correctly predicted by the model. A recall of 0.70 (70%) means that the model correctly identified 70% of customers who defaulted.

        # F1-Score: Combines precision and recall into one metric, providing a balanced evaluation of the model’s performance. A high F1-score (closer to 1) indicates good overall performance in predicting defaulters.

        # 4. Feature Importance:
        # Random Forest models have a unique advantage in that they provide feature importance, which shows which features are the most influential in making predictions.

        # For example, in the loan default use case, the most important features might be:

        # Credit Score: A low credit score might be the strongest indicator of default risk.
        # Loan Amount: Higher loan amounts could increase the likelihood of default.
        # Employment Years: More stable employment could reduce the chance of default.
        # The feature importance values can guide the bank in understanding what factors to focus on when assessing the risk of loan defaults.

        # Conclusion
        # The Random Forest model for predicting loan defaults in a bank provides several valuable outputs:

        # Predictions: It predicts whether a customer will default, with the option to see the default probability.
        # Accuracy: It gives an overall performance score (e.g., 85%), indicating how well it can generalize to unseen data.
        # Classification Report: Detailed metrics such as precision, recall, and F1-score help assess the balance between correctly identifying defaulters and minimizing false positives.
        # Feature Importance: This allows the bank to understand which factors (e.g., credit score, income, loan amount) are driving the model’s decisions, providing insights for better risk management.
        # Key Takeaways:
        # The model can help the bank predict loan default risk, assisting in decision-making for loan approvals.
        # It can provide a probability-based risk score, giving more granular control over loan assessments.
        # Feature importance can guide credit policies and customer management by focusing on the most influential risk factors.
        # This end-to-end system not only supports predictive analysis but also enhances the bank’s strategic decision-making capabilities by providing deeper insights into customer behavior.

        # Would you like help interpreting specific outputs, or should we explore how to fine-tune the model further?


# Fine-tuning a Random Forest model can help improve its performance by optimizing parameters and addressing any imbalance or overfitting. Here are the key steps and techniques for fine-tuning your Random Forest model in the loan default use case.

# Steps for Fine-Tuning:
# 1. Hyperparameter Tuning:
# Random Forest has several key hyperparameters that you can adjust to optimize performance. These include:

# n_estimators: The number of trees in the forest. Increasing this can improve accuracy but also increases computation time.
# max_depth: The maximum depth of each tree. Limiting the depth can prevent overfitting.
# min_samples_split: The minimum number of samples required to split a node. Increasing this can also prevent overfitting.
# max_features: The number of features to consider when looking for the best split. It can be a fraction (e.g., sqrt, log2) to reduce the model's variance.
# Example: Using GridSearchCV to find the best parameters.
# python
# Copy
# Edit
# from sklearn.model_selection import GridSearchCV

# # Define the parameter grid
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_split': [2, 5, 10],
#     'max_features': ['auto', 'sqrt', 'log2']
# }

# # Initialize the RandomForestClassifier
# rf = RandomForestClassifier(random_state=42)

# # Grid search for best parameters
# grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
# grid_search.fit(X_train, y_train)

# # Best parameters from grid search
# best_params = grid_search.best_params_
# print(f"Best Parameters: {best_params}")

# # Use the best model for prediction
# best_rf = grid_search.best_estimator_
# y_pred_best = best_rf.predict(X_test)

# # Evaluate the fine-tuned model
# accuracy_best = accuracy_score(y_test, y_pred_best)
# print(f"Tuned Model Accuracy: {accuracy_best:.2f}")
# Explanation:
# GridSearchCV systematically tests combinations of hyperparameters to find the best performing model.
# Cross-validation (cv=5) ensures the model generalizes well by evaluating it on different data subsets.
# You should see improvements in accuracy, precision, or recall after tuning the hyperparameters.

# 2. Handling Class Imbalance:
# If the default rate in the data is low, the model may focus more on predicting the majority class (e.g., "No Default") and ignore the minority class ("Default"). To handle this, we can:

# Class Weights: Assign more weight to the minority class to make the model pay more attention to it.
# Resampling Techniques: Use techniques like SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset.
# Example: Using Class Weights in Random Forest
# python
# Copy
# Edit
# # Initialize RandomForestClassifier with class weights
# rf_weighted = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)

# # Train the model
# rf_weighted.fit(X_train, y_train)

# # Predict and evaluate
# y_pred_weighted = rf_weighted.predict(X_test)
# accuracy_weighted = accuracy_score(y_test, y_pred_weighted)
# print(f"Weighted Model Accuracy: {accuracy_weighted:.2f}")
# Explanation:
# class_weight='balanced' adjusts weights based on the class distribution, ensuring the model pays more attention to underrepresented classes.
# 3. Cross-Validation:
# Using cross-validation helps to better evaluate the model's performance and prevent overfitting. It ensures that the model is tested on various subsets of data.

# Example: Using cross_val_score for performance evaluation
# python
# Copy
# Edit
# from sklearn.model_selection import cross_val_score

# # Perform cross-validation
# cv_scores = cross_val_score(best_rf, X, y, cv=5, scoring='accuracy')

# # Print the average accuracy across folds
# print(f"Cross-validated accuracy: {cv_scores.mean():.2f}")
# Explanation:
# cross_val_score provides a more robust measure of the model's performance by averaging accuracy over multiple data splits (folds).
# 4. Feature Selection:
# While Random Forest inherently handles feature selection by measuring feature importance, you can further improve performance by removing redundant or irrelevant features, especially if the dataset is large.

# Example: Removing less important features
# python
# Copy
# Edit
# # Select top N important features (e.g., top 4)
# important_features = feature_importances[feature_importances['importance'] > 0.1]['feature']
# X_train_important = X_train[important_features]
# X_test_important = X_test[important_features]

# # Train Random Forest with important features
# rf_important = RandomForestClassifier(n_estimators=200, random_state=42)
# rf_important.fit(X_train_important, y_train)

# # Predict and evaluate
# y_pred_important = rf_important.predict(X_test_important)
# accuracy_important = accuracy_score(y_test, y_pred_important)
# print(f"Model Accuracy with Important Features: {accuracy_important:.2f}")
# Explanation:
# Focusing on only the most important features can reduce noise and potentially improve accuracy or reduce computation time.
# 5. Model Interpretation:
# Random Forest models can be interpreted using SHAP values (SHapley Additive exPlanations), which explain the impact of each feature on individual predictions.

# Example: Using SHAP for feature impact visualization
# python
# Copy
# Edit
# import shap

# # Initialize SHAP explainer
# explainer = shap.TreeExplainer(best_rf)
# shap_values = explainer.shap_values(X_test)

# # Plot SHAP summary plot
# shap.summary_plot(shap_values[1], X_test)
# Explanation:
# SHAP summary plots provide a clear visualization of how each feature contributes to predictions. This helps banks understand model decisions, ensuring transparency and compliance.
# 6. Ensemble Methods:
# In addition to Random Forest, consider combining models (e.g., stacking or boosting) to improve performance. For example, you can combine Random Forest with models like Gradient Boosting for better results.

# Example: Stacking Classifier
# python
# Copy
# Edit
# from sklearn.ensemble import StackingClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC

# # Initialize base models and stacking classifier
# estimators = [('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
#               ('svm', SVC(probability=True))]
# stacking_clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

# # Train stacking classifier
# stacking_clf.fit(X_train, y_train)

# # Evaluate stacking model
# y_pred_stack = stacking_clf.predict(X_test)
# accuracy_stack = accuracy_score(y_test, y_pred_stack)
# print(f"Stacking Model Accuracy: {accuracy_stack:.2f}")
# Explanation:
# Stacking combines multiple models, and the final estimator (here, logistic regression) aggregates their predictions, often leading to better performance.
# Conclusion:
# To fine-tune a Random Forest model for predicting loan defaults:

# Hyperparameter Tuning: Use GridSearchCV to find the best set of parameters.
# Class Imbalance: Handle imbalance with class weights or resampling.
# Cross-Validation: Use cross-validation to ensure model generalization.
# Feature Selection: Remove less important features to improve performance.
# Model Interpretation: Use SHAP values for transparency and explainability.
# Ensemble Methods: Combine models (stacking/boosting) for better results.
# These strategies will help you improve model performance, enhance interpretability, and make the solution more robust for banking applications.

# Would you like to implement one of these techniques in more detail?










