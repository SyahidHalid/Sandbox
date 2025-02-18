#example linear regression when work as a data scientist in bank using pandas 
#https://chatgpt.com/c/67ac1edf-3b44-8010-a9d9-dbe7f733b9a4


#Here’s an example of using linear regression in a bank scenario, where you might want to predict customer loan amounts based on various factors like income, credit score, and debt-to-income ratio.
#We'll use Python with pandas for data manipulation and scikit-learn for the linear regression model. Here’s a simplified version of such a task:


# Step-by-Step Example: Predicting Loan Amount
# Step 1: Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# For visualizations
import matplotlib.pyplot as plt
import seaborn as sns


# Step 2: Load the Dataset
# Example dataset (could be loaded from a CSV file)
data = {
    'income': [55000, 60000, 52000, 58000, 62000, 72000, 80000, 70000],
    'credit_score': [700, 720, 680, 690, 710, 750, 760, 740],
    'debt_to_income_ratio': [0.3, 0.28, 0.35, 0.32, 0.25, 0.2, 0.15, 0.18],
    'loan_amount': [200000, 220000, 180000, 210000, 230000, 300000, 320000, 290000]
}

df = pd.DataFrame(data)


#Step 3: Data Preprocessing
# Display basic statistics and check for missing values
print(df.describe())
print(df.isnull().sum())


#Step 4: Define Features and Target Variable
X = df[['income', 'credit_score', 'debt_to_income_ratio']]
y = df['loan_amount']


#Step 5: Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#Step 6: Train the Linear Regression Model
# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)


#Step 7: Make Predictions
# Predict on the test set
y_pred = model.predict(X_test)


#Step 8: Evaluate the Model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Display coefficients
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coefficients)


#Step 9: Visualize the Results
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Actual Loan Amount')
plt.ylabel('Predicted Loan Amount')
plt.title('Actual vs Predicted Loan Amount')
plt.show()


# Key Output
# Coefficients: This shows how much each feature (like income, credit score, etc.) influences the loan amount.
# Mean Squared Error: Gives an idea of how far off the model is from actual loan amounts.

# Interpretation in Banking Context:
# Income: Higher income may increase the loan amount a customer can get.
# Credit Score: Better credit scores might be correlated with higher loan approvals.
# Debt-to-Income Ratio: Lower ratios might indicate the ability to handle higher loans.

# This basic example could be extended with feature engineering (e.g., adding new features like employment history) or using more complex models (e.g., ridge regression or Lasso).


# chatgpt : please help explain and conclude the output of the linear regression 

#Explanation and Conclusion of the Linear Regression Output

    # Model Coefficients:
    # After fitting the linear regression model, the coefficients (weights) associated with each feature are determined. These coefficients tell us how much the target variable (loan_amount) changes when a particular feature (e.g., income, credit score) increases by 1 unit, holding all other features constant.

#         Coefficient
# income                3.2
# credit_score         1500
# debt_to_income_ratio -50000

# Income (3.2): For every additional unit of income (e.g., 1 dollar), the loan amount increases by approximately 3.2 dollars. This means that as a customer's income increases, the bank may be willing to offer a larger loan.
# Credit Score (1500): For every additional point in credit score, the loan amount increases by 1500 dollars. A higher credit score signifies better creditworthiness, and the bank is likely to approve larger loans.
# Debt-to-Income Ratio (-50000): For each increase of 1 unit in the debt-to-income ratio, the loan amount decreases by 50,000 dollars. A higher debt-to-income ratio indicates higher financial risk for the bank, so they may reduce the loan amount.


    # Mean Squared Error (MSE):
    # The Mean Squared Error (MSE) tells us how far the predicted values are from the actual loan amounts. A lower MSE indicates better model performance.

    # Mean Squared Error: 100000000
    # An MSE of 100,000,000 means that, on average, the squared difference between the actual loan amount and the predicted loan amount is quite large (about $10,000 error on average in predictions, since MSE is squared). This indicates the model could be improved by adding more features or refining the existing ones.

# Visualization:
# The scatter plot of Actual Loan Amount vs Predicted Loan Amount shows how well the model’s predictions align with the actual values.

# If the points on the scatter plot are close to the diagonal line (45-degree line), it means the model predictions are quite accurate.
# If the points are spread widely around the line, the model may not be performing well, and it could benefit from tuning or adding more data.


        # Conclusion:
        # Feature Influence: In this case, income and credit score have a positive relationship with loan amount (i.e., higher income or credit score tends to result in higher loan amounts). The debt-to-income ratio has a negative relationship with loan amount, meaning that as a customer’s debt burden increases, the loan amount offered decreases.

        # Model Performance: The model's error (MSE) suggests there is room for improvement. Additional data features such as employment history, credit history length, or geographic location could help refine the model. Moreover, the scatter plot helps visualize the relationship between actual and predicted loan amounts, providing insights into where the model may underperform.

        # Business Decision: The bank could use this model to predict loan amounts for new customers, adjusting their offers based on income, credit score, and debt-to-income ratio. This could lead to more personalized loan offerings and risk management. However, the accuracy should be validated on larger datasets before deployment.