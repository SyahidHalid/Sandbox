# https://chatgpt.com/c/67a5c276-8384-8010-b8e7-d2786aab154f
# chatgpt : example SVM when work as a data scientist in bank using pandas 


# Here's an example of using Support Vector Machine (SVM) in a banking context where you might want to predict loan approval using a dataset. This example uses pandas for data handling and scikit-learn for applying the SVM model.


    # Let's assume you have a dataset containing the following features for loan applicants:

    # age: Applicant's age
    # income: Applicant's annual income
    # loan_amount: Amount of loan requested
    # credit_score: Applicant's credit score
    # loan_approved: Target variable indicating whether the loan was approved (1) or not (0)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# Sample dataset
data = {
    'age': [25, 45, 35, 50, 23, 40, 60, 48, 33, 39],
    'income': [50000, 100000, 75000, 120000, 45000, 90000, 130000, 105000, 68000, 80000],
    'loan_amount': [10000, 20000, 15000, 25000, 9000, 18000, 22000, 21000, 14000, 16000],
    'credit_score': [700, 800, 750, 820, 680, 770, 830, 790, 740, 760],
    'loan_approved': [1, 1, 0, 1, 0, 1, 1, 1, 0, 1]
}

# Load the dataset into a DataFrame
df = pd.DataFrame(data)

# Features and target variable
X = df[['age', 'income', 'loan_amount', 'credit_score']]
y = df['loan_approved']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# https://chatgpt.com/c/67a5c381-d0c8-8010-a153-adc5090f981e
# Chatgpt : what is StandardScaler() is python

# Scale the features for better performance of SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the SVM model
svm_model = SVC(kernel='linear')  # You can experiment with different kernels like 'rbf' or 'poly'
svm_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = svm_model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy * 100:.2f}%')
print('Confusion Matrix:')
print(conf_matrix)


    # Key Steps:
    # Data Preprocessing: The dataset is loaded into a Pandas DataFrame, and we split it into training and test sets.
    # Feature Scaling: SVM models are sensitive to the scale of features, so we use StandardScaler to standardize the features.
    # Model Training: We use the SVC class from sklearn with a linear kernel. Depending on the data, you could also try the 'rbf' or 'poly' kernels.
    # Model Evaluation: The accuracy score and confusion matrix are used to evaluate the model’s performance.

# Accuracy: 100.00%
# Confusion Matrix:
# [[1 0]
#  [0 2]]

# This is a basic implementation, but in real banking scenarios, you would work with larger datasets, tune hyperparameters, and use more advanced feature engineering techniques.


# Chatgpt : please help explain and conclude the output of the neural network

    # Accuracy is the ratio of correctly predicted observations to the total observations:
    # Accuracy = (TP+TN)/(TP+TN+FP+FN)
    # Accuracy gives an overall view of how well the model is performing. However, it can be misleading if the data is imbalanced (i.e., there are more positive or negative samples).

    # Precision is the ratio of correctly predicted positive observations to the total predicted positives:
    # Precision = TP/(TP+FP)
    # Precision answers the question: Of all the loans the model predicted to be approved, how many were actually approved?

    # Recall (Sensitivity) is the ratio of correctly predicted positive observations to the actual positives:
    # Recall = TP/(TP+FN)
    # Recall answers the question: Of all the loans that were actually approved, how many did the model correctly predict?

    # F1 Score is the harmonic mean of precision and recall, providing a balance between the two:
    # F1 Score = 2 * ((Precision*Recall)/(Precision+Recall))
    # The F1 score is particularly useful when there is an uneven class distribution (i.e., one class is much more common than the other).


# Example Output:
# Let’s assume after training and testing your neural network, you get the following output:
# Accuracy: 95%
# Precision: 92%
# Recall: 88%
# F1 Score: 90%

# Confusion Matrix:
# |                | Predicted No | Predicted Yes |
# |----------------|--------------|---------------|
# | Actual No      | 50           | 5             |
# | Actual Yes     | 6            | 39            |

# Accuracy (95%): The neural network correctly predicted 95% of the loan approval statuses. This suggests that the model is generally performing well across both classes (loan approved and not approved).

# Precision (92%): Out of all the loans the model predicted as approved, 92% were actually approved. This indicates that the model is good at avoiding false positives, meaning it doesn’t approve loans for applicants who shouldn’t receive them.

# Recall (88%): Of all the loans that should have been approved, the model successfully approved 88% of them. This indicates that the model is slightly less strong in terms of recall, meaning it misses some true positives (false negatives).

# F1 Score (90%): The F1 score, which balances precision and recall, is 90%, suggesting a good balance between these two metrics.

# Confusion Matrix:

# True Negatives (50): The model correctly predicted that 50 applicants should not receive loans.
# True Positives (39): The model correctly predicted that 39 applicants should receive loans.
# False Positives (5): The model wrongly approved 5 loans for applicants who should not have received them.
# False Negatives (6): The model missed 6 loan approvals that should have been granted.


    #Conclusion:

    # Overall, the neural network performs well in predicting loan approvals with high accuracy and a good balance between precision and recall. The low number of false positives (5) indicates the model is fairly conservative in its approvals, which could be favorable for reducing risk in a banking environment.
    # However, depending on the business priorities, further tuning may be required to increase recall (i.e., reduce the number of false negatives), ensuring that fewer qualified applicants are missed.
