#https://medium.com/@ethannabatchian/machine-learning-model-comparison-on-customer-churn-2e607b3ea3f0

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, PredefinedSplit
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, RocCurveDisplay
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


#Data import and Exploration
from xgboost import XGBClassifier
from xgboost import plot_importance

df = pd.read_csv('Churn_Modelling.csv')

df.info()

df['Exited'].value_counts()

average_churned_balance = df[df['Exited']==1]['Balance'].mean()
average_churned_balance

#Feature Engineering and Transformation
X = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis = 1)

X = pd.get_dummies(X, drop_first = True)

X = X.drop('Exited', axis = 1).copy()
y = df['Exited']

#Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size= 0.2, 
                                                    stratify = y, 
                                                    random_state= 21)

#Decision Tree Model
dt = DecisionTreeClassifier(random_state = 0)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

print("Accuracy:", "%.3f" % accuracy_score(y_test, dt_pred))
print("Precision:", "%.3f" % precision_score(y_test, dt_pred))
print("Recall:", "%.3f" % recall_score(y_test, dt_pred))
print("F1 Score:", "%.3f" % f1_score(y_test, dt_pred))

#Confusion Matrix and Decision Tree Visualisation
def confusion_matrix_plot(model, x_data, y_data):
    '''
    Accepts as argument model object, X data (test or validate), and y data(test or validate).
    Returns a plot of confusion matrix for predictions on y data.
    '''
    model_pred = model.predict(x_data)
    cm = confusion_matrix(y_data, model_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
    display_labels=model.classes_)
    disp.plot(values_format='') # `values_format=''` suppresses scientific notation

    plt.show()


plot_tree(dt, max_depth=2, 
          fontsize=14, feature_names= X.columns.tolist(),
          class_names=['stayed','churned'], filled=True)

plt.show()

#Hyperparameter tuning for decision tree (gridsearchCV)
tree_para = {
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'sqrt', 'log2'],
    'criterion': ['gini', 'entropy']
}

scoring = ['accuracy', 'precision', 'recall', 'f1']
clf = GridSearchCV(tuned_dt,
tree_para,
scoring = scoring,
cv=5,
refit="f1")

clf.fit(X_train, y_train)

clf.best_estimator_

def make_results(model_name, model_object):
    '''
    Accepts as arguments a model name (your choice - string) and
    a fit GridSearchCV model object.
    Returns a pandas df with the F1, recall, precision, and accuracy scores
    for the model with the best mean F1 score across all validation folds.
    '''
    # Get all the results from the CV and put them in a df
    cv_results = pd.DataFrame(model_object.cv_results_)
    # Isolate the row of the df with the max(mean f1 score)
    best_estimator_results = cv_results.iloc[cv_results['mean_test_f1'].idxmax(), :]

    # Extract accuracy, precision, recall, and f1 score from that row
    f1 = best_estimator_results.mean_test_f1
    recall = best_estimator_results.mean_test_recall
    precision = best_estimator_results.mean_test_precision
    accuracy = best_estimator_results.mean_test_accuracy
    
    # Create table of results
    table = pd.DataFrame()
    table = table.append({'Model': model_name,
    'F1': f1,
    'Recall': recall,
    'Precision': precision,
    'Accuracy': accuracy
    },
    ignore_index=True
    )
    return table

result_table = make_results("Tuned Decision Tree", clf)
result_table

#Random Forest Model and Cross-Validation

rf = RandomForestClassifier(random_state = 21)
cv_params = {'max_depth': [2,3,4,5, None],
            'min_samples_leaf': [1,2,3],
            'min_samples_split': [2,3,4],
            'max_features': [2,3,4],
            'n_estimators': [75, 100, 125, 150]
}
scoring = ['accuracy', 'precision', 'recall', 'f1']

rf_cv = GridSearchCV(rf, cv_params, scoring=scoring, cv=5, refit='f1')

rf_cv.fit(X_train, y_train)

rf_cv_results = make_results('Random Forest', rf_cv)
rf_cv_results


results = pd.concat([rd_cv_results, result_table])
results

#Cross Validation with separate Validation set
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, 
                                            test_size=0.2,
                                            stratify=y_train, 
                                            random_state=21)

split_index = [0 if x in X_val.index else -1 for x in X_train.index]

custom_split = PredefinedSplit(split_index)

rf_val = GridSearchCV(rf, cv_params, scoring=scoring, cv=custom_split, refit='f1')
rf_val.fit(X_train, y_train)

rf_val_results = make_results('Random Forest Validated', rf_val)

results = pd.concat([rf_val_results, results])

# Sort master results by F1 score in descending order
results.sort_values(by=['F1'], ascending=False)

#XGBoost Model
xgb = XGBClassifier(objective='binary:logistic', random_state=21)

cv_params = {'max_depth': [4,5,6,7,8],
            'min_child_weight': [1,2,3,4,5],
            'learning_rate': [0.1, 0.2, 0.3],
            'n_estimators': [75, 100, 125]
}

scoring = ['accuracy', 'precision', 'recall', 'f1']

xgb_cv = GridSearchCV(xgb, cv_params, scoring=scoring, cv=5, refit='f1')

xgb_cv.fit(X_train, y_train)

xgb_cv_results = make_results('XGBoost CV', xgb_cv)

results = pd.concat([xgb_cv_results, results]).sort_values(by=['F1'], ascending=False)

#Confusion Matrix for Champion Model (Random Forest Validated)
confusion_matrix_plot(rf_val, X_test, y_test)

#Feature Importance Plot
importances = rf_cv.best_estimator_.feature_importances_
rf_importances = pd.Series(importances, index=X_test.columns)

# Sort values in descending order
rf_importances_sorted = rf_importances.sort_values(ascending=False)

fig, ax = plt.subplots()
rf_importances_sorted.plot.bar(ax=ax)
ax.set_title('Feature importances')
ax.set_ylabel('Mean decrease in impurity')

# Rotate x-labels by 45 degrees
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

fig.tight_layout()


