
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris,load_breast_cancer,load_wine
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from scipy.stats import norm

data = load_wine()
x, y, column_names = data['data'], data['target'], data['feature_names']
x = pd.DataFrame(x, columns = column_names)

from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.33, random_state = 44)
#x.shape, x_train.shape, y_train.shape, x_val.shape, y_val.shape


# Create a Naive Bayes classifier
clf = GaussianNB()

# Train the classifier using the training data
clf.fit(x_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(x_val)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_val, y_pred)
print("Accuracy:", accuracy)


means = x_train.groupby(y_train).apply(np.mean)
stds = x_train.groupby(y_train).apply(np.std)

probs = x_train.groupby(y_train).apply(lambda x: len(x)/len(x_train))
probs

means

stds

#so P(class1) = 31.90%, P(class2) = 39.4%, P(class3) = 28.5%


y_pred = []

for elem in range(x_val.shape[0]):
    p = {}
    
    for c in np.unique(y_train):
        p[c] = probs.iloc[c]
        for index, param in enumerate(x_val.iloc[elem]):
            p[c]*= norm.pdf(param, stds.iloc[c, index], stds.iloc[c, index])
    y_pred.append(pd.Series(p).values.argmax())

print('Accuracy:', accuracy_score(y_val, y_pred))
#link


#notes. problem kat means sbb dia xjd dataset so ganti stds for index

#https://medium.com/@AbhiramiVS/bayes-theorem-with-python-ef2135d850f
