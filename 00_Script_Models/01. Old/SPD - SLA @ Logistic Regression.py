import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
pd.set_option("display.max_columns", None) 
pd.set_option("display.max_colwidth", 1000) #huruf dlm column
pd.set_option("display.max_rows", 100)
pd.set_option("display.precision", 2) #2 titik perpuluhan

Location = r"C:\\Users\\syahidhalid\\OneDrive - Export-Import Bank of Malaysia (EXIM Bank)\\Exim _ Syahid\\Analytics\\Loan Growth Model\\2024\\Above SLA Criteria"
File = "LDB 202402"

#df_ori = pd.read_excel(r"C:\Users\syahidhalid\OneDrive - Export-Import Bank of Malaysia (EXIM Bank)\Exim _ Syahid\Analytics\Loan Growth Model\2024\Above SLA Criteria\TAT - working.xlsx", sheet_name='working')
df_ori = pd.read_excel(str(Location)+"\\"+str(File)+".xlsx", sheet_name='Feb24', header=9)

df_ori.head()


df_ori.loc[df_ori['Sum of Disbrusement (days)'] < 134, 'L_Disbrused'] = 1
df_ori.loc[df_ori['Sum of Disbrusement (days)'] >= 134, 'L_Disbrused'] = 0

df = df_ori[['Approval Authority','Corporate Status','Syndicated / Club Deal',
             'Nature of Account','Facility','Facility Currency','BNM Main Sector',
             'Country Exposure','L_Disbrused']]

#----------------------------------------------------------------------------------

df['Approval Authority'].value_counts()

df['Approval Authority'] = df['Approval Authority'].str.replace(" ", "")
df['Approval Authority'] = df['Approval Authority'].astype(str)

def L_Approval(AA):
    if AA=='BCC':
        return '1'
    elif AA == 'BOD': 
        return '2'
    elif AA == 'MCC': 
        return '3'
    elif AA == 'FIIC': 
        return '4'
    elif AA == 'OversightMCC': 
        return '5'
    elif AA == 'Oversight-MCC': 
        return '5'
    elif AA == 'EXCO': 
        return '6'
    elif AA == 'CBO': 
        return '7'
    else: 
        return '8'
df['L_Approval'] = df.apply(lambda x: L_Approval(x['Approval Authority']), axis=1)

#
df['Corporate Status'].value_counts()

df['Corporate Status'] = df['Corporate Status'].str.replace(" ", "")
df['Corporate Status'] = df['Corporate Status'].astype(str)

def L_Approval(AA):
    if AA=='Non-SME':
        return '1'
    elif AA == 'SME-Medium ': 
        return '2'
    elif AA == 'SME-Small ': 
        return '3'
    else: 
        return '4'
df['L_Corporate'] = df.apply(lambda x: L_Approval(x['Corporate Status']), axis=1)

#
df['Syndicated / Club Deal'].value_counts()

df['Syndicated / Club Deal'] = df['Syndicated / Club Deal'].str.replace(" ", "")
df['Syndicated / Club Deal'] = df['Syndicated / Club Deal'].astype(str)

def L_Approval(AA):
    if AA=='NotApplicable ':
        return '1'
    elif AA == 'Syndicated ': 
        return '2'
    else: 
        return '3'
df['L_Syndicated'] = df.apply(lambda x: L_Approval(x['Syndicated / Club Deal']), axis=1)

#
df['Nature of Account'].value_counts()

df['Nature of Account'] = df['Nature of Account'].str.replace(" ", "")
df['Nature of Account'] = df['Nature of Account'].astype(str)

def L_Approval(AA):
    if AA=='NonTrade':
        return '1'
    elif AA == 'Trade': 
        return '2'
    elif AA == 'Trade-Guarantee': 
        return '3'
    else: 
        return '4'
df['L_Nature'] = df.apply(lambda x: L_Approval(x['Nature of Account']), axis=1)

df['L_Nature'].value_counts()


#
df['Facility'].value_counts()

df['Facility'] = df['Facility'].str.replace(" ", "")
df['Facility'] = df['Facility'].astype(str)

def L_Approval(AA):
    if AA=='TermFinancing-i':
        return '1'
    elif AA == 'SupplierFinancing-i': 
        return '2'
    elif AA == 'OverseasProjectFinancing': 
        return '3'
    elif AA == 'LetterofCredit/TrustReceipt-i': 
        return '4'
    elif AA == 'OverseasProjectFinancing-i': 
        return '5'
    elif AA == 'BankGuarantee': 
        return '6'
    elif AA == 'RevolvingCredit-i': 
        return '7'
    elif AA == 'LetterofCredit/TrustReceipt': 
        return '8'
    elif AA == 'BankGuarantee-i': 
        return '9'
    elif AA == 'VendorFinancingScheme-i': 
        return '10'
    elif AA == 'ContractFinancingOverseas-i': 
        return '11'
    elif AA == 'VendorFinancingScheme': 
        return '12'
    elif AA == 'OverseasInvestmentFinancing': 
        return '13'
    else: 
        return '14'
df['L_Facility'] = df.apply(lambda x: L_Approval(x['Facility']), axis=1)

df['L_Facility'].value_counts()

#----------------------------------------------------------------------------------

L_df = df[['L_Approval',
'L_Corporate',
'L_Syndicated',
'L_Nature',
'L_Facility',
'L_Disbrused']]

#define the predictor variables and the response variable
X = L_df[['L_Approval',
'L_Corporate',
'L_Syndicated',
'L_Nature',
'L_Facility']]
y = L_df['L_Disbrused']

#split the dataset into training (70%) and testing (30%) sets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0) 

#instantiate the model
log_regression = LogisticRegression()

#fit the model using the training data
log_regression.fit(X_train,y_train)

print(log_regression.intercept_)
print(log_regression.coef_)
print(log_regression.score(X, y))


import statsmodels.api as sm

X['L_Approval']  = X['L_Approval'].astype(int)
X['L_Corporate']  = X['L_Corporate'].astype(int)
X['L_Syndicated']  = X['L_Syndicated'].astype(int)
X['L_Nature']  = X['L_Nature'].astype(int)
X['L_Facility']  = X['L_Facility'].astype(int)

x = sm.add_constant(X)

#fit linear regression model
model = sm.Logit(y, x)

result = model.fit(method='newton')

#view model summary
print(result.summary())
result.summary()


# Modelling
from termcolor import colored as cl
lr = LogisticRegression(C = 0.1, solver = 'liblinear')
lr.fit(X_train,y_train)

print(cl(lr, attrs = ['bold']))


# Predictions

yhat = lr.predict(X_test)
yhat_prob = lr.predict_proba(X_test)

print(cl('yhat samples : ', attrs = ['bold']), yhat[:10])
print(cl('yhat_prob samples : ', attrs = ['bold']), yhat_prob[:10])

#model valuation
# 1. Jaccard Index
from sklearn.metrics import jaccard_score as jss
print(cl('Jaccard Similarity Score of our model is {}'.format(jss(y_test, yhat).round(2)), attrs = ['bold']))


# 2. Precision Score
from sklearn.metrics import precision_score # evaluation metric
print(cl('Precision Score of our model is {}'.format(precision_score(y_test, yhat).round(2)), attrs = ['bold']))


# 3. Log loss
from sklearn.metrics import log_loss # evaluation metric
print(cl('Log Loss of our model is {}'.format(log_loss(y_test, yhat)), attrs = ['bold']))

# 4. Classificaton report
from sklearn.metrics import classification_report # evaluation metric
print(cl(classification_report(y_test, yhat), attrs = ['bold']))


# 5. Confusion matrix
from sklearn.metrics import confusion_matrix # evaluation metric
import itertools # construct specialized tools
def plot_confusion_matrix(cm, classes,normalize = False, title = 'Confusion matrix', cmap = plt.cm.Blues):
    
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title, fontsize = 22)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45, fontsize = 13)
    plt.yticks(tick_marks, classes, fontsize = 13)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment = 'center',
                 fontsize = 15,
                 color = 'white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label', fontsize = 16)
    plt.xlabel('Predicted label', fontsize = 16)

# Compute confusion matrix

cnf_matrix = confusion_matrix(y_test, yhat, labels = [1,0])
np.set_printoptions(precision = 2)


# Plot non-normalized confusion matrix

plt.figure()
plot_confusion_matrix(cnf_matrix, classes = ['churn=1','churn=0'], normalize = False,  title = 'Confusion matrix')
plt.savefig('confusion_matrix.png')


#ROC
#define metrics
y_pred_proba = log_regression.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)

#create ROC curve
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()

#The AUC for this logistic regression model turns out to be 0.4508. Since this is close to 0.5, this confirms that the model does a poor job of classifying data.



#https://www.statology.org/plot-roc-curve-python/
#https://realpython.com/logistic-regression-python/
#https://medium.com/codex/machine-learning-logistic-regression-with-python-5ed4ded9d146#:~:text=Python%20Implementation%3A,-Output%3A&text=In%20logistic%20regression%2C%20the%20output,value%20between%200%20and%201.
#https://realpython.com/logistic-regression-python/