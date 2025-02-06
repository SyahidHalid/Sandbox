import pandas as pd #data manipulation and analysis
import numpy as np #linear Algebra
import matplotlib.pyplot as plt #data visualisation
import seaborn as sns # data visualisation
import warnings #ignore warnings
warnings.filterwarnings('ignore')

pd.set_option("display.max_columns", None) 
pd.set_option("display.max_colwidth", 1000) #huruf dlm column
pd.set_option("display.max_rows", 100)
pd.set_option("display.precision", 2) #2 titik perpuluhan

df_ori = pd.read_excel(r"C:\Users\syahidhalid\OneDrive - Export-Import Bank of Malaysia (EXIM Bank)\Exim _ Syahid\Analytics\Loan Growth Model\2024\Loan Growth - Working.xlsx", sheet_name='Internal')

df_ori

df = df_ori.iloc[np.where(df_ori['Position as At']!='2022-12-30')][['Gross Financing (Growth)','Asia US Emerging Markets','MY GDP','US GDP','Non Trade','Trade','Pending Disbursement','Fully Disburse (Exclude Impaired)','Cancelled']]

df

#EDA on Growth

df.describe() #for non object

################################################################
sns.pairplot(df,
                 markers="+",
                diag_kind='kde',
                kind='reg',
                plot_kws={'line_kws':{'color':'#aec6cf'},'scatter_kws':{'alpha':0.7,'color':'red'}},corner=True);

################################################################
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(),annot=True,square=True)
plt.title('Correlation Between Variables', size=18);
plt.xticks(size=13)
plt.yticks(size=13)
plt.show()


################################################################
fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(df_ori["Position as At"], df_ori["Total Outstanding Amount"], color ='maroon', 
        width = 0.6)
 
plt.xlabel("Month")
plt.ylabel("Gross Financing")
plt.title("Growth")

plt.show()


################################################################
from statsmodels.tsa.stattools import grangercausalitytests

maxlag = 3
def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):    
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table 
    are the P-Values. P-Values lesser than the significance level (0.05), implies 
    the Null Hypothesis that the coefficients of the corresponding past values is 
    zero, that is, the X does not cause Y can be rejected.

    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    """
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df

A = df[["Gross Financing (Growth)",'Asia US Emerging Markets']]
B = df[["Gross Financing (Growth)",'MY GDP']]
C = df[["Gross Financing (Growth)",'US GDP']]
D = df[["Gross Financing (Growth)",'Non Trade']]
E = df[["Gross Financing (Growth)",'Trade']]
F = df[["Gross Financing (Growth)",'Pending Disbursement']]
G = df[["Gross Financing (Growth)",'Fully Disburse (Exclude Impaired)']]
H = df[["Gross Financing (Growth)",'Cancelled']]

grangers_causation_matrix(A, variables = A.columns)
grangers_causation_matrix(B, variables = B.columns)
grangers_causation_matrix(C, variables = C.columns)
grangers_causation_matrix(D, variables = D.columns)
grangers_causation_matrix(E, variables = E.columns)
grangers_causation_matrix(F, variables = F.columns)
grangers_causation_matrix(G, variables = G.columns)
grangers_causation_matrix(H, variables = H.columns)


################################################################
from sklearn import linear_model

X = df[['Asia US Emerging Markets','MY GDP','US GDP','Non Trade','Trade','Pending Disbursement','Fully Disburse (Exclude Impaired)','Cancelled']]
y = df["Gross Financing (Growth)"]

regr = linear_model.LinearRegression()
regr.fit(X, y)

#regr.coef_

#predictedCO2 = regr.predict([[2300, 1300]])

#display regression coefficients and R-squared value of model
print(regr.intercept_)
print(regr.coef_)
print(regr.score(X, y))

#Using this output, we can write the equation for the fitted regression model:

#y = -38812985.09142425  -1340658.31852561x1 +6715055.10983107x2 +3804337.33990844x3 + 6826.30448118x4 -134233.1218451x5

#We can also see that the R2 value of the model is 0.93. 
#This means that 93% of the variation in the response variable can be explained by the two predictor variables in the model.
#Although this output is useful, we still don’t know the overall F-statistic of the model, the p-values of the individual regression coefficients, and other useful metrics that can help us understand how well the model fits the dataset.

import statsmodels.api as sm


#add constant to predictor variables
x = sm.add_constant(X)

#fit linear regression model
model = sm.OLS(y, x).fit()

#view model summary
print(model.summary())





intercept = 0.1221
x1 = 14.0824*df["Asia US Emerging Markets"]
x2 = 1.8885*df["MY GDP"]
x3 = 18.4037*df["US GDP"]
x4 = 0.0002*df["Non Trade"]
x5 = 0.0078*df["Trade"]
x6 = -0.0061*df["Pending Disbursement"]
x7 = 0.0079*df["Fully Disburse (Exclude Impaired)"]
x8 = -0.0227*df["Cancelled"]

df["Growth Predicted"] = intercept + x1 + x2 + x3 + x4 + x5 + x6  + x7 + x8

df[["Gross Financing (Growth)","Growth Predicted"]]

#Graph

plt.figure(figsize=(10,4))
plt.plot(df["Gross Financing (Growth)"], color='b')
plt.plot(df["Growth Predicted"], color='r')

plt.legend(['Growth','Predicted'], fontsize=8)

plt.show()