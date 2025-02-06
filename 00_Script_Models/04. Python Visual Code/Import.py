import streamlit
import pandas as pd
import numpy as np
import openpyxl
import matplotlib.pyplot as plt

x = [24,25,26]
y = [23,24,25]

plt.plot(x,y)
plt.show()

df = pd.read_excel('Compiled.xlsx', sheet_name='Sheet1')
df1 = pd.read_excel(r'C:\Users\syahidhalid\OneDrive - Export-Import Bank of Malaysia (EXIM Bank)\Desktop\Work\Customer Churn Prediction\Compiled.xlsx', sheet_name='Sheet1')


def transform(df):
    #df.head(2)
    #df.tail(2)
    #df.dtypes
    #df.describe()
    df['CIF Number1'] = df['CIF Number'][:4]
    return df

data_ransformed = transform(df)

data_ransformed



