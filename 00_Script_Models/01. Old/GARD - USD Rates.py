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

df_ori = pd.read_excel(r"C:\Users\syahidhalid\OneDrive - Export-Import Bank of Malaysia (EXIM Bank)\Exim _ Syahid\Analytics\Loan Growth Model\2024\Loan Growth - Working.xlsx", sheet_name='v3')

df_ori.head(1)

df_ori['YoY Growth'] = df_ori["Total Outstanding Amount"].pct_change(periods=1) * 100