import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt

#warnings.filterwarnings('ignore')
pd.set_option("display.max_columns", None) 
pd.set_option("display.max_colwidth", 1000) #huruf dlm column
pd.set_option("display.max_rows", 100)
pd.set_option("display.precision", 2) #2 titik perpuluhan

Location = r"C:\\Users\\syahidhalid\\Syahid PC\\Analytics\\Loan Database"
ECR1 = "ECR 2023"
ECR2 = "ECR 2024"
LDB1 = "LBD 202312 - 202301 (CSP Sector)"
LDB2 = "LBD 202412 - 202401 (CSP Sector)"

#df_ori = pd.read_excel(r"C:\Users\syahidhalid\OneDrive - Export-Import Bank of Malaysia (EXIM Bank)\Exim _ Syahid\Analytics\Loan Growth Model\2024\Above SLA Criteria\TAT - working.xlsx", sheet_name='working')
ECR_1 = pd.read_excel(str(Location)+"\\"+str(ECR1)+".xlsx", sheet_name='Oct23', header=0)
ECR_2 = pd.read_excel(str(Location)+"\\"+str(ECR1)+".xlsx", sheet_name='Nov23', header=0)
ECR_3 = pd.read_excel(str(Location)+"\\"+str(ECR1)+".xlsx", sheet_name='Dec23', header=0)
ECR_4 = pd.read_excel(str(Location)+"\\"+str(ECR2)+".xlsx", sheet_name='Jan24', header=0)
ECR_5 = pd.read_excel(str(Location)+"\\"+str(ECR2)+".xlsx", sheet_name='Feb24', header=0)
ECR_6 = pd.read_excel(str(Location)+"\\"+str(ECR2)+".xlsx", sheet_name='Mar24', header=0)

ECR_1.shape
ECR_2.shape
ECR_3.shape
ECR_4.shape
ECR_5.shape
ECR_6.shape

ECR_1['Position'] = '31/10/2023'
ECR_2['Position'] = '30/11/2023'
ECR_3['Position'] = '31/12/2023'
ECR_4['Position'] = '31/01/2024'
ECR_5['Position'] = '29/02/2024'
ECR_6['Position'] = '31/03/2024'


LDB_1 = pd.read_excel(str(Location)+"\\"+str(LDB1)+".xlsx", sheet_name='Oct23', header=0)
LDB_2 = pd.read_excel(str(Location)+"\\"+str(LDB1)+".xlsx", sheet_name='Nov23', header=0)
LDB_3 = pd.read_excel(str(Location)+"\\"+str(LDB1)+".xlsx", sheet_name='Dec23', header=0)
LDB_4 = pd.read_excel(str(Location)+"\\"+str(LDB2)+".xlsx", sheet_name='Jan24', header=0)
LDB_5 = pd.read_excel(str(Location)+"\\"+str(LDB2)+".xlsx", sheet_name='Feb24', header=0)
LDB_6 = pd.read_excel(str(Location)+"\\"+str(LDB2)+".xlsx", sheet_name='Mar24', header=0)

LDB_1.shape
LDB_2.shape
LDB_3.shape
LDB_4.shape
LDB_5.shape
LDB_6.shape

#df_ori.head()

ECR_1.columns=ECR_2.columns=ECR_3.columns=ECR_4.columns=ECR_5.columns=ECR_6.columns
LDB_1.columns=LDB_2.columns=LDB_3.columns=LDB_4.columns=LDB_5.columns=LDB_6.columns


ECRFIN = pd.concat([ECR_1,ECR_2,ECR_3,ECR_4,ECR_5,ECR_6])

LDBFIN = pd.concat([LDB_1,LDB_2,LDB_3,LDB_4,LDB_5,LDB_6])

ECRFIN.to_csv(r"C:\Users\syahidhalid\Syahid PC\FAD - Analytics\\ECR_POWERBI.txt", index = False)
LDBFIN.to_csv(r"C:\Users\syahidhalid\Syahid PC\FAD - Analytics\\LDB_POWERBI.txt", index = False)
