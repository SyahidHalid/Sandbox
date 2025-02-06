import pandas as pd
import numpy as np

Location = r"C:\Users\syahidhalid\OneDrive - Export-Import Bank of Malaysia (EXIM Bank)\Exim _ Syahid\Finance & Accounting\LDB\Data Source\202403\03. MARCH\Finance"
File = "Debtors Listing and Customer Balance Report as at March 2024-v4"
Sheet = "Other Debtors  Mar 2024"

df_ori = pd.read_excel(str(Location)+"\\"+str(File)+".xlsx", sheet_name=Sheet, header=0)

df_ori.head(20)