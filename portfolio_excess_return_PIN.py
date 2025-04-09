import numpy as np
import pandas as pd


PIN_data = np.genfromtxt(
    'PIN_results.txt',
    dtype = None,
    encoding = 'utf-8',
    delimiter = ' ',
    names = ['code', 'year', 'pin_a', 'pin_d', 'pin_eb', 'pin_es', 'pin_u' ,'pin_f', 'pin_rc', 'PIN']
)

# GPIN_data = np.genfromtxt(
#     'GPIN_results.txt',
#     dtype = None,
#     encoding = 'utf-8',
#     delimiter = ' ',
#     names = ['code', 'year', 'gpin_a', 'gpin_p', 'gpin_eta', 'gpin_r', 'gpin_d', 'gpin_th', 'gpin_f', 'gpin_rc', 'GPIN']
# )

SPM_data = np.genfromtxt(
    'SPMData.txt',
    dtype = None,
    encoding = 'utf-8',
    delimiter = ' ',
    names = ['code', 'year', 'LnMV', 'BPR', 'NI', 'CE', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
)


df_PIN_data = pd.DataFrame(PIN_data)
df_SPM_data = pd.DataFrame(SPM_data)
df_SPM_data["avg_return"] = df_SPM_data.loc[:, "Jul":"Jun"].mean(axis=1)

merged_data_PIN = df_SPM_data.merge(df_PIN_data, on = ['code', 'year'], how = 'inner')

merged_data_PIN_prev = merged_data_PIN[["code", "year", "LnMV", "PIN"]].copy()
merged_data_PIN_prev["year"] += 1
merged_data_PIN_prev['Size_group'] = pd.qcut(merged_data_PIN_prev['LnMV'], q = 5, labels = ["Small", "2", "3", "4", "Large"])
merged_data_PIN_prev['PIN_group'] = pd.qcut(merged_data_PIN_prev['PIN'], q = 3, labels = ["Low", "Medium", "High"])

merged_data_PIN = merged_data_PIN.merge(merged_data_PIN_prev[["code", "year", "Size_group", "PIN_group"]], on = ["code", "year"], how = "inner")

summary_PIN = merged_data_PIN.groupby(["Size_group", "PIN_group"]).agg(
    Excess_Returns = ("avg_return", "mean"), 
    Num_Stocks = ("code", "count"),
    Avg_PIN = ("PIN", "mean")
).reset_index()

pannel_A = summary_PIN.pivot(index = "Size_group", columns = "PIN_group", values = "Excess_Returns")
pannel_B = summary_PIN.pivot(index = "Size_group", columns = "PIN_group", values = "Num_Stocks")
pannel_C = summary_PIN.pivot(index = "Size_group", columns = "PIN_group", values = "Avg_PIN")

print("\nPanel A: Excess Returns")
print(pannel_A.to_string())

print("\nPannel B: Number of Stocks")
print(pannel_B.to_string())

print("\nPannel C: PIN")
print(pannel_C.to_string())
