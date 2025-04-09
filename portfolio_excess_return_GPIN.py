import numpy as np
import pandas as pd


GPIN_data = np.genfromtxt(
    'GPIN_results.txt',
    dtype = None,
    encoding = 'utf-8',
    delimiter = ' ',
    names = ['code', 'year', 'gpin_a', 'gpin_p', 'gpin_eta', 'gpin_r', 'gpin_d', 'gpin_th', 'gpin_f', 'gpin_rc', 'GPIN']
)

SPM_data = np.genfromtxt(
    'SPMData.txt',
    dtype = None,
    encoding = 'utf-8',
    delimiter = ' ',
    names = ['code', 'year', 'LnMV', 'BPR', 'NI', 'CE', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
)


df_GPIN_data = pd.DataFrame(GPIN_data)
df_SPM_data = pd.DataFrame(SPM_data)
df_SPM_data["avg_return"] = df_SPM_data.loc[:, "Jul":"Jun"].mean(axis=1)

merged_data_GPIN = df_SPM_data.merge(df_GPIN_data, on = ['code', 'year'], how = 'inner')

merged_data_GPIN_prev = merged_data_GPIN[["code", "year", "LnMV", "GPIN"]].copy()
merged_data_GPIN_prev["year"] += 1
merged_data_GPIN_prev['Size_group'] = pd.qcut(merged_data_GPIN_prev['LnMV'], q = 5, labels = ["Small", "2", "3", "4", "Large"])
merged_data_GPIN_prev['GPIN_group'] = pd.qcut(merged_data_GPIN_prev['GPIN'], q = 3, labels = ["Low", "Medium", "High"])

merged_data_GPIN = merged_data_GPIN.merge(merged_data_GPIN_prev[["code", "year", "Size_group", "GPIN_group"]], on = ["code", "year"], how = "inner")

summary_GPIN = merged_data_GPIN.groupby(["Size_group", "GPIN_group"]).agg(
    Excess_Returns = ("avg_return", "mean"), 
    Num_Stocks = ("code", "count"),
    Avg_GPIN = ("GPIN", "mean")
).reset_index()

pannel_A = summary_GPIN.pivot(index = "Size_group", columns = "GPIN_group", values = "Excess_Returns")
pannel_B = summary_GPIN.pivot(index = "Size_group", columns = "GPIN_group", values = "Num_Stocks")
pannel_C = summary_GPIN.pivot(index = "Size_group", columns = "GPIN_group", values = "Avg_GPIN")

print("\nPanel A: Excess Returns")
print(pannel_A.to_string())

print("\nPannel B: Number of Stocks")
print(pannel_B.to_string())

print("\nPannel C: GPIN")
print(pannel_C.to_string())
