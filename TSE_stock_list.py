import numpy as np
import pandas as pd


all_codes = np.genfromtxt(
    'list_of_stocks.txt',
    dtype = None,
    encoding = 'utf-8',
    delimiter = ' ',
    names = ['code']
)

TSE_codes = np.genfromtxt(
    'TSE_stock_list.txt',
    dtype = None,
    encoding = 'utf-8',
    delimiter = ' ',
    names = ['code']
)

SPMData = np.genfromtxt(
    'SPMData.txt',
    dtype = None,
    encoding = 'utf-8',
    delimiter = ' ',
    names = ['code', 'year', 'LnMV', 'BPR', 'NI', 'CE', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
)

all_code_set = set(all_codes['code'])
TSE_code_set = set(TSE_codes['code'])
SPM_code_set = set(SPMData['code'])

# print("In TSE but not in all code")
# in_TSE_not_in_all_code = TSE_code_set - all_code_set
# for code in in_TSE_not_in_all_code:
#     print(code)


# print("In SPM but not in TSE")
# in_SPM_not_in_TSE_code = SPM_code_set - TSE_code_set
# for code in in_SPM_not_in_TSE_code:
#     print(code)

print("In TSE but not in SPM")
in_TSE_not_in_SPM_code = TSE_code_set - SPM_code_set
in_TSE_not_in_SPM_code = sorted(in_TSE_not_in_SPM_code)
