import gpin_model
import eo_model
import numpy as np

data_file_path = 'PINBSN24SJun.txt'

data = np.genfromtxt(
    data_file_path,
    dtype = None,
    encoding = 'utf-8',
    delimiter= ' ',
    names = ['company_code', 'col2', 'col3', 'date', 'buy', 'sell']
)

#test run for toyota
toyota_data = data[data['company_code'] == 7203]

dates = toyota_data['date']
buys = toyota_data['buy']
sells = toyota_data['sell']

toyota_array = np.column_stack((dates, buys, sells))

print(gpin_model.fit(buys, sells))
