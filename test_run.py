import gpin_model
import eo_model
import numpy as np
import random

data_file_path = 'PINBSN24SJun.txt'

data = np.genfromtxt(
    data_file_path,
    dtype = None,
    encoding = 'utf-8',
    delimiter= ' ',
    names = ['company_code', 'col2', 'col3', 'date', 'buy', 'sell']
)

unique_stock_codes = np.unique(data['company_code'])
random_stock_codes = random.sample(list(unique_stock_codes), 10)

year_range = range(2001, 2024)
results = {}

for stock_code in random_stock_codes:
    stock_data = data[(data['company_code'] == stock_code) & (data['col2'] == 11)]

    if len(stock_data) == 0:
        print(f"No data for stock code {stock_code}. Skipping")
        continue

    years = np.array([int(str(d)[:4]) for d in stock_data['date']])
    results = {}

    for year in year_range:
        year_mask = years == year
        yearly_data = stock_data[year_mask]

        buys = yearly_data['buy']
        sells = yearly_data['sell']
        results[year] = eo_model.fit(buys, sells)
    
    file_name = f"{stock_code}_pin_results.txt"
    with open(file_name, "w") as file:
        file.write(f"Results for stock code {stock_code}:\n")
        for year, pin_result in results.items():
            file.write(f"{year}: {pin_result}\n")
    
    print(f"Results for stock code {stock_code} saved to {file_name}.")
