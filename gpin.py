import gpin_model
import numpy as np
import multiprocessing

def calculate_gpin(a, eta):
    return a * eta / (1 + eta)

data_file_path = "PINBSN24SJun.txt"

data = np.genfromtxt(
    data_file_path,
    dtype = None,
    encoding = 'utf-8',
    delimiter = ' ',
    names = ['company_code', 'col2', 'col3', 'date', 'buy', 'sell']
)

all_stock_codes_file_path = "TSE_stock_list.txt"

all_stock_codes = np.genfromtxt(
   all_stock_codes_file_path,
   dtype = None,
   encoding = 'utf-8',
   delimiter = ' ',
   names = ['company_code'] 
)

year_range = range(2000, 2024)
stock_codes = all_stock_codes['company_code']
output_file = "GPIN_results.txt"

def process_stock(stock_code):
    stock_data = data[(data['company_code'] == stock_code) & (data['col2'] == 11)]

    if len(stock_data )== 0:
        print(f"No data for stock code {stock_code}. Skipping")
        return

    years = np.array([int(str(d)[:4]) for d in stock_data['date']])
    months = np.array([int(str(d)[4:6]) for d in stock_data['date']])

    results_list = []
    for year in year_range:
        year_mask = ((years == year) & (months >= 7)) | ((years == year + 1) & (months <= 6))
        yearly_data = stock_data[year_mask]
        buys = yearly_data['buy']
        sells = yearly_data['sell']
                
        #Count days with non-zero trades
        buy_trading_days = np.sum(buys > 0)
        sell_trading_days = np.sum(sells > 0)

        if (buy_trading_days < 120) | (sell_trading_days < 120):
            print(f"Stock {stock_code}: less than 120 trading days in buy or sell in financial {year}. Skipped") 
            continue
        
        try:
            results = gpin_model.fit(buys, sells)
            final_gpin = calculate_gpin(results['a'], results['eta'])
            formatted_values = " ".join(f"{value:.5f}" for value in results.values())
            results_list.append(f"{stock_code} {year} {formatted_values} {final_gpin:.5f}")
            print(f"Processed successfully stock {stock_code} in financial year {year}")
        except Exception as e:
            print(f"Error in processing stock {stock_code} in financial year {year}")
        # file.write(f"{stock_code} {year} ")
        # formatted_values = " ".join(
        #     f"{value:.5f}" for value in results.values())
        # file.write(f"{formatted_values} {final_gpin:5f}\n")

    return results_list

if __name__ == "__main__":
    num_workers = multiprocessing.cpu_count()
    print(f"Using {num_workers} CPU cores")

    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(process_stock, stock_codes)

    with open(output_file, 'a') as file:
        for result in results:
            if result:
                file.write("\n".join(result) + "\n")
    
    print(f"All results written to {output_file}")
            