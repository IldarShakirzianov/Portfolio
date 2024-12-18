import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data_raw = pd.read_csv("C:/Users/Ildar Shakirzianov/Documents/SP 500 Stock Prices 2014-2017.csv")
#print(data.head())

#Adjusting the date
data_raw['date'] = pd.to_datetime(data_raw['date'])
data = data_raw.sort_values(by = ['symbol', 'date'])


data["returns"] = 0
columns = list(data.columns)
columns.insert(3, columns.pop(-1))
data = data[columns]

data['returns'] = data_raw.groupby('symbol')['close'].pct_change()
data = data[data['returns'].notnull()] 


# Plotting stock prices
"""plt.figure(figsize=(14,7))
for symbol in data['symbol'].unique():
    subset = data[data['symbol'] == symbol]
    plt.plot(subset['date'], subset['close'], label = symbol)

#plt.show()
"""
#Average returns
avg_returns_daily =  data.groupby('symbol')['returns'].mean()
avg_returns_daily = avg_returns_daily.reset_index()
avg_returns_daily.rename(columns={'returns': 'avg_return'}, inplace=True)
data = data.merge(avg_returns_daily, on = 'symbol')



#Volatility
volatility = data.groupby('symbol')['returns'].std()
volatility = volatility.reset_index()
volatility.rename(columns = {'returns':'volatility'}, inplace=True)
data = data.merge(volatility, on = 'symbol')

#Summary table

data_summary = data[['symbol']]
data_summary = data_summary.drop_duplicates()

data_summary = data_summary.merge(volatility, on = 'symbol')
data_summary = data_summary.merge(avg_returns_daily, on = 'symbol')

data_summary['avg_return'] = data_summary['avg_return']*252
data_summary['volatility'] = data_summary['volatility']*(252**0.5)


# Daily VaR Historical Simulation for 1 asset

data = data.sort_values(by = ['symbol', 'returns'])
confidence_level = 0.95
number_of_obs = len(data[data['symbol'] == 'A'])
var_return_pos = np.floor(number_of_obs * (1-confidence_level))


# Obtain worst returns based on the confidence level
data['index'] = data.groupby('symbol').cumcount() + 1
data_var = data[data['index']== var_return_pos]
# print(data_var.head())

# Calculate daily VaR for a portfolio of 2 assets
def portfolio_value_at_risk(stock_1, stock_2, weight_1, confidence_level):
    weight_2 = 1- weight_1
    data_stock_1 = data[data['symbol'].isin([stock_1])]
    data_stock_2 = data[data['symbol'].isin([stock_2])]

    data_portfolio = pd.merge(data_stock_1, data_stock_2, on = 'date')
    data_portfolio.rename(columns={'returns_x': 'returns_stock_1'}, inplace=True)
    data_portfolio.rename(columns={'returns_y': 'returns_stock_2'}, inplace=True)
    data_portfolio = data_portfolio[['returns_stock_1', 'returns_stock_2']]
    data_portfolio['portfolio_return'] = data_portfolio['returns_stock_1'] * weight_1\
        + data_portfolio['returns_stock_2'] * weight_2
    data_portfolio['index'] = range(1, len(data_portfolio)+1)
    var_return_index = np.floor(len(data_portfolio)+1*(1-confidence_level))
    portfolio_var = data_portfolio['portfolio_return'][data_portfolio['index'] == var_return_index]
    print(str(round(portfolio_var.iloc[0]*100, 2)) + "%")
    print("The worst return loss at " + str(confidence_level) + "% Confidence Level is " + str(round(portfolio_var.iloc[0]*100, 2)) + "%")

portfolio_value_at_risk('NVDA', 'AAPL', 0.4, 0.95)

