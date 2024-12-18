import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data_raw = pd.read_csv("C:/Users/Ildar Shakirzianov/Downloads/HistoricalData_1726401649896.csv")
number_of_days = 504

data_raw = data_raw.head(number_of_days)

# Calculate the main metrics

data = data_raw[['Date', 'Close/Last']]
data.rename(columns= {'Close/Last': 'Price'}, inplace = True)
data['Price'] = data['Price'].str.replace('$', '') #remove the $ sign
data['Price'] = data['Price'].astype(float) #make strings into float
data['logreturn'] = np.log(data['Price'] / data['Price'].shift(1)) #calcualte logreturns
data = data.drop(data.index[0])

print(data.head())

# Geometric Brownian Motion
sigma = data['logreturn'].std() #calcualte std
mu = data['logreturn'].mean()
S_0 = data.iloc[0,1]
T = 1
N = 252
M = 1000 # number of simulations
dt = 1 # since daily metrics are used

Z = np.random.normal(0,1, (M, N))
S = np.zeros((M, N+1))

S[:, 0] = S_0

for t in range(1, N+1):
    S[:, t] = S[:, t-1] * np.exp((mu - 0.5 * sigma**2)*dt + sigma*np.sqrt(dt) * Z[:, t-1])

S_T = S[:, -1]
mean_price = np.mean(S_T)
median_price = np.median(S_T)

percentile = 5
percentile_price = np.percentile(S_T, percentile)
Value_at_Risk = percentile_price - S_0
Value_at_Risk_return = (percentile_price - S_0)/S_0

# Print out the results
print(Value_at_Risk)
print("The worst return loss at " + str(100-percentile) + "% Confidence Level is " + str(round(Value_at_Risk_return,2)*100) + "%")
print("The maximum loss at " + str(100-percentile) + "% Confidence Level is " + str(Value_at_Risk))
print("the mean and median prices after " + str(T) + " years are " + str(round(mean_price, 2)) + 
      " and " + str(round(median_price, 2)) + " respectively")
