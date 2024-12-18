import math
import numpy as np
import matplotlib.pyplot as plt

mu = 0
T = 1
S0 = 35
sigma = 0.4
X = 35
r = 0.01
q = 0.03

def norm_dist_cdf(x): 
    return 0.5 * (1 + math.erf(x / np.sqrt(2))) #erf(x) is an integral of the standard normal distribution
  
d1 = (np.log(S0 / X) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
d2 = d1 - sigma * np.sqrt(T)

BS_option = X * np.exp(-r * T) * norm_dist_cdf(-d2) - S0* np.exp(-q * T) * norm_dist_cdf(-d1)

N_values = range(1, 300, 1)

option_payoffs = []


for N in N_values: #iterating over a range of N`s
    dt = T/N
    u = np.exp(sigma*np.sqrt(T/N))
    d = np.exp(-sigma*np.sqrt(T/N))
    p = (np.exp((r-q)*T/N)-d)/(u-d)
    payoff = 0

    for i in range(N+1):
    #Calculating the probability of each outcome using combinations formula
        probability = math.factorial(N) / (math.factorial(N-i)*math.factorial(i)) * p**i * (1-p)**(N-i)
        #Calculating the stock price at T
        S = S0 * (u)**i * (d)**(N-i)
        #Calculating the probability adjusted option payoff
        payoff += max(X-S, 0) * probability

    option_payoffs.append(payoff * np.exp(-r*T))

#Plotting a graph
plt.plot(N_values, option_payoffs, label = "Binomial Tree Option Pricing")
plt.xlabel('N')
plt.ylabel('Option Value')
plt.axhline(y=BS_option, color='r', label = "Black-Scholes Option Pricing")
plt.legend(loc='upper right') 
plt.show()

