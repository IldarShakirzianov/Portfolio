
import numpy as np
import math

mu = 0
T = 1
S0 = 45
sigma = 0.3
X = 40
r = 0.01
q = 0.05
def norm_dist_cdf(x): 
    return 0.5 * (1 + math.erf(x / np.sqrt(2))) #erf(x) is an integral of the standard normal distribution
  
d1 = (np.log(S0 / X) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
d2 = d1 - sigma * np.sqrt(T)

#put_option_price = X * np.exp(-r * T) * norm_dist_cdf(-d2) - S0* np.exp(-q * T) * norm_dist_cdf(-d1)
call_option_price = S0 * np.exp(-q * T) * norm_dist_cdf(d1) - X * np.exp(-r * T) * norm_dist_cdf(d2)

#print(put_option_price)
print(call_option_price)

#print(-norm_dist_cdf(-d1)) #delta of put option
#print(norm_dist_cdf(d1)) #delta of call option